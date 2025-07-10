import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class LayerNorm(nn.Module):
    def __init__(self, norm_shape, eps=1e-6):
        print(f"[LayerNorm] - dim ({norm_shape}), eps ({eps})")
        super().__init__()

        self.weight = nn.Parameter(torch.ones(norm_shape))
        self.bias = nn.Parameter(torch.zeros(norm_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        print(f"[DSConv] - in ({in_channel}), out ({out_channel}), k ({kernel_size})")
        super().__init__()

        self.proj = nn.Conv2d(in_channel, out_channel, 1)
        self.conv = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            padding=kernel_size // 2,
            groups=in_channel,
        )

    def forward(self, x):
        return self.proj(self.conv(x))


class WaveConv(nn.Module):
    def __init__(self, dim, hi_kernel, lo_kernel):
        assert dim % 4 == 0, "Channel must be divisible by 4"
        print(f"[WaveConv] - dim ({dim}), h_k ({hi_kernel}), l_k ({lo_kernel})")
        super().__init__()

        out_dim = dim // 4

        self.h_conv = nn.Conv2d(
            out_dim, out_dim, hi_kernel, padding=hi_kernel // 2, groups=out_dim
        )
        self.l_conv = nn.Conv2d(
            out_dim, out_dim, lo_kernel, padding=lo_kernel // 2, groups=out_dim
        )
        self.proj = nn.Conv2d(dim, out_dim, 1)
        self.linear = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.proj(x)

        h_feat = self.h_conv(x)
        l_feat = self.l_conv(x)

        hh_feat = self.h_conv(h_feat)
        hl_feat = self.l_conv(h_feat)
        lh_feat = self.h_conv(l_feat)
        ll_feat = self.l_conv(l_feat)

        x = self.linear(torch.cat([hh_feat, hl_feat, lh_feat, ll_feat], dim=1))
        return F.gelu(x)


class WavePool(nn.Module):
    def __init__(self, dim, hi_kernel, lo_kernel):
        print(f"[WavePool] - dim ({dim}), h_k ({hi_kernel}), l_k ({lo_kernel})")
        super().__init__()

        self.h1_conv = DSConv(dim, 2 * dim, hi_kernel)
        self.l1_conv = DSConv(dim, 2 * dim, lo_kernel)
        self.h2_conv = DSConv(2 * dim, 4 * dim, hi_kernel)
        self.l2_conv = DSConv(2 * dim, 4 * dim, lo_kernel)

        self.r_pool = nn.AvgPool2d((2, 2), (1, 2))
        self.c_pool = nn.AvgPool2d((2, 2), (2, 1))

        self.norm = nn.BatchNorm2d(8 * dim, 1e-6)

    def forward(self, x):
        h_feat = self.r_pool(self.h1_conv(x))
        l_feat = self.r_pool(self.l1_conv(x))

        hh_feat = self.c_pool(self.h2_conv(h_feat))
        hl_feat = self.c_pool(self.l2_conv(h_feat))
        lh_feat = self.c_pool(self.h2_conv(l_feat))
        ll_feat = self.c_pool(self.l2_conv(l_feat))

        x = torch.cat([hh_feat + ll_feat, hl_feat + lh_feat], dim=1)
        x = self.norm(F.gelu(x))

        return x


class WaveStem(nn.Module):
    def __init__(self, in_channel, out_channel, hi_kernel, lo_kernel):
        assert (hi_kernel - lo_kernel) % 2 == 0, (
            "Difference of kernel size must be even"
        )
        assert out_channel % 4 == 0, "Output Channel must be divisible by 4"
        print(
            f"[WaveStem] - in ({in_channel}), out ({out_channel}), h_k ({hi_kernel}), l_k ({lo_kernel})"
        )
        super().__init__()

        pad_h = (hi_kernel - lo_kernel) // 2
        out_dim = out_channel // 4

        self.proj = nn.Conv2d(in_channel, out_dim, 7, padding=3)
        self.l1_conv = nn.Conv2d(out_dim, out_dim, lo_kernel, stride=(1, 2))
        self.l2_conv = nn.Conv2d(out_dim, out_dim, lo_kernel, stride=(2, 1))

        self.h1_conv = nn.Conv2d(
            out_dim,
            out_dim,
            hi_kernel,
            stride=(1, 2),
            padding=(pad_h, pad_h),
        )
        self.h2_conv = nn.Conv2d(
            out_dim,
            out_dim,
            hi_kernel,
            stride=(2, 1),
            padding=(pad_h, pad_h),
        )

        self.linear = nn.Conv2d(out_channel, out_channel, 1)
        self.norm = nn.BatchNorm2d(out_channel, 1e-6)

    def forward(self, x):
        x = self.proj(x)

        h_feat = self.h1_conv(x)
        l_feat = self.l1_conv(x)

        hh_feat = self.h2_conv(h_feat)
        hl_feat = self.l2_conv(h_feat)
        lh_feat = self.h2_conv(l_feat)
        ll_feat = self.l2_conv(l_feat)

        x = self.linear(torch.cat([hh_feat, hl_feat, lh_feat, ll_feat], dim=1))
        x = self.norm(F.gelu(x))
        return x


class WaveModulation(nn.Module):
    def __init__(self, dim, hi_kernel, lo_kernel):
        print(f"[WaveMod] - dim ({dim}), h_k ({hi_kernel}), l_k ({lo_kernel})")
        super().__init__()

        self.a_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.conv = WaveConv(dim, hi_kernel, lo_kernel)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        a = self.conv(self.a_proj(x))
        x = self.proj(a * self.v_proj(x))
        return x


class WaveBlock(nn.Module):
    def __init__(self, dim, hi_kernel, lo_kernel, drop=0.0, pool=False):
        print(
            f"[WaveBlock] - dim ({dim}), h_k ({hi_kernel}), l_k ({lo_kernel}), drop ({drop}), pool ({pool})"
        )
        super().__init__()

        self.mod = WaveModulation(dim, hi_kernel, lo_kernel)
        self.conv = WaveConv(dim, hi_kernel, lo_kernel)
        self.mod_norm = LayerNorm(dim, 1e-6)
        self.conv_norm = LayerNorm(dim, 1e-6)
        self.drop = DropPath(drop) if drop > 0.0 else nn.Identity()
        self.pool = WavePool(dim, hi_kernel, lo_kernel) if pool else nn.Identity()

    def forward(self, x):
        x = x + self.drop(self.mod_norm(self.mod(x)))
        x = x + self.drop(self.conv_norm(self.conv(x)))
        return self.pool(x)


if __name__ == "__main__":
    pass

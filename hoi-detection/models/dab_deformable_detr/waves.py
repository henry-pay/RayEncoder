from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import distributed as dist

from .rays import RayBlock


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


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


class Wave(nn.Module):
    def __init__(
        self,
        in_channel,
        embed_dims=[],
        stems=[],
        blocks=[],
        depths=[],
        rays=[],
        return_intermediate=True,
    ):
        super().__init__()

        base_channel = in_channel
        blk_idx = []
        layers = []

        for layer, dim in zip(stems, embed_dims):
            layers.append(layer(base_channel, dim))
            base_channel = dim

        blk_idx.append(len(layers) - 1)

        for idx, (blk, depth) in enumerate(zip(blocks, depths), 1):
            if rays:
                layers.append(rays[idx - 1](dim=base_channel))

            for d in range(depth):
                requires_pool = (idx != len(blocks)) and (d == depth - 1)
                layers.append(blk(base_channel, pool=requires_pool))

            base_channel = base_channel if idx == len(blocks) else 8 * base_channel
            blk_idx.append(len(layers) - 1)

        self.idx = blk_idx
        self.model = nn.Sequential(*layers)
        self.channels = base_channel
        self.return_intermediate = return_intermediate
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Mapping keys
        for key in list(state_dict.keys()):
            if "model.0.model" in key:
                new_key = key[8:]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(self, state_dict)

    @staticmethod
    def load_state_dict(module, state_dict, strict=False):
        unexpected_keys = []
        all_missing_keys = []
        err_msg = []

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # use _load_from_state_dict to enable checkpoint version control
        def load(module, prefix=""):
            # recursively check parallel module in case that the model has a
            # complicated structure, e.g., nn.Module(nn.Module(DDP))
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                all_missing_keys,
                unexpected_keys,
                err_msg,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(module)
        load = None  # break load->load reference cycle

        # ignore "num_batches_tracked" of BN layers
        missing_keys = [
            key for key in all_missing_keys if "num_batches_tracked" not in key
        ]

        if unexpected_keys:
            err_msg.append(
                f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n"
            )
        if missing_keys:
            err_msg.append(
                f"missing keys in source state_dict: {', '.join(missing_keys)}\n"
            )

        rank, _ = get_dist_info()
        if len(err_msg) > 0 and rank == 0:
            err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
            err_msg = "\n".join(err_msg)
            if strict:
                raise RuntimeError(err_msg)
            else:
                print(err_msg)

    def forward(self, x):
        if not self.return_intermediate:
            x = self.model(x)
            return x
        else:
            cnt = 0
            res = dict()

            for idx, layer in enumerate(self.model):
                x = layer(x)

                if idx == self.idx[cnt]:
                    res[f"layer{cnt}"] = x
                    cnt += 1

            return res


def build_wave(
    in_channel,
    embed_dims=[32, 48, 64],
    depths=[6, 6],
    hi_kernel=5,
    lo_kernel=3,
    has_ray=False,
    point_no=[12, 12],
    point_scale=[10, 5],
    patch_size=[8, 8],
):
    stems = [
        partial(WaveStem, hi_kernel=hi_kernel, lo_kernel=lo_kernel)
        for _ in range(len(embed_dims))
    ]
    blocks = [
        partial(
            WaveBlock,
            hi_kernel=hi_kernel,
            lo_kernel=lo_kernel,
            drop=0.0,
        )
        for _ in range(len(depths))
    ]
    rays = (
        [
            partial(RayBlock, point_no=p_no, point_scale=p_scale, patch_size=p_size)
            for p_no, p_scale, p_size in zip(point_no, point_scale, patch_size)
        ]
        if has_ray
        else []
    )

    return Wave(in_channel, embed_dims, stems, blocks, depths, rays)

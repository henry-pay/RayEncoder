import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DAM(nn.Module):
    def __init__(self, in_channel):
        print(f"[DAM] - dim ({in_channel})")
        super().__init__()

        mu = torch.arange(in_channel) / in_channel * 5
        self.mu = nn.Parameter(mu.reshape(-1, in_channel, 1, 1), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        mask = F.relu(F.tanh(torch.sqrt(self.alpha) * (self.mu + self.beta)))
        return x * mask


class Ray(nn.Module):
    def __init__(self, dim, *, point_no=4, point_scale=1, patch_size=3):
        print(
            f"[Ray] - dim ({dim}), point_no ({point_no}), point_scale ({point_scale}), patch_size ({patch_size})"
        )
        super().__init__()

        # Initialize the ray based on quadrants
        point = torch.arange(1, (2 * point_no), 2) * (torch.pi / point_no)
        point = torch.stack((point.cos(), point.sin()), dim=1)
        self.point = nn.Parameter(point * point_scale, requires_grad=True)
        self.point_scale = point_scale

        # Cut the pixels into patches
        self.patch_size = patch_size

        # Parameters to model ray attenuation
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.var = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        x, h_p, w_p = self.get_patches(x)
        B, C, P, H, W = x.shape

        # Assign coordinates for pixels
        rows, cols = h_p * H, w_p * W
        rows = torch.arange(rows, device=x.device) - (rows / 2)
        cols = torch.arange(cols, device=x.device) - (cols / 2)
        p_coords = (torch.cartesian_prod(rows, cols) + 0.5).view(P, -1, 2)

        # Calculate attenuation based on euclidean distance
        pairwise_dist = self.get_norm(
            torch.cdist(self.random_rotation(self.point), p_coords, p=2).view(
                P, -1, H, W
            )
        )
        origin_dist = self.get_norm(
            torch.einsum("pni, pni -> pn", p_coords, p_coords).view(P, H, W)
        )
        point_attenuation = self.get_attenuation(pairwise_dist, "prij, prjk -> prik")
        origin_attenuation = self.get_attenuation(origin_dist, "pij, pjk -> pik")

        # Using FFT
        x = torch.fft.rfft2(x, dim=(3, 4), norm="ortho")
        h, w = x.shape[3:]

        point = point_attenuation.mean(dim=1).view(1, 1, P, H, W)
        origin = origin_attenuation.view(1, 1, P, H, W)
        if h != H or w != W:
            point = F.interpolate(point, size=x.shape[-3:], mode="trilinear")
            origin = F.interpolate(origin, size=x.shape[-3:], mode="trilinear")

        x = torch.cat([(x * point), (x * origin)], dim=1)
        x = torch.fft.irfft2(x, s=(H, W), dim=(3, 4), norm="ortho")
        x = x.view(B, 2 * C, h_p * H, w_p * W)

        return x

    def get_patches(self, x):
        # Get the sizes of input images
        B, C, H, W = x.shape

        # Calculate padding if needed
        pad_h = self.patch_size - (H % self.patch_size)
        pad_w = self.patch_size - (W % self.patch_size)

        # Cut the pixels into patches
        if pad_h == self.patch_size and pad_w == self.patch_size:
            h_patch = H // self.patch_size
            w_patch = W // self.patch_size
            return x.view(B, C, -1, self.patch_size, self.patch_size), h_patch, w_patch
        else:
            h_patch = (H + pad_h) // self.patch_size
            w_patch = (W + pad_w) // self.patch_size
            l_pad, u_pad = pad_h // 2, pad_w // 2
            r_pad, d_pad = pad_h - l_pad, pad_w - u_pad
            x = F.pad(x, (u_pad, d_pad, l_pad, r_pad), mode="circular")
            return x.view(B, C, -1, self.patch_size, self.patch_size), h_patch, w_patch

    def get_attenuation(self, dist, equation):
        ray_attenuation = self.beta * torch.exp(-self.alpha * dist)
        point_attenuation = torch.exp(-torch.square(dist) / (2 * self.var)) / (
            2 * torch.pi * self.var
        )
        attenuation = torch.einsum(equation, ray_attenuation, point_attenuation)
        attenuation = attenuation.softmax(dim=1)
        return attenuation

    @staticmethod
    def get_norm(x):
        if x.dim() == 3:
            min_value = x.min()
            max_value = x.max()
            range_value = max_value - min_value
            return (x - min_value) / range_value
        elif x.dim() == 4:
            return x.softmax(dim=1)
        else:
            raise NotImplementedError

    @staticmethod
    def random_rotation(x):
        if torch.rand(1) > 0.5:
            theta = torch.rand(1) * torch.pi * 2
            x_rot = theta.cos()
            y_rot = theta.sin()
            matrix = torch.tensor([[x_rot, y_rot], [-y_rot, x_rot]], device=x.device)
            x = torch.einsum("ij, jk -> ik", x, matrix)
        return x


class RayBlock(nn.Module):
    def __init__(self, dim, *, point_no=8, point_scale=1, patch_size=4):
        assert dim % 2 == 0, "Channel must be divisible by 2"
        print(
            f"[RayBlock] - dim ({dim}), point_no ({point_no}), point_scale ({point_scale}), patch_size ({patch_size})"
        )
        super().__init__()

        out_dim = dim // 2
        self.proj = nn.Conv2d(dim, out_dim, 1)
        self.ray = Ray(
            dim=out_dim,
            point_no=point_no,
            point_scale=point_scale,
            patch_size=patch_size,
        )
        self.linear = nn.Conv2d(dim, dim, 1)
        self.norm = LayerNorm(dim, 1e-6)

    def forward(self, x):
        x = self.proj(x)
        x = self.ray(x)
        x = self.linear(x)
        return self.norm(F.gelu(x))


class RayEncoder(nn.Module):
    def __init__(self, dim=512, point_no=8, point_scale=1, patch_size=4, depth=1):
        super().__init__()

        self.model = nn.Sequential(
            *[
                RayBlock(
                    dim=dim,
                    point_no=point_no,
                    point_scale=point_scale,
                    patch_size=patch_size,
                )
                for _ in range(depth)
            ]
        )
        self.dam = DAM(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = self.dam(x)
        return x


def build_encoder(dim, point_no=12, point_scale=3, patch_size=8, depth=3):
    return RayEncoder(
        dim=dim,
        point_no=point_no,
        point_scale=point_scale,
        patch_size=patch_size,
        depth=depth,
    )

import torch.nn as nn

from .modules.rays import RayBlock


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

    def forward(self, x):
        x = self.model(x)
        return x


def build_encoder(dim, config=None):
    return RayEncoder(dim=dim, **config)

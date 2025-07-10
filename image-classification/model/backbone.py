from functools import partial

import torch.nn as nn

from .modules.rays import RayBlock
from .modules.waves import WaveBlock, WaveStem


class Wave(nn.Module):
    def __init__(
        self,
        in_channel,
        embed_dims=[],
        stems=[],
        blocks=[],
        depths=[],
        rays=[],
    ):
        super().__init__()

        base_channel = in_channel
        layers = []

        for layer, dim in zip(stems, embed_dims):
            layers.append(layer(base_channel, dim))
            base_channel = dim

        for idx, (blk, depth) in enumerate(zip(blocks, depths), 1):
            if rays:
                layers.append(rays[idx - 1](dim=base_channel))

            for d in range(depth):
                requires_pool = (idx != len(blocks)) and (d == depth - 1)
                layers.append(blk(base_channel, pool=requires_pool))

            base_channel = base_channel if idx == len(blocks) else 8 * base_channel

        self.model = nn.Sequential(*layers)
        self.channels = base_channel

    def forward(self, x):
        x = self.model(x)
        return x


def build_wave(in_channel, config=None):
    embed_dims = config.embed_dims
    depths = config.depths
    stems = [
        partial(WaveStem, hi_kernel=stem.hi_kernel, lo_kernel=stem.lo_kernel)
        for stem in config.stems
    ]
    blocks = [
        partial(
            WaveBlock,
            hi_kernel=block.hi_kernel,
            lo_kernel=block.lo_kernel,
            drop=block.drop,
        )
        for block in config.blocks
    ]
    rays = [partial(RayBlock, **ray) for ray in config.rays] if "rays" in config else []

    return Wave(in_channel, embed_dims, stems, blocks, depths, rays)

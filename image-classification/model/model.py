import torch.nn as nn

from .backbone import build_wave
from .encoder import build_encoder
from .modules.rays import DAM


class Model(nn.Module):
    def __init__(self, in_channel, num_classes, config=None):
        super().__init__()

        layers = [build_wave(in_channel, config["backbone"])]
        dim = layers[0].channels

        if "encoder" in config:
            layers.append(build_encoder(dim, config["encoder"]))

        if "prune" in config:
            self.dam = DAM(dim)
            layers.append(self.dam)

        self.model = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = self.norm(x.mean([-2, -1]))
        return self.head(x)

    def get_beta(self):
        return self.dam.beta if hasattr(self, "dam") else None


def build_model(in_channel, num_classes, config=None):
    return Model(in_channel, num_classes, config)


if __name__ == "__main__":
    pass

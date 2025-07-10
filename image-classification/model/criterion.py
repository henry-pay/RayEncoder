import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy
from torch.nn import CrossEntropyLoss


class Criterion(nn.Module):
    def __init__(self, mixup=False, beta=None, scale=0.01, train=False):
        super().__init__()

        self.loss = SoftTargetCrossEntropy() if mixup else CrossEntropyLoss()
        self.beta = 0.0 if beta is None else beta
        self.scale = scale
        self.train = train

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss + self.scale * self.beta if self.train else loss


def build_criterion(mixup=False, beta=None, scale=0.01):
    return (
        Criterion(mixup=mixup, beta=beta, scale=scale, train=True),
        Criterion(mixup=False, beta=None, scale=0, train=False),
    )

import torch
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.optim import AdamW, Optimizer


class SAM(Optimizer):
    def __init__(self, params, base_optim, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho which should be negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optim = base_optim(self.param_groups, **kwargs)
        self.param_groups = self.base_optim.param_groups
        self.defaults.update(self.base_optim.defaults)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])

        self.base_optim.step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            )
        )


def set_weight_decay(model, skip_list=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith("bias") or name in skip_list:
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, distributed, config=None):
    skip = ()
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    parameters = set_weight_decay(model, skip)

    if config.use_sam:
        optimizer = (
            ZeroRedundancyOptimizer(parameters, SAM, base_optim=AdamW, **config.params)
            if distributed
            else SAM(parameters, AdamW, **config.params)
        )
    else:
        optimizer = (
            ZeroRedundancyOptimizer(parameters, AdamW, **config.params)
            if distributed
            else AdamW(parameters, **config.params)
        )

    return optimizer

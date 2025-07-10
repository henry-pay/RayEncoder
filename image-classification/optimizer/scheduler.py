from torch.optim.lr_scheduler import OneCycleLR


def build_scheduler(optim, epochs, n_iter_per_epoch, config=None):
    params = config.params
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(params.warmup_epoch * n_iter_per_epoch) / num_steps
    div_factor = params.max_lr / params.initial_lr
    final_div_factor = params.initial_lr / params.min_lr

    return OneCycleLR(
        optim,
        max_lr=params.max_lr,
        total_steps=num_steps,
        anneal_strategy=params.anneal_strategy,
        pct_start=warmup_steps,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        cycle_momentum=False,
    )

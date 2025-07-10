import os
import random
import warnings
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.distributed as dist
from dataloader.loader import build_dataloader
from engine.trainer import evaluate, train
from logs.logger import Logger
from model.criterion import build_criterion
from model.model import build_model
from optimizer.optim import build_optimizer
from optimizer.scheduler import build_scheduler


def save_checkpoint(
    rank, save_dir, save_prefix, epoch, model, optimizer, scheduler, scaler=None
):
    if rank == 0:
        print("Saving Model")
        save_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
        }

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"{save_prefix}_checkpoint_{epoch}.pth")
        torch.save(save_state, save_path)


def load_checkpoint(
    model_dir, load_file, model, optimizer, scheduler, scaler=None, resume=True
):
    print("Loading Pretrained Model")
    load_path = os.path.join(model_dir, load_file)
    if not os.path.exists(load_path):
        raise OSError("Not such file")

    if load_path.split(".")[-1] != "pth":
        latest_epoch = 0
        filename = ""
        for f in os.listdir(load_path):
            name, ext = f.split(".")
            epoch_count = int(name.split("_")[-1])
            if ext == "pth" and epoch_count > latest_epoch:
                latest_epoch = epoch_count
                filename = f
        load_path = os.path.join(load_path, filename)

    checkpoint = torch.load(load_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    epoch = 1

    if resume:
        epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    del checkpoint
    torch.cuda.empty_cache()

    return epoch


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config=None):
    if config.use_amp and config.optimizer.use_sam:
        raise ValueError("AMP and SAM should not be used together!!!")

    if config.distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = torch.cuda.amp.grad_scaler.GradScaler() if config.use_amp else None
    epochs = config.epochs
    update_freq = config.update_frequency
    max_norm = config.max_norm
    save_freq = config.save_frequency
    save_dir = os.path.join(
        config.model_dir, datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    )
    logger_dir = os.path.join(
        config.logger_dir, datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    )

    torch.manual_seed(config.torch_seed)
    torch.cuda.manual_seed(config.torch_seed)
    np.random.seed(config.numpy_seed)
    random.seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)

    train_data, test_data, mixup_fn = build_dataloader(
        config.distributed, config.dataset
    )
    model = build_model(
        config.dataset.channel,
        config.dataset.num_classes,
        config.model,
    ).to(device)
    optimizer = build_optimizer(model, config.distributed, config.optimizer)
    scheduler = build_scheduler(optimizer, epochs, len(train_data), config.scheduler)
    train_crit, test_crit = build_criterion(
        mixup_fn is not None, model.get_beta(), 1e-3
    )

    start_epoch = 1
    if config.load_file:
        start_epoch = load_checkpoint(
            config.model_dir,
            config.load_file,
            model,
            optimizer,
            scheduler,
            resume=config.resume,
        )

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )
        model_without_ddp = model.module
    else:
        model = model
        model_without_ddp = model

    train_logger = Logger(
        logger_dir,
        config.log_frequency,
        len(train_data),
        rank=rank,
        file_prefix=config.save_prefix,
    )
    test_logger = Logger(
        logger_dir,
        config.log_frequency,
        len(test_data),
        rank=rank,
        is_train=False,
        file_prefix=config.save_prefix,
    )

    if rank == 0:
        print(f"Training on {config.dataset.build.data_name} Dataset")
        print(f"Dataset Configuration\n{config.dataset}")
        print(f"Model Configuration:\n{config.model}")
        print(
            "Total Trainable Parameters :",
            sum(
                param.numel()
                for param in model_without_ddp.parameters()
                if param.requires_grad
            ),
        )
        print(
            f"Training Pipeline Configuration:\nOptimizer - {config.optimizer}\nScheduler - {config.scheduler}\n"
        )

    for epoch in range(start_epoch, epochs + 1):
        train(
            device,
            model,
            train_data,
            train_crit,
            optimizer,
            scheduler,
            epoch,
            mixup_fn,
            update_freq,
            max_norm,
            scaler,
            train_logger,
        )

        evaluate(
            device,
            model,
            test_data,
            test_crit,
            epoch,
            test_logger,
            scaler is not None,
        )

        if (
            (not config.distributed) and rank == 0
        ) or rank == dist.get_world_size() - 1:
            print(
                f"Train Loss : {train_logger['loss'].global_average:.3f}, Train Accuracy : {train_logger['acc_1'].global_average:.3f}"
            )
            print(
                f"Eval  Loss : {test_logger['loss'].global_average:.3f}, Eval  Accuracy : {test_logger['acc_1'].global_average:.3f}"
            )

        train_logger.reset()
        test_logger.reset()

        if epoch % save_freq == 0 or epoch == epochs:
            if config.distributed:
                optimizer.consolidate_state_dict(0)
            save_checkpoint(
                rank,
                save_dir,
                config.save_prefix,
                epoch,
                model_without_ddp,
                optimizer,
                scheduler,
            )

    if config.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    with warnings.catch_warnings(action="ignore"):
        main()

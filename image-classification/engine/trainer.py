import math
import os

import torch
from timm.utils import accuracy
from tqdm import tqdm


def train(
    device,
    model,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    mixup_fn=None,
    update_freq=1,
    max_norm=0,
    scaler=None,
    logger=None,
):
    model.train(True)
    optimizer.zero_grad()
    num_steps = len(data_loader)

    data = tqdm(data_loader, total=len(data_loader), leave=True)
    for step, (img, label) in enumerate(data, start=1):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        initial_label = label

        if mixup_fn is not None:
            img, label = mixup_fn(img, label)

        if scaler is not None:
            with torch.cuda.amp.autocast_mode.autocast():
                output = model(img.contiguous())
                loss = criterion(output, label)
        else:
            output = model(img.contiguous())
            loss = criterion(output, label)

        loss = loss / update_freq
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            torch.save(
                img,
                os.path.join(logger.logger_dir, "nan_input.pt"),
            )
            torch.save(
                output,
                os.path.join(logger.logger_dir, "nan_output.pt"),
            )
            torch.save(
                model.module.state_dict(),
                os.path.join(logger.logger_dir, "nan_model.pth"),
            )
            print(f"Non-finite Loss value at Batch {step}")
            continue

        acc_1, acc_5 = accuracy(output, initial_label, topk=(1, 5))
        if step % update_freq == 0 or step == num_steps:
            if scaler is None:
                loss.backward()

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                if hasattr(optimizer, "first_step"):
                    optimizer.first_step()
                    optimizer.zero_grad()

                    criterion(model(img), label).backward()
                    optimizer.second_step()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if not (scale > scaler.get_scale()):
                    scheduler.step()

        torch.cuda.synchronize()
        max_lr = 0
        min_lr = 10
        weight_decay = 0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

            if group["weight_decay"] > 0:
                weight_decay = group["weight_decay"]

        logger.update(
            epoch=epoch,
            step=step,
            loss=(loss_value, label.size(0)),
            acc_1=(acc_1, label.size(0)),
            acc_5=(acc_5, label.size(0)),
            lr=max_lr,
            min_lr=min_lr,
            weight_decay=weight_decay,
            beta=criterion.beta,
        )
        data.set_description(f"Train Epoch {epoch}")
        data.set_postfix(
            loss=f"{logger['loss'].average:.4f}",
            top_1=f"{logger['acc_1'].average:.3f}",
            top_5=f"{logger['acc_5'].average:.3f}",
        )


@torch.no_grad()
def evaluate(device, model, data_loader, criterion, epoch, logger=None, use_amp=False):
    model.eval()

    data = tqdm(data_loader, total=len(data_loader), leave=True)
    for step, (img, label) in enumerate(data, start=1):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast_mode.autocast():
                output = model(img)
                loss = criterion(output, label)
        else:
            output = model(img)
            loss = criterion(output, label)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            torch.save(
                img,
                os.path.join(logger.logger_dir, "nan_input.pt"),
            )
            torch.save(
                output,
                os.path.join(logger.logger_dir, "nan_output.pt"),
            )
            torch.save(
                model.module.state_dict(),
                os.path.join(logger.logger_dir, "nan_model.pth"),
            )
            print("Non Finite Loss")
            continue

        acc_1, acc_5 = accuracy(output, label, topk=(1, 5))
        logger.update(
            epoch=epoch,
            step=step,
            loss=(loss_value, label.size(0)),
            acc_1=(acc_1, label.size(0)),
            acc_5=(acc_5, label.size(0)),
        )
        data.set_description(f"Eval  Epoch {epoch}")
        data.set_postfix(
            loss=f"{logger['loss'].average:.4f}",
            top_1=f"{logger['acc_1'].average:.3f}",
            top_5=f"{logger['acc_5'].average:.3f}",
        )

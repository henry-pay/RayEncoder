import os
from pathlib import Path

import torch
import torch.distributed as dist
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Squeeze, ToTensor, ToTorchImage
from ffcv.writer import DatasetWriter
from timm.data import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageNet
from torchvision.transforms import v2 as transforms


def build_transform(is_train=True):
    if is_train:
        transform = transforms.Compose(
            [
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
                transforms.RandomErasing(0.25),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    return transform


def build_dataset(is_train, file_path, config=None):
    root = Path(config.root_path)
    dataset = ImageNet(root, "train" if is_train else "val")
    writer = DatasetWriter(
        file_path,
        {
            "image": RGBImageField(
                write_mode="smart", max_resolution=config.max_resolution
            ),
            "label": IntField(),
        },
        num_workers=config.num_workers,
    )
    writer.from_indexed_dataset(dataset, chunksize=100)


def build_dataloader(distributed=True, config=None):
    root = Path(config.file_path)
    train_path = root / "train.beton"
    eval_path = root / "eval.beton"

    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(train_path):
        if not distributed or dist.get_rank() == 0:
            build_dataset(True, train_path, config.build)

    if not os.path.exists(eval_path):
        if not distributed or dist.get_rank() == 0:
            build_dataset(False, eval_path, config.build)

    if distributed:
        dist.barrier()

    size = config.transform.image_size
    ratio = config.transform.image_size / config.transform.re_size
    train_decoder = RandomResizedCropRGBImageDecoder((size, size))
    train_img_pipeline = [
        train_decoder,
        ToTensor(),
        ToTorchImage(),
        build_transform(True),
    ]

    eval_decoder = CenterCropRGBImageDecoder((size, size), ratio=ratio)
    eval_img_pipeline = [
        eval_decoder,
        ToTensor(),
        ToTorchImage(),
        build_transform(False),
    ]
    label_pipeline = [IntDecoder(), ToTensor(), Squeeze()]

    train_loader = Loader(
        train_path,
        batch_size=config.build.batch_size,
        num_workers=config.build.num_workers,
        order=OrderOption.RANDOM,
        os_cache=True,
        drop_last=True,
        distributed=distributed,
        seed=config.build.seed,
        pipelines={"image": train_img_pipeline, "label": label_pipeline},
    )
    eval_loader = Loader(
        eval_path,
        batch_size=config.build.batch_size,
        num_workers=config.build.num_workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=False,
        distributed=distributed,
        pipelines={"image": eval_img_pipeline, "label": label_pipeline},
    )
    mixup_fn = (
        Mixup(**config.mixup_params, num_classes=config.num_classes)
        if "mixup_params" in config
        else None
    )
    return train_loader, eval_loader, mixup_fn

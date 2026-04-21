from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from .config import ProjectConfig


@dataclass(frozen=True)
class DataBundle:
    train_dataset: Subset[Dataset[Any]]
    val_dataset: Subset[Dataset[Any]]
    test_dataset: Dataset[Any]
    train_loader: DataLoader[Any]
    train_eval_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    test_loader: DataLoader[Any]
    class_distribution: dict[str, dict[str, int]]
    sample_shape: list[int]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _subset_distribution(dataset: Subset[Dataset[Any]], class_names: tuple[str, ...]) -> dict[str, int]:
    base_dataset = dataset.dataset
    indices = np.array(dataset.indices)
    targets = np.array(base_dataset.targets, dtype=np.int64)[indices]
    counts = np.bincount(targets, minlength=len(class_names))
    return {name: int(count) for name, count in zip(class_names, counts.tolist())}


def _dataset_distribution(dataset: datasets.CIFAR10, class_names: tuple[str, ...]) -> dict[str, int]:
    targets = np.array(dataset.targets, dtype=np.int64)
    counts = np.bincount(targets, minlength=len(class_names))
    return {name: int(count) for name, count in zip(class_names, counts.tolist())}


def build_runtime_config(config: ProjectConfig) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    num_workers = config.cuda_num_workers if device.type == "cuda" else config.cpu_num_workers
    pin_memory = device.type == "cuda"
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "device_type": device.type,
        "use_amp": device.type == "cuda",
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "non_blocking": pin_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
    }


def build_data_bundle(config: ProjectConfig, runtime: dict[str, Any]) -> DataBundle:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std),
        ]
    )

    train_full = datasets.CIFAR10(root=str(config.data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(config.data_dir), train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(train_full, [config.train_size, config.val_size], generator=generator)

    train_loader = _build_loader(train_dataset, config.batch_size, True, runtime["num_workers"], runtime["pin_memory"])
    train_eval_loader = _build_loader(train_dataset, config.batch_size, False, runtime["num_workers"], runtime["pin_memory"])
    val_loader = _build_loader(val_dataset, config.batch_size, False, runtime["num_workers"], runtime["pin_memory"])
    test_loader = _build_loader(test_dataset, config.batch_size, False, runtime["num_workers"], runtime["pin_memory"])

    sample_image, _ = train_dataset[0]
    class_distribution = {
        "train": _subset_distribution(train_dataset, config.class_names),
        "val": _subset_distribution(val_dataset, config.class_names),
        "test": _dataset_distribution(test_dataset, config.class_names),
    }
    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_distribution=class_distribution,
        sample_shape=list(sample_image.shape),
    )


def build_data_summary(config: ProjectConfig, data_bundle: DataBundle) -> dict[str, Any]:
    return {
        "dataset_name": "CIFAR-10",
        "class_names": list(config.class_names),
        "split_sizes": {
            "train": len(data_bundle.train_dataset),
            "val": len(data_bundle.val_dataset),
            "test": len(data_bundle.test_dataset),
        },
        "class_distribution": data_bundle.class_distribution,
        "sample_shape": data_bundle.sample_shape,
        "normalization": {
            "mean": list(config.normalize_mean),
            "std": list(config.normalize_std),
        },
        "batch_size": config.batch_size,
    }

"""Centralized DGM4 DataLoader factory."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from src.utils.config import get_key, require_key

from .collate import dgm4_collate
from .dataset import DGM4Dataset
from .sampler import IDAwareSampler


def _configure_data_worker(_: int, *, torch_threads: int) -> None:
    """Keep each loader worker single-threaded to avoid nested CPU oversubscription."""

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(torch_threads)


def _annotation_path(config: Mapping[str, Any], split: str) -> Path:
    key = {"train": "train_annotations", "val": "val_annotations", "test": "test_annotations"}[split]
    return Path(str(require_key(config, f"dataset.{key}")))


def create_dgm4_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    tokenizer: Any | None = None,
    max_samples: int | None = None,
    samples_per_class: int | None = None,
) -> DGM4Dataset:
    """Create a DGM4Dataset from YAML configuration."""

    return DGM4Dataset.from_annotation_file(
        annotation_path=_annotation_path(config, split),
        split=split,
        dataset_root=require_key(config, "dataset.root"),
        image_root=require_key(config, "dataset.image_root"),
        image_size=int(require_key(config, "image.size")),
        tokenizer=tokenizer,
        tokenizer_max_length=int(require_key(config, "tokenizer.max_length")),
        tokenizer_ignore_index=int(require_key(config, "tokenizer.ignore_index")),
        verify_schema=bool(get_key(config, "preflight.verify_schema", False)),
        verify_image_exists=bool(get_key(config, "dataset.verify_image_exists", True)),
        progress=bool(get_key(config, "progress.enabled", True)),
        max_samples=max_samples,
        samples_per_class=samples_per_class,
    )


def create_dataloader(
    config: Mapping[str, Any],
    *,
    split: str,
    tokenizer: Any | None = None,
    max_samples: int | None = None,
    samples_per_class: int | None = None,
) -> DataLoader:
    """Create a PyTorch DataLoader from YAML configuration."""

    dataset = create_dgm4_dataset(
        config,
        split=split,
        tokenizer=tokenizer,
        max_samples=max_samples,
        samples_per_class=samples_per_class,
    )
    batch_size = int(require_key(config, "dataloader.batch_size"))
    num_workers = int(require_key(config, "dataloader.num_workers"))
    pin_memory = bool(require_key(config, "dataloader.pin_memory"))
    persistent_workers = bool(require_key(config, "dataloader.persistent_workers"))
    prefetch_factor = get_key(config, "dataloader.prefetch_factor", None)
    worker_threads = int(get_key(config, "dataloader.worker_threads", 1))
    drop_last = bool(require_key(config, "dataloader.drop_last"))
    seed = int(require_key(config, "dataloader.seed"))
    shuffle_config = get_key(config, f"dataloader.shuffle.{split}", split == "train")
    shuffle = bool(shuffle_config)

    sampler = None
    if bool(get_key(config, "dataloader.id_aware_sampler.enabled", False)):
        sampler = IDAwareSampler(dataset.rows, seed=seed, shuffle=shuffle)
        shuffle = False

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": dgm4_collate,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        if worker_threads <= 0:
            raise ValueError("dataloader.worker_threads must be positive")
        kwargs["persistent_workers"] = persistent_workers
        kwargs["worker_init_fn"] = partial(
            _configure_data_worker,
            torch_threads=worker_threads,
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)

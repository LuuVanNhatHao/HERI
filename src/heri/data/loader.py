"""Centralized DGM4 DataLoader factory."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import torch
from torch.utils.data import DataLoader, RandomSampler

from heri.utils.config import get_key, require_key
from heri.utils.hashing import sha256_file

from .collate import multimodal_collate
from .dataset import DGM4Dataset
from .samm import SAMMDataset
from .sampler import IDAwareSampler


def _configure_data_worker(_: int, *, torch_threads: int) -> None:
    """Keep each loader worker single-threaded to avoid nested CPU oversubscription."""

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(torch_threads)


def _annotation_path(config: Mapping[str, Any], split: str) -> Path:
    key = {"train": "train_annotations", "val": "val_annotations", "test": "test_annotations"}[split]
    return Path(str(require_key(config, f"dataset.{key}")))


def verify_annotation_identity(config: Mapping[str, Any], split: str) -> str:
    """Measure annotation identity for this run instead of trusting YAML labels."""

    path = _annotation_path(config, split)
    observed = sha256_file(path, progress=False)
    expected = str(config.get("dataset_hashes", {}).get(split, ""))
    if bool(get_key(config, "dataset.verify_hashes", False)):
        if not expected or observed != expected:
            raise ValueError(
                f"{split} annotation SHA-256 mismatch for {path}: "
                f"observed={observed}, expected={expected or 'undeclared'}"
            )
    if isinstance(config, MutableMapping):
        config.setdefault("observed_dataset_hashes", {})[split] = observed
    return observed


def create_dgm4_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    tokenizer: Any | None = None,
    max_samples: int | None = None,
    samples_per_class: int | None = None,
) -> DGM4Dataset:
    """Create a DGM4Dataset from YAML configuration."""

    verify_annotation_identity(config, split)
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


def create_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    tokenizer: Any | None = None,
    max_samples: int | None = None,
    samples_per_class: int | None = None,
) -> DGM4Dataset | SAMMDataset:
    """Create the configured normalized dataset without branching in the model."""

    dataset_name = str(get_key(config, "dataset.name", "DGM4")).strip().upper()
    if dataset_name == "DGM4":
        return create_dgm4_dataset(
            config, split=split, tokenizer=tokenizer, max_samples=max_samples,
            samples_per_class=samples_per_class,
        )
    if dataset_name == "SAMM":
        verify_annotation_identity(config, split)
        return SAMMDataset.from_annotation_file(
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
            clip_out_of_bounds_boxes=bool(
                get_key(config, "dataset.annotation_policy.clip_boxes", True)
            ),
            mask_missing_image_boxes=bool(
                get_key(config, "dataset.annotation_policy.mask_missing_image_boxes", True)
            ),
            mask_invalid_text_positions=bool(
                get_key(config, "dataset.annotation_policy.mask_invalid_text_positions", True)
            ),
            jpeg_draft=bool(get_key(config, "dataset.jpeg_draft", True)),
            max_samples=max_samples,
            samples_per_class=samples_per_class,
        )
    raise ValueError(f"Unsupported dataset.name={dataset_name!r}; expected DGM4 or SAMM")


def create_dataloader(
    config: Mapping[str, Any],
    *,
    split: str,
    tokenizer: Any | None = None,
    max_samples: int | None = None,
    samples_per_class: int | None = None,
) -> DataLoader:
    """Create a PyTorch DataLoader from YAML configuration."""

    dataset = create_dataset(
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
    elif shuffle:
        # Worker base seeds must not consume the sampler's random stream when
        # persistent workers are recreated after an epoch-boundary resume.
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
        shuffle = False

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": multimodal_collate,
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

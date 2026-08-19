"""DataLoader factory shared by MultiBanFakeDetect and VLDBench."""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .collate import binary_benchmark_collate
from .data import MultiBanFakeDetectDataset, VLDBenchDataset


def _configure_worker(_: int, *, torch_threads: int) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(torch_threads)


def create_binary_benchmark_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    tokenizer: Any,
    max_samples: int | None = None,
) -> Dataset[dict[str, Any]]:
    """Construct the configured real benchmark adapter."""

    adapter = str(config["dataset"]["adapter"]).strip().casefold()
    if adapter == "multibanfakedetect":
        dataset: Dataset[dict[str, Any]] = MultiBanFakeDetectDataset(
            config,
            split=split,
            tokenizer=tokenizer,
        )
    elif adapter == "vldbench":
        dataset = VLDBenchDataset(config, split=split, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown binary benchmark adapter: {adapter!r}")
    if max_samples is not None:
        count = min(int(max_samples), len(dataset))
        if count <= 0:
            raise ValueError("max_samples must be positive when specified")
        dataset = Subset(dataset, range(count))
    return dataset


def create_binary_benchmark_dataloader(
    config: Mapping[str, Any],
    *,
    split: str,
    tokenizer: Any,
    max_samples: int | None = None,
) -> DataLoader:
    """Create a deterministic benchmark DataLoader."""

    dataset = create_binary_benchmark_dataset(
        config,
        split=split,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )
    loader = config["dataloader"]
    batch_size = int(loader["batch_size"])
    num_workers = int(loader["num_workers"])
    worker_threads = int(loader.get("worker_threads", 1))
    if batch_size <= 0 or num_workers < 0 or worker_threads <= 0:
        raise ValueError("Invalid DataLoader batch_size/num_workers/worker_threads")
    shuffle = bool(loader.get("shuffle", {}).get(split, split == "train"))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(loader["pin_memory"]),
        "drop_last": bool(loader["drop_last"]),
        "collate_fn": binary_benchmark_collate,
        "generator": torch.Generator().manual_seed(int(loader["seed"])),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader["persistent_workers"])
        kwargs["worker_init_fn"] = partial(
            _configure_worker, torch_threads=worker_threads
        )
        if loader.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(loader["prefetch_factor"])
    return DataLoader(**kwargs)

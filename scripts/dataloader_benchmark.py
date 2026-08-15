"""Benchmark host-side DGM4 loading across DataLoader worker counts."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compat.transformers_compat import load_fast_tokenizer
from src.data.loader import create_dataloader
from src.utils.config import get_key, load_yaml_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return int(os.cpu_count() or 1)


def _load_tokenizer(config: dict[str, Any]) -> Any | None:
    if not bool(get_key(config, "tokenizer.enabled", False)):
        return None
    return load_fast_tokenizer(
        name=str(get_key(config, "tokenizer.name")),
        use_fast=bool(get_key(config, "tokenizer.use_fast", True)),
        local_files_only=bool(get_key(config, "tokenizer.local_files_only", False)),
    )


def _benchmark_workers(
    base_config: dict[str, Any],
    *,
    tokenizer: Any | None,
    split: str,
    workers: int,
    prefetch_factor: int,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, float | int]:
    config = copy.deepcopy(base_config)
    config["dataloader"]["num_workers"] = int(workers)
    config["dataloader"]["worker_threads"] = 1
    config["dataloader"]["persistent_workers"] = workers > 0
    config["dataloader"]["prefetch_factor"] = int(prefetch_factor)
    batch_size = int(config["dataloader"]["batch_size"])
    requested_batches = warmup_batches + measured_batches
    loader = create_dataloader(
        config,
        split=split,
        tokenizer=tokenizer,
        max_samples=batch_size * requested_batches,
    )
    iterator = iter(loader)
    try:
        for _ in range(warmup_batches):
            next(iterator)
        started = time.perf_counter()
        samples = 0
        for _ in range(measured_batches):
            batch = next(iterator)
            samples += len(batch["id"])
        duration = time.perf_counter() - started
    except StopIteration as exc:
        raise RuntimeError(
            f"DataLoader produced fewer than {requested_batches} requested batches"
        ) from exc
    finally:
        del iterator
        del loader
        gc.collect()
    return {
        "num_workers": int(workers),
        "worker_threads": 1,
        "prefetch_factor": int(prefetch_factor),
        "warmup_batches": int(warmup_batches),
        "measured_batches": int(measured_batches),
        "samples": int(samples),
        "duration_seconds": float(duration),
        "batches_per_second": float(measured_batches / max(duration, 1e-9)),
        "samples_per_second": float(samples / max(duration, 1e-9)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="config/train_hera.yaml")
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/dataloader_benchmark.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.batches <= 0 or args.warmup_batches < 0 or args.prefetch_factor <= 0:
        raise ValueError("batches/prefetch-factor must be positive and warmup-batches non-negative")
    if not args.workers or any(workers < 0 for workers in args.workers):
        raise ValueError("workers must contain non-negative integers")
    config = load_yaml_config(args.config)
    setup_logging(str(get_key(config, "logging.level", "INFO")))
    set_seed(int(config["project"]["seed"]))
    tokenizer = _load_tokenizer(config)
    results = []
    for workers in dict.fromkeys(args.workers):
        result = _benchmark_workers(
            config,
            tokenizer=tokenizer,
            split=args.split,
            workers=workers,
            prefetch_factor=args.prefetch_factor,
            warmup_batches=args.warmup_batches,
            measured_batches=args.batches,
        )
        results.append(result)
        print(json.dumps(result, indent=2))
    best = max(results, key=lambda item: float(item["samples_per_second"]))
    report = {
        "config": args.config,
        "split": args.split,
        "available_cpu_count": _available_cpu_count(),
        "configured_batch_size": int(config["dataloader"]["batch_size"]),
        "best_num_workers": int(best["num_workers"]),
        "best_samples_per_second": float(best["samples_per_second"]),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved DataLoader benchmark: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

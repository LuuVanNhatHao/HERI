"""Prespecified matched studies, explicit design contrasts and artifact-bound results."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from heri.evaluation.protocol import EVALUATION_PROTOCOL
from heri.experiments.design import describe_model_design, design_delta
from heri.utils.hashing import source_tree_sha256
from heri.experiments.ablation import HERA_ABLATION_VARIANTS, apply_hera_ablation_variant
from heri.experiments.cycle_study import (
    HERA_CYCLE_STUDY_VARIANTS,
    apply_hera_cycle_study_variant,
)
from heri.utils.config import (
    dump_yaml,
    load_yaml_config,
    validate_hera_config,
    validate_baseline_config,
)

CORE_VARIANTS = (
    "b0",
    "b1",
    "b1_noisy_or",
    "learned_queries",
    "two_cycle_static",
    "full_hera",
)
METRICS = (
    "auc",
    "acc_fixed",
    "map",
    "cf1_fixed",
    "of1_fixed",
    "image_mean_iou",
    "image_mean_iou_positive_only",
    "text_f1",
    "text_word_f1",
)
OPTIONAL_METRICS = ("edc", "udr", "vsr_at_1_image", "vsr_at_1_text")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_digest(root: Path) -> str:
    return source_tree_sha256(root)


def materialize_study(suite: dict, root: Path) -> dict:
    """Materialize all jobs; isolate changed code/configs under a new content ID."""
    root = root.resolve()
    seeds = suite["seeds"]
    variants = suite["variants"]
    allowed = set(CORE_VARIANTS) | set(HERA_ABLATION_VARIANTS) | set(HERA_CYCLE_STUDY_VARIANTS)
    if not seeds or len(seeds) != len(set(seeds)) or not all(isinstance(s, int) for s in seeds):
        raise ValueError("seeds must be a nonempty unique list of integers")
    if not variants or len(variants) != len(set(variants)) or set(variants) - allowed:
        raise ValueError("variants must be unique registered controls")
    comparison = suite.get("comparison", "paired_control")
    if comparison not in {"paired_control", "progressive_construction"}:
        raise ValueError("comparison must be paired_control or progressive_construction")
    epochs = int(suite.get("epochs", 20))
    if epochs < 4:
        raise ValueError("Scientific runs require warm-up followed by backbone adaptation")
    hera = load_yaml_config(root / suite["hera_train_config"])
    b1 = load_yaml_config(root / suite["b1_train_config"])
    b0 = load_yaml_config(
        root / suite.get("b0_train_config", "config/datasets/dgm4/baselines/b0/train.yaml")
    )
    eval_template = load_yaml_config(
        root
        / suite.get(
            "hera_evaluate_config",
            f"config/datasets/{hera['dataset']['name'].lower()}/heri/evaluate.yaml",
        )
    )["evaluation"]
    prepared = []
    for variant in variants:
        for seed in seeds:
            cfg = copy.deepcopy(
                b0 if variant == "b0" else b1 if variant in {"b1", "b1_noisy_or"} else hera
            )
            if variant == "b1_noisy_or":
                cfg["model"]["binary_aggregation"] = "noisy_or"
            elif variant == "learned_queries":
                cfg["reasoning_control"] = {
                    "enabled": True,
                    "variant": "learned_queries",
                }
                cfg["graph"]["reasoning"].update(
                    dynamic_rebuild=False,
                    refresh_semantic=False,
                    refresh_hypothesis=False,
                )
                cfg["graph"]["reasoning"]["hypothesis_feedback"] = False
                cfg["graph"]["edges"]["feedback"]["enabled"] = False
                cfg["graph"]["edge_attention"]["enabled"] = False
            elif variant in HERA_CYCLE_STUDY_VARIANTS and variant != "full_hera":
                cfg = apply_hera_cycle_study_variant(cfg, variant)
            elif variant in HERA_ABLATION_VARIANTS:
                cfg = apply_hera_ablation_variant(cfg, variant)
            # Match data, backbones, optimizer, LR horizon and training stages.
            for key in (
                "dataset",
                "dataset_hashes",
                "image",
                "tokenizer",
                "backbones",
                "optimizer",
                "scheduler",
                "thresholding",
                "dataloader",
                "runtime",
                "training",
            ):
                cfg[key] = copy.deepcopy(hera[key])
            cfg["project"]["seed"] = seed
            cfg["dataloader"]["seed"] = seed
            cfg["training"]["epochs"] = epochs
            cfg["training"]["early_stopping"]["enabled"] = False
            selection = "samm_official_cf1" if cfg["dataset"]["name"] == "SAMM" else "cf1_fixed"
            cfg["training"]["early_stopping"]["metric"] = selection
            cfg["training"]["resume_from"] = None
            cfg["training"]["auto_resume"] = {
                "enabled": False,
                "run_dir": None,
                "checkpoint_name": "last.pt",
            }
            cfg["scheduler"]["resume_horizon_policy"] = "error"
            cfg["checkpointing"].update(
                selection_metric=selection,
                tie_breakers=["map", "auc"],
                top_k=0,
                keep_best=True,
                keep_last=True,
            )
            cfg.pop("smoke", None)
            for key in ("max_train_batches", "max_val_batches", "max_test_batches"):
                cfg["runtime"][key] = None
            prepared.append((variant, seed, cfg))
    identity = digest(
        {
            "protocol": EVALUATION_PROTOCOL,
            "source": source_digest(root),
            "configs": prepared,
        }
    )
    output = root / suite["output_root"] / identity[:16]
    jobs = []
    for variant, seed, cfg in prepared:
        job_root = output / variant / f"seed_{seed}"
        cfg["project"]["run_id"] = "train"
        cfg["output"]["root"] = job_root.as_posix()
        cfg["scientific_study"] = {
            "protocol_id": identity,
            "variant": variant,
            "seed": seed,
        }
        evaluation = copy.deepcopy(cfg)
        evaluation["project"]["run_id"] = "evaluation"
        evaluation["evaluation"] = copy.deepcopy(eval_template)
        evaluation["evaluation"].update(
            checkpoint_path=(job_root / "train/checkpoints/best.pt").as_posix(),
            thresholds_path=(job_root / "train/thresholds/best_thresholds.json").as_posix(),
            split="test",
        )
        validate = (
            validate_baseline_config
            if variant in {"b0", "b1", "b1_noisy_or"}
            else validate_hera_config
        )
        validate(cfg, mode="train")
        validate(evaluation, mode="evaluate")
        train_path, eval_path = (
            job_root / "configs/train.yaml",
            job_root / "configs/evaluate.yaml",
        )
        dump_yaml(cfg, train_path)
        dump_yaml(evaluation, eval_path)
        jobs.append(
            {
                "variant": variant,
                "seed": seed,
                "train_config": str(train_path),
                "evaluate_config": str(eval_path),
                "train_config_hash": digest(cfg),
                "evaluate_config_hash": digest(evaluation),
                "summary": str(job_root / "evaluation/summary.csv"),
                "design": describe_model_design(cfg),
                "run_evidence": str(job_root / "evaluation/run_evidence.json"),
            }
        )
    configurations = {(variant, seed): cfg for variant, seed, cfg in prepared}
    reference = "full_hera" if "full_hera" in variants else variants[0]
    pairs = (
        list(zip(variants[:-1], variants[1:]))
        if comparison == "progressive_construction"
        else [(variant, reference) for variant in variants if variant != reference]
    )
    comparisons = [
        {
            "seed": seed,
            "before_variant": before,
            "after_variant": after,
            "changes": design_delta(configurations[(before, seed)], configurations[(after, seed)]),
        }
        for seed in seeds
        for before, after in pairs
    ]
    manifest = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "evidence_policy": "artifact_consistency",
        "comparison": comparison,
        "comparisons": comparisons,
        "protocol_id": identity,
        "source_digest": source_digest(root),
        "epochs": epochs,
        "seeds": seeds,
        "jobs": jobs,
        "expected_test_samples": int(suite.get("expected_test_samples", 50705)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(output / "manifest.json")}


def mean_interval(values: list[float]) -> dict:
    """Seed-level Student-t interval; never infer significance from a single run."""
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Expected finite, nonempty observations")
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else None
    margin = (
        float(t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array))) if sd is not None else None
    )
    return {
        "n": len(array),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
    }


def aggregate_study(manifest: dict) -> dict:
    policy = manifest.get("evidence_policy")
    if policy not in {None, "artifact_consistency"}:
        raise ValueError("Unknown study evidence policy")
    rows, pending = {}, []
    for job in manifest["jobs"]:
        path = Path(job["summary"])
        if not path.is_file():
            pending.append({"variant": job["variant"], "seed": job["seed"]})
            continue
        if policy == "artifact_consistency":
            from heri.experiments.provenance import inspect_run_evidence

            evidence = inspect_run_evidence(
                job,
                protocol_id=manifest["protocol_id"],
                evaluation_protocol=manifest["evaluation_protocol"],
                expected_test_samples=manifest["expected_test_samples"],
                expected_source_sha256=manifest.get("source_digest"),
            )
            if evidence["status"] == "inconsistent":
                raise ValueError(f"Inconsistent run artifacts for {path}: {evidence['errors']}")
            if evidence["status"] != "internally_consistent":
                pending.append(
                    {"variant": job["variant"], "seed": job["seed"], "evidence": evidence}
                )
                continue
        with path.open(encoding="utf-8", newline="") as handle:
            found = list(csv.DictReader(handle))
        if len(found) != 1:
            raise ValueError(f"Expected one summary row: {path}")
        row = found[0]
        expected = {
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "checkpoint_evaluation_protocol": EVALUATION_PROTOCOL,
            "scientific_study_id": manifest["protocol_id"],
            "scientific_variant": job["variant"],
            "seed": str(job["seed"]),
            "split": "test",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"Unmatched {key} in {path}: {row.get(key)!r} != {value!r}")
        if not row.get("checkpoint_id"):
            raise ValueError(f"Unbound checkpoint: {path}")
        if int(row.get("sample_count", 0)) != manifest["expected_test_samples"]:
            raise ValueError(f"Partial or unexpected test split: {path}")
        # Missing localization annotations have explicit validity masks. They
        # must not reject a complete classification run; require original-word
        # metadata for available annotations independently of availability.
        metadata_complete = row.get(
            "text_original_word_metadata_complete",
            row.get("text_coverage_complete", ""),
        )
        if str(metadata_complete).lower() != "true":
            raise ValueError(f"Missing original-word annotation coverage: {path}")
        rows[(job["variant"], job["seed"])] = {metric: float(row[metric]) for metric in METRICS}
        for metric in OPTIONAL_METRICS:
            value = row.get(metric)
            rows[(job["variant"], job["seed"])][metric] = (
                float(value) if value not in (None, "") else None
            )
    variants = list(dict.fromkeys(job["variant"] for job in manifest["jobs"]))
    summaries, paired = {}, {}
    for variant in variants:
        complete = all((variant, seed) in rows for seed in manifest["seeds"])
        summaries[variant] = (
            {
                metric: mean_interval([rows[(variant, seed)][metric] for seed in manifest["seeds"]])
                if all(rows[(variant, seed)][metric] is not None for seed in manifest["seeds"])
                else {"available": False}
                for metric in (*METRICS, *OPTIONAL_METRICS)
            }
            if complete
            else {"status": "pending"}
        )
        if (
            variant != "full_hera"
            and complete
            and all(("full_hera", seed) in rows for seed in manifest["seeds"])
        ):
            paired[variant] = {
                metric: mean_interval(
                    [
                        rows[("full_hera", seed)][metric] - rows[(variant, seed)][metric]
                        for seed in manifest["seeds"]
                    ]
                )
                for metric in METRICS
            }
    return {
        "protocol_id": manifest["protocol_id"],
        "status": "pending" if pending else "complete" if policy else "unverified",
        "evidence_status": "artifact_consistency_required" if policy else "not_checked",
        "pending_jobs": pending,
        "variants": summaries,
        "paired_full_minus_control": paired,
        "inference_scope": "seed variation on one fixed test split; exploratory unadjusted intervals, not sample/generalization uncertainty",
    }

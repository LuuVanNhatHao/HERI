"""Final evaluation runner for canonical Phase-7 HERA-DGM."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

from src.data.loader import create_dataloader
from src.evaluation.evaluator import (
    compute_metrics_and_reports,
    run_inference,
    run_text_grounding_diagnostic_inference,
    text_grounding_metrics,
)
from src.evaluation.run import _load_thresholds
from src.evaluation.summary import write_predictions_csv, write_summary_csv
from src.experiments import (
    cycle_study_protocol,
    get_hera_ablation_variant,
    get_hera_cycle_study_variant,
    validate_hera_checkpoint_variant,
)
from src.losses.hera_components import HERAMultitaskLoss
from src.models.hera import HERADGM
from src.training.checkpoint import restore_checkpoint
from src.training.optimizer import ensure_no_uninitialized_parameters
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata


_MISSING_PATH_MARKERS = ("<run_id>", "RUN")


def resolve_hera_evaluation_artifacts(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Resolve the checkpoint and validation-threshold artifacts for HERA evaluation."""

    evaluation = config["evaluation"]
    checkpoint_path = evaluation.get("checkpoint_path")
    thresholds_path = evaluation.get("thresholds_path")
    if _is_explicit_artifact_path(checkpoint_path) and _is_explicit_artifact_path(thresholds_path):
        return Path(str(checkpoint_path)), Path(str(thresholds_path))

    resolution = evaluation.get("checkpoint_resolution")
    if not isinstance(resolution, Mapping) or not bool(resolution.get("enabled", False)):
        raise FileNotFoundError(
            "HERA evaluation checkpoint_path/thresholds_path are not explicit. "
            "Set both paths in YAML, or enable evaluation.checkpoint_resolution."
        )

    mode = str(resolution["mode"])
    checkpoint_name = str(resolution["checkpoint_name"])
    thresholds_name = str(resolution["thresholds_name"])
    if mode == "run_dir":
        run_dir_value = resolution.get("run_dir")
        if run_dir_value in (None, ""):
            raise ValueError("evaluation.checkpoint_resolution.run_dir is required when mode=run_dir")
        candidates = [_artifact_pair_from_run_dir(Path(str(run_dir_value)), checkpoint_name, thresholds_name)]
    elif mode == "latest":
        output_root = Path(str(resolution["output_root"]))
        candidates = _latest_artifact_candidates(output_root, checkpoint_name, thresholds_name)
    else:
        raise ValueError("evaluation.checkpoint_resolution.mode must be latest or run_dir")

    existing = [
        (checkpoint, thresholds)
        for checkpoint, thresholds in candidates
        if checkpoint.is_file() and thresholds.is_file()
    ]
    if not existing:
        raise FileNotFoundError(
            "No complete HERA evaluation artifact pair found. Expected each selected run to contain "
            f"checkpoints/{checkpoint_name} and thresholds/{thresholds_name}."
        )
    return existing[0]


def _is_explicit_artifact_path(value: Any) -> bool:
    if value in (None, ""):
        return False
    text = str(value)
    return not any(marker in text for marker in _MISSING_PATH_MARKERS)


def _artifact_pair_from_run_dir(
    run_dir: Path,
    checkpoint_name: str,
    thresholds_name: str,
) -> tuple[Path, Path]:
    return run_dir / "checkpoints" / checkpoint_name, run_dir / "thresholds" / thresholds_name


def _latest_artifact_candidates(
    output_root: Path,
    checkpoint_name: str,
    thresholds_name: str,
) -> list[tuple[Path, Path]]:
    if not output_root.exists():
        return []
    pairs = [
        _artifact_pair_from_run_dir(run_dir, checkpoint_name, thresholds_name)
        for run_dir in output_root.iterdir()
        if run_dir.is_dir()
    ]
    return sorted(
        pairs,
        key=lambda pair: (
            pair[0].stat().st_mtime if pair[0].exists() else float("-inf"),
            pair[1].stat().st_mtime if pair[1].exists() else float("-inf"),
            pair[0].as_posix(),
        ),
        reverse=True,
    )


def evaluate_hera(config: Mapping[str, Any]) -> Path:
    """Run final HERA test evaluation using validation-bound thresholds."""

    output_root = Path(str(config["output"]["root"]))
    run_id = str(config["project"].get("run_id", f"hera_evaluation_{int(time.time())}"))
    run_dir = output_root / run_id
    for subdir in ("predictions", "metrics", "logs", "router"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    dump_yaml(config, run_dir / "effective_config.yaml")
    environment = collect_environment_metadata(Path.cwd())
    environment["backbones"] = config["backbones"]
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, default=str), encoding="utf-8")

    device = torch.device(str(config["runtime"]["device"]))
    model = HERADGM.from_config(config, load_backbones=True).to(device)
    if model.deberta is None or not hasattr(model.deberta, "tokenizer"):
        raise RuntimeError("HERADGM evaluation requires an attached DeBERTa tokenizer")
    tokenizer = model.deberta.tokenizer
    loader = create_dataloader(
        config,
        split=str(config["evaluation"]["split"]),
        tokenizer=tokenizer,
        max_samples=config.get("smoke", {}).get("test_max_samples"),
        samples_per_class=config.get("smoke", {}).get("samples_per_class"),
    )
    precision = PrecisionManager(precision=str(config["runtime"]["precision"]), device=device)
    checkpoint_path, thresholds_path = resolve_hera_evaluation_artifacts(config)
    tqdm.write(f"Resolved evaluation checkpoint: {checkpoint_path}")
    tqdm.write(f"Resolved validation thresholds: {thresholds_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError("Evaluation checkpoint does not contain its effective HERA config")
    validate_hera_checkpoint_variant(
        config,
        checkpoint_config,
        context="Evaluation",
    )
    restore_checkpoint(checkpoint=checkpoint, model=model, restore_rng=False)
    ensure_no_uninitialized_parameters(model, context="HERA evaluation checkpoint load")
    thresholds = _load_thresholds(thresholds_path, checkpoint_path=checkpoint_path, checkpoint=checkpoint)
    bundle = run_inference(
        model=model,
        loader=loader,
        loss_fn=HERAMultitaskLoss(config["loss"], routing_config=config["routing"]),
        device=device,
        precision=precision,
        max_batches=config["runtime"].get("max_test_batches"),
        desc=f"{config['evaluation']['split']} inference",
        progress=bool(config["progress"]["enabled"]),
        text_grounding_aggregation=str(
            config["model"]["grounding"]["text"]["aggregation"]
        ),
    )
    reports = compute_metrics_and_reports(bundle, thresholds)
    intervention_config = config["evaluation"].get("image_deletion", {})
    if intervention_config.get("enabled", False):
        from src.evaluation.interventions import evaluate_image_deletion
        evaluate_image_deletion(
            model=model, loader=loader, device=device, precision=precision,
            path=run_dir / "metrics/image_deletion.json",
            random_repeats=int(intervention_config.get("random_repeats", 5)),
            seed=int(intervention_config.get("seed", 2026)),
            max_batches=intervention_config.get("max_batches"),
        )
    tqdm.write("Test binary classification_report:")
    tqdm.write(reports["binary_report"])
    tqdm.write("Test attribution classification_report:")
    tqdm.write(reports["attribution_report"])
    if reports.get("class_report"):
        tqdm.write("Test dataset-class classification_report:")
        tqdm.write(reports["class_report"])
    write_predictions_csv(
        bundle=bundle,
        thresholds=thresholds,
        path=run_dir / "predictions" / f"{config['evaluation']['split']}_predictions.csv",
    )
    (run_dir / "router" / f"{config['evaluation']['split']}_batch_summaries.json").write_text(
        json.dumps(bundle.router_batch_summaries, indent=2, default=str),
        encoding="utf-8",
    )
    ablation_variant = get_hera_ablation_variant(config)
    cycle_study_variant = get_hera_cycle_study_variant(config)
    cycle_protocol = (
        cycle_study_protocol(cycle_study_variant)
        if cycle_study_variant is not None
        else {
            "cycles": int(config["graph"]["reasoning"]["cycles"]),
            "dynamic_rebuild": bool(
                config["graph"]["reasoning"]["dynamic_rebuild"]
            ),
            "rebuild_mode": "state_dependent",
        }
    )
    training_micro_batch = int(checkpoint_config["dataloader"]["batch_size"])
    query_control = bool(config.get("reasoning_control", {}).get("enabled", False))
    if query_control:
        cycle_protocol["rebuild_mode"] = "none"
    training_accumulation = int(checkpoint_config["training"]["gradient_accumulation_steps"])
    run_info = {
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "checkpoint_evaluation_protocol": checkpoint.get("evaluation_protocol", "legacy"),
        "scientific_study_id": config.get("scientific_study", {}).get("protocol_id"),
        "scientific_variant": config.get("scientific_study", {}).get("variant"),
        "run_id": run_id,
        "model_name": config["project"]["name"],
        "dataset_name": config["dataset"]["name"],
        "dataset_root": config["dataset"]["root"],
        "architecture_version": config["project"]["architecture_version"],
        "image_grounding_protocol": config["model"]["grounding"]["image"]["protocol"],
        "grounding_routing": config["model"]["grounding"]["routing"],
        "text_grounding_aggregation": config["model"]["grounding"]["text"][
            "aggregation"
        ],
        "graph_reasoning_cycles": config["graph"]["reasoning"]["cycles"],
        "dynamic_graph_rebuild": config["graph"]["reasoning"]["dynamic_rebuild"],
        "graph_rebuild_mode": cycle_protocol["rebuild_mode"],
        "model_variant": (
            "hera_learned_queries" if query_control else f"hera_cycle_{cycle_study_variant}"
            if cycle_study_variant
            else (f"hera_{ablation_variant}" if ablation_variant else "hera")
        ),
        "reasoning_backend": "learned_queries" if query_control else config["graph"]["type"],
        "ablation_variant": ablation_variant or "canonical",
        "cycle_study_variant": cycle_study_variant or "none",
        "cycle_study_protocol_id": config.get("cycle_study", {}).get(
            "protocol_id"
        ),
        "seed": config["project"]["seed"],
        "checkpoint": str(checkpoint_path),
        "split": config["evaluation"]["split"],
        "config_name": str(config.get("project", {}).get("config_name", "evaluate_hera")),
        "precision_mode": precision.actual,
        "micro_batch_size": training_micro_batch,
        "gradient_accumulation_steps": training_accumulation,
        "effective_batch_size": training_micro_batch * training_accumulation,
        "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "training_duration": None,
        "git_commit": environment.get("git_commit"),
        "dataset_hashes": json.dumps(config.get("dataset_hashes", {})),
        "transformers_version": environment["packages"].get("transformers"),
        "torch_version": environment["packages"].get("torch"),
        "torch_geometric_version": environment["packages"].get("torch-geometric"),
        "backbone_identifiers": json.dumps(
            {key: value["name"] for key, value in config["backbones"].items()}
        ),
    }
    write_summary_csv(
        path=run_dir / "summary.csv",
        run_info=run_info,
        metrics=reports["metrics"],
        thresholds=thresholds,
        reports=reports,
    )
    return run_dir / "summary.csv"


def evaluate_hera_text_grounding_diagnostic(config: Mapping[str, Any]) -> Path:
    """Compare target-free SAMM text aggregation rules in one inference pass."""

    diagnostic = config["evaluation"].get("text_grounding_diagnostic", {})
    requested = tuple(diagnostic.get(
        "aggregations",
        ("relative_soft", "presence_gated", "token_noisy_or"),
    ))
    # Keep the released rule first so the complete bundle and reference metrics
    # are directly comparable with the canonical evaluation artifact.
    aggregations = tuple(dict.fromkeys(("relative_soft", *requested)))
    output_root = Path(str(config["output"]["root"]))
    run_id = str(
        config["project"].get(
            "run_id", f"text_grounding_diagnostic_{int(time.time())}"
        )
    )
    run_dir = output_root / run_id
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    dump_yaml(config, run_dir / "effective_config.yaml")
    environment = collect_environment_metadata(Path.cwd())
    environment["backbones"] = config["backbones"]
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, default=str), encoding="utf-8"
    )

    device = torch.device(str(config["runtime"]["device"]))
    model = HERADGM.from_config(config, load_backbones=True).to(device)
    if model.deberta is None or not hasattr(model.deberta, "tokenizer"):
        raise RuntimeError("HERADGM evaluation requires an attached DeBERTa tokenizer")
    loader = create_dataloader(
        config,
        split=str(config["evaluation"]["split"]),
        tokenizer=model.deberta.tokenizer,
        max_samples=config.get("smoke", {}).get("test_max_samples"),
        samples_per_class=config.get("smoke", {}).get("samples_per_class"),
    )
    precision = PrecisionManager(
        precision=str(config["runtime"]["precision"]), device=device
    )
    checkpoint_path, thresholds_path = resolve_hera_evaluation_artifacts(config)
    tqdm.write(f"Resolved diagnostic checkpoint: {checkpoint_path}")
    tqdm.write(f"Resolved validation thresholds: {thresholds_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError("Evaluation checkpoint does not contain its effective HERA config")
    validate_hera_checkpoint_variant(config, checkpoint_config, context="Evaluation")
    restore_checkpoint(checkpoint=checkpoint, model=model, restore_rng=False)
    ensure_no_uninitialized_parameters(
        model, context="HERA text-grounding diagnostic checkpoint load"
    )
    thresholds = _load_thresholds(
        thresholds_path, checkpoint_path=checkpoint_path, checkpoint=checkpoint
    )
    primary, bundles = run_text_grounding_diagnostic_inference(
        model=model,
        loader=loader,
        loss_fn=None,
        device=device,
        precision=precision,
        max_batches=config["runtime"].get("max_test_batches"),
        desc=f"{config['evaluation']['split']} text-grounding diagnostic",
        progress=bool(config["progress"]["enabled"]),
        aggregations=aggregations,
    )
    reference = compute_metrics_and_reports(primary, thresholds)
    reference_metrics = reference["metrics"]
    common = {
        "diagnostic_protocol": "hera_text_grounding_absence_v1",
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "run_id": run_id,
        "dataset_name": config["dataset"]["name"],
        "split": config["evaluation"]["split"],
        "seed": config["project"]["seed"],
        "checkpoint": str(checkpoint_path),
        "sample_count": len(primary.ids),
        "binary_acc_reference": reference_metrics.get("acc"),
        "binary_auc_reference": reference_metrics.get("auc"),
        "attribution_cf1_reference": reference_metrics.get("cf1"),
        "class_accuracy_reference": reference_metrics.get("class_accuracy"),
        "class_macro_f1_reference": reference_metrics.get("class_macro_f1"),
    }
    rows = [
        {**common, **text_grounding_metrics(bundles[mode])}
        for mode in aggregations
    ]
    summary_path = run_dir / "diagnostic_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "metrics" / "diagnostic_summary.json").write_text(
        json.dumps({"reference": common, "aggregations": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "metrics" / "classification_reference.json").write_text(
        json.dumps(
            {
                "binary": reference["binary_report_dict"],
                "attribution": reference["attribution_report_dict"],
                "dataset_class": reference.get("class_report_dict"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    tqdm.write("Text-grounding absence diagnostic (word level):")
    tqdm.write(
        "aggregation       all_F1  positive_F1  absent_word_FP  absent_sample_FP_rate"
    )
    for row in rows:
        tqdm.write(
            f"{row['text_grounding_aggregation']:<17} "
            f"{float(row['text_word_f1']):>7.4f}  "
            f"{float(row['text_f1_positive_only']):>11.4f}  "
            f"{int(row['text_absent_word_predicted_positive_count']):>14d}  "
            f"{float(row['text_absent_sample_false_positive_rate']):>21.4f}"
        )
    return summary_path

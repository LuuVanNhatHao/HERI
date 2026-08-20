"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised for invalid project configuration."""


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config as a dictionary."""

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def require_key(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Return a required dotted key or raise a precise config error."""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing mandatory config key: {dotted_key}")
        current = current[part]
    return current


def get_key(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a dotted config key with a default."""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def validate_prepare_config(config: Mapping[str, Any]) -> None:
    """Validate mandatory Phase-1 configuration fields."""

    mandatory = (
        "project.name",
        "project.seed",
        "dataset.root",
        "dataset.train_annotations",
        "dataset.val_annotations",
        "dataset.test_annotations",
        "dataset.image_root",
        "image.size",
        "image.preserve_aspect_ratio",
        "image.padding",
        "tokenizer.enabled",
        "tokenizer.name",
        "tokenizer.use_fast",
        "tokenizer.max_length",
        "tokenizer.ignore_index",
        "dataloader.batch_size",
        "dataloader.num_workers",
        "dataloader.pin_memory",
        "dataloader.persistent_workers",
        "dataloader.drop_last",
        "dataloader.seed",
        "preflight.verify_schema",
        "preflight.verify_paths",
        "preflight.verify_image_headers",
        "preflight.verify_full_decode",
        "preflight.verify_boxes",
        "preflight.verify_mtcnn_match",
        "preflight.verify_text_positions",
        "preflight.verify_split_counts",
        "output.root",
        "progress.enabled",
    )
    for key in mandatory:
        require_key(config, key)
    if Path(str(require_key(config, "dataset.root"))).as_posix() != "data/DGM4":
        raise ConfigError("dataset.root must be the canonical value: data/DGM4")
    if Path(str(require_key(config, "dataset.image_root"))).as_posix() != "data/DGM4/images":
        raise ConfigError("dataset.image_root must be the canonical value: data/DGM4/images")
    if int(require_key(config, "image.size")) != 384:
        raise ConfigError("image.size must be the canonical Phase-1 value: 384")
    if not bool(require_key(config, "image.preserve_aspect_ratio")):
        raise ConfigError("image.preserve_aspect_ratio must be true")
    if not bool(require_key(config, "image.padding")):
        raise ConfigError("image.padding must be true")
    if bool(require_key(config, "tokenizer.enabled")) and not bool(
        require_key(config, "tokenizer.use_fast")
    ):
        raise ConfigError("tokenizer.use_fast must be true when tokenizer.enabled is true")
    if int(require_key(config, "dataloader.batch_size")) <= 0:
        raise ConfigError("dataloader.batch_size must be positive")
    if int(require_key(config, "dataloader.num_workers")) < 0:
        raise ConfigError("dataloader.num_workers must be non-negative")


def validate_baseline_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate mandatory Phase-2 baseline train/eval/smoke configuration."""

    mandatory = [
        "project.name",
        "project.seed",
        "project.architecture_version",
        "dataset.root",
        "dataset.train_annotations",
        "dataset.val_annotations",
        "dataset.test_annotations",
        "dataset.image_root",
        "dataset.use_mtcnn_boxes",
        "image.size",
        "image.preserve_aspect_ratio",
        "image.padding",
        "tokenizer.enabled",
        "tokenizer.name",
        "tokenizer.use_fast",
        "tokenizer.max_length",
        "tokenizer.ignore_index",
        "backbones.siglip2.name",
        "backbones.siglip2.image_mean",
        "backbones.siglip2.image_std",
        "backbones.siglip2.text_max_length",
        "backbones.dinov3.name",
        "backbones.dinov3.image_mean",
        "backbones.dinov3.image_std",
        "backbones.deberta.name",
        "model.baseline_type",
        "model.fusion_dim",
        "model.head_hidden_dim",
        "model.dropout",
        "loss.weights.binary",
        "loss.weights.attribution",
        "loss.weights.image_grounding",
        "loss.weights.text_grounding",
        "training.epochs",
        "training.gradient_accumulation_steps",
        "training.stages.stage_a_freeze_epochs",
        "optimizer.type",
        "optimizer.lr_new",
        "optimizer.lr_backbones.siglip2",
        "optimizer.lr_backbones.dinov3",
        "optimizer.lr_backbones.deberta",
        "optimizer.weight_decay",
        "scheduler.type",
        "thresholding.enabled",
        "checkpointing.selection_metric",
        "dataloader.batch_size",
        "dataloader.num_workers",
        "dataloader.pin_memory",
        "dataloader.persistent_workers",
        "dataloader.drop_last",
        "dataloader.seed",
        "runtime.device",
        "runtime.precision",
        "runtime.model_dtype",
        "progress.enabled",
        "logging.level",
        "output.root",
    ]
    if mode == "evaluate":
        mandatory.extend(
            [
                "evaluation.checkpoint_path",
                "evaluation.thresholds_path",
                "evaluation.split",
                "evaluation.checkpoint_resolution.enabled",
                "evaluation.checkpoint_resolution.mode",
                "evaluation.checkpoint_resolution.output_root",
                "evaluation.checkpoint_resolution.run_dir",
                "evaluation.checkpoint_resolution.checkpoint_name",
                "evaluation.checkpoint_resolution.thresholds_name",
            ]
        )
    if mode in {"train", "evaluate"}:
        mandatory.extend(
            [
                "training.micro_batch_size",
                "training.auto_resume.enabled",
                "training.auto_resume.run_dir",
                "training.auto_resume.checkpoint_name",
                "training.early_stopping.enabled",
                "training.early_stopping.metric",
                "training.early_stopping.mode",
                "training.early_stopping.patience",
                "training.early_stopping.min_delta",
                "training.early_stopping.minimum_backbone_adaptation_epochs",
                "training.stages.stage_b_full_unfreeze.enabled",
                "training.stages.stage_b_full_unfreeze.start_epoch",
                "scheduler.resume_horizon_policy",
                "checkpointing.top_k",
                "runtime.cuda_memory_logging.enabled",
                "runtime.gradient_checkpointing.enabled",
                "runtime.gradient_checkpointing.only_trainable_backbones",
                "runtime.gradient_checkpointing.use_reentrant",
                "runtime.gradient_checkpointing.backbones.siglip2",
                "runtime.gradient_checkpointing.backbones.dinov3",
                "runtime.gradient_checkpointing.backbones.deberta",
            ]
        )
    else:
        mandatory.extend(
            [
                "training.stages.stage_b_unfreeze_last_blocks.siglip2",
                "training.stages.stage_b_unfreeze_last_blocks.dinov3",
                "training.stages.stage_b_unfreeze_last_blocks.deberta",
            ]
        )
    for key in mandatory:
        require_key(config, key)
    if bool(require_key(config, "dataset.use_mtcnn_boxes")):
        raise ConfigError("Canonical baselines forbid annotation-derived dataset.mtcnn_boxes")
    if str(require_key(config, "project.architecture_version")) != "baseline_dense_bbox_v1":
        raise ConfigError("Baseline project.architecture_version must be baseline_dense_bbox_v1")
    if Path(str(require_key(config, "dataset.root"))).as_posix() != "data/DGM4":
        raise ConfigError("dataset.root must be the canonical value: data/DGM4")
    if Path(str(require_key(config, "dataset.image_root"))).as_posix() != "data/DGM4/images":
        raise ConfigError("dataset.image_root must be the canonical value: data/DGM4/images")
    if int(require_key(config, "image.size")) != 384:
        raise ConfigError("image.size must be the canonical Phase-2 value: 384")
    if not bool(require_key(config, "image.preserve_aspect_ratio")):
        raise ConfigError("image.preserve_aspect_ratio must be true")
    if not bool(require_key(config, "image.padding")):
        raise ConfigError("image.padding must be true")
    expected_backbones = {
        "backbones.siglip2.name": "google/siglip2-so400m-patch14-384",
        "backbones.dinov3.name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "backbones.deberta.name": "microsoft/deberta-v3-large",
        "tokenizer.name": "microsoft/deberta-v3-large",
    }
    for key, expected in expected_backbones.items():
        if str(require_key(config, key)) != expected:
            raise ConfigError(f"{key} must be canonical value: {expected}")
    baseline_type = str(require_key(config, "model.baseline_type"))
    if baseline_type == "b0_global":
        require_key(config, "model.fusion_hidden_dim")
    elif baseline_type == "b1_cross_attention":
        require_key(config, "model.num_attention_layers")
        require_key(config, "model.num_heads")
    else:
        raise ConfigError(f"model.baseline_type must be b0_global or b1_cross_attention, got {baseline_type!r}")
    if bool(require_key(config, "tokenizer.enabled")) and not bool(require_key(config, "tokenizer.use_fast")):
        raise ConfigError("tokenizer.use_fast must be true when tokenizer.enabled is true")
    if int(require_key(config, "training.gradient_accumulation_steps")) <= 0:
        raise ConfigError("training.gradient_accumulation_steps must be positive")
    if int(require_key(config, "dataloader.batch_size")) <= 0:
        raise ConfigError("dataloader.batch_size must be positive")
    if int(require_key(config, "dataloader.num_workers")) < 0:
        raise ConfigError("dataloader.num_workers must be non-negative")
    precision = str(require_key(config, "runtime.precision")).lower()
    model_dtype = str(require_key(config, "runtime.model_dtype")).lower()
    if precision in {"bf16", "bfloat16", "fp16", "float16"} and model_dtype not in {"fp32", "float32"}:
        raise ConfigError(
            "runtime.model_dtype must be fp32 for AMP training/inference; runtime.precision controls autocast"
        )
    if mode in {"train", "evaluate"}:
        freeze_epochs = int(require_key(config, "training.stages.stage_a_freeze_epochs"))
        full_unfreeze_enabled = bool(require_key(config, "training.stages.stage_b_full_unfreeze.enabled"))
        full_unfreeze_start = int(require_key(config, "training.stages.stage_b_full_unfreeze.start_epoch"))
        if not full_unfreeze_enabled:
            raise ConfigError("Controlled B0/B1 training requires full-backbone fine-tuning after warm-up")
        if full_unfreeze_start != freeze_epochs + 1:
            raise ConfigError(
                "training.stages.stage_b_full_unfreeze.start_epoch must immediately follow "
                "training.stages.stage_a_freeze_epochs"
            )
        if int(require_key(config, "training.micro_batch_size")) != int(
            require_key(config, "dataloader.batch_size")
        ):
            raise ConfigError("training.micro_batch_size must equal dataloader.batch_size")
        if str(require_key(config, "training.early_stopping.mode")) not in {"max", "min"}:
            raise ConfigError("training.early_stopping.mode must be max or min")
        if int(require_key(config, "training.early_stopping.patience")) < 0:
            raise ConfigError("training.early_stopping.patience must be non-negative")
        if float(require_key(config, "training.early_stopping.min_delta")) < 0:
            raise ConfigError("training.early_stopping.min_delta must be non-negative")
        if int(require_key(config, "training.early_stopping.minimum_backbone_adaptation_epochs")) < 1:
            raise ConfigError("training.early_stopping.minimum_backbone_adaptation_epochs must be positive")
        if str(require_key(config, "scheduler.resume_horizon_policy")) not in {
            "error",
            "continue_from_checkpoint_lr",
        }:
            raise ConfigError(
                "scheduler.resume_horizon_policy must be error or continue_from_checkpoint_lr"
            )
    if mode == "evaluate":
        checkpoint_path = require_key(config, "evaluation.checkpoint_path")
        thresholds_path = require_key(config, "evaluation.thresholds_path")
        explicit_checkpoint = _is_explicit_evaluation_path(checkpoint_path)
        explicit_thresholds = _is_explicit_evaluation_path(thresholds_path)
        if explicit_checkpoint != explicit_thresholds:
            raise ConfigError(
                "evaluation.checkpoint_path and evaluation.thresholds_path must both be explicit or both be resolved"
            )
        resolution_enabled = bool(require_key(config, "evaluation.checkpoint_resolution.enabled"))
        if not explicit_checkpoint and not resolution_enabled:
            raise ConfigError(
                "evaluation checkpoint paths are missing or contain placeholders; enable evaluation.checkpoint_resolution"
            )
        resolution_mode = str(require_key(config, "evaluation.checkpoint_resolution.mode"))
        if resolution_mode not in {"latest", "run_dir"}:
            raise ConfigError("evaluation.checkpoint_resolution.mode must be latest or run_dir")
        if resolution_mode == "run_dir" and require_key(
            config, "evaluation.checkpoint_resolution.run_dir"
        ) in (None, ""):
            raise ConfigError(
                "evaluation.checkpoint_resolution.run_dir is required when mode=run_dir"
            )


def validate_evidence_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate mandatory Phase-3 evidence smoke/inspection configuration."""

    mandatory = [
        "project.name",
        "project.seed",
        "dataset.root",
        "dataset.train_annotations",
        "dataset.val_annotations",
        "dataset.test_annotations",
        "dataset.image_root",
        "image.size",
        "image.preserve_aspect_ratio",
        "image.padding",
        "tokenizer.enabled",
        "tokenizer.name",
        "tokenizer.use_fast",
        "tokenizer.max_length",
        "tokenizer.ignore_index",
        "backbones.siglip2.name",
        "backbones.siglip2.image_mean",
        "backbones.siglip2.image_std",
        "backbones.siglip2.text_max_length",
        "backbones.dinov3.name",
        "backbones.dinov3.image_mean",
        "backbones.dinov3.image_std",
        "backbones.deberta.name",
        "evidence.hidden_dim",
        "evidence.visual_semantic.enabled",
        "evidence.visual_semantic.max_nodes",
        "evidence.visual_semantic.budgeting",
        "evidence.visual_forensic.enabled",
        "evidence.visual_forensic.max_nodes",
        "evidence.visual_forensic.budgeting",
        "evidence.face.enabled",
        "evidence.face.truncate",
        "evidence.text_alignment.enabled",
        "evidence.text_alignment.max_nodes",
        "evidence.text_context.enabled",
        "evidence.text_context.max_nodes",
        "evidence.cross_modal.enabled",
        "evidence.cross_modal.similarity",
        "dataloader.batch_size",
        "dataloader.num_workers",
        "dataloader.pin_memory",
        "dataloader.persistent_workers",
        "dataloader.drop_last",
        "dataloader.seed",
        "runtime.device",
        "runtime.precision",
        "runtime.model_dtype",
        "progress.enabled",
        "logging.level",
        "output.root",
    ]
    if mode == "smoke":
        mandatory.extend(["smoke.split", "smoke.max_samples", "smoke.inspect_batches"])
    elif mode == "inspect":
        mandatory.extend(["inspection.split", "inspection.max_samples", "inspection.inspect_batches"])
    else:
        raise ConfigError(f"Unknown evidence config mode: {mode!r}")
    for key in mandatory:
        require_key(config, key)
    if Path(str(require_key(config, "dataset.root"))).as_posix() != "data/DGM4":
        raise ConfigError("dataset.root must be the canonical value: data/DGM4")
    if Path(str(require_key(config, "dataset.image_root"))).as_posix() != "data/DGM4/images":
        raise ConfigError("dataset.image_root must be the canonical value: data/DGM4/images")
    if int(require_key(config, "image.size")) != 384:
        raise ConfigError("image.size must be the canonical Phase-3 value: 384")
    if not bool(require_key(config, "image.preserve_aspect_ratio")):
        raise ConfigError("image.preserve_aspect_ratio must be true")
    if not bool(require_key(config, "image.padding")):
        raise ConfigError("image.padding must be true")
    expected_backbones = {
        "backbones.siglip2.name": "google/siglip2-so400m-patch14-384",
        "backbones.dinov3.name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "backbones.deberta.name": "microsoft/deberta-v3-large",
        "tokenizer.name": "microsoft/deberta-v3-large",
    }
    for key, expected in expected_backbones.items():
        if str(require_key(config, key)) != expected:
            raise ConfigError(f"{key} must be canonical value: {expected}")
    if bool(require_key(config, "tokenizer.enabled")) and not bool(require_key(config, "tokenizer.use_fast")):
        raise ConfigError("tokenizer.use_fast must be true when tokenizer.enabled is true")
    if int(require_key(config, "evidence.hidden_dim")) <= 0:
        raise ConfigError("evidence.hidden_dim must be positive")
    for family in ("visual_semantic", "visual_forensic", "text_alignment", "text_context"):
        if int(require_key(config, f"evidence.{family}.max_nodes")) <= 0:
            raise ConfigError(f"evidence.{family}.max_nodes must be positive")
    if bool(require_key(config, "evidence.face.truncate")):
        max_nodes = get_key(config, "evidence.face.max_nodes")
        if max_nodes is None or int(max_nodes) <= 0:
            raise ConfigError("evidence.face.max_nodes must be positive when face.truncate is true")
    if str(require_key(config, "evidence.cross_modal.similarity")) != "cosine":
        raise ConfigError("evidence.cross_modal.similarity currently supports only cosine")
    if int(require_key(config, "dataloader.batch_size")) <= 0:
        raise ConfigError("dataloader.batch_size must be positive")
    if int(require_key(config, "dataloader.num_workers")) < 0:
        raise ConfigError("dataloader.num_workers must be non-negative")


def validate_graph_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate Phase-4 graph smoke/inspection configuration."""

    validate_evidence_config(config, mode="smoke" if mode == "smoke" else "inspect")
    mandatory = [
        "graph.type",
        "graph.hidden_dim",
        "graph.layers",
        "graph.heads",
        "graph.dropout",
        "graph.hypothesis_init_std",
        "graph.reasoning.cycles",
        "graph.reasoning.hypothesis_feedback",
        "graph.edges.visual_semantic_visual_forensic.method",
        "graph.edges.visual_semantic_visual_forensic.k",
        "graph.edges.visual_semantic_visual_forensic.bidirectional",
        "graph.edges.visual_semantic_text_alignment.method",
        "graph.edges.visual_semantic_text_alignment.k",
        "graph.edges.visual_semantic_text_alignment.bidirectional",
        "graph.edges.text_alignment_text_context.method",
        "graph.edges.text_alignment_text_context.bidirectional",
        "graph.edges.semantic_to_hypotheses.enabled",
        "graph.edges.feedback.enabled",
        "graph.edges.feedback.semantic_feedback",
    ]
    for key in mandatory:
        require_key(config, key)
    if str(require_key(config, "graph.type")).lower() != "hgt":
        raise ConfigError("graph.type must be hgt in Phase 4")
    hidden_dim = int(require_key(config, "graph.hidden_dim"))
    heads = int(require_key(config, "graph.heads"))
    layers = int(require_key(config, "graph.layers"))
    cycles = int(require_key(config, "graph.reasoning.cycles"))
    if hidden_dim <= 0:
        raise ConfigError("graph.hidden_dim must be positive")
    if heads <= 0:
        raise ConfigError("graph.heads must be positive")
    if hidden_dim % heads != 0:
        raise ConfigError("graph.hidden_dim must be divisible by graph.heads")
    if layers < 1 or layers > 3:
        raise ConfigError("graph.layers must be in the Phase-4 supported range [1,3]")
    if cycles < 1 or cycles > 2:
        raise ConfigError("graph.reasoning.cycles must be 1 or 2")
    if float(require_key(config, "graph.dropout")) < 0:
        raise ConfigError("graph.dropout must be non-negative")
    allowed_edge_keys = {
        "visual_semantic_visual_forensic",
        "visual_semantic_text_alignment",
        "text_alignment_text_context",
        "semantic_to_hypotheses",
        "feedback",
    }
    actual_edge_keys = set(require_key(config, "graph.edges").keys())
    unknown_edge_keys = sorted(actual_edge_keys - allowed_edge_keys)
    if unknown_edge_keys:
        raise ConfigError(f"Unknown graph edge config keys: {unknown_edge_keys}")
    methods = {
        "graph.edges.visual_semantic_visual_forensic.method": "spatial_knn",
        "graph.edges.visual_semantic_text_alignment.method": "similarity_topk",
        "graph.edges.text_alignment_text_context.method": "word_aligned",
    }
    for key, expected in methods.items():
        actual = str(require_key(config, key))
        if actual != expected:
            raise ConfigError(f"{key} must be {expected}, got {actual!r}")
    for key in (
        "graph.edges.visual_semantic_visual_forensic.k",
        "graph.edges.visual_semantic_text_alignment.k",
    ):
        if int(require_key(config, key)) <= 0:
            raise ConfigError(f"{key} must be positive")


def validate_decision_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate Phase-5 decision smoke/inspection configuration."""

    validate_graph_config(config, mode=mode)
    mandatory = [
        "model.hidden_dim",
        "model.hypotheses.order",
        "model.attribution.hidden_dim",
        "model.attribution.dropout",
        "model.grounding.image.scoring",
        "model.grounding.image.dropout",
        "model.grounding.text.scoring",
        "model.grounding.text.dropout",
        "model.grounding.text.full_token_store",
        "model.grounding.text.token_input_dim",
        "model.binary.aggregation",
        "loss.lambda_attr",
        "loss.lambda_bin",
        "loss.lambda_img",
        "loss.lambda_txt",
    ]
    for key in mandatory:
        require_key(config, key)
    expected_order = ["FS", "FA", "TS", "TA"]
    if list(require_key(config, "model.hypotheses.order")) != expected_order:
        raise ConfigError(f"model.hypotheses.order must be {expected_order}")
    graph_hidden_dim = int(require_key(config, "graph.hidden_dim"))
    model_hidden_dim = int(require_key(config, "model.hidden_dim"))
    attribution_hidden_dim = int(require_key(config, "model.attribution.hidden_dim"))
    if model_hidden_dim != graph_hidden_dim:
        raise ConfigError("model.hidden_dim must match graph.hidden_dim")
    if attribution_hidden_dim != graph_hidden_dim:
        raise ConfigError("model.attribution.hidden_dim must match graph.hidden_dim")
    if str(require_key(config, "model.binary.aggregation")) != "noisy_or":
        raise ConfigError("model.binary.aggregation must be noisy_or")
    if str(require_key(config, "model.grounding.image.scoring")) != "dot_product":
        raise ConfigError("model.grounding.image.scoring must be dot_product")
    if str(require_key(config, "model.grounding.text.scoring")) != "dot_product":
        raise ConfigError("model.grounding.text.scoring must be dot_product")
    if not bool(require_key(config, "model.grounding.text.full_token_store")):
        raise ConfigError("model.grounding.text.full_token_store must be true")
    if int(require_key(config, "model.grounding.text.token_input_dim")) <= 0:
        raise ConfigError("model.grounding.text.token_input_dim must be positive")
    for key in (
        "model.attribution.dropout",
        "model.grounding.image.dropout",
        "model.grounding.text.dropout",
        "loss.lambda_attr",
        "loss.lambda_bin",
        "loss.lambda_img",
        "loss.lambda_txt",
    ):
        if float(require_key(config, key)) < 0:
            raise ConfigError(f"{key} must be non-negative")


def validate_routing_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate Phase-6 routing smoke/inspection configuration."""

    validate_decision_config(config, mode=mode)
    mandatory = [
        "routing.enabled",
        "routing.mode",
        "routing.feature_dim",
        "routing.estimator.hidden_dim",
        "routing.estimator.dropout",
        "routing.estimator.initial_bias",
        "routing.estimator.output_weight_std",
        "routing.visual_semantic.enabled",
        "routing.visual_forensic.enabled",
        "routing.face.enabled",
        "routing.text_alignment.enabled",
        "routing.text_context.enabled",
        "routing.hard_eval.enabled",
        "routing.hard_eval.threshold",
        "routing.diagnostics.low_threshold",
        "routing.diagnostics.high_threshold",
        "routing.regularization.enabled",
        "routing.regularization.weight",
    ]
    for key in mandatory:
        require_key(config, key)
    feature_dim = int(require_key(config, "routing.feature_dim"))
    evidence_dim = int(require_key(config, "evidence.hidden_dim"))
    graph_dim = int(require_key(config, "graph.hidden_dim"))
    if feature_dim != evidence_dim or feature_dim != graph_dim:
        raise ConfigError("routing.feature_dim must match evidence.hidden_dim and graph.hidden_dim")
    if int(require_key(config, "routing.estimator.hidden_dim")) <= 0:
        raise ConfigError("routing.estimator.hidden_dim must be positive")
    for key in ("routing.estimator.dropout", "routing.estimator.output_weight_std"):
        if float(require_key(config, key)) < 0:
            raise ConfigError(f"{key} must be non-negative")
    route_mode = str(require_key(config, "routing.mode"))
    if route_mode not in {"soft", "hard"}:
        raise ConfigError("routing.mode must be soft or hard")
    if route_mode == "hard" and not bool(require_key(config, "routing.hard_eval.enabled")):
        raise ConfigError("routing.mode=hard requires routing.hard_eval.enabled=true")
    for key in (
        "routing.hard_eval.threshold",
        "routing.diagnostics.low_threshold",
        "routing.diagnostics.high_threshold",
    ):
        value = float(require_key(config, key))
        if value < 0 or value > 1:
            raise ConfigError(f"{key} must be in [0,1]")
    if float(require_key(config, "routing.diagnostics.low_threshold")) >= float(
        require_key(config, "routing.diagnostics.high_threshold")
    ):
        raise ConfigError("routing diagnostic low_threshold must be < high_threshold")
    if bool(require_key(config, "routing.regularization.enabled")) and float(
        require_key(config, "routing.regularization.weight")
    ) <= 0:
        raise ConfigError("routing.regularization.weight must be positive when enabled")
    if bool(require_key(config, "routing.regularization.enabled")):
        target_mean = float(require_key(config, "routing.regularization.target_mean"))
        min_std = float(require_key(config, "routing.regularization.min_std"))
        if target_mean <= 0 or target_mean > 1:
            raise ConfigError("routing.regularization.target_mean must be in (0,1]")
        if min_std < 0 or min_std >= 0.5:
            raise ConfigError("routing.regularization.min_std must be in [0,0.5)")
    if not bool(require_key(config, "routing.regularization.enabled")) and float(
        require_key(config, "routing.regularization.weight")
    ) != 0.0:
        raise ConfigError("routing.regularization.weight must be 0.0 when regularization is disabled")


def validate_hera_config(config: Mapping[str, Any], *, mode: str) -> None:
    """Validate canonical HERA or one registered production ablation."""

    mandatory = [
        "project.name",
        "project.seed",
        "project.architecture_version",
        "dataset.root",
        "dataset.train_annotations",
        "dataset.val_annotations",
        "dataset.test_annotations",
        "dataset.image_root",
        "dataset.use_mtcnn_boxes",
        "image.size",
        "image.preserve_aspect_ratio",
        "image.padding",
        "tokenizer.enabled",
        "tokenizer.name",
        "tokenizer.use_fast",
        "tokenizer.max_length",
        "tokenizer.ignore_index",
        "backbones.siglip2.name",
        "backbones.siglip2.image_mean",
        "backbones.siglip2.image_std",
        "backbones.siglip2.text_max_length",
        "backbones.dinov3.name",
        "backbones.dinov3.image_mean",
        "backbones.dinov3.image_std",
        "backbones.deberta.name",
        "evidence.hidden_dim",
        "evidence.visual_semantic.enabled",
        "evidence.visual_semantic.max_nodes",
        "evidence.visual_forensic.enabled",
        "evidence.visual_forensic.max_nodes",
        "evidence.face.enabled",
        "evidence.text_alignment.enabled",
        "evidence.text_alignment.max_nodes",
        "evidence.text_context.enabled",
        "evidence.text_context.max_nodes",
        "evidence.cross_modal.enabled",
        "evidence.cross_modal.similarity",
        "routing.enabled",
        "routing.mode",
        "routing.feature_dim",
        "routing.estimator.hidden_dim",
        "routing.estimator.dropout",
        "routing.estimator.initial_bias",
        "routing.estimator.output_weight_std",
        "routing.visual_semantic.enabled",
        "routing.visual_forensic.enabled",
        "routing.face.enabled",
        "routing.text_alignment.enabled",
        "routing.text_context.enabled",
        "routing.hard_eval.enabled",
        "routing.hard_eval.threshold",
        "routing.diagnostics.low_threshold",
        "routing.diagnostics.high_threshold",
        "routing.regularization.enabled",
        "routing.regularization.weight",
        "graph.type",
        "graph.hidden_dim",
        "graph.layers",
        "graph.heads",
        "graph.dropout",
        "graph.hypothesis_init_std",
        "graph.reasoning.cycles",
        "graph.reasoning.hypothesis_feedback",
        "graph.reasoning.dynamic_rebuild",
        "graph.reasoning.hypothesis_edge_reweighting",
        "graph.edge_attention.enabled",
        "graph.edge_attention.confidence_gate",
        "graph.edges.visual_semantic_visual_forensic.method",
        "graph.edges.visual_semantic_visual_forensic.k",
        "graph.edges.visual_semantic_visual_forensic.bidirectional",
        "graph.edges.visual_semantic_text_alignment.method",
        "graph.edges.visual_semantic_text_alignment.k",
        "graph.edges.visual_semantic_text_alignment.bidirectional",
        "graph.edges.text_alignment_text_context.method",
        "graph.edges.text_alignment_text_context.bidirectional",
        "graph.edges.semantic_to_hypotheses.enabled",
        "graph.edges.feedback.enabled",
        "graph.edges.feedback.semantic_feedback",
        "model.model_type",
        "model.hidden_dim",
        "model.hypotheses.order",
        "model.attribution.hidden_dim",
        "model.attribution.dropout",
        "model.grounding.image.scoring",
        "model.grounding.image.dropout",
        "model.grounding.image.protocol",
        "model.grounding.routing",
        "model.grounding.routing_temperature",
        "model.grounding.text.scoring",
        "model.grounding.text.dropout",
        "model.grounding.text.full_token_store",
        "model.grounding.text.token_input_dim",
        "model.binary.aggregation",
        "loss.lambda_attr",
        "loss.lambda_bin",
        "loss.lambda_img",
        "loss.lambda_txt",
        "loss.lambda_ev",
        "loss.lambda_align",
        "loss.image_grounding.l1_weight",
        "loss.image_grounding.giou_weight",
        "loss.image_grounding.heatmap_weight",
        "training.epochs",
        "training.micro_batch_size",
        "training.gradient_accumulation_steps",
        "training.auto_resume.enabled",
        "training.auto_resume.run_dir",
        "training.auto_resume.checkpoint_name",
        "training.early_stopping.enabled",
        "training.early_stopping.metric",
        "training.early_stopping.mode",
        "training.early_stopping.patience",
        "training.early_stopping.min_delta",
        "training.early_stopping.minimum_backbone_adaptation_epochs",
        "training.stages.stage_a_freeze_epochs",
        "training.stages.stage_b_full_unfreeze.enabled",
        "training.stages.stage_b_full_unfreeze.start_epoch",
        "optimizer.type",
        "optimizer.lr_new",
        "optimizer.lr_backbones.siglip2",
        "optimizer.lr_backbones.dinov3",
        "optimizer.lr_backbones.deberta",
        "optimizer.weight_decay",
        "scheduler.type",
        "scheduler.resume_horizon_policy",
        "thresholding.enabled",
        "checkpointing.selection_metric",
        "checkpointing.top_k",
        "dataloader.batch_size",
        "dataloader.num_workers",
        "dataloader.pin_memory",
        "dataloader.persistent_workers",
        "dataloader.drop_last",
        "dataloader.seed",
        "runtime.device",
        "runtime.precision",
        "runtime.model_dtype",
        "runtime.cuda_memory_logging.enabled",
        "runtime.gradient_checkpointing.enabled",
        "runtime.gradient_checkpointing.only_trainable_backbones",
        "runtime.gradient_checkpointing.use_reentrant",
        "runtime.gradient_checkpointing.backbones.siglip2",
        "runtime.gradient_checkpointing.backbones.dinov3",
        "runtime.gradient_checkpointing.backbones.deberta",
        "progress.enabled",
        "logging.level",
        "output.root",
    ]
    if mode == "evaluate":
        mandatory.extend(
            [
                "evaluation.checkpoint_path",
                "evaluation.thresholds_path",
                "evaluation.split",
                "evaluation.checkpoint_resolution.enabled",
                "evaluation.checkpoint_resolution.mode",
                "evaluation.checkpoint_resolution.output_root",
                "evaluation.checkpoint_resolution.run_dir",
                "evaluation.checkpoint_resolution.checkpoint_name",
                "evaluation.checkpoint_resolution.thresholds_name",
            ]
        )
    elif mode == "smoke":
        mandatory.extend(["smoke.split", "smoke.max_samples", "smoke.inspect_batches"])
    elif mode != "train":
        raise ConfigError(f"Unknown HERA config mode: {mode!r}")
    for key in mandatory:
        require_key(config, key)
    if bool(require_key(config, "dataset.use_mtcnn_boxes")):
        raise ConfigError("Canonical HERA forbids annotation-derived dataset.mtcnn_boxes")
    if str(require_key(config, "project.architecture_version")) != "hera_final_v1":
        raise ConfigError("project.architecture_version must be hera_final_v1")
    _validate_common_canonical_data_and_backbones(config, phase_name="Phase-7")
    if str(require_key(config, "model.model_type")) != "hera":
        raise ConfigError("model.model_type must be hera for canonical Phase-7 HERA")
    if list(require_key(config, "model.hypotheses.order")) != ["FS", "FA", "TS", "TA"]:
        raise ConfigError("model.hypotheses.order must be ['FS', 'FA', 'TS', 'TA']")
    hidden_dim = int(require_key(config, "model.hidden_dim"))
    for key in ("evidence.hidden_dim", "routing.feature_dim", "graph.hidden_dim", "model.attribution.hidden_dim"):
        if int(require_key(config, key)) != hidden_dim:
            raise ConfigError(f"{key} must match model.hidden_dim")
    ablation = config.get("ablation", {})
    ablation_enabled = isinstance(ablation, Mapping) and bool(ablation.get("enabled", False))
    graph_type = str(require_key(config, "graph.type")).lower()
    evidence_mode = str(config["graph"].get("evidence_mode", "heterogeneous")).lower()
    grounding_mode = str(config["model"]["grounding"].get("mode", "hypothesis_conditioned"))
    binary_aggregation = str(require_key(config, "model.binary.aggregation"))
    if ablation_enabled:
        from src.experiments import validate_hera_ablation_contract

        try:
            validate_hera_ablation_contract(config)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    else:
        if graph_type != "hgt":
            raise ConfigError("graph.type must be hgt for canonical HERA")
        if evidence_mode != "heterogeneous":
            raise ConfigError("graph.evidence_mode must be heterogeneous for canonical HERA")
        if grounding_mode != "hypothesis_conditioned":
            raise ConfigError("model.grounding.mode must be hypothesis_conditioned for canonical HERA")
        if binary_aggregation != "noisy_or":
            raise ConfigError("model.binary.aggregation must be noisy_or")
    if bool(require_key(config, "evidence.face.enabled")):
        raise ConfigError("Final HERA requires evidence.face.enabled=false under dense bbox grounding")
    if bool(require_key(config, "routing.face.enabled")):
        raise ConfigError("Final HERA requires routing.face.enabled=false under dense bbox grounding")
    if int(require_key(config, "graph.reasoning.cycles")) != 2:
        raise ConfigError("Final HERA requires exactly two iterative reasoning cycles")
    if not bool(require_key(config, "graph.reasoning.dynamic_rebuild")):
        raise ConfigError("Final HERA requires dynamic graph rebuild between reasoning cycles")
    if str(require_key(config, "graph.reasoning.hypothesis_edge_reweighting")) != "cosine":
        raise ConfigError("graph.reasoning.hypothesis_edge_reweighting must be cosine")
    if not bool(require_key(config, "graph.edge_attention.enabled")):
        raise ConfigError("Final HERA requires reliability-aware edge attention")
    if str(require_key(config, "graph.edge_attention.confidence_gate")) != "direct":
        raise ConfigError("graph.edge_attention.confidence_gate must be direct")
    if str(require_key(config, "model.grounding.image.protocol")) != "dense_bbox":
        raise ConfigError("Final HERA image grounding protocol must be dense_bbox")
    if graph_type not in {"hgt", "gatv2"}:
        raise ConfigError("graph.type must be hgt or gatv2")
    if evidence_mode not in {"heterogeneous", "homogeneous"}:
        raise ConfigError("graph.evidence_mode must be heterogeneous or homogeneous")
    if grounding_mode not in {"hypothesis_conditioned", "generic"}:
        raise ConfigError("model.grounding.mode must be hypothesis_conditioned or generic")
    if str(require_key(config, "model.grounding.routing")) not in {"soft", "hard"}:
        raise ConfigError("model.grounding.routing must be soft or hard")
    if float(require_key(config, "model.grounding.routing_temperature")) <= 0:
        raise ConfigError("model.grounding.routing_temperature must be positive")
    if binary_aggregation not in {"noisy_or", "independent_head"}:
        raise ConfigError("model.binary.aggregation must be noisy_or or independent_head")
    if not bool(require_key(config, "model.grounding.text.full_token_store")):
        raise ConfigError("model.grounding.text.full_token_store must be true")
    freeze_epochs = int(require_key(config, "training.stages.stage_a_freeze_epochs"))
    full_unfreeze_enabled = bool(require_key(config, "training.stages.stage_b_full_unfreeze.enabled"))
    full_unfreeze_start = int(require_key(config, "training.stages.stage_b_full_unfreeze.start_epoch"))
    if not full_unfreeze_enabled:
        raise ConfigError("Canonical HERA training requires full-backbone fine-tuning after warm-up")
    if full_unfreeze_start != freeze_epochs + 1:
        raise ConfigError(
            "training.stages.stage_b_full_unfreeze.start_epoch must immediately follow "
            "training.stages.stage_a_freeze_epochs"
        )
    if int(require_key(config, "training.gradient_accumulation_steps")) <= 0:
        raise ConfigError("training.gradient_accumulation_steps must be positive")
    if int(require_key(config, "training.micro_batch_size")) != int(require_key(config, "dataloader.batch_size")):
        raise ConfigError("training.micro_batch_size must equal dataloader.batch_size")
    if str(require_key(config, "training.early_stopping.mode")) not in {"max", "min"}:
        raise ConfigError("training.early_stopping.mode must be max or min")
    if int(require_key(config, "training.early_stopping.patience")) < 0:
        raise ConfigError("training.early_stopping.patience must be non-negative")
    if float(require_key(config, "training.early_stopping.min_delta")) < 0:
        raise ConfigError("training.early_stopping.min_delta must be non-negative")
    if int(require_key(config, "training.early_stopping.minimum_backbone_adaptation_epochs")) < 1:
        raise ConfigError("training.early_stopping.minimum_backbone_adaptation_epochs must be positive")
    if str(require_key(config, "scheduler.resume_horizon_policy")) not in {
        "error",
        "continue_from_checkpoint_lr",
    }:
        raise ConfigError(
            "scheduler.resume_horizon_policy must be error or continue_from_checkpoint_lr"
        )
    for key in (
        "loss.lambda_attr",
        "loss.lambda_bin",
        "loss.lambda_img",
        "loss.lambda_txt",
        "loss.lambda_ev",
        "loss.lambda_align",
        "loss.image_grounding.l1_weight",
        "loss.image_grounding.giou_weight",
        "loss.image_grounding.heatmap_weight",
        "model.attribution.dropout",
        "model.grounding.image.dropout",
        "model.grounding.text.dropout",
        "optimizer.weight_decay",
    ):
        if float(require_key(config, key)) < 0:
            raise ConfigError(f"{key} must be non-negative")
    if int(require_key(config, "dataloader.batch_size")) <= 0:
        raise ConfigError("dataloader.batch_size must be positive")
    if int(require_key(config, "dataloader.num_workers")) < 0:
        raise ConfigError("dataloader.num_workers must be non-negative")
    low = float(require_key(config, "routing.diagnostics.low_threshold"))
    high = float(require_key(config, "routing.diagnostics.high_threshold"))
    if low < 0 or high > 1 or low >= high:
        raise ConfigError("routing diagnostics thresholds must satisfy 0 <= low < high <= 1")
    regularization_enabled = bool(require_key(config, "routing.regularization.enabled"))
    regularization_weight = float(require_key(config, "routing.regularization.weight"))
    if regularization_enabled and regularization_weight <= 0:
        raise ConfigError("routing.regularization.weight must be positive when enabled")
    if not regularization_enabled and regularization_weight != 0.0:
        raise ConfigError("routing.regularization.weight must be 0.0 when disabled")
    if mode == "evaluate":
        resolution = require_key(config, "evaluation.checkpoint_resolution")
        resolution_enabled = bool(require_key(config, "evaluation.checkpoint_resolution.enabled"))
        checkpoint_path = require_key(config, "evaluation.checkpoint_path")
        thresholds_path = require_key(config, "evaluation.thresholds_path")
        explicit_checkpoint = _is_explicit_evaluation_path(checkpoint_path)
        explicit_thresholds = _is_explicit_evaluation_path(thresholds_path)
        if explicit_checkpoint != explicit_thresholds:
            raise ConfigError("evaluation.checkpoint_path and evaluation.thresholds_path must both be explicit or both be resolved")
        if not explicit_checkpoint and not resolution_enabled:
            raise ConfigError(
                "evaluation checkpoint paths are missing or contain placeholders; enable evaluation.checkpoint_resolution"
            )
        resolution_mode = str(require_key(config, "evaluation.checkpoint_resolution.mode"))
        if resolution_mode not in {"latest", "run_dir"}:
            raise ConfigError("evaluation.checkpoint_resolution.mode must be latest or run_dir")
        if resolution_mode == "run_dir" and require_key(config, "evaluation.checkpoint_resolution.run_dir") in (None, ""):
            raise ConfigError("evaluation.checkpoint_resolution.run_dir is required when mode=run_dir")
        for key in (
            "evaluation.checkpoint_resolution.output_root",
            "evaluation.checkpoint_resolution.checkpoint_name",
            "evaluation.checkpoint_resolution.thresholds_name",
        ):
            if require_key(config, key) in (None, ""):
                raise ConfigError(f"{key} must be non-empty")
        if not isinstance(resolution, Mapping):
            raise ConfigError("evaluation.checkpoint_resolution must be a mapping")


def validate_server_validation_config(config: Mapping[str, Any]) -> None:
    """Validate the cumulative server-validation YAML."""

    mandatory = [
        "project.name",
        "project.seed",
        "output.root",
        "progress.enabled",
        "server.expected_gpu_vram_gb",
        "server.expected_disk_gb",
        "server.cuda_environment",
        "validation.stop_on_failure",
        "validation.resume_completed_checks",
        "validation.resume_require_command_match",
        "validation.resume_require_config_hash_match",
        "validation.resume_require_git_commit_match",
        "validation.force_rerun_all",
        "validation.phases",
    ]
    for key in mandatory:
        require_key(config, key)
    phases = require_key(config, "validation.phases")
    if not isinstance(phases, list) or not phases:
        raise ConfigError("validation.phases must be a non-empty list")
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ConfigError("Each server-validation phase must be a mapping")
        for key in ("name", "output_file", "checks"):
            if key not in phase:
                raise ConfigError(f"Server-validation phase is missing {key!r}")
        checks = phase["checks"]
        if not isinstance(checks, list) or not checks:
            raise ConfigError(f"Server-validation phase {phase['name']!r} must contain checks")
        for check in checks:
            if not isinstance(check, Mapping):
                raise ConfigError(f"Check in phase {phase['name']!r} must be a mapping")
            for key in ("name", "command", "required_for_training"):
                if key not in check:
                    raise ConfigError(f"Check in phase {phase['name']!r} is missing {key!r}")
            command = check["command"]
            if not isinstance(command, list) or len(command) < 2:
                raise ConfigError(f"Check {check['name']!r} command must be a list like ['python', 'script.py', 'config.yaml']")


def _validate_common_canonical_data_and_backbones(config: Mapping[str, Any], *, phase_name: str) -> None:
    if Path(str(require_key(config, "dataset.root"))).as_posix() != "data/DGM4":
        raise ConfigError("dataset.root must be the canonical value: data/DGM4")
    if Path(str(require_key(config, "dataset.image_root"))).as_posix() != "data/DGM4/images":
        raise ConfigError("dataset.image_root must be the canonical value: data/DGM4/images")
    if int(require_key(config, "image.size")) != 384:
        raise ConfigError(f"image.size must be the canonical {phase_name} value: 384")
    if not bool(require_key(config, "image.preserve_aspect_ratio")):
        raise ConfigError("image.preserve_aspect_ratio must be true")
    if not bool(require_key(config, "image.padding")):
        raise ConfigError("image.padding must be true")
    expected_backbones = {
        "backbones.siglip2.name": "google/siglip2-so400m-patch14-384",
        "backbones.dinov3.name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "backbones.deberta.name": "microsoft/deberta-v3-large",
        "tokenizer.name": "microsoft/deberta-v3-large",
    }
    for key, expected in expected_backbones.items():
        if str(require_key(config, key)) != expected:
            raise ConfigError(f"{key} must be canonical value: {expected}")
    if bool(require_key(config, "tokenizer.enabled")) and not bool(require_key(config, "tokenizer.use_fast")):
        raise ConfigError("tokenizer.use_fast must be true when tokenizer.enabled is true")
    precision = str(require_key(config, "runtime.precision")).lower()
    model_dtype = str(require_key(config, "runtime.model_dtype")).lower()
    if precision in {"bf16", "bfloat16", "fp16", "float16"} and model_dtype not in {"fp32", "float32"}:
        raise ConfigError(
            "runtime.model_dtype must be fp32 for AMP training/inference; runtime.precision controls autocast"
        )


def _is_explicit_evaluation_path(value: Any) -> bool:
    if value in (None, ""):
        return False
    text = str(value)
    return "<run_id>" not in text and "RUN" not in text


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a mapping to YAML."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)

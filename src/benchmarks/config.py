"""Validation for the two standalone HERA binary benchmark YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.utils.config import ConfigError, require_key

from .contracts import PRIMARY_TASK, SUPPORTED_DATASETS, SignalContract


def validate_binary_benchmark_config(
    config: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    """Validate a train, evaluate, or prepare binary benchmark configuration."""

    if mode not in {"prepare", "train", "evaluate"}:
        raise ConfigError(f"Unknown binary benchmark mode: {mode!r}")
    required = [
        "project.name",
        "project.seed",
        "task.type",
        "task.primary_target",
        "task.negative_name",
        "task.positive_name",
        "task.signals.binary",
        "task.signals.attribution",
        "task.signals.image_grounding",
        "task.signals.text_grounding",
        "task.signals.subtype",
        "task.signals.text_label",
        "dataset.name",
        "dataset.adapter",
        "dataset.root",
        "dataset.verify_image_exists",
        "dataset.split_protocol",
        "dataset.cross_split_duplicate_policy",
        "dataset.preflight.full_decode",
        "dataset.text.fields",
        "dataset.text.separator",
        "image.size",
        "image.preserve_aspect_ratio",
        "image.padding",
        "tokenizer.enabled",
        "tokenizer.name",
        "tokenizer.use_fast",
        "tokenizer.max_length",
        "tokenizer.ignore_index",
        "backbones.siglip2.name",
        "backbones.dinov3.name",
        "backbones.deberta.name",
        "model.model_type",
        "model.latent_slots.count",
        "model.latent_slots.roles",
        "model.binary.aggregation",
        "model.binary.dropout",
        "loss.lambda_bin",
        "loss.lambda_align",
        "loss.lambda_router",
        "loss.lambda_subtype",
        "loss.lambda_text_label",
        "training.epochs",
        "training.micro_batch_size",
        "training.gradient_accumulation_steps",
        "training.auto_resume.enabled",
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
        "scheduler.warmup_steps",
        "scheduler.resume_horizon_policy",
        "thresholding.binary.min",
        "thresholding.binary.max",
        "thresholding.binary.step",
        "checkpointing.selection_metric",
        "checkpointing.keep_last",
        "checkpointing.keep_best",
        "checkpointing.best_weights_only",
        "checkpointing.atomic_write",
        "checkpointing.minimum_free_gib",
        "dataloader.batch_size",
        "dataloader.num_workers",
        "dataloader.worker_threads",
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
        "runtime.gradient_audit.enabled",
        "runtime.gradient_audit.minimum_element_coverage",
        "progress.enabled",
        "logging.level",
        "output.root",
    ]
    for key in required:
        require_key(config, key)

    if str(require_key(config, "task.type")) != PRIMARY_TASK:
        raise ConfigError(f"task.type must be {PRIMARY_TASK!r}")
    adapter = str(require_key(config, "dataset.adapter")).strip().casefold()
    if adapter not in SUPPORTED_DATASETS:
        raise ConfigError(
            f"dataset.adapter must be one of {sorted(SUPPORTED_DATASETS)}"
        )
    duplicate_policy = str(
        require_key(config, "dataset.cross_split_duplicate_policy")
    ).casefold()
    if duplicate_policy not in {"report_only", "group_exclusive"}:
        raise ConfigError(
            "dataset.cross_split_duplicate_policy must be report_only or group_exclusive"
        )
    if not bool(require_key(config, "dataset.preflight.full_decode")):
        raise ConfigError(
            "Canonical benchmark preparation requires full image decoding"
        )
    signals = SignalContract.from_config(config)
    if not signals.binary:
        raise ConfigError("Binary benchmark training requires task.signals.binary=true")
    for forbidden in ("attribution", "image_grounding", "text_grounding"):
        if getattr(signals, forbidden):
            raise ConfigError(
                f"task.signals.{forbidden} must remain false: the released benchmark has no such target"
            )
    if signals.subtype and adapter != "multibanfakedetect":
        raise ConfigError(
            "Subtype supervision is available only for MultiBanFakeDetect"
        )
    if signals.text_label and adapter != "vldbench":
        raise ConfigError("text_label supervision is available only for VLDBench")
    if (
        not str(require_key(config, "task.negative_name")).strip()
        or not str(require_key(config, "task.positive_name")).strip()
    ):
        raise ConfigError("Binary benchmark class names must be non-empty")

    if int(require_key(config, "image.size")) != 384:
        raise ConfigError(
            "Binary benchmark image.size must remain 384 for the backbones"
        )
    if not bool(require_key(config, "image.preserve_aspect_ratio")) or not bool(
        require_key(config, "image.padding")
    ):
        raise ConfigError(
            "Binary benchmark images require aspect-ratio resize and padding"
        )
    if not bool(require_key(config, "tokenizer.enabled")) or not bool(
        require_key(config, "tokenizer.use_fast")
    ):
        raise ConfigError("Binary benchmark text requires an enabled fast tokenizer")

    if str(require_key(config, "model.model_type")) != "hera_binary_benchmark":
        raise ConfigError("model.model_type must be hera_binary_benchmark")
    if str(require_key(config, "model.binary.aggregation")) != "independent_head":
        raise ConfigError(
            "Binary-only datasets require an independent hypothesis-aggregation head; "
            "DGM4 Noisy-OR attribution semantics are unavailable"
        )
    if int(require_key(config, "model.latent_slots.count")) != 4:
        raise ConfigError(
            "HERA binary reasoning requires exactly four latent evidence slots"
        )
    roles = list(require_key(config, "model.latent_slots.roles"))
    if roles != ["visual_slot_1", "visual_slot_2", "text_slot_1", "text_slot_2"]:
        raise ConfigError(
            "model.latent_slots.roles must declare the canonical modality-aligned slots"
        )
    dropout = float(require_key(config, "model.binary.dropout"))
    if not 0 <= dropout < 1:
        raise ConfigError("model.binary.dropout must be in [0,1)")
    if str(require_key(config, "tokenizer.name")) != str(
        require_key(config, "backbones.deberta.name")
    ):
        raise ConfigError("tokenizer.name must equal backbones.deberta.name")

    for key in (
        "loss.lambda_bin",
        "loss.lambda_align",
        "loss.lambda_router",
        "loss.lambda_subtype",
        "loss.lambda_text_label",
    ):
        if float(require_key(config, key)) < 0:
            raise ConfigError(f"{key} must be non-negative")
    if float(require_key(config, "loss.lambda_bin")) <= 0:
        raise ConfigError("loss.lambda_bin must be positive")
    if signals.subtype != (float(require_key(config, "loss.lambda_subtype")) > 0):
        raise ConfigError(
            "task.signals.subtype and loss.lambda_subtype must be enabled together"
        )
    if signals.text_label != (float(require_key(config, "loss.lambda_text_label")) > 0):
        raise ConfigError(
            "task.signals.text_label and loss.lambda_text_label must be enabled together"
        )
    router_regularization = bool(require_key(config, "routing.regularization.enabled"))
    if router_regularization != (float(require_key(config, "loss.lambda_router")) > 0):
        raise ConfigError(
            "routing.regularization.enabled and loss.lambda_router must be enabled together"
        )

    freeze_epochs = int(require_key(config, "training.stages.stage_a_freeze_epochs"))
    if not bool(require_key(config, "training.stages.stage_b_full_unfreeze.enabled")):
        raise ConfigError(
            "Binary benchmark HERA runs require full-backbone fine-tuning after warm-up"
        )
    if (
        int(require_key(config, "training.stages.stage_b_full_unfreeze.start_epoch"))
        != freeze_epochs + 1
    ):
        raise ConfigError(
            "Full fine-tuning must begin immediately after the frozen warm-up"
        )
    if int(require_key(config, "training.micro_batch_size")) != int(
        require_key(config, "dataloader.batch_size")
    ):
        raise ConfigError("training.micro_batch_size must equal dataloader.batch_size")
    if int(require_key(config, "training.gradient_accumulation_steps")) <= 0:
        raise ConfigError("training.gradient_accumulation_steps must be positive")
    if int(require_key(config, "training.epochs")) <= freeze_epochs:
        raise ConfigError(
            "training.epochs must include at least one full-fine-tuning epoch"
        )
    if str(require_key(config, "training.early_stopping.metric")) != "macro_f1":
        raise ConfigError("Binary benchmark early stopping must monitor macro_f1")
    if str(require_key(config, "training.early_stopping.mode")) != "max":
        raise ConfigError("Binary benchmark early stopping mode must be max")
    if str(require_key(config, "checkpointing.selection_metric")) != "macro_f1":
        raise ConfigError(
            "Binary benchmark best checkpoint must be selected by macro_f1"
        )
    if not bool(require_key(config, "checkpointing.keep_last")) or not bool(
        require_key(config, "checkpointing.keep_best")
    ):
        raise ConfigError(
            "Binary benchmark runs must retain both last and best checkpoints"
        )
    if not bool(require_key(config, "checkpointing.best_weights_only")):
        raise ConfigError("The best benchmark artifact must remain model-only")
    if not bool(require_key(config, "checkpointing.atomic_write")):
        raise ConfigError("Resume checkpoint writes must remain atomic")
    if float(require_key(config, "checkpointing.minimum_free_gib")) < 0:
        raise ConfigError("checkpointing.minimum_free_gib must be non-negative")
    if int(require_key(config, "dataloader.num_workers")) < 0:
        raise ConfigError("dataloader.num_workers must be non-negative")
    if int(require_key(config, "dataloader.worker_threads")) <= 0:
        raise ConfigError("dataloader.worker_threads must be positive")
    if int(require_key(config, "tokenizer.max_length")) <= 0:
        raise ConfigError("tokenizer.max_length must be positive")

    precision = str(require_key(config, "runtime.precision")).casefold()
    model_dtype = str(require_key(config, "runtime.model_dtype")).casefold()
    if precision in {"bf16", "bfloat16", "fp16", "float16"} and model_dtype not in {
        "fp32",
        "float32",
    }:
        raise ConfigError("runtime.model_dtype must remain fp32 when AMP is enabled")
    minimum_gradient_coverage = float(
        require_key(config, "runtime.gradient_audit.minimum_element_coverage")
    )
    if not 0 < minimum_gradient_coverage <= 1:
        raise ConfigError(
            "runtime.gradient_audit.minimum_element_coverage must be in (0,1]"
        )

    _validate_dataset_specific(config, adapter=adapter, mode=mode)
    if mode == "evaluate":
        require_key(config, "evaluation.split")
        require_key(config, "evaluation.checkpoint_path")
        require_key(config, "evaluation.run_dir")


def _validate_dataset_specific(
    config: Mapping[str, Any],
    *,
    adapter: str,
    mode: str,
) -> None:
    root = Path(str(require_key(config, "dataset.root")))
    if adapter == "multibanfakedetect":
        if str(require_key(config, "task.primary_target")) != "label":
            raise ConfigError("MultiBanFakeDetect primary target must be label")
        if str(require_key(config, "dataset.split_protocol")) != "official_80_10_10":
            raise ConfigError(
                "MultiBanFakeDetect must use the released official_80_10_10 split"
            )
        if (
            str(require_key(config, "dataset.cross_split_duplicate_policy"))
            != "report_only"
        ):
            raise ConfigError(
                "The official MultiBanFakeDetect protocol preserves released duplicates; "
                "cross_split_duplicate_policy must be report_only"
            )
        for key in (
            "train_csv",
            "val_csv",
            "test_csv",
            "train_images",
            "val_images",
            "test_images",
        ):
            require_key(config, f"dataset.{key}")
        if mode != "prepare" and not root.exists():
            raise ConfigError(f"MultiBanFakeDetect root does not exist: {root}")
    else:
        if str(require_key(config, "task.primary_target")) != "multimodal_label":
            raise ConfigError("VLDBench primary target must be multimodal_label")
        if (
            str(require_key(config, "dataset.split_protocol"))
            != "stratified_image_group_70_15_15"
        ):
            raise ConfigError(
                "VLDBench must use the deterministic stratified_image_group_70_15_15 protocol"
            )
        if (
            str(require_key(config, "dataset.cross_split_duplicate_policy"))
            != "group_exclusive"
        ):
            raise ConfigError("VLDBench split construction must be group_exclusive")
        parquet = Path(str(require_key(config, "dataset.parquet")))
        manifest = Path(str(require_key(config, "dataset.split_manifest.path")))
        require_key(config, "dataset.split_manifest.seed")
        require_key(config, "dataset.split_manifest.folds")
        require_key(config, "dataset.split_manifest.train_folds")
        require_key(config, "dataset.split_manifest.val_folds")
        require_key(config, "dataset.split_manifest.test_folds")
        if mode != "prepare" and not parquet.exists():
            raise ConfigError(f"VLDBench parquet does not exist: {parquet}")
        if mode in {"train", "evaluate"} and not manifest.exists():
            raise ConfigError(
                f"VLDBench split manifest is missing: {manifest}. Run "
                "python scripts/prepare_binary_benchmark.py <config.yaml> first."
            )

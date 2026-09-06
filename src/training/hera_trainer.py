"""Phase-7 trainer for the full canonical HERA-DGM model."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from tqdm.auto import tqdm

from src.compat.transformers_compat import set_gradient_checkpointing
from src.data.loader import create_dataloader
from src.evaluation.evaluator import compute_metrics_and_reports, move_batch_to_device, run_inference
from src.experiments import validate_hera_checkpoint_variant
from src.losses.hera_components import HERAMultitaskLoss
from src.models.hera import HERADGM
from src.thresholding.tuner import Thresholds, default_thresholds_for_hypotheses, tune_thresholds
from src.training.checkpoint import restore_checkpoint, save_checkpoint
from src.training.optimizer import (
    apply_training_stage,
    create_optimizer,
    load_compatible_optimizer_state,
    migrate_optimizer_state,
    parameter_accounting,
)
from src.training.precision import PrecisionManager
from src.training.scheduler import (
    create_continuation_scheduler_plan,
    create_scheduler_from_plan,
    create_scheduler_plan,
)
from src.training.state import TrainingState
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata


BYTES_PER_GIB = 1024**3


def create_hera_run_dir(config: Mapping[str, Any], *, resume_checkpoint: Path | None = None) -> Path:
    """Create a unique HERA run directory with canonical subdirectories."""

    if resume_checkpoint is not None and resume_checkpoint.parent.name == "checkpoints":
        run_dir = resume_checkpoint.parent.parent
        for subdir in ("checkpoints", "metrics", "logs", "thresholds", "predictions", "router"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        return run_dir

    output_root = Path(str(config["output"]["root"]))
    seed = int(config["project"]["seed"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(config["project"].get("run_id") or f"hera_seed{seed}_{timestamp}")
    run_dir = output_root / run_id
    for subdir in ("checkpoints", "metrics", "logs", "thresholds", "predictions", "router"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_hera_resume_checkpoint(config: Mapping[str, Any]) -> Path | None:
    """Resolve explicit or YAML-enabled auto-resume checkpoint paths."""

    explicit = config["training"].get("resume_from")
    if explicit not in (None, ""):
        path = Path(str(explicit))
        if not path.exists():
            raise FileNotFoundError(f"Configured training.resume_from checkpoint does not exist: {path}")
        return path

    auto_resume = config["training"]["auto_resume"]
    if not bool(auto_resume["enabled"]):
        return None
    checkpoint_name = str(auto_resume["checkpoint_name"])
    run_dir = auto_resume.get("run_dir")
    if run_dir not in (None, ""):
        path = Path(str(run_dir)) / "checkpoints" / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"Configured training.auto_resume.run_dir has no checkpoint: {path}")
        return path
    return _latest_hera_resume_checkpoint(Path(str(config["output"]["root"])), checkpoint_name=checkpoint_name)


def _latest_hera_resume_checkpoint(output_root: Path, *, checkpoint_name: str) -> Path | None:
    if not output_root.exists():
        return None
    candidates = [
        checkpoint
        for checkpoint in output_root.glob(f"*/checkpoints/{checkpoint_name}")
        if checkpoint.is_file()
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
        reverse=True,
    )[0]


def _cuda_device_index(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(device.index) if device.index is not None else int(torch.cuda.current_device())


def _cuda_memory_summary(device: torch.device) -> dict[str, Any] | None:
    """Return PyTorch CUDA memory state for the configured device."""

    index = _cuda_device_index(device)
    if index is None:
        return None
    props = torch.cuda.get_device_properties(index)
    return {
        "device": f"cuda:{index}",
        "device_name": torch.cuda.get_device_name(index),
        "allocated_gb": float(torch.cuda.memory_allocated(index) / BYTES_PER_GIB),
        "reserved_gb": float(torch.cuda.memory_reserved(index) / BYTES_PER_GIB),
        "max_allocated_gb": float(torch.cuda.max_memory_allocated(index) / BYTES_PER_GIB),
        "total_gb": float(props.total_memory / BYTES_PER_GIB),
    }


def _cuda_memory_postfix(device: torch.device) -> str | None:
    summary = _cuda_memory_summary(device)
    if summary is None:
        return None
    return f"{summary['allocated_gb']:.1f}/{summary['reserved_gb']:.1f}GB"


def _first_parameter_device(model: nn.Module) -> str:
    first_parameter = next(model.parameters(), None)
    return "no_parameters" if first_parameter is None else str(first_parameter.device)


class HERATrainer:
    """Train/validate the complete HERA-DGM model with Phase-7 contracts."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.device = torch.device(str(config["runtime"]["device"]))
        self.progress = bool(config["progress"]["enabled"])
        self.cuda_memory_logging = bool(config["runtime"]["cuda_memory_logging"]["enabled"])
        self.resume_checkpoint_path = _resolve_hera_resume_checkpoint(config)
        self.run_dir = create_hera_run_dir(config, resume_checkpoint=self.resume_checkpoint_path)
        if self.resume_checkpoint_path is None:
            dump_yaml(config, self.run_dir / "effective_config.yaml")
        environment = collect_environment_metadata(Path.cwd())
        environment["backbones"] = config["backbones"]
        (self.run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, default=str),
            encoding="utf-8",
        )
        self.model = HERADGM.from_config(config, load_backbones=True).to(self.device)
        self.gradient_checkpointing = apply_configured_gradient_checkpointing(self.model, config)
        self._log_cuda_runtime_state("after model load")
        if self.resume_checkpoint_path is not None:
            tqdm.write(f"Resume checkpoint selected: {self.resume_checkpoint_path}")
        self.loss_fn = HERAMultitaskLoss(config["loss"], routing_config=config["routing"])
        self.precision = PrecisionManager(
            precision=str(config["runtime"]["precision"]),
            device=self.device,
        )
        if self.model.deberta is None or not hasattr(self.model.deberta, "tokenizer"):
            raise RuntimeError("HERADGM requires an attached DeBERTa fast tokenizer for dataset loading")
        tokenizer = self.model.deberta.tokenizer
        smoke = config.get("smoke", {})
        self.train_loader = create_dataloader(
            config,
            split="train",
            tokenizer=tokenizer,
            max_samples=smoke.get("train_max_samples"),
            samples_per_class=smoke.get("samples_per_class"),
        )
        self.val_loader = create_dataloader(
            config,
            split="val",
            tokenizer=tokenizer,
            max_samples=smoke.get("val_max_samples"),
            samples_per_class=smoke.get("samples_per_class"),
        )
        hypothesis_names = tuple(config["model"]["hypotheses"]["order"])
        self.state = TrainingState(
            thresholds=default_thresholds_for_hypotheses(hypothesis_names)
        )
        self.history_path = self.run_dir / "metrics" / "history.csv"
        self._lazy_initialized = False
        self._active_stage: str | None = None
        self._best_score_tuple: tuple[float, ...] | None = None
        self._ranked_checkpoints: list[tuple[tuple[float, ...], Path]] = []
        self._last_parameter_counts: dict[str, dict[str, int]] = {}
        self._scheduler_plan: dict[str, Any] | None = None

    def run(self) -> Path:
        """Run configured full HERA training."""

        total_epochs = int(self.config["training"]["epochs"])
        max_train_batches = self.config["runtime"].get("max_train_batches")
        max_val_batches = self.config["runtime"].get("max_val_batches")
        train_batches = min(len(self.train_loader), max_train_batches or len(self.train_loader))
        accumulation = int(self.config["training"]["gradient_accumulation_steps"])
        optimizer_steps_per_epoch = max(1, math.ceil(max(1, train_batches) / accumulation))
        total_steps = max(1, total_epochs * optimizer_steps_per_epoch)

        optimizer: torch.optim.Optimizer | None = None
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        resume_path = self.resume_checkpoint_path
        resume_payload: dict[str, Any] | None = None
        start_epoch = 1
        resume_checkpoint_epoch: int | None = None
        stop_reason = "max_epochs"
        if resume_path:
            resume_payload, loaded_resume_path = _load_hera_resume_payload(resume_path)
            if loaded_resume_path != resume_path:
                tqdm.write(
                    f"Primary resume checkpoint was unreadable; recovered from: {loaded_resume_path}"
                )
                resume_path = loaded_resume_path
            checkpoint_config = resume_payload.get("config")
            if not isinstance(checkpoint_config, Mapping):
                raise RuntimeError("Resume checkpoint does not contain its effective HERA config")
            validate_hera_checkpoint_variant(
                self.config,
                checkpoint_config,
                context="Resume",
            )
            resume_checkpoint_epoch = int(resume_payload["epoch"])
            start_epoch = resume_checkpoint_epoch + 1
            if start_epoch > total_epochs:
                best_checkpoint = _completed_checkpoint_best_path(
                    resume_payload,
                    run_dir=self.run_dir,
                )
                self.state.epoch = resume_checkpoint_epoch
                self._write_training_completion(
                    best_checkpoint=best_checkpoint,
                    stop_reason="max_epochs",
                )
                tqdm.write(
                    "Training checkpoint already reached the configured horizon; "
                    f"reusing best checkpoint: {best_checkpoint}"
                )
                return best_checkpoint
            _persist_resume_config_snapshots(
                run_dir=self.run_dir,
                checkpoint=resume_payload,
                current_config=self.config,
                resume_epoch=resume_checkpoint_epoch,
                target_total_epochs=total_epochs,
            )

        for epoch in range(start_epoch, total_epochs + 1):
            self.state.epoch = epoch
            self._apply_stage(epoch)
            if not self._lazy_initialized:
                self._initialize_lazy_modules()
            if optimizer is None or self._stage_changed_requires_optimizer_rebuild(epoch):
                previous_optimizer = optimizer
                optimizer, group_summaries = create_optimizer(self.model, self.config)
                if resume_payload is not None and resume_checkpoint_epoch is not None:
                    checkpoint_stage = _stage_name(self.config, resume_checkpoint_epoch)
                    target_stage = _stage_name(
                        self.config,
                        epoch,
                    )
                    crosses_stage_boundary = checkpoint_stage != target_stage
                    scheduler = self._restore_resume_checkpoint(
                        checkpoint=resume_payload,
                        optimizer=optimizer,
                        total_epochs=total_epochs,
                        total_steps=total_steps,
                        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                        resume_checkpoint_epoch=resume_checkpoint_epoch,
                        crosses_stage_boundary=crosses_stage_boundary,
                    )
                    self._restore_training_state(resume_payload)
                    self.state.epoch = epoch
                    if crosses_stage_boundary:
                        tqdm.write(
                            f"Resume crosses STAGE {checkpoint_stage} -> STAGE {target_stage}; "
                            "optimizer was rebuilt for the new stage."
                        )
                    resume_payload = None
                    resume_path = None
                elif previous_optimizer is not None:
                    migrated = migrate_optimizer_state(previous_optimizer, optimizer)
                    remaining_steps = max(1, (total_epochs - epoch + 1) * optimizer_steps_per_epoch)
                    anchor_lrs = [float(group["lr"]) for group in optimizer.param_groups]
                    self._scheduler_plan = create_continuation_scheduler_plan(
                        self.config,
                        remaining_steps=remaining_steps,
                        source_total_epochs=total_epochs,
                        resume_epoch=epoch - 1,
                        target_total_epochs=total_epochs,
                        start_global_step=self.state.global_step,
                        anchor_lrs=anchor_lrs,
                    )
                    scheduler = create_scheduler_from_plan(
                        optimizer, self._scheduler_plan, anchor_current_lrs=True
                    )
                    tqdm.write(
                        "Stage B optimizer migration: "
                        f"preserved_state_for={migrated} parameters, anchor_lrs={anchor_lrs}"
                    )
                else:
                    self._scheduler_plan = create_scheduler_plan(self.config, total_steps=total_steps)
                    scheduler = create_scheduler_from_plan(optimizer, self._scheduler_plan)
                self._last_parameter_counts = parameter_accounting(self.model)
                tqdm.write(f"Parameter accounting: {self._last_parameter_counts}")
                tqdm.write(f"Optimizer groups: {[summary.__dict__ for summary in group_summaries]}")
            train_losses = self._train_one_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                max_batches=max_train_batches,
            )
            val_bundle = run_inference(
                model=self.model,
                loader=self.val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                precision=self.precision,
                max_batches=max_val_batches,
                desc=f"Epoch {epoch} validation",
                progress=self.progress,
            )
            thresholds = tune_thresholds(
                binary_targets=torch.tensor(val_bundle.binary_targets).numpy(),
                binary_scores=torch.tensor(val_bundle.binary_scores).numpy(),
                atomic_targets=torch.tensor(val_bundle.atomic_targets).numpy(),
                atomic_scores=torch.tensor(val_bundle.atomic_scores).numpy(),
                config=self.config["thresholding"],
                progress=self.progress,
                hypothesis_names=tuple(self.config["model"]["hypotheses"]["order"]),
            )
            self.state.thresholds = thresholds
            reports = compute_metrics_and_reports(val_bundle, thresholds)
            default_reports = compute_metrics_and_reports(
                val_bundle,
                Thresholds(
                    binary=0.5,
                    atomic=(0.5,) * len(thresholds.atomic),
                    hypothesis_names=thresholds.hypothesis_names,
                ),
            )
            router_summary = _aggregate_router_summaries(val_bundle.router_batch_summaries)
            _warn_router_collapse(router_summary)
            tqdm.write("Validation binary classification_report:")
            tqdm.write(reports["binary_report"])
            tqdm.write("Validation attribution classification_report:")
            tqdm.write(reports["attribution_report"])
            tqdm.write(f"Validation thresholds: {thresholds.as_dict()}")
            early_stop = self._update_early_stopping(reports["metrics"])
            epoch_metrics = {
                "epoch": epoch,
                "stage": _stage_name(self.config, epoch),
                "train": train_losses,
                "val": reports["metrics"],
                "default_threshold_val": default_reports["metrics"],
                "thresholds": thresholds.as_dict(),
                "router": router_summary,
                "parameter_counts": self._last_parameter_counts,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "precision": self.precision.actual,
                "effective_batch_size": _effective_batch_size(self.config),
                "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "early_stopping": _early_stopping_status(self.config, self.state, reports["metrics"]),
            }
            self._persist_epoch_metrics(epoch_metrics)
            (self.run_dir / "router" / f"epoch_{epoch:03d}.json").write_text(
                json.dumps(router_summary, indent=2, default=str),
                encoding="utf-8",
            )
            self._checkpoint_decision(
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=reports["metrics"],
                epoch_metrics=epoch_metrics,
            )
            if early_stop:
                stop_reason = "early_stopping"
                tqdm.write(
                    "Early stopping triggered: "
                    f"epoch={epoch}, "
                    f"metric={self.config['training']['early_stopping']['metric']}, "
                    f"bad_epochs={self.state.early_stopping_bad_epochs}, "
                    f"patience={self.config['training']['early_stopping']['patience']}"
                )
                break
        if self.state.best_checkpoint is None:
            raise RuntimeError("HERA training finished without producing a best checkpoint")
        self._write_training_completion(
            best_checkpoint=self.state.best_checkpoint,
            stop_reason=stop_reason,
        )
        return self.state.best_checkpoint

    def _write_training_completion(
        self,
        *,
        best_checkpoint: Path,
        stop_reason: str,
    ) -> None:
        """Atomically mark a run complete for idempotent suite restarts."""

        payload = {
            "status": "complete",
            "final_epoch": int(self.state.epoch),
            "stop_reason": str(stop_reason),
            "best_checkpoint": str(best_checkpoint),
            "ablation_variant": self.model.ablation_variant,
            "cycle_study_variant": getattr(self.model, "cycle_study_variant", None),
            "cycle_study_protocol_id": getattr(
                self.model, "cycle_study_protocol_id", None
            ),
        }
        destination = self.run_dir / "training_complete.json"
        temporary = self.run_dir / ".training_complete.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(destination)

    def _restore_resume_checkpoint(
        self,
        *,
        checkpoint: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        total_steps: int,
        optimizer_steps_per_epoch: int,
        resume_checkpoint_epoch: int,
        crosses_stage_boundary: bool,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if crosses_stage_boundary:
            restore_checkpoint(checkpoint=checkpoint, model=self.model)
            loaded = load_compatible_optimizer_state(optimizer, checkpoint["optimizer"])
            remaining_steps = max(1, (total_epochs - resume_checkpoint_epoch) * optimizer_steps_per_epoch)
            anchor_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            self._scheduler_plan = create_continuation_scheduler_plan(
                self.config,
                remaining_steps=remaining_steps,
                source_total_epochs=_checkpoint_total_epochs(checkpoint),
                resume_epoch=resume_checkpoint_epoch,
                target_total_epochs=total_epochs,
                start_global_step=int(checkpoint["global_step"]),
                anchor_lrs=anchor_lrs,
            )
            scheduler = create_scheduler_from_plan(
                optimizer, self._scheduler_plan, anchor_current_lrs=True
            )
            tqdm.write(
                "Cross-stage resume preserved compatible optimizer state: "
                f"parameters={loaded}, anchor_lrs={anchor_lrs}"
            )
            return scheduler

        source_total_epochs = _checkpoint_total_epochs(checkpoint)
        if total_epochs < source_total_epochs:
            raise RuntimeError(
                "Configured training horizon is shorter than the resumed checkpoint plan: "
                f"checkpoint_total_epochs={source_total_epochs}, configured_total_epochs={total_epochs}"
            )
        if total_epochs > source_total_epochs:
            policy = str(self.config["scheduler"]["resume_horizon_policy"])
            if policy != "continue_from_checkpoint_lr":
                raise RuntimeError(
                    "Training horizon changed during resume but scheduler.resume_horizon_policy does not allow "
                    f"a no-jump continuation: policy={policy!r}"
                )
            restore_checkpoint(
                checkpoint=checkpoint,
                model=self.model,
                optimizer=optimizer,
                scaler=self.precision.scaler,
            )
            remaining_steps = max(1, (total_epochs - resume_checkpoint_epoch) * optimizer_steps_per_epoch)
            anchor_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            self._scheduler_plan = create_continuation_scheduler_plan(
                self.config,
                remaining_steps=remaining_steps,
                source_total_epochs=source_total_epochs,
                resume_epoch=resume_checkpoint_epoch,
                target_total_epochs=total_epochs,
                start_global_step=int(checkpoint["global_step"]),
                anchor_lrs=anchor_lrs,
            )
            scheduler = create_scheduler_from_plan(
                optimizer,
                self._scheduler_plan,
                anchor_current_lrs=True,
            )
            tqdm.write(
                "Extended training horizon with no-jump LR continuation: "
                f"epochs={source_total_epochs}->{total_epochs}, "
                f"resume_epoch={resume_checkpoint_epoch}, remaining_optimizer_steps={remaining_steps}, "
                f"anchor_lrs={anchor_lrs}"
            )
            return scheduler

        checkpoint_plan = checkpoint.get("scheduler_plan")
        self._scheduler_plan = (
            dict(checkpoint_plan)
            if isinstance(checkpoint_plan, Mapping)
            else create_scheduler_plan(self.config, total_steps=total_steps)
        )
        scheduler = create_scheduler_from_plan(optimizer, self._scheduler_plan)
        restore_checkpoint(
            checkpoint=checkpoint,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=self.precision.scaler,
        )
        return scheduler

    def _apply_stage(self, epoch: int) -> None:
        stage = _stage_name(self.config, epoch)
        if self._active_stage is not None and self._active_stage != stage:
            tqdm.write(f"STAGE {self._active_stage} -> STAGE {stage}")
        apply_training_stage(self.model, self.config, epoch=epoch)
        self.gradient_checkpointing = apply_configured_gradient_checkpointing(self.model, self.config)
        if stage == "A":
            _assert_stage_a_backbones_frozen(self.model)
        elif stage == "B":
            _assert_stage_b_backbones_fully_trainable(self.model)
        self._active_stage = stage

    def _stage_changed_requires_optimizer_rebuild(self, epoch: int) -> bool:
        return epoch > 1 and _stage_name(self.config, epoch) != _stage_name(self.config, epoch - 1)

    def _initialize_lazy_modules(self) -> None:
        """Run one real HERA forward pass before optimizer/count/checkpoint operations."""

        raw_batch = next(iter(self.train_loader))
        batch = move_batch_to_device(raw_batch, self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            with self.precision.autocast():
                output = self.model(batch)
                output.validate()
        self.model.train(was_training)
        self._lazy_initialized = True
        self._log_cuda_runtime_state("after lazy initialization")

    def _train_one_epoch(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        max_batches: int | None,
    ) -> dict[str, float]:
        self.model.train()
        accumulation = int(self.config["training"]["gradient_accumulation_steps"])
        max_grad_norm = self.config["training"].get("max_grad_norm")
        total = len(self.train_loader) if max_batches is None else min(max_batches, len(self.train_loader))
        iterator = tqdm(
            self.train_loader,
            total=total,
            desc=f"Epoch {self.state.epoch} train",
            disable=not self.progress,
        )
        sums: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            batch = move_batch_to_device(raw_batch, self.device)
            with self.precision.autocast():
                output = self.model(batch)
                losses = self.loss_fn(output, batch)
                _ensure_finite_losses(losses.as_dict(), epoch=self.state.epoch, batch_index=batch_index, config=self.config)
                scaled_total = losses.total / accumulation
            self.precision.backward(scaled_total)
            for key, value in losses.as_dict().items():
                sums[key] = sums.get(key, 0.0) + value
            if batch_index % accumulation == 0:
                self._optimizer_step(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    max_grad_norm=max_grad_norm,
                )
            postfix = {
                "total": f"{losses.total.detach().item():.4f}",
                "L_attr": f"{losses.attribution.detach().item():.4f}",
                "L_bin": f"{losses.binary.detach().item():.4f}",
                "L_img": f"{losses.image_grounding.detach().item():.4f}",
                "L_txt": f"{losses.text_grounding.detach().item():.4f}",
                "L_ev": f"{losses.evidence.detach().item():.4f}",
                "L_align": f"{losses.alignment.detach().item():.4f}",
                "L_router": f"{losses.router.detach().item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "stage": _stage_name(self.config, self.state.epoch),
            }
            if self.cuda_memory_logging:
                vram = _cuda_memory_postfix(self.device)
                if vram is not None:
                    postfix["vram"] = vram
            iterator.set_postfix(**postfix)
        if total % accumulation != 0:
            self._optimizer_step(
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=max_grad_norm,
            )
        denom = max(1, total)
        result = {key: value / denom for key, value in sums.items()}
        result["duration_seconds"] = time.time() - started
        return result

    def _optimizer_step(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        max_grad_norm: float | None,
    ) -> None:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(max_grad_norm),
                error_if_nonfinite=True,
            )
        self.precision.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self.state.global_step += 1

    def _persist_epoch_metrics(self, epoch_metrics: Mapping[str, Any]) -> None:
        path = self.run_dir / "metrics" / f"epoch_{int(epoch_metrics['epoch']):03d}.json"
        path.write_text(json.dumps(epoch_metrics, indent=2, default=str), encoding="utf-8")
        flat = {
            "epoch": epoch_metrics["epoch"],
            "stage": epoch_metrics["stage"],
            "train_total_loss": epoch_metrics["train"].get("total"),
            "train_binary_loss": epoch_metrics["train"].get("binary"),
            "train_attribution_loss": epoch_metrics["train"].get("attribution"),
            "train_image_grounding_loss": epoch_metrics["train"].get("image_grounding"),
            "train_text_grounding_loss": epoch_metrics["train"].get("text_grounding"),
            "train_evidence_loss": epoch_metrics["train"].get("evidence"),
            "train_alignment_loss": epoch_metrics["train"].get("alignment"),
            "train_router_loss": epoch_metrics["train"].get("router"),
            "val_cf1": epoch_metrics["val"].get("cf1"),
            "val_map": epoch_metrics["val"].get("map"),
            "val_auc": epoch_metrics["val"].get("auc"),
            "val_text_f1": epoch_metrics["val"].get("text_f1"),
            "val_image_mean_iou": epoch_metrics["val"].get("image_mean_iou"),
            "effective_batch_size": epoch_metrics["effective_batch_size"],
            "peak_vram": epoch_metrics["peak_vram"],
            "early_stopping_metric": epoch_metrics["early_stopping"]["metric"],
            "early_stopping_best": epoch_metrics["early_stopping"]["best"],
            "early_stopping_bad_epochs": epoch_metrics["early_stopping"]["bad_epochs"],
            **epoch_metrics["thresholds"],
        }
        exists = self.history_path.exists()
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat))
            if not exists:
                writer.writeheader()
            writer.writerow(flat)

    def _checkpoint_decision(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        metrics: Mapping[str, Any],
        epoch_metrics: Mapping[str, Any],
    ) -> None:
        primary = str(self.config["checkpointing"]["selection_metric"])
        score_tuple = _selection_score_tuple(metrics, self.config)
        improved = self._best_score_tuple is None or score_tuple > self._best_score_tuple
        keep_best = bool(self.config["checkpointing"].get("keep_best", True))
        if improved and keep_best:
            self._best_score_tuple = score_tuple
            self.state.best_metric = float(score_tuple[0])
            self.state.best_score = score_tuple
            self.state.best_checkpoint = self.run_dir / "checkpoints" / "best.pt"
        initial_best_written = False
        if improved and keep_best and self.state.best_checkpoint is not None:
            if not self.state.best_checkpoint.is_file():
                best_checkpoint_id = save_checkpoint(
                    path=self.state.best_checkpoint,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=self.precision.scaler,
                    state=self.state,
                    metrics=epoch_metrics,
                    config=self.config,
                    precision=self.precision.actual,
                    scheduler_plan=self._scheduler_plan,
                )
                initial_best_written = True
                tqdm.write(f"Saved initial recovery checkpoint: {self.state.best_checkpoint}")
        last_path = self.run_dir / "checkpoints" / "last.pt"
        if bool(self.config["checkpointing"].get("keep_last", True)):
            save_checkpoint(
                path=last_path,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=self.precision.scaler,
                state=self.state,
                metrics=epoch_metrics,
                config=self.config,
                precision=self.precision.actual,
                scheduler_plan=self._scheduler_plan,
            )
            tqdm.write(f"Saved last checkpoint: {last_path}")
        epoch_threshold_path = self.run_dir / "thresholds" / f"epoch_{self.state.epoch:03d}_thresholds.json"
        epoch_threshold_path.write_text(
            json.dumps(
                {
                    **self.state.thresholds.as_dict(),
                    "epoch": self.state.epoch,
                    "selection_metric": primary,
                    "selection_metric_value": score_tuple[0],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        top_k = int(self.config["checkpointing"].get("top_k", 1))
        if top_k > 0:
            ranked_path = self.run_dir / "checkpoints" / f"epoch_{self.state.epoch:03d}.pt"
            save_checkpoint(
                path=ranked_path,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=self.precision.scaler,
                state=self.state,
                metrics=epoch_metrics,
                config=self.config,
                precision=self.precision.actual,
                scheduler_plan=self._scheduler_plan,
            )
            tqdm.write(f"Saved ranked checkpoint: {ranked_path}")
            self._ranked_checkpoints.append((score_tuple, ranked_path))
            self._ranked_checkpoints.sort(key=lambda item: item[0], reverse=True)
            for _, stale_path in self._ranked_checkpoints[top_k:]:
                if stale_path.exists():
                    stale_path.unlink()
            self._ranked_checkpoints = self._ranked_checkpoints[:top_k]
        if improved and keep_best:
            best_path = self.state.best_checkpoint
            if best_path is None:
                raise RuntimeError("Best checkpoint path was not initialized")
            if not initial_best_written:
                best_checkpoint_id = save_checkpoint(
                    path=best_path,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=self.precision.scaler,
                    state=self.state,
                    metrics=epoch_metrics,
                    config=self.config,
                    precision=self.precision.actual,
                    scheduler_plan=self._scheduler_plan,
                )
                tqdm.write(f"Saved best checkpoint: {best_path}")
            (self.run_dir / "thresholds" / "best_thresholds.json").write_text(
                json.dumps(
                    {
                        **self.state.thresholds.as_dict(),
                        "checkpoint_path": str(best_path),
                        "checkpoint_id": best_checkpoint_id,
                        "epoch": self.state.epoch,
                        "selection_metric": primary,
                        "selection_metric_value": self.state.best_metric,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _restore_training_state(self, checkpoint: Mapping[str, Any]) -> None:
        self.state.global_step = int(checkpoint["global_step"])
        self.state.best_metric = checkpoint.get("best_metric")
        best_score = checkpoint.get("best_score")
        self.state.best_score = tuple(float(value) for value in best_score) if best_score is not None else None
        self.state.best_checkpoint = (
            Path(str(checkpoint["best_checkpoint"]))
            if checkpoint.get("best_checkpoint") is not None
            else None
        )
        threshold_data = checkpoint.get("thresholds")
        if threshold_data:
            self.state.thresholds = _thresholds_from_mapping(threshold_data)
        if self.state.best_score is not None:
            self._best_score_tuple = self.state.best_score
        elif self.state.best_metric is not None:
            self._best_score_tuple = (float(self.state.best_metric),)
        if "early_stopping_best_metric" in checkpoint and "early_stopping_bad_epochs" in checkpoint:
            early_best = checkpoint.get("early_stopping_best_metric")
            self.state.early_stopping_best_metric = float(early_best) if early_best is not None else None
            self.state.early_stopping_bad_epochs = int(checkpoint["early_stopping_bad_epochs"])
        else:
            early_best, bad_epochs = _recover_early_stopping_state(
                self.history_path,
                self.config["training"]["early_stopping"],
            )
            if early_best is None:
                early_best = float(self.state.best_metric) if self.state.best_metric is not None else None
            self.state.early_stopping_best_metric = early_best
            self.state.early_stopping_bad_epochs = bad_epochs
            tqdm.write(
                "Recovered legacy early-stopping state: "
                f"best={early_best}, bad_epochs={bad_epochs}"
            )

    def _log_cuda_runtime_state(self, label: str) -> None:
        if not self.cuda_memory_logging:
            return
        summary = _cuda_memory_summary(self.device)
        if summary is None:
            tqdm.write(
                "CUDA runtime "
                f"({label}): requested_device={self.device}, "
                f"torch_cuda_available={torch.cuda.is_available()}"
            )
            return
        tqdm.write(
            "CUDA runtime "
            f"({label}): requested_device={self.device}, "
            f"active_device={summary['device']}, "
            f"gpu={summary['device_name']}, "
            f"model_device={_first_parameter_device(self.model)}, "
            f"allocated={summary['allocated_gb']:.2f}GB, "
            f"reserved={summary['reserved_gb']:.2f}GB, "
            f"max_allocated={summary['max_allocated_gb']:.2f}GB, "
            f"total={summary['total_gb']:.2f}GB"
        )

    def _update_early_stopping(self, metrics: Mapping[str, Any]) -> bool:
        settings = self.config["training"]["early_stopping"]
        if not bool(settings["enabled"]):
            return False
        metric = str(settings["metric"])
        value = metrics.get(metric)
        if value is None:
            raise ValueError(f"Early stopping metric {metric!r} is missing from validation metrics")
        current = float(value)
        adaptation_start_epoch = _backbone_adaptation_start_epoch(self.config)
        adaptation_epoch = self.state.epoch - adaptation_start_epoch + 1
        minimum_adaptation_epochs = int(settings["minimum_backbone_adaptation_epochs"])
        if self.state.epoch < adaptation_start_epoch:
            if _early_stopping_improved(
                current=current,
                best=self.state.early_stopping_best_metric,
                mode=str(settings["mode"]),
                min_delta=float(settings["min_delta"]),
            ):
                self.state.early_stopping_best_metric = current
            self.state.early_stopping_bad_epochs = 0
            return False
        if adaptation_epoch == 1:
            self.state.early_stopping_best_metric = None
            self.state.early_stopping_bad_epochs = 0
        improved = _early_stopping_improved(
            current=current,
            best=self.state.early_stopping_best_metric,
            mode=str(settings["mode"]),
            min_delta=float(settings["min_delta"]),
        )
        grace_active = adaptation_epoch <= minimum_adaptation_epochs
        if improved:
            self.state.early_stopping_best_metric = current
            self.state.early_stopping_bad_epochs = 0
        elif not grace_active:
            self.state.early_stopping_bad_epochs += 1
        patience = int(settings["patience"])
        tqdm.write(
            "Early stopping monitor: "
            f"metric={metric}, value={current:.6f}, "
            f"best={self.state.early_stopping_best_metric:.6f}, "
            f"bad_epochs={self.state.early_stopping_bad_epochs}/{patience}, "
            f"mode={settings['mode']}, min_delta={float(settings['min_delta']):.6f}"
        )
        return (
            not grace_active
            and self.state.early_stopping_bad_epochs >= patience
            and not improved
        )


def _checkpoint_total_epochs(checkpoint: Mapping[str, Any]) -> int:
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError("Resume checkpoint is missing its effective config")
    training = checkpoint_config.get("training")
    if not isinstance(training, Mapping) or "epochs" not in training:
        raise RuntimeError("Resume checkpoint config is missing training.epochs")
    return int(training["epochs"])


def _persist_resume_config_snapshots(
    *,
    run_dir: Path,
    checkpoint: Mapping[str, Any],
    current_config: Mapping[str, Any],
    resume_epoch: int,
    target_total_epochs: int,
) -> None:
    """Preserve both checkpoint and continuation configs in a resumed run."""

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError("Resume checkpoint is missing its effective config")
    history_dir = run_dir / "config_history"
    dump_yaml(
        checkpoint_config,
        history_dir / f"checkpoint_epoch_{resume_epoch:03d}_config.yaml",
    )
    dump_yaml(
        current_config,
        history_dir / f"resume_epoch_{resume_epoch + 1:03d}_target_{target_total_epochs:03d}_config.yaml",
    )
    dump_yaml(current_config, run_dir / "effective_config.yaml")


def _load_hera_resume_payload(path: Path) -> tuple[dict[str, Any], Path]:
    """Load last.pt and fall back to best.pt if an interrupted write corrupted it."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False), path
    except Exception as primary_error:
        fallback = path.parent / "best.pt"
        if fallback == path or not fallback.is_file():
            raise RuntimeError(f"Failed to load HERA resume checkpoint: {path}") from primary_error
        try:
            return torch.load(fallback, map_location="cpu", weights_only=False), fallback
        except Exception as fallback_error:
            raise RuntimeError(
                f"Both HERA resume checkpoints are unreadable: {path} and {fallback}"
            ) from fallback_error


def _completed_checkpoint_best_path(
    checkpoint: Mapping[str, Any],
    *,
    run_dir: Path,
) -> Path:
    """Resolve the best artifact when a resumed run already reached its horizon."""

    configured = checkpoint.get("best_checkpoint")
    candidates = []
    if configured not in (None, ""):
        candidates.append(Path(str(configured)))
    candidates.append(run_dir / "checkpoints" / "best.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Completed HERA checkpoint has no reusable best.pt artifact in its run directory"
    )


def _recover_early_stopping_state(
    history_path: Path,
    settings: Mapping[str, Any],
) -> tuple[float | None, int]:
    """Reconstruct early-stop state for checkpoints written before it was persisted."""

    if not history_path.exists():
        return None, 0
    metric = str(settings["metric"])
    column = f"val_{metric}"
    best: float | None = None
    bad_epochs = 0
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_value = row.get(column)
            if raw_value in (None, ""):
                continue
            current = float(raw_value)
            improved = _early_stopping_improved(
                current=current,
                best=best,
                mode=str(settings["mode"]),
                min_delta=float(settings["min_delta"]),
            )
            if improved:
                best = current
                bad_epochs = 0
            else:
                bad_epochs += 1
    return best, bad_epochs


def apply_configured_gradient_checkpointing(model: HERADGM, config: Mapping[str, Any]) -> dict[str, bool]:
    """Apply YAML-controlled gradient checkpointing to attached HF backbones."""

    settings = config["runtime"]["gradient_checkpointing"]
    enabled = bool(settings["enabled"])
    only_trainable = bool(settings["only_trainable_backbones"])
    use_reentrant = bool(settings.get("use_reentrant", False))
    requested_backbones = settings["backbones"]
    actual: dict[str, bool] = {}
    for name, module in model.backbone_modules().items():
        requested = enabled and bool(requested_backbones.get(name, False))
        if requested and only_trainable and not _has_trainable_parameters(module):
            requested = False
        actual[name] = set_gradient_checkpointing(
            module,
            enabled=requested,
            stage=name,
            use_reentrant=use_reentrant,
        )
    return actual


def _has_trainable_parameters(module: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


def _stage_name(config: Mapping[str, Any], epoch: int) -> str:
    stages = config["training"]["stages"]
    freeze_epochs = int(stages["stage_a_freeze_epochs"])
    epoch_int = int(epoch)
    if epoch_int <= freeze_epochs:
        return "A"
    full_unfreeze = stages.get("stage_b_full_unfreeze", {})
    if bool(full_unfreeze.get("enabled", False)):
        start_epoch = int(full_unfreeze.get("start_epoch", freeze_epochs + 1))
        if epoch_int >= start_epoch:
            return "B"
    return "B"


def _backbone_adaptation_start_epoch(config: Mapping[str, Any]) -> int:
    """Return the first epoch whose backbone policy should seed early stopping."""

    stages = config["training"]["stages"]
    freeze_epochs = int(stages["stage_a_freeze_epochs"])
    full_unfreeze = stages.get("stage_b_full_unfreeze", {})
    if bool(full_unfreeze.get("enabled", False)):
        return int(full_unfreeze.get("start_epoch", freeze_epochs + 1))
    return freeze_epochs + 1


def _assert_stage_a_backbones_frozen(model: HERADGM) -> None:
    for name, module in model.backbone_modules().items():
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        if trainable:
            raise RuntimeError(f"Stage A must keep {name} frozen, but {trainable} parameters are trainable")


def _assert_stage_b_backbones_fully_trainable(model: HERADGM) -> None:
    """Fail fast if a nominal full-fine-tune stage leaves any backbone parameter frozen."""

    backbones = model.backbone_modules()
    expected = {"siglip2", "dinov3", "deberta"}
    missing = expected.difference(backbones)
    if missing:
        raise RuntimeError(f"Stage B is missing required backbones: {sorted(missing)}")
    for name, module in backbones.items():
        total = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        if total == 0 or trainable != total:
            raise RuntimeError(
                f"Stage B must fully fine-tune {name}, but only {trainable}/{total} parameters are trainable"
            )


def _selection_score_tuple(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, ...]:
    keys = [str(config["checkpointing"]["selection_metric"])]
    keys.extend(str(key) for key in config["checkpointing"].get("tie_breakers", []))
    values: list[float] = []
    for key in keys:
        value = metrics.get(key)
        values.append(float(value) if value is not None else float("-inf"))
    return tuple(values)


def _thresholds_from_mapping(data: Mapping[str, Any]) -> Thresholds:
    names = tuple(data.get("hypothesis_names", ("FS", "FA", "TS", "TA")))
    values = data.get("atomic_thresholds")
    if values is None:
        values = [data[f"{name.lower()}_threshold"] for name in names]
    return Thresholds(
        binary=float(data["binary_threshold"]),
        atomic=tuple(float(value) for value in values),
        hypothesis_names=names,
    )


def _effective_batch_size(config: Mapping[str, Any]) -> int:
    return int(config["dataloader"]["batch_size"]) * int(config["training"]["gradient_accumulation_steps"])


def _early_stopping_improved(
    *,
    current: float,
    best: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    if mode == "min":
        return current < best - min_delta
    raise ValueError(f"Unsupported early stopping mode: {mode!r}")


def _early_stopping_status(
    config: Mapping[str, Any],
    state: TrainingState,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    settings = config["training"]["early_stopping"]
    metric = str(settings["metric"])
    return {
        "enabled": bool(settings["enabled"]),
        "metric": metric,
        "mode": str(settings["mode"]),
        "patience": int(settings["patience"]),
        "min_delta": float(settings["min_delta"]),
        "minimum_backbone_adaptation_epochs": int(settings["minimum_backbone_adaptation_epochs"]),
        "current": float(metrics[metric]) if metric in metrics and metrics[metric] is not None else None,
        "best": state.early_stopping_best_metric,
        "bad_epochs": state.early_stopping_bad_epochs,
    }


def _aggregate_router_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    result: dict[str, Any] = {}
    family_names = summaries[0].get("families", {}).keys()
    for family in family_names:
        family_stats = [summary["families"][family] for summary in summaries if family in summary.get("families", {})]
        if not family_stats:
            continue
        result[family] = {
            "mean": _mean([stats["reliability_mean"] for stats in family_stats]),
            "std": _mean([stats["reliability_std"] for stats in family_stats]),
            "min": min(float(stats["reliability_min"]) for stats in family_stats),
            "max": max(float(stats["reliability_max"]) for stats in family_stats),
            "fraction_low": _mean([stats["fraction_below_low_threshold"] for stats in family_stats]),
            "fraction_high": _mean([stats["fraction_above_high_threshold"] for stats in family_stats]),
        }
    return result


def _mean(values: list[float]) -> float:
    return float(sum(float(value) for value in values) / max(1, len(values)))


def _warn_router_collapse(summary: Mapping[str, Any]) -> None:
    for family, stats in summary.items():
        if float(stats.get("fraction_low", 0.0)) > 0.98:
            tqdm.write(f"WARNING: router family {family} is nearly all low-reliability")
        if float(stats.get("fraction_high", 0.0)) > 0.98:
            tqdm.write(f"WARNING: router family {family} is nearly all high-reliability")


def _ensure_finite_losses(
    losses: Mapping[str, float],
    *,
    epoch: int,
    batch_index: int,
    config: Mapping[str, Any],
) -> None:
    for name, value in losses.items():
        if not math.isfinite(float(value)):
            raise FloatingPointError(
                "Non-finite HERA loss detected: "
                f"epoch={epoch}, batch={batch_index}, loss={name}, value={value}, "
                f"stage={_stage_name(config, epoch)}"
            )

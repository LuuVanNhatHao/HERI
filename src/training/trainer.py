"""Phase-2 baseline trainer."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

from src.data.loader import create_dataloader
from src.evaluation.evaluator import (
    compute_metrics_and_reports,
    move_batch_to_device,
    run_inference,
)
from src.losses.baseline_losses import BaselineMultitaskLoss
from src.models.baselines import build_baseline_model
from src.thresholding.tuner import Thresholds, default_thresholds_for_hypotheses, tune_thresholds
from src.training.checkpoint import (
    restore_checkpoint,
    save_checkpoint,
    validate_baseline_checkpoint_config,
)
from src.training.hera_trainer import (
    _assert_stage_a_backbones_frozen,
    _assert_stage_b_backbones_fully_trainable,
    _backbone_adaptation_start_epoch,
    _checkpoint_total_epochs,
    _completed_checkpoint_best_path,
    _cuda_memory_postfix,
    _cuda_memory_summary,
    _early_stopping_improved,
    _early_stopping_status,
    _effective_batch_size,
    _first_parameter_device,
    _load_hera_resume_payload,
    _persist_resume_config_snapshots,
    _recover_early_stopping_state,
    _selection_score_tuple,
    _stage_name,
    _thresholds_from_mapping,
    apply_configured_gradient_checkpointing,
)
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
from src.training.validation_reporting import emit_validation_classification_reports
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata


def create_run_dir(
    config: Mapping[str, Any],
    *,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Create a unique run directory and standard subdirectories."""

    if resume_checkpoint is not None and resume_checkpoint.parent.name == "checkpoints":
        run_dir = resume_checkpoint.parent.parent
        for subdir in ("checkpoints", "metrics", "logs", "thresholds", "predictions"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        return run_dir

    output_root = Path(str(config["output"]["root"]))
    seed = int(config["project"]["seed"])
    baseline = str(config["model"]["baseline_type"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / str(config["project"].get("run_id") or f"{baseline}_seed{seed}_{timestamp}")
    for subdir in ("checkpoints", "metrics", "logs", "thresholds", "predictions"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_baseline_resume_checkpoint(config: Mapping[str, Any]) -> Path | None:
    """Resolve an explicit checkpoint or the latest matching B0/B1 run."""

    explicit = config["training"].get("resume_from")
    if explicit not in (None, ""):
        path = Path(str(explicit))
        if not path.is_file():
            raise FileNotFoundError(f"Configured training.resume_from checkpoint does not exist: {path}")
        return path

    auto_resume = config["training"]["auto_resume"]
    if not bool(auto_resume["enabled"]):
        return None
    checkpoint_name = str(auto_resume["checkpoint_name"])
    run_dir = auto_resume.get("run_dir")
    if run_dir not in (None, ""):
        path = Path(str(run_dir)) / "checkpoints" / checkpoint_name
        if not path.is_file():
            raise FileNotFoundError(f"Configured training.auto_resume.run_dir has no checkpoint: {path}")
        return path

    output_root = Path(str(config["output"]["root"]))
    baseline = str(config["model"]["baseline_type"])
    seed = int(config["project"]["seed"])
    candidates = [
        checkpoint
        for checkpoint in output_root.glob(f"{baseline}_seed{seed}_*/checkpoints/{checkpoint_name}")
        if checkpoint.is_file()
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
        reverse=True,
    )[0]


class BaselineTrainer:
    """Train and validate B0/B1 baselines with shared infrastructure."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.device = torch.device(str(config["runtime"]["device"]))
        self.progress = bool(config["progress"]["enabled"])
        self.cuda_memory_logging = bool(config["runtime"]["cuda_memory_logging"]["enabled"])
        self.resume_checkpoint_path = _resolve_baseline_resume_checkpoint(config)
        self.run_dir = create_run_dir(config, resume_checkpoint=self.resume_checkpoint_path)
        if self.resume_checkpoint_path is None:
            dump_yaml(config, self.run_dir / "effective_config.yaml")
        environment = collect_environment_metadata(Path.cwd())
        environment["backbones"] = config["backbones"]
        (self.run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, default=str),
            encoding="utf-8",
        )
        self.model = build_baseline_model(config, load_backbones=True).to(self.device)
        self.gradient_checkpointing = apply_configured_gradient_checkpointing(self.model, config)
        self._log_cuda_runtime_state("after model load")
        if self.resume_checkpoint_path is not None:
            tqdm.write(f"Resume checkpoint selected: {self.resume_checkpoint_path}")
        self.loss_fn = BaselineMultitaskLoss(config["loss"]["weights"])
        self.precision = PrecisionManager(
            precision=str(config["runtime"]["precision"]),
            device=self.device,
        )
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
        hypothesis_names = tuple(
            config.get("model", {}).get("hypotheses", {}).get(
                "order", ("FS", "FA", "TS", "TA")
            )
        )
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
        """Run full configured training."""

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
        if resume_path is not None:
            resume_payload, loaded_resume_path = _load_hera_resume_payload(resume_path)
            if loaded_resume_path != resume_path:
                tqdm.write(
                    "Primary baseline resume checkpoint was unreadable; "
                    f"recovered from: {loaded_resume_path}"
                )
                resume_path = loaded_resume_path
            checkpoint_config = resume_payload.get("config")
            if not isinstance(checkpoint_config, Mapping):
                raise RuntimeError("Resume checkpoint does not contain its effective baseline config")
            validate_baseline_checkpoint_config(
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
                    "Baseline checkpoint already reached the configured horizon; "
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
                    crosses_stage_boundary = _stage_name(
                        self.config, resume_checkpoint_epoch
                    ) != _stage_name(self.config, epoch)
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
                        optimizer,
                        self._scheduler_plan,
                        anchor_current_lrs=True,
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
                hypothesis_names=tuple(
                    self.config["model"].get("hypotheses", {}).get(
                        "order", ("FS", "FA", "TS", "TA")
                    )
                ),
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
            emit_validation_classification_reports(
                reports,
                epoch=epoch,
                metrics_dir=self.run_dir / "metrics",
            )
            tqdm.write(f"Validation thresholds: {thresholds.as_dict()}")
            early_stop = self._update_early_stopping(reports["metrics"])
            epoch_metrics = {
                "epoch": epoch,
                "stage": _stage_name(self.config, epoch),
                "train": train_losses,
                "val": reports["metrics"],
                "default_threshold_val": default_reports["metrics"],
                "thresholds": thresholds.as_dict(),
                "parameter_counts": self._last_parameter_counts,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "precision": self.precision.actual,
                "effective_batch_size": _effective_batch_size(self.config),
                "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "early_stopping": _early_stopping_status(
                    self.config,
                    self.state,
                    reports["metrics"],
                ),
            }
            self._persist_epoch_metrics(epoch_metrics)
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
            raise RuntimeError("Training finished without producing a best checkpoint")
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
        """Atomically mark a baseline run complete for resumable suites."""

        payload = {
            "status": "complete",
            "final_epoch": int(self.state.epoch),
            "stop_reason": str(stop_reason),
            "best_checkpoint": str(best_checkpoint),
            "baseline_type": str(self.config["model"]["baseline_type"]),
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
            load_compatible_optimizer_state(optimizer, checkpoint["optimizer"])
            remaining_steps = max(
                1,
                (total_epochs - resume_checkpoint_epoch) * optimizer_steps_per_epoch,
            )
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
            return create_scheduler_from_plan(
                optimizer,
                self._scheduler_plan,
                anchor_current_lrs=True,
            )

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
                    "Training horizon changed during resume but scheduler.resume_horizon_policy "
                    f"does not allow continuation: policy={policy!r}"
                )
            restore_checkpoint(
                checkpoint=checkpoint,
                model=self.model,
                optimizer=optimizer,
                scaler=self.precision.scaler,
            )
            remaining_steps = max(
                1,
                (total_epochs - resume_checkpoint_epoch) * optimizer_steps_per_epoch,
            )
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
            return create_scheduler_from_plan(
                optimizer,
                self._scheduler_plan,
                anchor_current_lrs=True,
            )

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
        else:
            _assert_stage_b_backbones_fully_trainable(self.model)
        self._active_stage = stage

    def _stage_changed_requires_optimizer_rebuild(self, epoch: int) -> bool:
        return epoch > 1 and _stage_name(self.config, epoch) != _stage_name(
            self.config,
            epoch - 1,
        )

    def _initialize_lazy_modules(self) -> None:
        """Run one real forward pass to initialize lazy projection dimensions."""

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
                loss_values = losses.as_dict()
                if bool(self.config["runtime"].get("fail_on_nan", True)):
                    for name, value in loss_values.items():
                        if not math.isfinite(float(value)):
                            raise FloatingPointError(
                                "Non-finite baseline loss detected: "
                                f"epoch={self.state.epoch}, batch={batch_index}, "
                                f"loss={name}, value={value}, "
                                f"stage={_stage_name(self.config, self.state.epoch)}"
                            )
                scaled_total = losses.total / accumulation
            self.precision.backward(scaled_total)
            for key, value in loss_values.items():
                sums[key] = sums.get(key, 0.0) + value
            if batch_index % accumulation == 0:
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
            postfix = dict(
                loss=f"{losses.total.detach().item():.4f}",
                bin=f"{losses.binary.detach().item():.4f}",
                attr=f"{losses.attribution.detach().item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                stage=_stage_name(self.config, self.state.epoch),
            )
            if self.cuda_memory_logging:
                vram = _cuda_memory_postfix(self.device)
                if vram is not None:
                    postfix["vram"] = vram
            iterator.set_postfix(**postfix)
        if total % accumulation != 0:
            self.precision.optimizer_step(optimizer)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            self.state.global_step += 1
        denom = max(1, total)
        result = {key: value / denom for key, value in sums.items()}
        result["duration_seconds"] = time.time() - started
        return result

    def _persist_epoch_metrics(self, epoch_metrics: Mapping[str, Any]) -> None:
        path = self.run_dir / "metrics" / f"epoch_{int(epoch_metrics['epoch']):03d}.json"
        path.write_text(json.dumps(epoch_metrics, indent=2, default=str), encoding="utf-8")
        flat = {
            "epoch": epoch_metrics["epoch"],
            "train_total_loss": epoch_metrics["train"].get("total"),
            "train_binary_loss": epoch_metrics["train"].get("binary"),
            "train_attribution_loss": epoch_metrics["train"].get("attribution"),
            "train_image_grounding_loss": epoch_metrics["train"].get("image_grounding"),
            "train_text_grounding_loss": epoch_metrics["train"].get("text_grounding"),
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
        epoch_threshold_path = (
            self.run_dir / "thresholds" / f"epoch_{self.state.epoch:03d}_thresholds.json"
        )
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
        self.state.best_score = (
            tuple(float(value) for value in best_score)
            if best_score is not None
            else None
        )
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
            self.state.early_stopping_best_metric = (
                float(early_best) if early_best is not None else None
            )
            self.state.early_stopping_bad_epochs = int(checkpoint["early_stopping_bad_epochs"])
        else:
            early_best, bad_epochs = _recover_early_stopping_state(
                self.history_path,
                self.config["training"]["early_stopping"],
            )
            self.state.early_stopping_best_metric = early_best
            self.state.early_stopping_bad_epochs = bad_epochs

    def _log_cuda_runtime_state(self, label: str) -> None:
        if not self.cuda_memory_logging:
            return
        summary = _cuda_memory_summary(self.device)
        if summary is None:
            tqdm.write(
                f"CUDA runtime ({label}): requested_device={self.device}, "
                f"torch_cuda_available={torch.cuda.is_available()}"
            )
            return
        tqdm.write(
            f"CUDA runtime ({label}): requested_device={self.device}, "
            f"active_device={summary['device']}, gpu={summary['device_name']}, "
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

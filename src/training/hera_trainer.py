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
from tqdm.auto import tqdm

from src.compat.transformers_compat import set_gradient_checkpointing
from src.data.loader import create_dataloader
from src.evaluation.evaluator import compute_metrics_and_reports, move_batch_to_device, run_inference
from src.losses.hera_components import HERAMultitaskLoss
from src.models.hera import HERADGM
from src.thresholding.tuner import DEFAULT_THRESHOLDS, Thresholds, tune_thresholds
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optimizer import apply_training_stage, create_optimizer, parameter_accounting
from src.training.precision import PrecisionManager
from src.training.scheduler import create_scheduler
from src.training.state import TrainingState
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata


def create_hera_run_dir(config: Mapping[str, Any]) -> Path:
    """Create a unique HERA run directory with canonical subdirectories."""

    output_root = Path(str(config["output"]["root"]))
    seed = int(config["project"]["seed"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(config["project"].get("run_id") or f"hera_seed{seed}_{timestamp}")
    run_dir = output_root / run_id
    for subdir in ("checkpoints", "metrics", "logs", "thresholds", "predictions", "router"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_dir


class HERATrainer:
    """Train/validate the complete HERA-DGM model with Phase-7 contracts."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.device = torch.device(str(config["runtime"]["device"]))
        self.progress = bool(config["progress"]["enabled"])
        self.run_dir = create_hera_run_dir(config)
        dump_yaml(config, self.run_dir / "effective_config.yaml")
        environment = collect_environment_metadata(Path.cwd())
        environment["backbones"] = config["backbones"]
        (self.run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, default=str),
            encoding="utf-8",
        )
        self.model = HERADGM.from_config(config, load_backbones=True).to(self.device)
        self.gradient_checkpointing = apply_configured_gradient_checkpointing(self.model, config)
        self.loss_fn = HERAMultitaskLoss(config["loss"])
        self.precision = PrecisionManager(
            precision=str(config["runtime"]["precision"]),
            device=self.device,
        )
        if self.model.deberta is None or not hasattr(self.model.deberta, "tokenizer"):
            raise RuntimeError("HERADGM requires an attached DeBERTa fast tokenizer for DGM4 loading")
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
        self.state = TrainingState()
        self.history_path = self.run_dir / "metrics" / "history.csv"
        self._lazy_initialized = False
        self._active_stage: str | None = None
        self._best_score_tuple: tuple[float, ...] | None = None
        self._ranked_checkpoints: list[tuple[tuple[float, ...], Path]] = []
        self._last_parameter_counts: dict[str, dict[str, int]] = {}

    def run(self) -> Path:
        """Run configured full HERA training."""

        total_epochs = int(self.config["training"]["epochs"])
        max_train_batches = self.config["runtime"].get("max_train_batches")
        max_val_batches = self.config["runtime"].get("max_val_batches")
        train_batches = min(len(self.train_loader), max_train_batches or len(self.train_loader))
        accumulation = int(self.config["training"]["gradient_accumulation_steps"])
        total_steps = max(1, total_epochs * math.ceil(max(1, train_batches) / accumulation))

        optimizer: torch.optim.Optimizer | None = None
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        resume_path = self.config["training"].get("resume_from")
        start_epoch = 1
        resume_checkpoint_epoch: int | None = None
        if resume_path:
            resume_payload = torch.load(Path(str(resume_path)), map_location="cpu", weights_only=False)
            resume_checkpoint_epoch = int(resume_payload["epoch"])
            start_epoch = resume_checkpoint_epoch + 1
            if start_epoch > total_epochs:
                raise RuntimeError(
                    f"Resume checkpoint epoch {resume_checkpoint_epoch} already reached configured total_epochs={total_epochs}"
                )

        for epoch in range(start_epoch, total_epochs + 1):
            self.state.epoch = epoch
            self._apply_stage(epoch)
            if not self._lazy_initialized:
                self._initialize_lazy_modules()
            if optimizer is None or self._stage_changed_requires_optimizer_rebuild(epoch):
                optimizer, group_summaries = create_optimizer(self.model, self.config)
                scheduler = create_scheduler(optimizer, self.config, total_steps=total_steps)
                self._last_parameter_counts = parameter_accounting(self.model)
                tqdm.write(f"Parameter accounting: {self._last_parameter_counts}")
                tqdm.write(f"Optimizer groups: {[summary.__dict__ for summary in group_summaries]}")
            if resume_path and resume_checkpoint_epoch is not None:
                crosses_stage_boundary = _stage_name(self.config, resume_checkpoint_epoch) != _stage_name(
                    self.config,
                    epoch,
                )
                checkpoint = load_checkpoint(
                    path=Path(str(resume_path)),
                    model=self.model,
                    optimizer=None if crosses_stage_boundary else optimizer,
                    scheduler=None if crosses_stage_boundary else scheduler,
                    scaler=None if crosses_stage_boundary else self.precision.scaler,
                )
                self._restore_training_state(checkpoint)
                self.state.epoch = epoch
                if crosses_stage_boundary:
                    tqdm.write("Resume crosses STAGE A -> STAGE B; optimizer was rebuilt for the new stage.")
                resume_path = None
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
            )
            self.state.thresholds = thresholds
            reports = compute_metrics_and_reports(val_bundle, thresholds)
            default_reports = compute_metrics_and_reports(val_bundle, DEFAULT_THRESHOLDS)
            router_summary = _aggregate_router_summaries(val_bundle.router_batch_summaries)
            _warn_router_collapse(router_summary)
            tqdm.write("Validation binary classification_report:")
            tqdm.write(reports["binary_report"])
            tqdm.write("Validation attribution classification_report:")
            tqdm.write(reports["attribution_report"])
            tqdm.write(f"Validation thresholds: {thresholds.as_dict()}")
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
        if self.state.best_checkpoint is None:
            raise RuntimeError("HERA training finished without producing a best checkpoint")
        return self.state.best_checkpoint

    def _apply_stage(self, epoch: int) -> None:
        stage = _stage_name(self.config, epoch)
        if self._active_stage is not None and self._active_stage != stage:
            tqdm.write(f"STAGE {self._active_stage} -> STAGE {stage}")
        apply_training_stage(self.model, self.config, epoch=epoch)
        if stage == "A":
            _assert_stage_a_backbones_frozen(self.model)
        self._active_stage = stage

    def _stage_changed_requires_optimizer_rebuild(self, epoch: int) -> bool:
        freeze_epochs = int(self.config["training"]["stages"]["stage_a_freeze_epochs"])
        return epoch == freeze_epochs + 1

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
            iterator.set_postfix(
                total=f"{losses.total.detach().item():.4f}",
                L_attr=f"{losses.attribution.detach().item():.4f}",
                L_bin=f"{losses.binary.detach().item():.4f}",
                L_img=f"{losses.image_grounding.detach().item():.4f}",
                L_txt=f"{losses.text_grounding.detach().item():.4f}",
                L_ev=f"{losses.evidence.detach().item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                stage=_stage_name(self.config, self.state.epoch),
            )
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_grad_norm))
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
            "val_cf1": epoch_metrics["val"].get("cf1"),
            "val_map": epoch_metrics["val"].get("map"),
            "val_auc": epoch_metrics["val"].get("auc"),
            "val_text_f1": epoch_metrics["val"].get("text_f1"),
            "val_image_mean_iou": epoch_metrics["val"].get("image_mean_iou"),
            "effective_batch_size": epoch_metrics["effective_batch_size"],
            "peak_vram": epoch_metrics["peak_vram"],
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
            )
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
            )
            self._ranked_checkpoints.append((score_tuple, ranked_path))
            self._ranked_checkpoints.sort(key=lambda item: item[0], reverse=True)
            for _, stale_path in self._ranked_checkpoints[top_k:]:
                if stale_path.exists():
                    stale_path.unlink()
            self._ranked_checkpoints = self._ranked_checkpoints[:top_k]
        if improved and bool(self.config["checkpointing"].get("keep_best", True)):
            self._best_score_tuple = score_tuple
            self.state.best_metric = float(score_tuple[0])
            best_path = self.run_dir / "checkpoints" / "best.pt"
            self.state.best_checkpoint = best_path
            save_checkpoint(
                path=best_path,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=self.precision.scaler,
                state=self.state,
                metrics=epoch_metrics,
                config=self.config,
                precision=self.precision.actual,
            )
            (self.run_dir / "thresholds" / "best_thresholds.json").write_text(
                json.dumps(
                    {
                        **self.state.thresholds.as_dict(),
                        "checkpoint_path": str(best_path),
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
        self.state.best_checkpoint = (
            Path(str(checkpoint["best_checkpoint"]))
            if checkpoint.get("best_checkpoint") is not None
            else None
        )
        threshold_data = checkpoint.get("thresholds")
        if threshold_data:
            self.state.thresholds = _thresholds_from_mapping(threshold_data)
        if self.state.best_metric is not None:
            self._best_score_tuple = (float(self.state.best_metric),)


def apply_configured_gradient_checkpointing(model: HERADGM, config: Mapping[str, Any]) -> dict[str, bool]:
    """Apply YAML-controlled gradient checkpointing to attached HF backbones."""

    settings = config["runtime"]["gradient_checkpointing"]
    enabled = bool(settings["enabled"])
    requested_backbones = settings["backbones"]
    actual: dict[str, bool] = {}
    for name, module in model.backbone_modules().items():
        requested = enabled and bool(requested_backbones.get(name, False))
        actual[name] = set_gradient_checkpointing(module, enabled=requested, stage=name)
    return actual


def _stage_name(config: Mapping[str, Any], epoch: int) -> str:
    freeze_epochs = int(config["training"]["stages"]["stage_a_freeze_epochs"])
    return "A" if int(epoch) <= freeze_epochs else "B"


def _assert_stage_a_backbones_frozen(model: HERADGM) -> None:
    for name, module in model.backbone_modules().items():
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        if trainable:
            raise RuntimeError(f"Stage A must keep {name} frozen, but {trainable} parameters are trainable")


def _selection_score_tuple(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, ...]:
    keys = [str(config["checkpointing"]["selection_metric"])]
    keys.extend(str(key) for key in config["checkpointing"].get("tie_breakers", []))
    values: list[float] = []
    for key in keys:
        value = metrics.get(key)
        values.append(float(value) if value is not None else float("-inf"))
    return tuple(values)


def _thresholds_from_mapping(data: Mapping[str, Any]) -> Thresholds:
    return Thresholds(
        binary=float(data["binary_threshold"]),
        atomic=(
            float(data["fs_threshold"]),
            float(data["fa_threshold"]),
            float(data["ts_threshold"]),
            float(data["ta_threshold"]),
        ),
    )


def _effective_batch_size(config: Mapping[str, Any]) -> int:
    return int(config["dataloader"]["batch_size"]) * int(config["training"]["gradient_accumulation_steps"])


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

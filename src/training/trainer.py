"""Phase-2 baseline trainer."""

from __future__ import annotations

import csv
import json
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
from src.thresholding.tuner import DEFAULT_THRESHOLDS, tune_thresholds
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optimizer import apply_training_stage, create_optimizer, parameter_accounting
from src.training.precision import PrecisionManager
from src.training.scheduler import create_scheduler
from src.training.state import TrainingState
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata


def create_run_dir(config: Mapping[str, Any]) -> Path:
    """Create a unique run directory and standard subdirectories."""

    output_root = Path(str(config["output"]["root"]))
    seed = int(config["project"]["seed"])
    baseline = str(config["model"]["baseline_type"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{baseline}_seed{seed}_{timestamp}"
    for subdir in ("checkpoints", "metrics", "logs", "thresholds", "predictions"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_dir


class BaselineTrainer:
    """Train and validate B0/B1 baselines with shared infrastructure."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.device = torch.device(str(config["runtime"]["device"]))
        self.progress = bool(config["progress"]["enabled"])
        self.run_dir = create_run_dir(config)
        dump_yaml(config, self.run_dir / "effective_config.yaml")
        environment = collect_environment_metadata(Path.cwd())
        environment["backbones"] = config["backbones"]
        (self.run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, default=str),
            encoding="utf-8",
        )
        self.model = build_baseline_model(config, load_backbones=True).to(self.device)
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
        self.state = TrainingState()
        self.history_path = self.run_dir / "metrics" / "history.csv"
        self._lazy_initialized = False
        self._ranked_checkpoints: list[tuple[float, Path]] = []

    def run(self) -> Path:
        """Run full configured training."""

        total_epochs = int(self.config["training"]["epochs"])
        max_train_batches = self.config["runtime"].get("max_train_batches")
        max_val_batches = self.config["runtime"].get("max_val_batches")
        total_steps = max(1, total_epochs * min(len(self.train_loader), max_train_batches or len(self.train_loader)))

        optimizer: torch.optim.Optimizer | None = None
        scheduler = None
        resume_path = self.config["training"].get("resume_from")
        freeze_epochs = int(self.config["training"]["stages"]["stage_a_freeze_epochs"])
        for epoch in range(1, total_epochs + 1):
            self.state.epoch = epoch
            apply_training_stage(self.model, self.config, epoch=epoch)
            if not self._lazy_initialized:
                self._initialize_lazy_modules()
            if optimizer is None or epoch == freeze_epochs + 1:
                optimizer, group_summaries = create_optimizer(self.model, self.config)
                scheduler = create_scheduler(optimizer, self.config, total_steps=total_steps)
                tqdm.write(f"Parameter accounting: {parameter_accounting(self.model)}")
                tqdm.write(f"Optimizer groups: {[summary.__dict__ for summary in group_summaries]}")
            if epoch == 1:
                if resume_path:
                    checkpoint = load_checkpoint(
                        path=Path(str(resume_path)),
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=self.precision.scaler,
                    )
                    self.state.epoch = int(checkpoint["epoch"])
                    self.state.global_step = int(checkpoint["global_step"])
                    self.state.best_metric = checkpoint.get("best_metric")
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
            tqdm.write("Validation binary classification_report:")
            tqdm.write(reports["binary_report"])
            tqdm.write("Validation attribution classification_report:")
            tqdm.write(reports["attribution_report"])
            tqdm.write(f"Validation thresholds: {thresholds.as_dict()}")
            epoch_metrics = {
                "epoch": epoch,
                "train": train_losses,
                "val": reports["metrics"],
                "default_threshold_val": default_reports["metrics"],
                "thresholds": thresholds.as_dict(),
            }
            self._persist_epoch_metrics(epoch_metrics)
            self._checkpoint_decision(
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=reports["metrics"],
                epoch_metrics=epoch_metrics,
            )
        if self.state.best_checkpoint is None:
            raise RuntimeError("Training finished without producing a best checkpoint")
        return self.state.best_checkpoint

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
                scaled_total = losses.total / accumulation
            self.precision.backward(scaled_total)
            for key, value in losses.as_dict().items():
                sums[key] = sums.get(key, 0.0) + value
            if batch_index % accumulation == 0:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_grad_norm))
                self.precision.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1
            iterator.set_postfix(
                loss=f"{losses.total.detach().item():.4f}",
                bin=f"{losses.binary.detach().item():.4f}",
                attr=f"{losses.attribution.detach().item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
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
        observed = metrics.get(primary)
        if observed is None:
            observed = -1.0
        improved = self.state.best_metric is None or float(observed) > float(self.state.best_metric)
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
            self._ranked_checkpoints.append((float(observed), ranked_path))
            self._ranked_checkpoints.sort(key=lambda item: item[0], reverse=True)
            for _, stale_path in self._ranked_checkpoints[top_k:]:
                if stale_path.exists():
                    stale_path.unlink()
            self._ranked_checkpoints = self._ranked_checkpoints[:top_k]
        if improved and bool(self.config["checkpointing"].get("keep_best", True)):
            self.state.best_metric = float(observed)
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

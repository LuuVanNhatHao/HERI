"""Resume-capable full-fine-tuning trainer for standalone HERA binary benchmarks."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

from src.evaluation.evaluator import move_batch_to_device
from src.training.hera_trainer import apply_configured_gradient_checkpointing
from src.training.optimizer import (
    apply_training_stage,
    create_optimizer,
    migrate_optimizer_state,
    parameter_accounting,
)
from src.training.precision import PrecisionManager
from src.training.scheduler import (
    create_continuation_scheduler_plan,
    create_scheduler_from_plan,
    create_scheduler_plan,
)
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata

from .artifacts import (
    assert_checkpoint_signature,
    create_benchmark_run_dir,
    load_checkpoint_payload,
    resolve_resume_checkpoint,
    save_best_model,
    save_last_checkpoint,
)
from .inference import run_binary_inference
from .loader import create_binary_benchmark_dataloader
from .loss import BinaryBenchmarkLoss
from .metrics import auxiliary_metrics, binary_metrics_and_report, tune_binary_threshold
from .model import BinaryBenchmarkHERA
from .reporting import persist_epoch, write_training_complete
from .resume import restore_resume_runtime
from .state import BenchmarkTrainingState
from .training_utils import (
    checkpoint_stage_is_eligible,
    cuda_memory_postfix,
    full_ft_gradient_coverage,
    stage_name,
)


class BinaryBenchmarkTrainer:
    """Train HERA without inventing unavailable attribution or grounding labels."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.device = torch.device(str(config["runtime"]["device"]))
        self.progress = bool(config["progress"]["enabled"])
        self.resume_checkpoint = resolve_resume_checkpoint(config)
        self.run_dir = create_benchmark_run_dir(
            config, resume_checkpoint=self.resume_checkpoint
        )
        if self.resume_checkpoint is None:
            dump_yaml(config, self.run_dir / "effective_config.yaml")
        else:
            history = self.run_dir / "config_history"
            history.mkdir(parents=True, exist_ok=True)
            dump_yaml(
                config,
                history / f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
            )
        environment = collect_environment_metadata(Path.cwd())
        environment["benchmark"] = {
            "dataset": config["dataset"]["name"],
            "adapter": config["dataset"]["adapter"],
            "task": config["task"],
        }
        environment_path = self.run_dir / "environment.json"
        if environment_path.exists():
            environment_path = self.run_dir / (
                f"environment_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        environment_path.write_text(
            json.dumps(environment, indent=2, default=str),
            encoding="utf-8",
        )
        self.model = BinaryBenchmarkHERA.from_config(config, load_backbones=True).to(
            self.device
        )
        self.gradient_checkpointing = apply_configured_gradient_checkpointing(
            self.model, config
        )
        if self.model.deberta is None or not hasattr(self.model.deberta, "tokenizer"):
            raise RuntimeError(
                "Binary benchmark HERA requires an attached fast text tokenizer"
            )
        tokenizer = self.model.deberta.tokenizer
        limits = config.get("runtime", {}).get("dataset_max_samples", {})
        self.train_loader = create_binary_benchmark_dataloader(
            config,
            split="train",
            tokenizer=tokenizer,
            max_samples=limits.get("train"),
        )
        self.val_loader = create_binary_benchmark_dataloader(
            config,
            split="val",
            tokenizer=tokenizer,
            max_samples=limits.get("val"),
        )
        self.loss_fn = BinaryBenchmarkLoss(config)
        self.precision = PrecisionManager(
            precision=str(config["runtime"]["precision"]),
            device=self.device,
        )
        self.state = BenchmarkTrainingState()
        self.history_path = self.run_dir / "metrics" / "history.csv"
        self.scheduler_plan: dict[str, Any] | None = None
        self._lazy_initialized = False
        self._active_stage: str | None = None
        self._stage_b_gradient_audited = False
        self._log_cuda("after model load")

    def run(self) -> Path:
        total_epochs = int(self.config["training"]["epochs"])
        max_train_batches = self.config["runtime"].get("max_train_batches")
        max_val_batches = self.config["runtime"].get("max_val_batches")
        train_batches = min(
            len(self.train_loader), max_train_batches or len(self.train_loader)
        )
        accumulation = int(self.config["training"]["gradient_accumulation_steps"])
        optimizer_steps_per_epoch = max(
            1, math.ceil(max(1, train_batches) / accumulation)
        )
        total_steps = max(1, total_epochs * optimizer_steps_per_epoch)

        resume_payload = (
            load_checkpoint_payload(self.resume_checkpoint)
            if self.resume_checkpoint is not None
            else None
        )
        start_epoch = 1
        resume_epoch: int | None = None
        if resume_payload is not None:
            assert_checkpoint_signature(resume_payload, self.config)
            resume_epoch = int(resume_payload["epoch"])
            start_epoch = resume_epoch + 1
            self.state = BenchmarkTrainingState.from_checkpoint(resume_payload)
            tqdm.write(f"Resume checkpoint selected: {self.resume_checkpoint}")
            completion = self.run_dir / "training_complete.json"
            if completion.is_file():
                status = json.loads(completion.read_text(encoding="utf-8"))
                if status.get("status") == "complete":
                    best = self._existing_best_model()
                    tqdm.write(
                        f"Training run is already complete ({status.get('stop_reason')}): {best}"
                    )
                    return best
            if start_epoch > total_epochs:
                best = self._existing_best_model()
                write_training_complete(
                    run_dir=self.run_dir,
                    dataset_name=str(self.config["dataset"]["name"]),
                    state=self.state,
                    best=best,
                    stop_reason="max_epochs",
                )
                return best

        optimizer: torch.optim.Optimizer | None = None
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        stop_reason = "max_epochs"
        for epoch in range(start_epoch, total_epochs + 1):
            self.state.epoch = epoch
            self._apply_stage(epoch)
            if not self._lazy_initialized:
                self._initialize_lazy_modules()
            if optimizer is None or stage_name(self.config, epoch) != stage_name(
                self.config, epoch - 1
            ):
                previous_optimizer = optimizer
                optimizer, summaries = create_optimizer(self.model, self.config)
                if resume_payload is not None and resume_epoch is not None:
                    restored = restore_resume_runtime(
                        model=self.model,
                        precision=self.precision,
                        config=self.config,
                        state=self.state,
                        payload=resume_payload,
                        optimizer=optimizer,
                        total_epochs=total_epochs,
                        total_steps=total_steps,
                        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                        resume_epoch=resume_epoch,
                        target_epoch=epoch,
                    )
                    scheduler = restored.scheduler
                    self.scheduler_plan = restored.scheduler_plan
                    if restored.message is not None:
                        tqdm.write(restored.message)
                    resume_payload = None
                elif previous_optimizer is not None:
                    migrated = migrate_optimizer_state(previous_optimizer, optimizer)
                    remaining_steps = max(
                        1, (total_epochs - epoch + 1) * optimizer_steps_per_epoch
                    )
                    anchors = [float(group["lr"]) for group in optimizer.param_groups]
                    self.scheduler_plan = create_continuation_scheduler_plan(
                        self.config,
                        remaining_steps=remaining_steps,
                        source_total_epochs=total_epochs,
                        resume_epoch=epoch - 1,
                        target_total_epochs=total_epochs,
                        start_global_step=self.state.global_step,
                        anchor_lrs=anchors,
                    )
                    scheduler = create_scheduler_from_plan(
                        optimizer,
                        self.scheduler_plan,
                        anchor_current_lrs=True,
                    )
                    tqdm.write(
                        f"Stage optimizer migration: preserved_state_for={migrated}, anchor_lrs={anchors}"
                    )
                else:
                    self.scheduler_plan = create_scheduler_plan(
                        self.config, total_steps=total_steps
                    )
                    scheduler = create_scheduler_from_plan(
                        optimizer, self.scheduler_plan
                    )
                tqdm.write(f"Parameter accounting: {parameter_accounting(self.model)}")
                tqdm.write(
                    f"Optimizer groups: {[summary.__dict__ for summary in summaries]}"
                )

            train_metrics = self._train_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                max_batches=max_train_batches,
            )
            validation = run_binary_inference(
                model=self.model,
                loader=self.val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                precision=self.precision,
                max_batches=max_val_batches,
                desc=f"Epoch {epoch} validation",
                progress=self.progress,
            )
            self.state.threshold = tune_binary_threshold(
                validation.targets,
                validation.scores,
                self.config["thresholding"],
            )
            val_metrics, report = binary_metrics_and_report(
                validation,
                threshold=self.state.threshold,
                negative_name=str(self.config["task"]["negative_name"]),
                positive_name=str(self.config["task"]["positive_name"]),
            )
            val_metrics.update(auxiliary_metrics(validation))
            tqdm.write("Validation binary classification_report:")
            tqdm.write(report)
            tqdm.write(f"Validation binary threshold: {self.state.threshold:.4f}")
            should_stop = self._update_early_stopping(val_metrics)
            epoch_metrics = {
                "epoch": epoch,
                "stage": stage_name(self.config, epoch),
                "train": train_metrics,
                "val": val_metrics,
                "binary_threshold": self.state.threshold,
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
                "effective_batch_size": int(self.config["training"]["micro_batch_size"])
                * accumulation,
                "precision": self.precision.actual,
                "peak_vram_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                ),
                "signals": self.model.signals.as_dict(),
            }
            persist_epoch(
                run_dir=self.run_dir,
                history_path=self.history_path,
                metrics=epoch_metrics,
                report=report,
            )
            self._checkpoint(
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=val_metrics,
                epoch_metrics=epoch_metrics,
            )
            if should_stop:
                stop_reason = "early_stopping"
                tqdm.write(
                    f"Early stopping triggered at epoch={epoch}, "
                    f"bad_epochs={self.state.early_stopping_bad_epochs}"
                )
                break
        best = self._existing_best_model()
        write_training_complete(
            run_dir=self.run_dir,
            dataset_name=str(self.config["dataset"]["name"]),
            state=self.state,
            best=best,
            stop_reason=stop_reason,
        )
        return best

    def _apply_stage(self, epoch: int) -> None:
        stage = stage_name(self.config, epoch)
        if self._active_stage is not None and self._active_stage != stage:
            tqdm.write(f"STAGE {self._active_stage} -> STAGE {stage}")
        apply_training_stage(self.model, self.config, epoch=epoch)
        self.gradient_checkpointing = apply_configured_gradient_checkpointing(
            self.model, self.config
        )
        backbones = self.model.backbone_modules()
        if stage == "A" and any(
            parameter.requires_grad
            for module in backbones.values()
            for parameter in module.parameters()
        ):
            raise RuntimeError(
                "Stage A requires every backbone parameter to remain frozen"
            )
        if stage == "B" and any(
            not parameter.requires_grad
            for module in backbones.values()
            for parameter in module.parameters()
        ):
            raise RuntimeError(
                "Stage B requires every backbone parameter to be trainable"
            )
        self._active_stage = stage

    def _initialize_lazy_modules(self) -> None:
        batch = move_batch_to_device(next(iter(self.train_loader)), self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), self.precision.autocast():
            self.model(batch).validate()
        self.model.train(was_training)
        self._lazy_initialized = True
        self._log_cuda("after lazy initialization")

    def _train_epoch(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        max_batches: int | None,
    ) -> dict[str, float]:
        self.model.train()
        accumulation = int(self.config["training"]["gradient_accumulation_steps"])
        total = (
            len(self.train_loader)
            if max_batches is None
            else min(len(self.train_loader), int(max_batches))
        )
        iterator = tqdm(
            self.train_loader,
            total=total,
            desc=f"Epoch {self.state.epoch} train",
            disable=not self.progress,
        )
        sums: dict[str, float] = {}
        processed = 0
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > int(max_batches):
                break
            processed += 1
            batch = move_batch_to_device(raw_batch, self.device)
            with self.precision.autocast():
                output = self.model(batch)
                losses = self.loss_fn(output, batch)
                values = losses.as_dict()
                if bool(self.config["runtime"].get("fail_on_nan", True)) and not all(
                    math.isfinite(value) for value in values.values()
                ):
                    raise FloatingPointError(
                        f"Non-finite benchmark loss at epoch={self.state.epoch} batch={batch_index}: {values}"
                    )
                scaled = losses.total / accumulation
            self.precision.backward(scaled)
            if (
                stage_name(self.config, self.state.epoch) == "B"
                and not self._stage_b_gradient_audited
                and bool(self.config["runtime"]["gradient_audit"]["enabled"])
            ):
                coverage = full_ft_gradient_coverage(
                    self.model,
                    minimum_element_coverage=float(
                        self.config["runtime"]["gradient_audit"][
                            "minimum_element_coverage"
                        ]
                    ),
                )
                audit_path = self.run_dir / "metrics" / "full_ft_gradient_audit.json"
                audit_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
                tqdm.write(f"Full-FT backbone gradient audit: {coverage}")
                self._stage_b_gradient_audited = True
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + value
            if batch_index % accumulation == 0:
                self._optimizer_step(optimizer, scheduler)
            postfix = {
                "loss": f"{values['total']:.4f}",
                "L_bin": f"{values['binary']:.4f}",
                "L_align": f"{values['alignment']:.4f}",
                "L_router": f"{values['router']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "stage": stage_name(self.config, self.state.epoch),
            }
            memory = cuda_memory_postfix(self.device)
            if memory is not None:
                postfix["vram"] = memory
            iterator.set_postfix(**postfix)
        if processed and processed % accumulation:
            self._optimizer_step(optimizer, scheduler)
        denominator = max(1, processed)
        result = {key: value / denominator for key, value in sums.items()}
        result["duration_seconds"] = time.time() - started
        result["processed_batches"] = processed
        return result

    def _optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        max_grad_norm = self.config["training"].get("max_grad_norm")
        if max_grad_norm is not None:
            if self.precision.actual == "fp16":
                self.precision.scaler.unscale_(optimizer)
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

    def _update_early_stopping(self, metrics: Mapping[str, float]) -> bool:
        settings = self.config["training"]["early_stopping"]
        if not bool(settings["enabled"]):
            return False
        value = float(metrics[str(settings["metric"])])
        freeze_epochs = int(self.config["training"]["stages"]["stage_a_freeze_epochs"])
        adaptation_epoch = self.state.epoch - freeze_epochs
        if adaptation_epoch <= 0:
            self.state.early_stopping_best = max(
                value,
                self.state.early_stopping_best
                if self.state.early_stopping_best is not None
                else value,
            )
            self.state.early_stopping_bad_epochs = 0
            return False
        if adaptation_epoch == 1:
            self.state.early_stopping_best = None
            self.state.early_stopping_bad_epochs = 0
        best = self.state.early_stopping_best
        delta = float(settings["min_delta"])
        mode = str(settings["mode"])
        improved = best is None or (
            value > best + delta if mode == "max" else value < best - delta
        )
        grace = adaptation_epoch <= int(settings["minimum_backbone_adaptation_epochs"])
        if improved:
            self.state.early_stopping_best = value
            self.state.early_stopping_bad_epochs = 0
        elif not grace:
            self.state.early_stopping_bad_epochs += 1
        tqdm.write(
            f"Early stopping: metric={settings['metric']}, value={value:.6f}, "
            f"best={self.state.early_stopping_best:.6f}, "
            f"bad_epochs={self.state.early_stopping_bad_epochs}/{settings['patience']}"
        )
        return (
            not grace
            and not improved
            and self.state.early_stopping_bad_epochs >= int(settings["patience"])
        )

    def _checkpoint(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        metrics: Mapping[str, float],
        epoch_metrics: Mapping[str, Any],
    ) -> None:
        selection = str(self.config["checkpointing"]["selection_metric"])
        score = float(metrics[selection])
        minimum_stage = str(self.config["checkpointing"]["minimum_stage"])
        current_stage = stage_name(self.config, self.state.epoch)
        eligible = checkpoint_stage_is_eligible(self.config, self.state.epoch)
        improved = eligible and (
            self.state.best_metric is None or score > self.state.best_metric
        )
        if improved and bool(self.config["checkpointing"].get("keep_best", True)):
            self.state.best_metric = score
            self.state.best_threshold = self.state.threshold
            self.state.best_model_path = self.run_dir / "checkpoints" / "best_model.pt"
            save_best_model(
                path=self.state.best_model_path,
                model=self.model,
                state=self.state.checkpoint_dict(),
                metrics=epoch_metrics,
                config=self.config,
                precision=self.precision.actual,
            )
            (self.run_dir / "thresholds" / "best_threshold.json").write_text(
                json.dumps(
                    {
                        "binary_threshold": self.state.threshold,
                        "epoch": self.state.epoch,
                        "selection_metric": selection,
                        "selection_metric_value": score,
                        "checkpoint_path": str(self.state.best_model_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            tqdm.write(
                f"Saved best model-only checkpoint: {self.state.best_model_path}"
            )
        elif not eligible:
            tqdm.write(
                "Best-checkpoint selection deferred: "
                f"stage={current_stage}, minimum_stage={minimum_stage}"
            )
        if bool(self.config["checkpointing"].get("keep_last", True)):
            last = self.run_dir / "checkpoints" / "last.pt"
            save_last_checkpoint(
                path=last,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=self.precision.scaler,
                state=self.state.checkpoint_dict(),
                metrics=epoch_metrics,
                config=self.config,
                precision=self.precision.actual,
                scheduler_plan=self.scheduler_plan,
            )
            tqdm.write(f"Saved resume checkpoint: {last}")

    def _existing_best_model(self) -> Path:
        candidates = []
        if self.state.best_model_path is not None:
            candidates.append(self.state.best_model_path)
        candidates.append(self.run_dir / "checkpoints" / "best_model.pt")
        for path in candidates:
            if path.is_file():
                return path
        raise RuntimeError(f"Training has no valid best model checkpoint: {candidates}")

    def _log_cuda(self, label: str) -> None:
        if not bool(self.config["runtime"]["cuda_memory_logging"]["enabled"]):
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            tqdm.write(
                f"CUDA runtime ({label}): requested={self.device}, available={torch.cuda.is_available()}"
            )
            return
        index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        tqdm.write(
            f"CUDA runtime ({label}): gpu={torch.cuda.get_device_name(index)}, "
            f"allocated={torch.cuda.memory_allocated(index) / 1024**3:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved(index) / 1024**3:.2f} GiB, total={total:.2f} GiB"
        )

"""Inference loop for binary HERA benchmark datasets."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from src.evaluation.evaluator import move_batch_to_device
from src.training.precision import PrecisionManager

from .loss import BinaryBenchmarkLoss
from .metrics import BinaryPredictionBundle
from .model import BinaryBenchmarkHERA


def run_binary_inference(
    *,
    model: BinaryBenchmarkHERA,
    loader: torch.utils.data.DataLoader,
    loss_fn: BinaryBenchmarkLoss | None,
    device: torch.device,
    precision: PrecisionManager,
    max_batches: int | None,
    desc: str,
    progress: bool,
) -> BinaryPredictionBundle:
    model.eval()
    bundle = BinaryPredictionBundle()
    total = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    iterator = tqdm(loader, total=total, desc=desc, disable=not progress)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > int(max_batches):
                break
            batch = move_batch_to_device(raw_batch, device)
            with precision.autocast():
                output = model(batch)
                losses = loss_fn(output, batch) if loss_fn is not None else None
            if losses is not None:
                bundle.add_losses(losses.as_dict())
            bundle.ids.extend(str(value) for value in raw_batch["id"])
            bundle.targets.extend(
                float(value) for value in raw_batch["binary_target"].tolist()
            )
            bundle.scores.extend(
                float(value)
                for value in output.binary_probability.detach().cpu().tolist()
            )
            if output.subtype_logits is not None:
                subtype_predictions = (
                    output.subtype_logits.argmax(dim=1).detach().cpu().tolist()
                )
            else:
                subtype_predictions = [-100] * len(raw_batch["id"])
            bundle.subtype_targets.extend(
                int(value) for value in raw_batch["subtype_target"].tolist()
            )
            bundle.subtype_predictions.extend(
                int(value) for value in subtype_predictions
            )
            bundle.text_label_targets.extend(
                float(value) if available else float("nan")
                for value, available in zip(
                    raw_batch["text_label_target"].tolist(),
                    raw_batch["text_label_available"].tolist(),
                )
            )
            if output.text_label_logit is not None:
                text_scores = (
                    torch.sigmoid(output.text_label_logit).detach().cpu().tolist()
                )
            else:
                text_scores = [float("nan")] * len(raw_batch["id"])
            bundle.text_label_scores.extend(float(value) for value in text_scores)
            bundle.metadata.extend(dict(value) for value in raw_batch["metadata"])
    return bundle

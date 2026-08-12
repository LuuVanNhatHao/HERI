"""Shared inference, metrics, and report computation for Phase-2 baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.bbox import iou
from src.metrics.attribution import attribution_metrics
from src.metrics.detection import binary_detection_metrics
from src.metrics.reports import (
    attribution_classification_report_text,
    attribution_report_dict,
    binary_classification_report_text,
    binary_report_dict,
)
from src.thresholding.tuner import Thresholds
from src.training.precision import PrecisionManager


@dataclass
class PredictionBundle:
    """Accumulated predictions and targets for one split."""

    ids: list[int] = field(default_factory=list)
    fake_cls: list[str] = field(default_factory=list)
    generator: list[str | None] = field(default_factory=list)
    image_source: list[str | None] = field(default_factory=list)
    annotation_image_path: list[str | None] = field(default_factory=list)
    has_image_manipulation: list[bool] = field(default_factory=list)
    has_text_manipulation: list[bool] = field(default_factory=list)
    binary_targets: list[float] = field(default_factory=list)
    binary_scores: list[float] = field(default_factory=list)
    atomic_targets: list[list[float]] = field(default_factory=list)
    atomic_scores: list[list[float]] = field(default_factory=list)
    image_ious: list[float] = field(default_factory=list)
    image_grounding_pred_indices: list[int | None] = field(default_factory=list)
    image_grounding_pred_ious: list[float | None] = field(default_factory=list)
    face_area_fraction: list[float | None] = field(default_factory=list)
    face_size_bucket: list[str | None] = field(default_factory=list)
    text_target_word_ids: list[str] = field(default_factory=list)
    text_pred_word_ids: list[str] = field(default_factory=list)
    router_diagnostics: list[str] = field(default_factory=list)
    router_batch_summaries: list[dict[str, Any]] = field(default_factory=list)
    text_tp: int = 0
    text_fp: int = 0
    text_fn: int = 0
    loss_sums: dict[str, float] = field(default_factory=dict)
    batches: int = 0

    def add_loss(self, losses: Mapping[str, float]) -> None:
        for key, value in losses.items():
            self.loss_sums[key] = self.loss_sums.get(key, 0.0) + float(value)
        self.batches += 1

    def mean_losses(self) -> dict[str, float]:
        denom = max(1, self.batches)
        return {key: value / denom for key, value in self.loss_sums.items()}


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a structured batch to device."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def run_inference(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: Any | None,
    device: torch.device,
    precision: PrecisionManager,
    max_batches: int | None,
    desc: str,
    progress: bool,
) -> PredictionBundle:
    """Run split inference with tqdm and accumulate predictions."""

    model.eval()
    bundle = PredictionBundle()
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    iterator = tqdm(loader, total=total, desc=desc, disable=not progress)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with precision.autocast():
                output = model(batch)
                if loss_fn is not None:
                    losses = loss_fn(output, batch)
                    bundle.add_loss(losses.as_dict())
            _accumulate_batch(bundle, output, batch)
    return bundle


def _accumulate_batch(bundle: PredictionBundle, output: Any, batch: Mapping[str, Any]) -> None:
    binary_scores = output.binary_probability.detach().cpu()
    atomic_scores = output.atomic_probabilities.detach().cpu()
    binary_targets = batch["binary_target"].detach().cpu()
    atomic_targets = batch["atomic_target"].detach().cpu()
    bundle.ids.extend(int(value) for value in batch["id"])
    bundle.fake_cls.extend(str(value) for value in batch["fake_cls"])
    bundle.generator.extend(batch["generator"])
    batch_size = binary_targets.shape[0]
    bundle.image_source.extend(list(batch.get("image_source", [None] * batch_size)))
    bundle.annotation_image_path.extend(list(batch.get("annotation_image_path", [None] * batch_size)))
    bundle.has_image_manipulation.extend(bool(v) for v in batch["has_image_manipulation"].detach().cpu())
    bundle.has_text_manipulation.extend(bool(v) for v in batch["has_text_manipulation"].detach().cpu())
    bundle.binary_targets.extend(float(v) for v in binary_targets)
    bundle.binary_scores.extend(float(v) for v in binary_scores)
    bundle.atomic_targets.extend(atomic_targets.tolist())
    bundle.atomic_scores.extend(atomic_scores.tolist())
    bundle.router_diagnostics.extend(_router_sample_diagnostics(output, batch_size))
    router_batch_summary = getattr(output, "metadata", {}).get("router_batch_summary")
    if router_batch_summary:
        bundle.router_batch_summaries.append(router_batch_summary)
    _accumulate_face_size_metadata(bundle, batch)
    _accumulate_image_ious(bundle, output, batch)
    _accumulate_text_counts(bundle, output, batch)


def _accumulate_image_ious(bundle: PredictionBundle, output: Any, batch: Mapping[str, Any]) -> None:
    logits = output.image_grounding_logits.detach().cpu()
    masks = output.image_grounding_mask.detach().cpu().to(torch.bool)
    boxes = batch["candidate_face_boxes"].detach().cpu()
    targets = batch["fake_image_box_target"].detach().cpu()
    applies = batch["has_image_manipulation"].detach().cpu().to(torch.bool)
    atomic_targets = batch["atomic_target"].detach().cpu()
    for idx in range(logits.shape[0]):
        if not bool(applies[idx]) or not bool(masks[idx].any()):
            bundle.image_grounding_pred_indices.append(None)
            bundle.image_grounding_pred_ious.append(None)
            continue
        sample_logits = _select_image_grounding_logits(logits, atomic_targets, idx)
        if sample_logits is None:
            bundle.image_grounding_pred_indices.append(None)
            bundle.image_grounding_pred_ious.append(None)
            continue
        pred_idx = int(torch.argmax(sample_logits.masked_fill(~masks[idx], -1e4)).item())
        pred_iou = iou(boxes[idx, pred_idx].tolist(), targets[idx].tolist())
        bundle.image_ious.append(pred_iou)
        bundle.image_grounding_pred_indices.append(pred_idx)
        bundle.image_grounding_pred_ious.append(pred_iou)


def _accumulate_text_counts(bundle: PredictionBundle, output: Any, batch: Mapping[str, Any]) -> None:
    logits = output.text_grounding_logits.detach().cpu()
    targets = batch["text_grounding_target"].detach().cpu()
    masks = output.text_grounding_mask.detach().cpu().to(torch.bool)
    output_word_ids = getattr(output, "text_word_ids", None)
    word_ids = (
        output_word_ids.detach().cpu()
        if isinstance(output_word_ids, torch.Tensor)
        else batch["text_word_ids"].detach().cpu()
    )
    applies = batch["has_text_manipulation"].detach().cpu().to(torch.bool)
    atomic_targets = batch["atomic_target"].detach().cpu()
    probs = (
        output.text_grounding_probabilities.detach().cpu()
        if hasattr(output, "text_grounding_probabilities")
        else torch.sigmoid(logits)
    )
    for sample_idx in range(logits.shape[0]):
        if not bool(applies[sample_idx]):
            bundle.text_target_word_ids.append("")
            bundle.text_pred_word_ids.append("")
            continue
        sample_logits, sample_probs = _select_text_grounding_scores(logits, probs, atomic_targets, sample_idx)
        if sample_logits is None or sample_probs is None:
            bundle.text_target_word_ids.append("")
            bundle.text_pred_word_ids.append("")
            continue
        valid = masks[sample_idx] & (targets[sample_idx] >= 0)
        if not bool(valid.any()):
            bundle.text_target_word_ids.append("")
            bundle.text_pred_word_ids.append("")
            continue
        target_words: list[int] = []
        pred_words: list[int] = []
        sample_word_ids = sorted(
            int(word_id)
            for word_id in torch.unique(word_ids[sample_idx][valid]).tolist()
            if int(word_id) >= 0
        )
        for word_id in sample_word_ids:
            word_mask = valid & (word_ids[sample_idx] == word_id)
            y_true = int((targets[sample_idx][word_mask] == 1).any().item())
            y_pred = int((sample_probs[word_mask].max() >= 0.5).item())
            if y_true:
                target_words.append(word_id)
            if y_pred:
                pred_words.append(word_id)
            bundle.text_tp += int(y_true == 1 and y_pred == 1)
            bundle.text_fp += int(y_true == 0 and y_pred == 1)
            bundle.text_fn += int(y_true == 1 and y_pred == 0)
        bundle.text_target_word_ids.append(json.dumps(target_words))
        bundle.text_pred_word_ids.append(json.dumps(pred_words))


def _select_image_grounding_logits(
    logits: torch.Tensor,
    atomic_targets: torch.Tensor,
    sample_idx: int,
) -> torch.Tensor | None:
    if logits.ndim == 2:
        return logits[sample_idx]
    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError("image_grounding_logits must be [B,N] or [B,2,N]")
    if atomic_targets[sample_idx, 0] > 0.5:
        return logits[sample_idx, 0, :]
    if atomic_targets[sample_idx, 1] > 0.5:
        return logits[sample_idx, 1, :]
    return None


def _select_text_grounding_scores(
    logits: torch.Tensor,
    probs: torch.Tensor,
    atomic_targets: torch.Tensor,
    sample_idx: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if logits.ndim == 2:
        return logits[sample_idx], probs[sample_idx]
    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError("text_grounding_logits must be [B,N] or [B,2,N]")
    if atomic_targets[sample_idx, 2] > 0.5:
        return logits[sample_idx, 0, :], probs[sample_idx, 0, :]
    if atomic_targets[sample_idx, 3] > 0.5:
        return logits[sample_idx, 1, :], probs[sample_idx, 1, :]
    return None, None


def _router_sample_diagnostics(output: Any, batch_size: int) -> list[str]:
    means = getattr(output, "metadata", {}).get("router_family_mean_reliability")
    if not means:
        return [""] * batch_size
    rows: list[str] = []
    for sample_idx in range(batch_size):
        rows.append(
            json.dumps(
                {
                    family: float(values[sample_idx])
                    for family, values in means.items()
                    if sample_idx < len(values)
                },
                sort_keys=True,
            )
        )
    return rows


def _accumulate_face_size_metadata(bundle: PredictionBundle, batch: Mapping[str, Any]) -> None:
    targets = batch["fake_image_box_target"].detach().cpu()
    applies = batch["has_image_manipulation"].detach().cpu().to(torch.bool)
    image = batch.get("image")
    if isinstance(image, torch.Tensor):
        canvas_area = float(image.shape[-1] * image.shape[-2])
    else:
        canvas_area = 384.0 * 384.0
    for idx in range(targets.shape[0]):
        if not bool(applies[idx]):
            bundle.face_area_fraction.append(None)
            bundle.face_size_bucket.append(None)
            continue
        box = targets[idx]
        width = max(0.0, float(box[2] - box[0]))
        height = max(0.0, float(box[3] - box[1]))
        fraction = (width * height) / max(canvas_area, 1.0)
        bundle.face_area_fraction.append(fraction)
        bundle.face_size_bucket.append(_face_size_bucket(fraction))


def _face_size_bucket(area_fraction: float) -> str:
    if area_fraction < 0.05:
        return "small"
    if area_fraction < 0.15:
        return "medium"
    return "large"


def compute_metrics_and_reports(
    bundle: PredictionBundle,
    thresholds: Thresholds,
) -> dict[str, Any]:
    """Compute DGM4 metrics and sklearn reports for accumulated predictions."""

    binary_targets = np.asarray(bundle.binary_targets, dtype=int)
    binary_scores = np.asarray(bundle.binary_scores, dtype=float)
    atomic_targets = np.asarray(bundle.atomic_targets, dtype=int)
    atomic_scores = np.asarray(bundle.atomic_scores, dtype=float)
    binary_pred = (binary_scores >= thresholds.binary).astype(int)
    atomic_pred = (atomic_scores >= np.asarray(thresholds.atomic).reshape(1, -1)).astype(int)

    image_metrics = _image_metrics_from_ious(bundle.image_ious)
    text_metrics = _text_metrics_from_counts(bundle.text_tp, bundle.text_fp, bundle.text_fn)
    metrics = {
        **binary_detection_metrics(binary_targets, binary_scores, threshold=thresholds.binary),
        **attribution_metrics(atomic_targets, atomic_scores, thresholds=np.asarray(thresholds.atomic)),
        **image_metrics,
        **text_metrics,
        "losses": bundle.mean_losses(),
    }
    return {
        "metrics": metrics,
        "binary_report": binary_classification_report_text(binary_targets, binary_pred),
        "attribution_report": attribution_classification_report_text(atomic_targets, atomic_pred),
        "binary_report_dict": binary_report_dict(binary_targets, binary_pred),
        "attribution_report_dict": attribution_report_dict(atomic_targets, atomic_pred),
        "binary_pred": binary_pred,
        "atomic_pred": atomic_pred,
    }


def _image_metrics_from_ious(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"image_mean_iou": None, "image_iou50": None, "image_iou75": None}
    arr = np.asarray(values, dtype=float)
    return {
        "image_mean_iou": float(arr.mean()),
        "image_iou50": float((arr >= 0.5).mean()),
        "image_iou75": float((arr >= 0.75).mean()),
    }


def _text_metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"text_precision": precision, "text_recall": recall, "text_f1": f1}

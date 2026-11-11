"""Shared inference, metrics, and report computation for baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

from heri.data.bbox import iou
from heri.metrics.attribution import attribution_metrics
from heri.metrics.detection import binary_detection_metrics
from heri.metrics.consistency import (
    evidence_decision_metrics,
    unavailable_evidence_decision_metrics,
)
from heri.metrics.verified_support import verified_support_units
from heri.losses.box_grounding import normalized_target_boxes, patch_overlap_targets
from heri.metrics.reports import (
    attribution_classification_report_text,
    attribution_report_dict,
    binary_classification_report_text,
    binary_report_dict,
)
from heri.thresholding.tuner import Thresholds
from heri.training.precision import PrecisionManager
from heri.evaluation.protocol import protocol_metadata
from heri.evaluation.samm_protocol import (
    SAMM_RELEASED_LABEL_ORDER,
    is_samm_reporting_ontology,
    samm_released_metrics,
    samm_released_scores,
    samm_released_targets,
)
from heri.evaluation.hypothesis_grounding import (
    TEXT_GROUNDING_AGGREGATIONS,
    image_grounding_presence_probability,
    route_image_grounding_boxes,
    route_text_grounding_probabilities,
)


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
    image_grounding_available: list[bool] = field(default_factory=list)
    text_grounding_available: list[bool] = field(default_factory=list)
    binary_targets: list[float] = field(default_factory=list)
    binary_scores: list[float] = field(default_factory=list)
    atomic_targets: list[list[float]] = field(default_factory=list)
    atomic_scores: list[list[float]] = field(default_factory=list)
    evidence_support_scores: list[list[float]] = field(default_factory=list)
    evidence_support_by_modality_scores: list[list[list[float]]] = field(
        default_factory=list
    )
    evidence_support_modality_mask: list[list[bool]] = field(default_factory=list)
    evidence_support_modality_names: list[str] = field(default_factory=list)
    class_targets: list[int] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    hypothesis_names: list[str] = field(default_factory=list)
    image_ious: list[float] = field(default_factory=list)
    image_all_ious: list[float] = field(default_factory=list)
    image_grounding_pred_indices: list[int | None] = field(default_factory=list)
    image_grounding_pred_ious: list[float | None] = field(default_factory=list)
    image_grounding_pred_boxes: list[str] = field(default_factory=list)
    image_grounding_pred_boxes_original: list[str] = field(default_factory=list)
    image_presence_scores: list[float] = field(default_factory=list)
    image_presence_hypothesis_names: list[str] = field(default_factory=list)
    verified_support_hits: list[list[list[bool]]] = field(default_factory=list)
    verified_support_eligible: list[list[list[bool]]] = field(default_factory=list)
    face_area_fraction: list[float | None] = field(default_factory=list)
    face_size_bucket: list[str | None] = field(default_factory=list)
    text_target_word_ids: list[str] = field(default_factory=list)
    text_pred_word_ids: list[str] = field(default_factory=list)
    router_diagnostics: list[str] = field(default_factory=list)
    router_batch_summaries: list[dict[str, Any]] = field(default_factory=list)
    hypothesis_probabilities_by_cycle: list[str] = field(default_factory=list)
    text_tp: int = 0
    text_fp: int = 0
    text_fn: int = 0
    text_token_tp: int = 0
    text_token_fp: int = 0
    text_token_fn: int = 0
    text_positive_tp: int = 0
    text_positive_fp: int = 0
    text_positive_fn: int = 0
    text_total_words: int = 0
    text_visible_words: int = 0
    text_total_positive_words: int = 0
    text_unseen_positive_words: int = 0
    text_coverage_complete: bool = True
    text_original_word_metadata_complete: bool = True
    text_token_count: int = 0
    text_evaluated_sample_count: int = 0
    text_manipulated_evaluated_sample_count: int = 0
    text_absent_sample_count: int = 0
    text_absent_sample_with_prediction_count: int = 0
    text_absent_token_count: int = 0
    text_absent_token_predicted_positive_count: int = 0
    text_absent_word_count: int = 0
    text_absent_word_predicted_positive_count: int = 0
    text_grounding_aggregation: str = "relative_soft"
    text_sample_counts: list[dict[str, int]] = field(default_factory=list)
    text_word_probabilities: list[str] = field(default_factory=list)
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

    non_blocking = device.type == "cuda"
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = (
            value.to(device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
        )
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
    text_grounding_aggregation: str = "relative_soft",
) -> PredictionBundle:
    """Run split inference with tqdm and accumulate predictions."""

    model.eval()
    bundle = PredictionBundle(text_grounding_aggregation=text_grounding_aggregation)
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
            _accumulate_batch(
                bundle,
                output,
                batch,
                text_grounding_aggregation=text_grounding_aggregation,
            )
    return bundle


def run_text_grounding_diagnostic_inference(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: Any | None,
    device: torch.device,
    precision: PrecisionManager,
    max_batches: int | None,
    desc: str,
    progress: bool,
    aggregations: Sequence[str] = TEXT_GROUNDING_AGGREGATIONS,
) -> tuple[PredictionBundle, dict[str, PredictionBundle]]:
    """Evaluate several text aggregation rules with one model forward pass.

    The first aggregation receives the complete prediction bundle.  Remaining
    bundles retain aggregate text counts only, which avoids tripling the large
    per-sample prediction payload on SAMM-20K.
    """

    modes = tuple(dict.fromkeys(str(value).strip().lower() for value in aggregations))
    if not modes:
        raise ValueError("At least one text grounding aggregation is required")
    invalid = [value for value in modes if value not in TEXT_GROUNDING_AGGREGATIONS]
    if invalid:
        raise ValueError(f"Unknown text grounding aggregation(s): {invalid}")
    primary = PredictionBundle(text_grounding_aggregation=modes[0])
    text_bundles = {
        mode: (primary if index == 0 else PredictionBundle(text_grounding_aggregation=mode))
        for index, mode in enumerate(modes)
    }
    model.eval()
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    iterator = tqdm(loader, total=total, desc=desc, disable=not progress)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with precision.autocast():
                output = model(batch)
                losses = loss_fn(output, batch) if loss_fn is not None else None
            if losses is not None:
                primary.add_loss(losses.as_dict())
            _accumulate_batch(
                primary,
                output,
                batch,
                text_grounding_aggregation=modes[0],
            )
            for mode in modes[1:]:
                _accumulate_text_counts(
                    text_bundles[mode],
                    output,
                    batch,
                    text_grounding_aggregation=mode,
                    store_details=False,
                )
    return primary, text_bundles


def _accumulate_batch(
    bundle: PredictionBundle,
    output: Any,
    batch: Mapping[str, Any],
    *,
    text_grounding_aggregation: str = "relative_soft",
) -> None:
    binary_scores = output.binary_probability.detach().cpu()
    atomic_scores = output.atomic_probabilities.detach().cpu()
    binary_targets = batch["binary_target"].detach().cpu()
    atomic_targets = batch["atomic_target"].detach().cpu()
    bundle.ids.extend(int(value) for value in batch["id"])
    bundle.fake_cls.extend(str(value) for value in batch["fake_cls"])
    bundle.generator.extend(batch["generator"])
    batch_size = binary_targets.shape[0]
    cycle_probabilities = getattr(output, "metadata", {}).get("hypothesis_probabilities_by_cycle", [])
    bundle.hypothesis_probabilities_by_cycle.extend(
        json.dumps([cycle[index] for cycle in cycle_probabilities]) if cycle_probabilities else ""
        for index in range(batch_size)
    )
    bundle.image_source.extend(list(batch.get("image_source", [None] * batch_size)))
    bundle.annotation_image_path.extend(list(batch.get("annotation_image_path", [None] * batch_size)))
    bundle.has_image_manipulation.extend(bool(v) for v in batch["has_image_manipulation"].detach().cpu())
    bundle.has_text_manipulation.extend(bool(v) for v in batch["has_text_manipulation"].detach().cpu())
    image_available = batch.get("image_grounding_available")
    text_available = batch.get("text_grounding_available")
    bundle.image_grounding_available.extend(
        [True] * batch_size
        if not isinstance(image_available, torch.Tensor)
        else [bool(value) for value in image_available.detach().cpu()]
    )
    bundle.text_grounding_available.extend(
        [True] * batch_size
        if not isinstance(text_available, torch.Tensor)
        else [bool(value) for value in text_available.detach().cpu()]
    )
    bundle.binary_targets.extend(float(v) for v in binary_targets)
    bundle.binary_scores.extend(float(v) for v in binary_scores)
    bundle.atomic_targets.extend(atomic_targets.tolist())
    bundle.atomic_scores.extend(atomic_scores.tolist())
    evidence_support = getattr(output, "evidence_support", None)
    if isinstance(evidence_support, torch.Tensor):
        support = evidence_support.detach().cpu()
        if support.shape != atomic_scores.shape:
            raise ValueError("evidence_support must align with atomic probabilities [B,K]")
        bundle.evidence_support_scores.extend(support.tolist())
    support_by_modality = getattr(output, "evidence_support_by_modality", None)
    support_modality_mask = getattr(output, "evidence_support_modality_mask", None)
    if isinstance(support_by_modality, torch.Tensor):
        support_by_modality = support_by_modality.detach().cpu()
        if support_by_modality.shape[:2] != atomic_scores.shape:
            raise ValueError(
                "evidence_support_by_modality must align with [B,K,M]"
            )
        if not isinstance(support_modality_mask, torch.Tensor):
            raise ValueError("Per-modality support requires a modality mask")
        mask = support_modality_mask.detach().cpu().to(torch.bool)
        if mask.shape != support_by_modality.shape[1:]:
            raise ValueError("evidence_support_modality_mask must be [K,M]")
        if bundle.evidence_support_modality_mask:
            if bundle.evidence_support_modality_mask != mask.tolist():
                raise ValueError("Evidence modality mask changed between batches")
        else:
            bundle.evidence_support_modality_mask = mask.tolist()
            names = getattr(output, "metadata", {}).get(
                "evidence_modality_order", ("image", "text")
            )
            bundle.evidence_support_modality_names = [str(value) for value in names]
        bundle.evidence_support_by_modality_scores.extend(
            support_by_modality.tolist()
        )
    if not bundle.hypothesis_names:
        bundle.hypothesis_names = list(
            getattr(output, "metadata", {}).get(
                "hypothesis_order", batch.get("hypothesis_names", [])
            )
        )
    if not bundle.class_names:
        bundle.class_names = list(batch.get("class_names", []))
    class_target = batch.get("class_target")
    if isinstance(class_target, torch.Tensor):
        bundle.class_targets.extend(int(value) for value in class_target.detach().cpu())
    bundle.router_diagnostics.extend(_router_sample_diagnostics(output, batch_size))
    router_batch_summary = getattr(output, "metadata", {}).get("router_batch_summary")
    if router_batch_summary:
        bundle.router_batch_summaries.append(router_batch_summary)
    _accumulate_face_size_metadata(bundle, batch)
    _accumulate_image_ious(bundle, output, batch)
    _accumulate_verified_support(bundle, output, batch)
    _accumulate_text_counts(
        bundle,
        output,
        batch,
        text_grounding_aggregation=text_grounding_aggregation,
    )


def _accumulate_image_ious(bundle: PredictionBundle, output: Any, batch: Mapping[str, Any]) -> None:
    targets = batch["fake_image_box_target"].detach().cpu()
    applies = batch["has_image_manipulation"].detach().cpu().to(torch.bool)
    available = batch.get("image_grounding_available")
    available = (
        torch.ones_like(applies)
        if available is None
        else available.detach().cpu().to(torch.bool)
    )
    image = batch["image"]
    height, width = int(image.shape[-2]), int(image.shape[-1])
    scale = targets.new_tensor([width, height, width, height])
    normalized_targets = (targets / scale).clamp(0.0, 1.0)
    if hasattr(output, "image_grounding_boxes"):
        if output.image_grounding_boxes.ndim == 3:
            predicted_boxes, route_weights = route_image_grounding_boxes(output)
        else:
            present = image_grounding_presence_probability(output) >= 0.5
            predicted_boxes = output.image_grounding_boxes * present[:, None]
            route_weights = output.atomic_probabilities[:, :2]
    else:
        raise ValueError("Protocol-clean evaluation requires direct image_grounding_boxes")
    predicted_boxes = predicted_boxes.detach().float().cpu()
    route_weights = route_weights.detach().cpu()
    presence_scores = image_grounding_presence_probability(output).detach().cpu()
    if presence_scores.shape != (predicted_boxes.shape[0],):
        raise ValueError("image presence scores must be [B]")
    bundle.image_presence_scores.extend(float(value) for value in presence_scores)
    if not bundle.image_presence_hypothesis_names:
        metadata = getattr(output, "metadata", {})
        hypothesis_order = tuple(metadata.get("hypothesis_order", ()))
        image_indices = tuple(metadata.get("image_hypothesis_indices", (0, 1)))
        if hypothesis_order and max(image_indices, default=-1) < len(hypothesis_order):
            bundle.image_presence_hypothesis_names = [
                str(hypothesis_order[index]) for index in image_indices
            ]
    for idx in range(predicted_boxes.shape[0]):
        # Eq.22 clips to valid (unpadded) image before removing padding and
        # restoring original pixels. Structural geometry is independent of GT.
        metadata = batch.get("image_transform_meta", [])
        meta = metadata[idx] if idx < len(metadata) else None
        box, original_box = _clip_and_restore_box(predicted_boxes[idx], meta, width, height)
        predicted_boxes[idx] = box
        target_box, original_target = _clip_and_restore_box(normalized_targets[idx], meta, width, height)
        route_index = int(route_weights[idx].argmax().item())
        bundle.image_grounding_pred_indices.append(route_index)
        bundle.image_grounding_pred_boxes.append(json.dumps(predicted_boxes[idx].tolist()))
        bundle.image_grounding_pred_boxes_original.append(json.dumps(original_box.tolist()))
        if not bool(available[idx]):
            bundle.image_grounding_pred_ious.append(None)
            continue
        if not bool(applies[idx]):
            # Match the released evaluator's empty-target convention. Never use
            # GT subtype to zero a prediction or choose a grounding head.
            pred_iou = float(bool((predicted_boxes[idx] < 1e-4).all()))
            bundle.image_all_ious.append(pred_iou)
            bundle.image_grounding_pred_ious.append(pred_iou)
            continue
        # Eq.22 deliberately emits the empty box on abstention. Clipping an
        # off-image prediction can also produce zero area; both are valid
        # prediction failures with IoU zero, not invalid target annotations.
        pred_extent = original_box[2:] - original_box[:2]
        pred_iou = (
            iou(original_box.tolist(), original_target.tolist())
            if bool((pred_extent > 0).all()) else 0.0
        )
        bundle.image_ious.append(pred_iou)
        bundle.image_all_ious.append(pred_iou)
        bundle.image_grounding_pred_ious.append(pred_iou)


def _clip_and_restore_box(box: torch.Tensor, meta: Mapping[str, Any] | None,
                          width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip normalized canvas xyxy to content; return canvas and original pixels."""
    result = box.float().clone()
    if bool((result == 0).all()):
        return result, result.clone()
    if meta is None:
        result = result.clamp(0.0, 1.0)
        return result, result * result.new_tensor([width, height, width, height])
    left, top = float(meta["pad_left"]), float(meta["pad_top"])
    right, bottom = left + float(meta["resized_width"]), top + float(meta["resized_height"])
    pixels = result * result.new_tensor([width, height, width, height])
    pixels[[0, 2]] = pixels[[0, 2]].clamp(left, right)
    pixels[[1, 3]] = pixels[[1, 3]].clamp(top, bottom)
    original = (pixels - pixels.new_tensor([left, top, left, top])) / float(meta["scale"])
    original[[0, 2]] = original[[0, 2]].clamp(0.0, float(meta["original_width"]))
    original[[1, 3]] = original[[1, 3]].clamp(0.0, float(meta["original_height"]))
    return pixels / pixels.new_tensor([width, height, width, height]), original


def _accumulate_verified_support(bundle: PredictionBundle, output: Any,
                                batch: Mapping[str, Any]) -> None:
    if getattr(output, "evidence_support_by_modality", None) is None:
        return
    scores = output.atomic_probabilities
    hits = torch.zeros(scores.shape + (2,), dtype=torch.bool, device=scores.device)
    eligible = torch.zeros_like(hits)
    metadata = output.metadata
    target_boxes = normalized_target_boxes(batch, device=scores.device, dtype=torch.float32)
    image_targets = patch_overlap_targets(output.image_grounding_patch_boxes.float(), target_boxes)
    for modality, name, local_targets in ((0, "image", image_targets),
                                          (1, "text", batch["text_grounding_target"] > 0)):
        available = batch.get(f"{name}_grounding_available", torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device))
        unit_hits, unit_eligible = verified_support_units(
            atomic_probabilities=scores,
            atomic_targets=batch["atomic_target"],
            grounding_logits=getattr(output, f"{name}_grounding_logits"),
            grounding_mask=getattr(output, f"{name}_grounding_mask"),
            localization_targets=local_targets,
            hypothesis_indices=tuple(metadata[f"{name}_hypothesis_indices"]),
            annotation_available=available,
        )
        hits[:, :, modality] = unit_hits
        eligible[:, :, modality] = unit_eligible
    bundle.verified_support_hits.extend(hits.cpu().tolist())
    bundle.verified_support_eligible.extend(eligible.cpu().tolist())


def _accumulate_text_counts(
    bundle: PredictionBundle,
    output: Any,
    batch: Mapping[str, Any],
    *,
    text_grounding_aggregation: str = "relative_soft",
    store_details: bool = True,
) -> None:
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
    available = batch.get("text_grounding_available")
    available = (
        torch.ones_like(applies)
        if available is None
        else available.detach().cpu().to(torch.bool)
    )
    if hasattr(output, "text_grounding_probabilities"):
        bundle.text_grounding_aggregation = text_grounding_aggregation
        probs, _ = route_text_grounding_probabilities(
            output, aggregation=text_grounding_aggregation
        )
        probs = probs.detach().cpu()
    else:
        bundle.text_grounding_aggregation = "generic_head"
        probs = torch.sigmoid(logits)
    for sample_idx in range(logits.shape[0]):
        sample_probs = probs[sample_idx]
        if not bool(available[sample_idx]):
            bundle.text_coverage_complete = False
            if store_details:
                bundle.text_target_word_ids.append("")
                bundle.text_pred_word_ids.append("")
                bundle.text_word_probabilities.append("")
                bundle.text_sample_counts.append({
                    "word_tp": 0, "word_fp": 0, "word_fn": 0,
                    "token_tp": 0, "token_fp": 0, "token_fn": 0,
                    "words": len(batch.get("words", [[]])[sample_idx]),
                    "visible_words": 0, "unseen_positive_words": 0,
                })
            continue
        bundle.text_evaluated_sample_count += 1
        if bool(applies[sample_idx]):
            bundle.text_manipulated_evaluated_sample_count += 1
        valid = masks[sample_idx] & (targets[sample_idx] >= 0)
        token_true = targets[sample_idx][valid] == 1
        token_pred = sample_probs[valid] >= 0.5
        token_tp = int((token_true & token_pred).sum())
        token_fp = int((~token_true & token_pred).sum())
        token_fn = int((token_true & ~token_pred).sum())
        bundle.text_token_tp += token_tp
        bundle.text_token_fp += token_fp
        bundle.text_token_fn += token_fn
        bundle.text_token_count += int(valid.sum())
        visible_ids = {
            int(v) for v in word_ids[sample_idx][valid].unique().tolist() if int(v) >= 0
        }
        word_probs = {}
        visible_targets = set()
        for word_id in sorted(visible_ids):
            word_mask = valid & (word_ids[sample_idx] == word_id)
            word_probs[word_id] = float(sample_probs[word_mask].max())
            if bool((targets[sample_idx][word_mask] == 1).any()):
                visible_targets.add(word_id)
        if "fake_text_pos" in batch and "words" in batch:
            positions = batch["fake_text_pos"][sample_idx]
            target_words = {int(v) for v in positions.tolist() if int(v) >= 0}
            word_count = len(batch["words"][sample_idx])
            if any(v >= word_count for v in target_words | visible_ids):
                raise ValueError("Grounding word IDs exceed the original word sequence")
        else:
            # Synthetic/legacy batches cannot establish truncation coverage.
            bundle.text_coverage_complete = False
            bundle.text_original_word_metadata_complete = False
            target_words = visible_targets
            word_count = len(visible_ids)
        pred_words = {word_id for word_id, p in word_probs.items() if p >= 0.5}
        tp = len(target_words & pred_words)
        fp = len(pred_words - target_words)
        fn = len(target_words - pred_words)
        bundle.text_tp += tp
        bundle.text_fp += fp
        bundle.text_fn += fn
        if bool(applies[sample_idx]):
            bundle.text_positive_tp += len(visible_targets & pred_words)
            bundle.text_positive_fp += len(pred_words - visible_targets)
            bundle.text_positive_fn += len(visible_targets - pred_words)
        else:
            bundle.text_absent_sample_count += 1
            bundle.text_absent_sample_with_prediction_count += int(bool(pred_words))
            bundle.text_absent_token_count += int(valid.sum())
            bundle.text_absent_token_predicted_positive_count += int(token_pred.sum())
            bundle.text_absent_word_count += word_count
            bundle.text_absent_word_predicted_positive_count += len(pred_words)
        unseen_positive = len(target_words - visible_ids)
        bundle.text_total_words += word_count
        bundle.text_visible_words += len(visible_ids)
        bundle.text_total_positive_words += len(target_words)
        bundle.text_unseen_positive_words += unseen_positive
        if store_details:
            bundle.text_target_word_ids.append(json.dumps(sorted(target_words)))
            bundle.text_pred_word_ids.append(json.dumps(sorted(pred_words)))
            bundle.text_word_probabilities.append(json.dumps(word_probs, sort_keys=True))
            bundle.text_sample_counts.append({
                "word_tp": tp, "word_fp": fp, "word_fn": fn,
                "token_tp": token_tp, "token_fp": token_fp, "token_fn": token_fn,
                "words": word_count, "visible_words": len(visible_ids),
                "unseen_positive_words": unseen_positive,
            })


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
    available = batch.get("image_grounding_available")
    if isinstance(available, torch.Tensor):
        applies = applies & available.detach().cpu().to(torch.bool)
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
    atomic_thresholds = np.asarray(thresholds.atomic, dtype=float)
    if atomic_thresholds.shape != (atomic_targets.shape[1],):
        if atomic_thresholds.size and bool(np.allclose(atomic_thresholds, 0.5)):
            atomic_thresholds = np.full(atomic_targets.shape[1], 0.5)
        else:
            raise ValueError("Atomic thresholds must align with attribution targets")
    binary_pred = (binary_scores >= thresholds.binary).astype(int)
    atomic_pred = (atomic_scores >= atomic_thresholds.reshape(1, -1)).astype(int)

    gated_image_all_ious, gated_image_positive_ious = _gated_image_ious(
        bundle, thresholds.image_presence
    )
    image_metrics = _image_metrics_from_ious(gated_image_all_ious)
    ungated_image_metrics = _image_metrics_from_ious(bundle.image_all_ious)
    grounding_metrics = text_grounding_metrics(bundle)
    hypothesis_names = bundle.hypothesis_names or [
        f"hypothesis_{index}" for index in range(atomic_targets.shape[1])
    ]
    if len(hypothesis_names) != atomic_targets.shape[1]:
        raise ValueError("hypothesis_names must align with attribution targets")
    display_names = [
        {"FS": "face_swap", "FA": "face_attribute", "TS": "text_swap", "TA": "text_attribute"}.get(
            name, name.lower()
        )
        for name in hypothesis_names
    ]
    primary_attr = attribution_metrics(
        atomic_targets, atomic_scores, thresholds=atomic_thresholds
    )
    fixed_attr = attribution_metrics(
        atomic_targets, atomic_scores, thresholds=np.full(atomic_targets.shape[1], 0.5)
    )
    fixed_binary = binary_detection_metrics(binary_targets, binary_scores, threshold=0.5)
    image_availability = (
        bundle.image_grounding_available
        if len(bundle.image_grounding_available) == len(bundle.ids)
        else [True] * len(bundle.ids)
    )
    text_availability = (
        bundle.text_grounding_available
        if len(bundle.text_grounding_available) == len(bundle.ids)
        else [True] * len(bundle.ids)
    )
    consistency_metrics = unavailable_evidence_decision_metrics()
    if bundle.evidence_support_by_modality_scores:
        if len(bundle.evidence_support_by_modality_scores) != len(
            bundle.atomic_scores
        ):
            raise ValueError(
                "Per-modality evidence support must exist for every sample"
            )
        modality_names = tuple(bundle.evidence_support_modality_names)
        availability_by_name = {
            "image": image_availability,
            "text": text_availability,
        }
        try:
            modality_availability = np.stack(
                [availability_by_name[name] for name in modality_names], axis=1
            )
        except KeyError as exc:
            raise ValueError(f"Unknown evidence modality: {exc.args[0]!r}") from exc
        consistency_metrics = evidence_decision_metrics(
            atomic_scores,
            np.asarray(bundle.evidence_support_by_modality_scores, dtype=float),
            np.full(atomic_targets.shape[1], 0.5, dtype=float),
            modality_applicability=np.asarray(
                bundle.evidence_support_modality_mask, dtype=bool
            ),
            modality_availability=modality_availability,
            modality_names=modality_names,
        )
    elif bundle.evidence_support_scores:
        raise ValueError(
            "Current HERA evaluation requires explicit per-modality evidence support"
        )
    verified_metrics: dict[str, Any] = {
        "verified_support_metrics_available": bool(bundle.verified_support_eligible),
        "vsr_at_1_image": None, "vsr_at_1_text": None,
        "vsr_at_1_image_unit_count": None, "vsr_at_1_text_unit_count": None,
    }
    if bundle.verified_support_eligible:
        eligible = np.asarray(bundle.verified_support_eligible, dtype=bool)
        hits = np.asarray(bundle.verified_support_hits, dtype=bool)
        for index, name in enumerate(("image", "text")):
            count = int(eligible[:, :, index].sum())
            verified_metrics[f"vsr_at_1_{name}"] = float(hits[:, :, index].sum() / count) if count else None
            verified_metrics[f"vsr_at_1_{name}_unit_count"] = count
    metrics = {
        **protocol_metadata(
            decision_threshold_source=thresholds.decision_source,
            image_presence_source=thresholds.image_presence_source,
        ),
        **binary_detection_metrics(binary_targets, binary_scores, threshold=thresholds.binary),
        **primary_attr,
        **image_metrics,
        **grounding_metrics,
        **consistency_metrics,
        **verified_metrics,
        **{f"native_{key}": value for key, value in primary_attr.items()},
        **{f"{key}_fixed": value for key, value in fixed_attr.items() if key != "map"},
        **{f"native_{key}_fixed": value for key, value in fixed_attr.items()},
        "acc_fixed": fixed_binary["acc"],
        **{
            f"{key}_positive_only": value
            for key, value in _image_metrics_from_ious(gated_image_positive_ious).items()
        },
        **{
            f"{key}_ungated": value
            for key, value in ungated_image_metrics.items()
        },
        **_image_presence_metrics(bundle, thresholds.image_presence),
        "image_presence_threshold": thresholds.image_presence,
        "image_presence_threshold_source": thresholds.image_presence_source,
        "image_presence_aggregation": "atomic_any_fixed_0.5",
        "image_presence_hypotheses": ",".join(bundle.image_presence_hypothesis_names),
        "sample_count": len(bundle.ids),
        "image_sample_count": len(bundle.image_all_ious),
        "image_positive_sample_count": len(bundle.image_ious),
        "text_positive_sample_count": sum(
            manipulated and available
            for manipulated, available in zip(
                bundle.has_text_manipulation, text_availability
            )
        ),
        "image_grounding_unavailable_count": sum(
            not value for value in image_availability
        ),
        "text_grounding_unavailable_count": sum(
            not value for value in text_availability
        ),
        "text_token_count": bundle.text_token_count,
        "text_word_count": bundle.text_total_words,
        "text_visible_word_count": bundle.text_visible_words,
        "text_positive_word_count": bundle.text_total_positive_words,
        "text_unseen_positive_word_count": bundle.text_unseen_positive_words,
        "text_coverage_complete": bundle.text_coverage_complete,
        "text_original_word_metadata_complete": bundle.text_original_word_metadata_complete,
        "losses": bundle.mean_losses(),
    }
    samm_protocol = is_samm_reporting_ontology(bundle.fake_cls, hypothesis_names)
    samm_official_report = None
    samm_official_report_text = None
    if samm_protocol:
        metrics.update(
            samm_released_metrics(
                fake_cls=bundle.fake_cls,
                atomic_scores=atomic_scores,
                binary_scores=binary_scores,
                hypothesis_names=hypothesis_names,
            )
        )
        official_targets = samm_released_targets(bundle.fake_cls)
        official_scores = samm_released_scores(
            atomic_scores, binary_scores, hypothesis_names
        )
        official_pred = (official_scores >= 0.5).astype(int)
        official_names = list(SAMM_RELEASED_LABEL_ORDER)
        samm_official_report = attribution_report_dict(
            official_targets, official_pred, official_names
        )
        samm_official_report_text = attribution_classification_report_text(
            official_targets, official_pred, official_names
        )
        released_image_ious = _samm_released_all_sample_image_ious(
            bundle, thresholds.image_presence
        )
        metrics.update(
            {
                f"samm_released_all_sample_{key}": value
                for key, value in _image_metrics_from_ious(
                    released_image_ious
                ).items()
            }
        )
        metrics["samm_released_all_sample_image_count"] = len(
            released_image_ious
        )
    class_report = None
    valid_class_targets = bool(
        len(bundle.class_targets) == len(bundle.ids)
        and bundle.class_targets
        and min(bundle.class_targets) >= 0
        and len(bundle.class_names) == atomic_scores.shape[1] + 1
    )
    class_pred = None
    class_scores = None
    class_report_text = None
    if valid_class_targets:
        class_targets = np.asarray(bundle.class_targets, dtype=int)
        class_scores = exclusive_class_decision_scores(atomic_scores)
        class_pred = class_scores.argmax(axis=1)
        metrics["class_accuracy"] = float(accuracy_score(class_targets, class_pred))
        metrics["class_macro_f1"] = float(
            f1_score(class_targets, class_pred, average="macro", zero_division=0)
        )
        class_report = classification_report(
            class_targets, class_pred, labels=list(range(len(bundle.class_names))),
            target_names=bundle.class_names, digits=4, zero_division=0, output_dict=True,
        )
        class_report_text = classification_report(
            class_targets, class_pred, labels=list(range(len(bundle.class_names))),
            target_names=bundle.class_names, digits=4, zero_division=0,
        )
    return {
        "metrics": metrics,
        "binary_report": binary_classification_report_text(binary_targets, binary_pred),
        "attribution_report": attribution_classification_report_text(
            atomic_targets, atomic_pred, display_names
        ),
        "binary_report_dict": binary_report_dict(binary_targets, binary_pred),
        "attribution_report_dict": attribution_report_dict(
            atomic_targets, atomic_pred, display_names
        ),
        "hypothesis_names": hypothesis_names,
        "hypothesis_report_names": display_names,
        "class_report_dict": class_report,
        "class_report": class_report_text,
        "samm_official_report_dict": samm_official_report,
        "samm_official_report": samm_official_report_text,
        "class_scores": class_scores,
        "class_pred": class_pred,
        "binary_pred": binary_pred,
        "atomic_pred": atomic_pred,
    }


def text_grounding_metrics(bundle: PredictionBundle) -> dict[str, Any]:
    """Return text-localization metrics, including absence false positives."""

    token_metrics = _text_metrics_from_counts(
        bundle.text_token_tp, bundle.text_token_fp, bundle.text_token_fn
    )
    word_metrics = _text_metrics_from_counts(
        bundle.text_tp, bundle.text_fp, bundle.text_fn
    )
    positive_metrics = _text_metrics_from_counts(
        bundle.text_positive_tp,
        bundle.text_positive_fp,
        bundle.text_positive_fn,
    )
    return {
        **token_metrics,
        **{key.replace("text_", "text_word_"): value for key, value in word_metrics.items()},
        **{f"{key}_positive_only": value for key, value in positive_metrics.items()},
        "text_grounding_aggregation": bundle.text_grounding_aggregation,
        "text_evaluated_sample_count": bundle.text_evaluated_sample_count,
        "text_manipulated_evaluated_sample_count": (
            bundle.text_manipulated_evaluated_sample_count
        ),
        "text_absent_sample_count": bundle.text_absent_sample_count,
        "text_absent_sample_with_prediction_count": (
            bundle.text_absent_sample_with_prediction_count
        ),
        "text_absent_sample_false_positive_rate": _safe_ratio(
            bundle.text_absent_sample_with_prediction_count,
            bundle.text_absent_sample_count,
        ),
        "text_absent_token_count": bundle.text_absent_token_count,
        "text_absent_token_predicted_positive_count": (
            bundle.text_absent_token_predicted_positive_count
        ),
        "text_absent_token_false_positive_rate": _safe_ratio(
            bundle.text_absent_token_predicted_positive_count,
            bundle.text_absent_token_count,
        ),
        "text_absent_word_count": bundle.text_absent_word_count,
        "text_absent_word_predicted_positive_count": (
            bundle.text_absent_word_predicted_positive_count
        ),
        "text_absent_word_false_positive_rate": _safe_ratio(
            bundle.text_absent_word_predicted_positive_count,
            bundle.text_absent_word_count,
        ),
        "text_token_count": bundle.text_token_count,
        "text_word_count": bundle.text_total_words,
        "text_visible_word_count": bundle.text_visible_words,
        "text_positive_word_count": bundle.text_total_positive_words,
        "text_unseen_positive_word_count": bundle.text_unseen_positive_words,
        "text_coverage_complete": bundle.text_coverage_complete,
        "text_original_word_metadata_complete": bundle.text_original_word_metadata_complete,
    }


def exclusive_class_decision_scores(atomic_scores: np.ndarray) -> np.ndarray:
    """Build Real + mutually exclusive manipulation decision scores.

    SAMM annotations are exclusive while HERA remains multilabel internally.  The
    real score therefore follows Noisy-OR's complement and the manipulation
    scores remain the corresponding hypothesis probabilities.  These are
    decision scores, not a normalized probability simplex.
    """

    scores = np.asarray(atomic_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("atomic_scores must be [N,K] with K >= 1")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("atomic_scores must contain finite probabilities in [0,1]")
    real_scores = np.prod(1.0 - scores, axis=1, keepdims=True)
    return np.concatenate([real_scores, scores], axis=1)


def _image_metrics_from_ious(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "image_mean_iou": None,
            "image_iou50": None,
            "image_iou75": None,
            "image_iou95": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "image_mean_iou": float(arr.mean()),
        "image_iou50": float((arr > 0.5).mean()),
        "image_iou75": float((arr > 0.75).mean()),
        "image_iou95": float((arr > 0.95).mean()),
    }


def _gated_image_ious(
    bundle: PredictionBundle,
    threshold: float | None,
) -> tuple[list[float], list[float]]:
    """Apply target-free empty-box abstention to cached ungated IoUs."""

    if threshold is None:
        return list(bundle.image_all_ious), list(bundle.image_ious)
    if len(bundle.image_presence_scores) != len(bundle.ids):
        raise ValueError("Image presence scores must align with prediction IDs")
    if len(bundle.image_grounding_pred_ious) != len(bundle.ids):
        raise ValueError("Image grounding IoUs must align with prediction IDs")
    all_ious: list[float] = []
    positive_ious: list[float] = []
    for applies, available, score, ungated_iou in zip(
        bundle.has_image_manipulation,
        bundle.image_grounding_available,
        bundle.image_presence_scores,
        bundle.image_grounding_pred_ious,
    ):
        if not available or ungated_iou is None:
            continue
        predicts_presence = float(score) >= float(threshold)
        if applies:
            value = float(ungated_iou) if predicts_presence else 0.0
            positive_ious.append(value)
        else:
            # An abstention emits the official empty box [0,0,0,0].  If the
            # raw head was already empty, retain its valid empty-target match.
            value = float(ungated_iou) if predicts_presence else 1.0
        all_ious.append(value)
    return all_ious, positive_ious


def _samm_released_all_sample_image_ious(
    bundle: PredictionBundle,
    threshold: float | None,
) -> list[float]:
    """Score every SAMM row using the released empty-target convention.

    SAMM contains a small number of manipulated rows with no box annotation.
    The released loader converts those targets to ``[0,0,0,0]`` rather than
    dropping them, so official-compatible all-sample IoU must retain them.
    The presence gate remains target-free; annotation availability is consulted
    only afterwards to construct the evaluation target.
    """

    sample_count = len(bundle.ids)
    if not sample_count or len(bundle.image_grounding_pred_boxes) != sample_count:
        return []
    if len(bundle.image_presence_scores) != sample_count:
        raise ValueError("Image presence scores must align with SAMM prediction IDs")
    availability = (
        bundle.image_grounding_available
        if len(bundle.image_grounding_available) == sample_count
        else [True] * sample_count
    )
    if len(bundle.image_grounding_pred_ious) != sample_count:
        raise ValueError("Image grounding IoUs must align with SAMM prediction IDs")
    values: list[float] = []
    for index in range(sample_count):
        target_has_box = bool(
            bundle.has_image_manipulation[index] and availability[index]
        )
        predicts_presence = (
            True
            if threshold is None
            else bundle.image_presence_scores[index] >= float(threshold)
        )
        if target_has_box:
            raw_iou = bundle.image_grounding_pred_ious[index]
            if raw_iou is None:
                raise ValueError("Annotated SAMM image target is missing its cached IoU")
            values.append(float(raw_iou) if predicts_presence else 0.0)
            continue
        if not predicts_presence:
            values.append(1.0)
            continue
        raw_box = np.asarray(
            json.loads(bundle.image_grounding_pred_boxes[index]), dtype=float
        )
        values.append(float(bool((raw_box < 1e-4).all())))
    return values


def gated_image_iou_for_sample(
    bundle: PredictionBundle,
    index: int,
    threshold: float | None,
) -> float | None:
    """Return the per-sample IoU after ontology-aware empty-box abstention."""

    if index >= len(bundle.image_grounding_pred_ious):
        return None
    value = bundle.image_grounding_pred_ious[index]
    if value is None or threshold is None:
        return value
    if index >= len(bundle.image_presence_scores):
        raise ValueError("Image presence scores must align with grounding predictions")
    predicts_presence = bundle.image_presence_scores[index] >= float(threshold)
    if bundle.has_image_manipulation[index]:
        return float(value) if predicts_presence else 0.0
    return float(value) if predicts_presence else 1.0


def _image_presence_metrics(
    bundle: PredictionBundle,
    threshold: float | None,
) -> dict[str, float | None]:
    if threshold is None or not bundle.image_presence_scores:
        return {
            "image_presence_precision": None,
            "image_presence_recall": None,
            "image_presence_f1": None,
        }
    available = np.asarray(bundle.image_grounding_available, dtype=bool)
    targets = np.asarray(bundle.has_image_manipulation, dtype=int)[available]
    scores = np.asarray(bundle.image_presence_scores, dtype=float)[available]
    predictions = (scores >= float(threshold)).astype(int)
    tp = int(((targets == 1) & (predictions == 1)).sum())
    fp = int(((targets == 0) & (predictions == 1)).sum())
    fn = int(((targets == 1) & (predictions == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "image_presence_precision": precision,
        "image_presence_recall": recall,
        "image_presence_f1": f1,
    }


def _text_metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"text_precision": precision, "text_recall": recall, "text_f1": f1}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0

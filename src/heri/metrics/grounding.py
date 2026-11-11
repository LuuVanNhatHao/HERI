"""Image and text grounding metrics."""

from __future__ import annotations

import numpy as np
import torch

from heri.data.bbox import iou


def image_grounding_metrics(
    *,
    candidate_boxes: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_boxes: torch.Tensor,
    applies: torch.Tensor,
    logits: torch.Tensor,
) -> dict[str, float | None]:
    """Compute candidate-selected image grounding IoU metrics."""

    ious: list[float] = []
    scores = logits.detach().cpu()
    boxes = candidate_boxes.detach().cpu()
    masks = candidate_mask.detach().cpu().to(torch.bool)
    targets = target_boxes.detach().cpu()
    applicable = applies.detach().cpu().to(torch.bool)
    for idx in range(scores.shape[0]):
        if not bool(applicable[idx]):
            continue
        if not bool(masks[idx].any()):
            continue
        masked_scores = scores[idx].masked_fill(~masks[idx], -1e4)
        pred_idx = int(torch.argmax(masked_scores).item())
        ious.append(iou(boxes[idx, pred_idx].tolist(), targets[idx].tolist()))
    if not ious:
        return {"image_mean_iou": None, "image_iou50": None, "image_iou75": None}
    arr = np.asarray(ious, dtype=float)
    return {
        "image_mean_iou": float(arr.mean()),
        "image_iou50": float((arr >= 0.5).mean()),
        "image_iou75": float((arr >= 0.75).mean()),
    }


def text_grounding_metrics(
    *,
    targets: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    applies: torch.Tensor,
    word_ids: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Compute word-level precision/recall/F1 for DGM4 text grounding."""

    valid = mask.detach().cpu().to(torch.bool) & applies.detach().cpu().to(torch.bool).unsqueeze(1)
    target = targets.detach().cpu()
    valid = valid & (target >= 0)
    if not bool(valid.any()):
        return {"text_precision": None, "text_recall": None, "text_f1": None}
    probs = torch.sigmoid(logits.detach().cpu())
    ids = word_ids.detach().cpu() if word_ids is not None else _identity_word_ids(target)
    tp = fp = fn = 0
    for sample_idx in range(target.shape[0]):
        if not bool(applies.detach().cpu().to(torch.bool)[sample_idx]):
            continue
        sample_word_ids = sorted(
            int(word_id)
            for word_id in torch.unique(ids[sample_idx][valid[sample_idx]]).tolist()
            if int(word_id) >= 0
        )
        for word_id in sample_word_ids:
            word_mask = valid[sample_idx] & (ids[sample_idx] == word_id)
            y_true = int((target[sample_idx][word_mask] == 1).any().item())
            y_pred = int((probs[sample_idx][word_mask].max() >= threshold).item())
            tp += int(y_true == 1 and y_pred == 1)
            fp += int(y_true == 0 and y_pred == 1)
            fn += int(y_true == 1 and y_pred == 0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"text_precision": precision, "text_recall": recall, "text_f1": f1}


def _identity_word_ids(targets: torch.Tensor) -> torch.Tensor:
    return torch.arange(targets.shape[1], dtype=torch.long).unsqueeze(0).expand_as(targets)

"""Predicted-box image deletion with matched random rectangles; no GT selection."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .evaluator import move_batch_to_device
from .hypothesis_grounding import route_image_grounding_boxes
from .protocol import EVALUATION_PROTOCOL


def rectangle_masks(boxes, content_mask, *, generator, randomize=False):
    """Intersect prediction with content; random controls have identical shape/area.

    Content must be the rectangular resize-and-pad support from the data loader.
    Empty predicted rectangles remain empty in both conditions.
    """
    content = content_mask.detach().cpu().bool()
    if content.ndim == 4:
        content = content[:, 0]
    masks = torch.zeros_like(content)
    height, width = content.shape[-2:]
    for i, box in enumerate(boxes.detach().cpu()):
        ys, xs = content[i].nonzero(as_tuple=True)
        if not len(xs):
            continue
        left, right, top, bottom = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        if int(content[i].sum()) != (right - left) * (bottom - top):
            raise ValueError("Matched-rectangle deletion requires rectangular image content")
        x0 = min(right, max(left, math.floor(float(box[0]) * width)))
        y0 = min(bottom, max(top, math.floor(float(box[1]) * height)))
        x1 = min(right, max(left, math.ceil(float(box[2]) * width)))
        y1 = min(bottom, max(top, math.ceil(float(box[3]) * height)))
        w, h = max(0, x1 - x0), max(0, y1 - y0)
        if randomize and w and h:
            x0 = int(torch.randint(left, right - w + 1, (), generator=generator))
            y0 = int(torch.randint(top, bottom - h + 1, (), generator=generator))
        masks[i, y0:y0+h, x0:x0+w] = True
    return masks.to(content_mask.device)


def deleted_batch(batch, masks):
    image = batch["image"]
    content = batch["image_content_mask"]
    if content.ndim == 3:
        content = content.unsqueeze(1)
    fill = (image * content).sum((-1, -2), keepdim=True) / content.sum((-1, -2), keepdim=True).clamp_min(1)
    return {**batch, "image": torch.where(masks.unsqueeze(1), fill, image)}


@torch.no_grad()
def evaluate_image_deletion(*, model, loader, device, precision, path: Path, random_repeats=5, seed=2026, max_batches=None):
    if random_repeats < 1:
        raise ValueError("random_repeats must be positive")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for index, raw in enumerate(tqdm(loader, desc="predicted-box deletion")):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch_to_device(raw, device)
        with precision.autocast():
            clean = model(batch)
            boxes, _ = route_image_grounding_boxes(clean)
            selected = clean.atomic_probabilities[:, :2].argmax(-1)
            mask = rectangle_masks(boxes, batch["image_content_mask"], generator=generator)
            targeted = model(deleted_batch(batch, mask))
            random_outputs = []
            for _ in range(random_repeats):
                random_mask = rectangle_masks(boxes, batch["image_content_mask"], generator=generator, randomize=True)
                assert torch.equal(mask.sum((-1, -2)), random_mask.sum((-1, -2)))
                random_output = model(deleted_batch(batch, random_mask))
                random_outputs.append((random_output.atomic_probabilities, random_output.binary_probability))
                del random_output
        random_atomic = torch.stack([out[0] for out in random_outputs]).mean(0)
        random_binary = torch.stack([out[1] for out in random_outputs]).mean(0)
        for i, sample_id in enumerate(raw["id"]):
            h = int(selected[i])
            rows.append({"id": int(sample_id), "selected_visual_hypothesis": h,
                "deleted_pixels": int(mask[i].sum()), "content_pixels": int(batch["image_content_mask"][i].sum()),
                "clean_atomic": float(clean.atomic_probabilities[i, h]),
                "targeted_atomic_drop": float(clean.atomic_probabilities[i, h] - targeted.atomic_probabilities[i, h]),
                "random_atomic_drop": float(clean.atomic_probabilities[i, h] - random_atomic[i, h]),
                "targeted_binary_drop": float(clean.binary_probability[i] - targeted.binary_probability[i]),
                "random_binary_drop": float(clean.binary_probability[i] - random_binary[i])})
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"evaluation_protocol": EVALUATION_PROTOCOL, "selection": "predicted_visual_hypothesis_and_predicted_box_no_ground_truth",
              "fill": "per_image_content_channel_mean", "random_repeats": random_repeats, "seed": seed,
              "max_batches": max_batches, "sample_count": len(rows), "rows": rows,
              "interpretation": "input perturbation sensitivity; replacement distribution shift and box size limit causal interpretation"}
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result

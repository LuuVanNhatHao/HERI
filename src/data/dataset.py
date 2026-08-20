"""DGM4 Dataset implementation for Phase 1."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .bbox import validate_box
from .constants import ABSENT_BOX
from .labels import (
    atomic_target,
    binary_target,
    has_image_manipulation,
    has_text_manipulation,
)
from .paths import resolve_image_path
from .schema import validate_row, validate_text_positions
from .tokenizer_alignment import align_word_positions_to_subwords
from .transforms import (
    image_content_mask,
    preprocess_image,
    transform_box_with_meta,
)


def load_annotations(path: str | Path, *, split: str) -> list[dict[str, Any]]:
    """Load DGM4 annotations from JSON."""

    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"DGM4 {split} annotation file must contain a list: {path}")
    return data


def select_smoke_rows(
    rows: Sequence[dict[str, Any]],
    *,
    max_samples: int | None = None,
    samples_per_class: int | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic small subset for smoke inspection."""

    if samples_per_class is None:
        return list(rows[:max_samples]) if max_samples is not None else list(rows)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        label = row["fake_cls"]
        if counts[label] < samples_per_class:
            selected.append(row)
            counts[label] += 1
        if max_samples is not None and len(selected) >= max_samples:
            break
    return selected


class DGM4Dataset(Dataset[dict[str, Any]]):
    """Dataset exposing the frozen DGM4 data contract as structured dicts."""

    def __init__(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        split: str,
        dataset_root: str | Path,
        image_size: int,
        image_root: str | Path | None = None,
        tokenizer: Any | None = None,
        tokenizer_max_length: int | None = None,
        tokenizer_ignore_index: int = -100,
        verify_schema: bool = False,
        verify_image_exists: bool = True,
        progress: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.image_root = Path(image_root) if image_root is not None else self.dataset_root / "images"
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_ignore_index = tokenizer_ignore_index
        self.verify_image_exists = verify_image_exists
        if verify_schema:
            iterator = tqdm(
                enumerate(self.rows),
                total=len(self.rows),
                desc=f"DGM4 {split} dataset schema",
                disable=not progress,
            )
            for index, row in iterator:
                validate_row(row, split=split, row_index=index)

    @classmethod
    def from_annotation_file(
        cls,
        *,
        annotation_path: str | Path,
        split: str,
        dataset_root: str | Path,
        image_size: int,
        image_root: str | Path | None = None,
        tokenizer: Any | None = None,
        tokenizer_max_length: int | None = None,
        tokenizer_ignore_index: int = -100,
        verify_schema: bool = False,
        verify_image_exists: bool = True,
        progress: bool = False,
        max_samples: int | None = None,
        samples_per_class: int | None = None,
    ) -> "DGM4Dataset":
        """Build a dataset from a DGM4 JSON annotation file."""

        rows = load_annotations(annotation_path, split=split)
        rows = select_smoke_rows(
            rows,
            max_samples=max_samples,
            samples_per_class=samples_per_class,
        )
        return cls(
            rows=rows,
            split=split,
            dataset_root=dataset_root,
            image_root=image_root,
            image_size=image_size,
            tokenizer=tokenizer,
            tokenizer_max_length=tokenizer_max_length,
            tokenizer_ignore_index=tokenizer_ignore_index,
            verify_schema=verify_schema,
            verify_image_exists=verify_image_exists,
            progress=progress,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        validate_row(row, split=self.split, row_index=index)
        fake_cls = row["fake_cls"]
        resolved = resolve_image_path(
            row["image"],
            self.dataset_root,
            image_root=self.image_root,
            verify_exists=self.verify_image_exists,
        )
        with Image.open(resolved.physical_path) as image:
            original_size = image.size
            image_tensor, transform_meta = preprocess_image(
                image,
                target_size=self.image_size,
            )

        image_size = (original_size[0], original_size[1])
        image_manipulated = has_image_manipulation(fake_cls)
        fake_image_box_target = torch.tensor(ABSENT_BOX, dtype=torch.float32)
        if image_manipulated:
            fake_image_box_original = validate_box(
                row["fake_image_box"],
                image_size=image_size,
                field_name="fake_image_box",
            )
            transformed_box = transform_box_with_meta(fake_image_box_original, transform_meta)
            fake_image_box_target = torch.tensor(transformed_box, dtype=torch.float32)

        text = row["text"]
        words = text.split()
        fake_text_pos = list(row["fake_text_pos"])
        validate_text_positions(
            words,
            fake_text_pos,
            context=f"split={self.split} index={index} id={row['id']}",
        )
        text_alignment = align_word_positions_to_subwords(
            words=words,
            fake_text_pos=fake_text_pos,
            tokenizer=self.tokenizer,
            max_length=self.tokenizer_max_length,
            ignore_index=self.tokenizer_ignore_index,
        )

        item: dict[str, Any] = {
            "id": row["id"],
            "split": self.split,
            "image": image_tensor,
            "image_content_mask": image_content_mask(transform_meta),
            "image_path": str(resolved.physical_path),
            "annotation_image_path": row["image"],
            "image_transform_meta": transform_meta.to_dict(),
            "text": text,
            "words": words,
            "binary_target": torch.tensor(binary_target(fake_cls), dtype=torch.float32),
            "atomic_target": torch.tensor(atomic_target(fake_cls), dtype=torch.float32),
            "fake_image_box_target": fake_image_box_target,
            "has_image_manipulation": torch.tensor(image_manipulated, dtype=torch.bool),
            "fake_text_pos": torch.tensor(fake_text_pos, dtype=torch.long),
            "text_grounding_target": torch.tensor(text_alignment.labels, dtype=torch.long),
            "text_grounding_mask": torch.tensor(text_alignment.mask, dtype=torch.bool),
            "text_word_ids": torch.tensor(
                [-1 if word_id is None else int(word_id) for word_id in text_alignment.word_ids],
                dtype=torch.long,
            ),
            "has_text_manipulation": torch.tensor(
                has_text_manipulation(fake_cls),
                dtype=torch.bool,
            ),
            "fake_cls": fake_cls,
            "generator": resolved.generator,
            "image_source": resolved.source,
        }

        encoding = text_alignment.encoding
        if encoding is not None and "input_ids" in encoding:
            item["text_input_ids"] = torch.tensor(encoding["input_ids"], dtype=torch.long)
            item["text_attention_mask"] = torch.tensor(
                encoding.get("attention_mask", [1] * len(encoding["input_ids"])),
                dtype=torch.bool,
            )
        else:
            item["text_input_ids"] = torch.arange(len(words), dtype=torch.long)
            item["text_attention_mask"] = torch.ones(len(words), dtype=torch.bool)
        return item


def group_indices_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    """Return row indices grouped by sample ID."""

    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["id"])].append(index)
    return groups

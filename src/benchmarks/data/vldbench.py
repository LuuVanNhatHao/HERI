"""Memory-mapped VLDBench Parquet adapter."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from src.benchmarks.contracts import binary_label_from_vldbench

from .base import BinaryBenchmarkDataset, BinaryBenchmarkRecord


class VLDBenchDataset(BinaryBenchmarkDataset):
    """VLDBench adapter using a deterministic external split manifest."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        split: str,
        tokenizer: Any,
    ) -> None:
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            tokenizer_max_length=int(config["tokenizer"]["max_length"]),
            image_size=int(config["image"]["size"]),
        )
        dataset = config["dataset"]
        self.parquet_path = Path(str(dataset["parquet"]))
        manifest_path = Path(str(dataset["split_manifest"]["path"]))
        if not self.parquet_path.is_file():
            raise FileNotFoundError(
                f"VLDBench parquet does not exist: {self.parquet_path}"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"VLDBench split manifest does not exist: {manifest_path}. Run the prepare script first."
            )
        manifest = pd.read_csv(manifest_path)
        required_manifest = {
            "unique_id",
            "source_row",
            "split",
            "image_sha256",
            "multimodal_label",
            "text_label",
        }
        missing_manifest = sorted(required_manifest - set(manifest.columns))
        if missing_manifest:
            raise ValueError(
                f"VLDBench manifest is missing columns: {missing_manifest}"
            )
        selected = manifest[manifest["split"].eq(split)].copy()
        if selected.empty:
            raise ValueError(f"VLDBench manifest contains no rows for split={split!r}")
        self.source_rows = selected["source_row"].astype(int).tolist()
        self.manifest_ids = selected["unique_id"].astype(str).tolist()
        self.manifest_multimodal_labels = (
            selected["multimodal_label"].astype(str).tolist()
        )
        self.manifest_text_labels = selected["text_label"].astype(str).tolist()
        self.table = pq.read_table(self.parquet_path, memory_map=True)
        if max(self.source_rows) >= len(self.table) or min(self.source_rows) < 0:
            raise ValueError(
                "VLDBench manifest source_row falls outside the Parquet table"
            )
        self.text_fields = tuple(str(value) for value in dataset["text"]["fields"])
        self.separator = str(dataset["text"]["separator"])
        unknown = sorted(set(self.text_fields) - set(self.table.column_names))
        if unknown:
            raise ValueError(f"Unknown VLDBench text fields: {unknown}")

    def __len__(self) -> int:
        return len(self.source_rows)

    def raw_record(self, index: int) -> BinaryBenchmarkRecord:
        source_row = self.source_rows[int(index)]
        row = {
            name: self.table[name][source_row].as_py()
            for name in (
                "unique_id",
                "image",
                "multimodal_label",
                "text_label",
                "news_category",
                "outlet",
                *self.text_fields,
            )
        }
        sample_id = str(row["unique_id"])
        if sample_id != self.manifest_ids[int(index)]:
            raise ValueError(
                f"VLDBench manifest/Parquet mismatch at source_row={source_row}: "
                f"manifest={self.manifest_ids[int(index)]!r}, parquet={sample_id!r}"
            )
        expected_multimodal = self.manifest_multimodal_labels[int(index)]
        expected_text = self.manifest_text_labels[int(index)]
        if str(row["multimodal_label"]) != expected_multimodal:
            raise ValueError(
                f"VLDBench multimodal label changed at source_row={source_row}: "
                f"manifest={expected_multimodal!r}, parquet={row['multimodal_label']!r}"
            )
        if str(row["text_label"]) != expected_text:
            raise ValueError(
                f"VLDBench text label changed at source_row={source_row}: "
                f"manifest={expected_text!r}, parquet={row['text_label']!r}"
            )
        image_object = row["image"]
        image_bytes = None if image_object is None else image_object.get("bytes")
        if not image_bytes:
            raise ValueError(f"VLDBench image bytes are missing: id={sample_id}")
        text_values = [row[field] for field in self.text_fields]
        if any(value is None or not str(value).strip() for value in text_values):
            raise ValueError(f"VLDBench has empty configured text: id={sample_id}")
        text = self.separator.join(str(value).strip() for value in text_values)
        return BinaryBenchmarkRecord(
            sample_id=sample_id,
            text=text,
            image=Image.open(BytesIO(image_bytes)),
            image_reference=f"{self.parquet_path.as_posix()}#row={source_row}",
            binary_target=binary_label_from_vldbench(row["multimodal_label"]),
            text_label_target=binary_label_from_vldbench(row["text_label"]),
            text_label_available=True,
            metadata={
                "category": str(row["news_category"]),
                "outlet": str(row["outlet"]),
                "source_row": source_row,
            },
        )

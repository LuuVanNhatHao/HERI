from pathlib import Path

from conftest import write_split
from src.data.preflight import run_preflight


def test_preflight_small_fixture(tmp_path: Path, tiny_dgm4_root: Path, sample_rows: list[dict]) -> None:
    train = tiny_dgm4_root / "train.json"
    val = tiny_dgm4_root / "val.json"
    test = tiny_dgm4_root / "test.json"
    write_split(train, sample_rows[:2])
    write_split(val, sample_rows[2:3])
    write_split(test, sample_rows[3:])
    config = {
        "project": {"name": "HERA-DGM", "seed": 42},
        "dataset": {
            "root": str(tiny_dgm4_root),
            "train_annotations": str(train),
            "val_annotations": str(val),
            "test_annotations": str(test),
            "image_root": str(tiny_dgm4_root / "images"),
            "verify_hashes": False,
        },
        "preflight": {
            "verify_schema": True,
            "verify_paths": True,
            "verify_image_headers": True,
            "verify_full_decode": False,
            "verify_boxes": True,
            "verify_mtcnn_match": True,
            "verify_text_positions": True,
            "verify_split_counts": False,
            "write_manifest": True,
            "worker_threads": 2,
        },
        "output": {"root": str(tmp_path / "outputs")},
        "progress": {"enabled": False},
    }
    audit = run_preflight(config)
    assert audit["overall_status"] == "PASS_WITH_DISABLED_CHECKS"
    assert audit["path_validation"]["errors"] == 0
    assert audit["bbox_validation"]["bbox_errors"] == 0
    assert audit["mtcnn_matching"]["errors"] == 0

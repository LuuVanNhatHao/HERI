from __future__ import annotations

from heri.training.validation_reporting import emit_validation_classification_reports


def test_emit_validation_classification_reports_includes_samm_classes(tmp_path, capsys) -> None:
    reports = {
        "binary_report": "binary report body",
        "attribution_report": "attribution report body",
        "class_report": ("orig\nswap_manipulation\nattribute_manipulation"),
    }

    written = emit_validation_classification_reports(
        reports,
        epoch=3,
        metrics_dir=tmp_path,
    )

    assert set(written) == {"binary", "attribution", "dataset_class"}
    assert written["binary"].name == "epoch_003_binary_classification_report.txt"
    assert written["attribution"].name == ("epoch_003_attribution_classification_report.txt")
    assert written["dataset_class"].name == ("epoch_003_dataset_class_classification_report.txt")
    assert written["dataset_class"].read_text(encoding="utf-8") == (
        "orig\nswap_manipulation\nattribute_manipulation\n"
    )
    output = capsys.readouterr().out
    assert "Validation binary classification_report:" in output
    assert "Validation attribution classification_report:" in output
    assert "Validation dataset-class classification_report:" in output
    assert "attribute_manipulation" in output


def test_emit_validation_classification_reports_omits_unavailable_dataset_class(
    tmp_path, capsys
) -> None:
    written = emit_validation_classification_reports(
        {
            "binary_report": "binary report body",
            "attribution_report": "attribution report body",
            "class_report": None,
        },
        epoch=1,
        metrics_dir=tmp_path,
    )

    assert set(written) == {"binary", "attribution"}
    assert not (tmp_path / "epoch_001_dataset_class_classification_report.txt").exists()
    assert "dataset-class" not in capsys.readouterr().out

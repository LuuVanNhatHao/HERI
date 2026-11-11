"""Console and artifact reporting for per-epoch validation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tqdm.auto import tqdm


_VALIDATION_REPORTS = (
    (
        "binary_report",
        "binary",
        "Validation binary classification_report:",
    ),
    (
        "attribution_report",
        "attribution",
        "Validation attribution classification_report:",
    ),
    (
        "class_report",
        "dataset_class",
        "Validation dataset-class classification_report:",
    ),
    (
        "samm_official_report",
        "samm_official_multilabel",
        "Validation SAMM released-protocol three-label classification_report:",
    ),
)


def emit_validation_classification_reports(
    reports: Mapping[str, Any],
    *,
    epoch: int,
    metrics_dir: str | Path,
) -> dict[str, Path]:
    """Print and persist every available sklearn validation report.

    ``class_report`` is dataset-dependent.  In particular, SAMM supplies the
    mutually exclusive ``orig``, ``swap_manipulation``, and
    ``attribute_manipulation`` targets, while DGM4 does not expose an
    equivalent exclusive class target through this evaluation path.
    """

    destination = Path(metrics_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for report_key, artifact_name, heading in _VALIDATION_REPORTS:
        report = reports.get(report_key)
        if not isinstance(report, str) or not report.strip():
            continue
        rendered = report.rstrip() + "\n"
        tqdm.write(heading)
        tqdm.write(rendered.rstrip())
        path = destination / (
            f"epoch_{int(epoch):03d}_{artifact_name}_classification_report.txt"
        )
        path.write_text(rendered, encoding="utf-8")
        written[artifact_name] = path
    return written

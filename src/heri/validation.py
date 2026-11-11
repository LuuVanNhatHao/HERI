"""Small offline checks for paper equations, runtime identity and data flow."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET


SELECTIONS = {
    "equations": [
        "contracts/test_paper_graph.py",
        "contracts/test_grounding_formulas.py",
        "unit/models/test_noisy_or.py::test_noisy_or_extreme_logits_have_finite_gradients",
        "unit/losses/test_heri_components.py::test_binary_loss_does_not_push_absent_hypotheses_up_on_fake_rows",
        "integration/test_samm.py::test_samm_per_modality_evidence_loss_exposes_weak_text_support",
        "integration/test_samm.py::test_samm_per_modality_evidence_loss_masks_unavailable_annotations",
        "integration/test_samm.py::test_samm_per_modality_evidence_loss_empty_domain_is_differentiable_zero",
    ],
    "runtime": ["contracts/test_runtime.py"],
    "integration": ["integration/test_end_to_end.py"],
    "config": [
        "contracts/test_paper_experiments.py",
        "contracts/test_repository_layout.py",
        "contracts/test_cli.py",
    ],
}


def run_checks(root: Path, *, section: str = "all") -> int:
    if not (root / "tests").is_dir():
        raise FileNotFoundError("Run validation from a HERI source checkout containing tests/")
    selections = (
        [name for group in SELECTIONS.values() for name in group]
        if section == "all"
        else SELECTIONS[section]
    )
    report_dir = root / "artifacts" / "validation" / "latest"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "results.xml"
    settings = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    saved = {key: os.environ.get(key) for key in (*settings, "TEMP", "TMP")}
    saved_tempdir = tempfile.tempdir
    sys.dont_write_bytecode = True
    os.environ.update(settings)
    try:
        # Scratch stays on the repository drive and is removed on success or failure.
        with tempfile.TemporaryDirectory(prefix=".work-", dir=report_dir.parent) as work:
            system_tmp = Path(work) / "system"
            system_tmp.mkdir()
            os.environ["TEMP"] = os.environ["TMP"] = str(system_tmp)
            tempfile.tempdir = str(system_tmp)
            import torch
            import pytest

            torch.set_num_threads(1)
            print(
                "HERI checks: offline CPU, synthetic samples, no pretrained downloads.", flush=True
            )
            result = int(
                pytest.main(
                    [
                        "-q",
                        "--tb=short",
                        "-p",
                        "no:cacheprovider",
                        "-o",
                        "addopts=",
                        "--basetemp",
                        str(Path(work) / "pytest"),
                        "--junitxml",
                        str(report),
                        *[str(root / "tests" / name) for name in selections],
                    ]
                )
            )
        summary = {
            "section": section,
            "exit_code": result,
            "device": "cpu",
            "pretrained_weights": False,
        }
        if report.is_file():
            suite = ET.parse(report).getroot().find("testsuite")
            if suite is not None:
                summary.update(
                    {
                        key: suite.get(key)
                        for key in ("tests", "failures", "errors", "skipped", "time")
                    }
                )
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Validation reports: {report_dir}", flush=True)
        return result
    finally:
        tempfile.tempdir = saved_tempdir
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

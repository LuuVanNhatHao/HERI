from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from heri.evaluation.routing_analysis import (
    ROUTER_FAMILIES,
    compare_evidence_routing,
    load_routing_predictions,
)


def _predictions(path: Path, *, shift: float) -> None:
    rows = []
    classes = (("orig", 0), ("swap", 1), ("attribute", 1), ("orig", 0))
    for index, (class_name, target) in enumerate(classes):
        values = {
            family: min(0.99, 0.30 + shift + 0.08 * family_index + 0.01 * index)
            for family_index, family in enumerate(ROUTER_FAMILIES)
        }
        rows.append(
            {
                "id": index,
                "binary_target": target,
                "class_name": class_name,
                "fake_cls": class_name,
                "router_diagnostics": json.dumps(values),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_routing_comparison_writes_stratified_scientific_artifacts(tmp_path: Path) -> None:
    dgm4 = tmp_path / "dgm4.csv"
    samm = tmp_path / "samm.csv"
    _predictions(dgm4, shift=0.0)
    _predictions(samm, shift=0.1)

    report = compare_evidence_routing(
        inputs={"DGM4": dgm4, "SAMM": samm},
        output_dir=tmp_path / "analysis",
        reference="DGM4",
        comparison="SAMM",
        bootstrap_samples=100,
        seed=777,
    )

    contrast = pd.read_csv(report["artifacts"]["contrast"])
    grouped = pd.read_csv(report["artifacts"]["by_dataset_and_class"])
    assert len(contrast) == 3 * len(ROUTER_FAMILIES)
    assert set(contrast["scope"]) == {"all", "real", "fake"}
    assert (contrast["reliability_delta"] > 0).all()
    assert "class:swap" in set(grouped["scope"])
    assert set(report["overall_family_rankings"]) == {"DGM4", "SAMM"}
    assert Path(report["manifest"]).is_file()


def test_routing_loader_rejects_incomplete_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        [{"id": 1, "binary_target": 0, "router_diagnostics": '{"visual_semantic": 0.5}'}]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks families"):
        load_routing_predictions(path, dataset_name="bad")


def test_routing_loader_rejects_uniform_bypass(tmp_path: Path) -> None:
    path = tmp_path / "bypass.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "binary_target": 0,
                "router_diagnostics": json.dumps({family: 1.0 for family in ROUTER_FAMILIES}),
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="uniform bypass"):
        load_routing_predictions(path, dataset_name="bypass")

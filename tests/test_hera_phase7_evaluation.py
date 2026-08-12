from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from decision_fixtures import synthetic_graph_output
from phase2_fixtures import synthetic_batch
from src.evaluation.evaluator import compute_metrics_and_reports, run_inference
from src.evaluation.summary import write_predictions_csv, write_summary_csv
from src.models.hera import HERADecisionModule
from src.thresholding.tuner import DEFAULT_THRESHOLDS
from src.training.precision import PrecisionManager


class StaticHERAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        output = HERADecisionModule(
            {
                "model": {
                    "hidden_dim": 12,
                    "hypotheses": {"order": ["FS", "FA", "TS", "TA"]},
                    "attribution": {"hidden_dim": 12, "dropout": 0.0},
                    "grounding": {
                        "image": {"scoring": "dot_product", "dropout": 0.0},
                        "text": {
                            "scoring": "dot_product",
                            "dropout": 0.0,
                            "full_token_store": True,
                            "token_input_dim": 19,
                        },
                    },
                    "binary": {"aggregation": "noisy_or"},
                }
            }
        )(synthetic_graph_output())
        output.metadata["router_family_mean_reliability"] = {
            "face": [0.9, 0.8, 0.7],
            "text_context": [0.6, 0.5, 0.4],
        }
        self.output = output

    def forward(self, batch):
        return self.output


def test_hera_prediction_export_contains_phase8_9_10_metadata(tmp_path: Path) -> None:
    batch = synthetic_batch()
    bundle = run_inference(
        model=StaticHERAModel(),
        loader=[batch],
        loss_fn=None,
        device=torch.device("cpu"),
        precision=PrecisionManager(precision="fp32", device=torch.device("cpu")),
        max_batches=None,
        desc="fixture inference",
        progress=False,
    )
    reports = compute_metrics_and_reports(bundle, DEFAULT_THRESHOLDS)
    predictions = tmp_path / "predictions.csv"
    summary = tmp_path / "summary.csv"
    write_predictions_csv(bundle=bundle, thresholds=DEFAULT_THRESHOLDS, path=predictions)
    write_summary_csv(
        path=summary,
        run_info={"run_id": "fixture", "model_name": "HERA-DGM", "seed": 42},
        metrics=reports["metrics"],
        thresholds=DEFAULT_THRESHOLDS,
        reports=reports,
    )

    prediction_df = pd.read_csv(predictions)
    summary_df = pd.read_csv(summary)
    for column in (
        "generator",
        "image_source",
        "annotation_image_path",
        "face_size_bucket",
        "image_grounding_iou",
        "text_grounding_target_word_ids",
        "text_grounding_pred_word_ids",
        "router_diagnostics",
    ):
        assert column in prediction_df.columns
    assert "cf1" in summary_df.columns
    assert "real_f1" in summary_df.columns

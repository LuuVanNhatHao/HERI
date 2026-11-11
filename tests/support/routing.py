from __future__ import annotations

import torch

from heri.evidence.outputs import (
    CrossModalMetadata,
    EvidenceSet,
    HeterogeneousEvidenceBatch,
    PAD_SOURCE_INDEX,
    PAD_WORD_ID,
)


def routing_config(
    hidden_dim: int = 12, *, enabled: bool = True, mode: str = "soft", initial_bias: float = 1.0
) -> dict:
    return {
        "routing": {
            "enabled": enabled,
            "mode": mode,
            "feature_dim": hidden_dim,
            "estimator": {
                "hidden_dim": max(4, hidden_dim),
                "dropout": 0.0,
                "initial_bias": initial_bias,
                "output_weight_std": 0.001,
            },
            "visual_semantic": {"enabled": True},
            "visual_forensic": {"enabled": True},
            "text_alignment": {"enabled": True},
            "text_context": {"enabled": True},
            "hard_eval": {"enabled": mode == "hard", "threshold": 0.5},
            "diagnostics": {"low_threshold": 0.25, "high_threshold": 0.75},
            "regularization": {"enabled": False, "weight": 0.0},
        }
    }


def minimal_evidence(hidden_dim: int = 2) -> HeterogeneousEvidenceBatch:
    return HeterogeneousEvidenceBatch(
        visual_semantic=_evidence_set("visual_semantic", hidden_dim=hidden_dim),
        visual_forensic=_evidence_set("visual_forensic", hidden_dim=hidden_dim),
        text_alignment=_evidence_set("text_alignment", hidden_dim=hidden_dim, words=True),
        text_context=_text_context_set(hidden_dim=hidden_dim),
        cross_modal_metadata=CrossModalMetadata(method="disabled"),
    )


def _evidence_set(
    family: str,
    *,
    hidden_dim: int,
    spatial: bool = False,
    words: bool = False,
) -> EvidenceSet:
    features = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], dtype=torch.float32)
    if hidden_dim != 2:
        features = torch.randn(1, 2, hidden_dim)
    mask = torch.tensor([[True, True]])
    source_indices = torch.tensor([[0, 1]], dtype=torch.long)
    kwargs = {}
    if spatial:
        boxes = torch.tensor([[[0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 1.0, 1.0]]])
        kwargs.update(
            spatial_boxes=boxes * 384.0,
            normalized_boxes=boxes,
            spatial_centers=torch.tensor([[[76.8, 76.8], [288.0, 288.0]]]),
            normalized_centers=torch.tensor([[[0.2, 0.2], [0.75, 0.75]]]),
        )
    if words:
        kwargs["word_ids"] = torch.tensor([[0, 1]], dtype=torch.long)
    return EvidenceSet(
        family=family,
        features=features,
        mask=mask,
        source_indices=source_indices,
        provenance={"source": family, "fake_text_pos_used": False},
        **kwargs,
    )


def _text_context_set(hidden_dim: int) -> EvidenceSet:
    features = torch.randn(1, 2, hidden_dim)
    mask = torch.tensor([[True, True]])
    source_indices = torch.tensor([[0, 1]], dtype=torch.long)
    return EvidenceSet(
        family="text_context",
        features=features,
        mask=mask,
        source_indices=source_indices,
        word_ids=torch.tensor([[0, 1]], dtype=torch.long),
        full_features=torch.randn(1, 5, 19),
        full_mask=torch.tensor([[True, True, True, False, False]]),
        full_source_indices=torch.tensor(
            [[0, 1, 2, PAD_SOURCE_INDEX, PAD_SOURCE_INDEX]], dtype=torch.long
        ),
        full_word_ids=torch.tensor([[0, 1, 2, PAD_WORD_ID, PAD_WORD_ID]], dtype=torch.long),
        provenance={"source": "text_context", "fake_text_pos_used": False},
    )

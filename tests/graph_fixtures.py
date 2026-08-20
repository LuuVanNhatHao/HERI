from __future__ import annotations

import torch

from evidence_fixtures import evidence_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from src.evidence.outputs import HeterogeneousEvidenceBatch


def graph_config(
    hidden_dim: int = 12,
    *,
    cycles: int = 1,
    feedback: bool = True,
    dynamic_rebuild: bool = False,
) -> dict:
    return {
        "type": "hgt",
        "hidden_dim": hidden_dim,
        "layers": 1,
        "heads": 3,
        "dropout": 0.0,
        "hypothesis_init_std": 0.02,
        "input_dims": {},
        "reasoning": {
            "cycles": cycles,
            "hypothesis_feedback": feedback,
            "dynamic_rebuild": dynamic_rebuild,
            "hypothesis_edge_reweighting": "cosine",
        },
        "edges": {
            "visual_semantic_visual_forensic": {
                "method": "spatial_knn",
                "k": 2,
                "bidirectional": True,
            },
            "visual_semantic_text_alignment": {
                "method": "similarity_topk",
                "k": 2,
                "bidirectional": True,
            },
            "text_alignment_text_context": {
                "method": "word_aligned",
                "bidirectional": True,
            },
            "semantic_to_hypotheses": {"enabled": True},
            "feedback": {"enabled": feedback, "semantic_feedback": False},
        },
    }


def synthetic_evidence(hidden_dim: int = 12) -> HeterogeneousEvidenceBatch:
    torch.manual_seed(17)
    extractor = HeterogeneousEvidenceExtractor(evidence_config(hidden_dim=hidden_dim))
    return extractor(synthetic_features(), EvidenceStructuralMetadata.from_batch(synthetic_batch()))

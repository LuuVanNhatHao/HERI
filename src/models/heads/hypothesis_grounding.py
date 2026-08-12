"""Hypothesis-conditioned grounding heads for canonical HERA decisions."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.evidence.outputs import EvidenceSet, PAD_WORD_ID
from src.graph.schema import HYPOTHESIS_ORDER


IMAGE_HYPOTHESIS_INDICES = (0, 1)
TEXT_HYPOTHESIS_INDICES = (2, 3)


class HypothesisConditionedImageGroundingHead(nn.Module):
    """Score every legal MTCNN face candidate conditioned on H_FS and H_FA."""

    def __init__(self, *, hidden_dim: int, dropout: float = 0.0, scoring: str = "dot_product") -> None:
        super().__init__()
        if scoring != "dot_product":
            raise ValueError("Phase-5 image grounding currently supports scoring='dot_product'")
        self.hidden_dim = int(hidden_dim)
        self.face_key = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.hypothesis_queries = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in IMAGE_HYPOTHESIS_INDICES
            ]
        )

    def forward(
        self,
        *,
        hypothesis_features: torch.Tensor,
        face_evidence: EvidenceSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits/probabilities [B,2,N_face] and mask [B,N_face]."""

        _validate_hypotheses(hypothesis_features, self.hidden_dim)
        face_evidence.validate()
        face_features = face_evidence.features
        if face_features.shape[0] != hypothesis_features.shape[0]:
            raise ValueError("face_evidence batch size must match hypothesis_features")
        if face_features.shape[-1] != self.hidden_dim:
            raise ValueError("face_evidence hidden dimension mismatch")
        mask = face_evidence.mask.to(torch.bool)
        keys = self.face_key(face_features)
        logits = []
        scale = math.sqrt(float(self.hidden_dim))
        for local_head_index, hypothesis_index in enumerate(IMAGE_HYPOTHESIS_INDICES):
            query = self.hypothesis_queries[local_head_index](hypothesis_features[:, hypothesis_index, :])
            score = torch.sum(keys * query.unsqueeze(1), dim=-1) / scale
            logits.append(score)
        stacked_logits = torch.stack(logits, dim=1).masked_fill(~mask.unsqueeze(1), -1e4)
        probabilities = _masked_softmax(stacked_logits, mask.unsqueeze(1))
        return stacked_logits, probabilities, mask


class HypothesisConditionedTextGroundingHead(nn.Module):
    """Score the full DeBERTa token store conditioned on H_TS and H_TA."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_input_dim: int,
        dropout: float = 0.0,
        scoring: str = "dot_product",
    ) -> None:
        super().__init__()
        if scoring != "dot_product":
            raise ValueError("Phase-5 text grounding currently supports scoring='dot_product'")
        if int(token_input_dim) <= 0:
            raise ValueError("token_input_dim must be positive and YAML-provided")
        self.hidden_dim = int(hidden_dim)
        self.token_input_dim = int(token_input_dim)
        self.token_key = nn.Sequential(
            nn.LayerNorm(self.token_input_dim),
            nn.Linear(self.token_input_dim, self.hidden_dim),
        )
        self.hypothesis_queries = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in TEXT_HYPOTHESIS_INDICES
            ]
        )

    def forward(
        self,
        *,
        hypothesis_features: torch.Tensor,
        text_context: EvidenceSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full-token logits/probabilities [B,2,N_full_text] and mask."""

        _validate_hypotheses(hypothesis_features, self.hidden_dim)
        text_context.validate()
        if text_context.full_features is None:
            raise ValueError("text_context.full_features is mandatory for Phase-5 full-text grounding")
        if text_context.full_word_ids is None:
            raise ValueError("text_context.full_word_ids is mandatory for Phase-5 word-aware grounding")
        full_features = text_context.full_features
        if full_features.shape[0] != hypothesis_features.shape[0]:
            raise ValueError("text_context full store batch size must match hypothesis_features")
        if full_features.shape[-1] != self.token_input_dim:
            raise ValueError(
                f"text_context.full_features last dim must be token_input_dim={self.token_input_dim}, "
                f"got {full_features.shape[-1]}"
            )
        full_mask = (
            torch.ones(full_features.shape[:2], dtype=torch.bool, device=full_features.device)
            if text_context.full_mask is None
            else text_context.full_mask.to(torch.bool)
        )
        word_mask = text_context.full_word_ids.to(device=full_features.device, dtype=torch.long) != PAD_WORD_ID
        mask = full_mask & word_mask
        keys = self.token_key(full_features)
        logits = []
        scale = math.sqrt(float(self.hidden_dim))
        for local_head_index, hypothesis_index in enumerate(TEXT_HYPOTHESIS_INDICES):
            query = self.hypothesis_queries[local_head_index](hypothesis_features[:, hypothesis_index, :])
            score = torch.sum(keys * query.unsqueeze(1), dim=-1) / scale
            logits.append(score)
        stacked_logits = torch.stack(logits, dim=1).masked_fill(~mask.unsqueeze(1), -1e4)
        probabilities = torch.sigmoid(stacked_logits).masked_fill(~mask.unsqueeze(1), 0.0)
        return stacked_logits, probabilities, mask


def _validate_hypotheses(hypothesis_features: torch.Tensor, hidden_dim: int) -> None:
    if hypothesis_features.ndim != 3:
        raise ValueError(f"hypothesis_features must be [B,4,D], got {tuple(hypothesis_features.shape)}")
    if hypothesis_features.shape[1] != len(HYPOTHESIS_ORDER):
        raise ValueError("hypothesis_features must preserve frozen [FS,FA,TS,TA] order")
    if hypothesis_features.shape[-1] != hidden_dim:
        raise ValueError(f"hypothesis hidden_dim mismatch: expected {hidden_dim}, got {hypothesis_features.shape[-1]}")
    if not torch.isfinite(hypothesis_features).all():
        raise FloatingPointError("hypothesis_features contain non-finite values")


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.bool)
    probabilities = torch.softmax(logits, dim=-1)
    probabilities = probabilities.masked_fill(~mask, 0.0)
    denominator = probabilities.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).eps)
    return torch.where(mask.any(dim=-1, keepdim=True), probabilities / denominator, probabilities)

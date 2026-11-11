"""Hypothesis-conditioned grounding heads for canonical HERA decisions."""

from __future__ import annotations

import torch
from torch import nn

from heri.evidence.outputs import EvidenceSet, PAD_WORD_ID
from heri.graph.schema import HYPOTHESIS_ORDER
from heri.models.fusion.projections import LazyFeatureProjection


IMAGE_HYPOTHESIS_INDICES = (0, 1)
TEXT_HYPOTHESIS_INDICES = (2, 3)


class HypothesisConditionedImageGroundingHead(nn.Module):
    """Predict dense manipulation heatmaps and boxes from raw-image DINO patches."""

    def __init__(self, *, hidden_dim: int, dropout: float = 0.0, scoring: str = "dot_product",
                 hypothesis_indices: tuple[int, ...] = IMAGE_HYPOTHESIS_INDICES,
                 hypothesis_count: int = len(HYPOTHESIS_ORDER)) -> None:
        super().__init__()
        if scoring not in {"bilinear", "dot_product"}:
            raise ValueError("Image grounding requires bilinear scoring")
        self.hidden_dim = int(hidden_dim)
        self.hypothesis_indices = tuple(int(index) for index in hypothesis_indices)
        self.hypothesis_count = int(hypothesis_count)
        _validate_hypothesis_indices(self.hypothesis_indices, self.hypothesis_count)
        # Eq.18--20: modality-specific transforms shared across hypotheses.
        self.patch_key = nn.LazyLinear(self.hidden_dim, bias=False)
        self.hypothesis_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.grounding_bias = nn.Parameter(torch.zeros(()))
        self.box_regressor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 4),
        )

    def forward(
        self,
        *,
        hypothesis_features: torch.Tensor,
        image_evidence: EvidenceSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return patch logits/probabilities, mask, and normalized boxes [B,2,4]."""

        _validate_hypotheses(hypothesis_features, self.hidden_dim, self.hypothesis_count)
        image_evidence.validate()
        full_features, mask, _ = _dense_image_store(image_evidence)
        if full_features.shape[0] != hypothesis_features.shape[0]:
            raise ValueError("image_evidence batch size must match hypothesis_features")
        keys = self.patch_key(full_features)
        logits = []
        boxes = []
        for hypothesis_index in self.hypothesis_indices:
            hypothesis = hypothesis_features[:, hypothesis_index, :]
            query = self.hypothesis_query(hypothesis)
            score = torch.sum(keys.float() * query.float().unsqueeze(1), dim=-1) + self.grounding_bias.float()
            logits.append(score)
            attention = _masked_softmax(score, mask)
            context = torch.sum(attention.unsqueeze(-1) * keys.float(), dim=1).to(hypothesis.dtype)
            parameters = self.box_regressor(torch.cat([hypothesis, context], dim=-1))
            boxes.append(_normalized_center_size_box(parameters))
        stacked_logits = torch.stack(logits, dim=1).masked_fill(~mask.unsqueeze(1), -1e4)
        probabilities = torch.sigmoid(stacked_logits).masked_fill(~mask.unsqueeze(1), 0.0)
        return stacked_logits, probabilities, mask, torch.stack(boxes, dim=1)


class HypothesisConditionedTextGroundingHead(nn.Module):
    """Score the full DeBERTa token store conditioned on H_TS and H_TA."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_input_dim: int,
        dropout: float = 0.0,
        scoring: str = "dot_product",
        hypothesis_indices: tuple[int, ...] = TEXT_HYPOTHESIS_INDICES,
        hypothesis_count: int = len(HYPOTHESIS_ORDER),
    ) -> None:
        super().__init__()
        if scoring not in {"bilinear", "dot_product"}:
            raise ValueError("Text grounding requires bilinear scoring")
        if int(token_input_dim) <= 0:
            raise ValueError("token_input_dim must be positive and YAML-provided")
        self.hidden_dim = int(hidden_dim)
        self.token_input_dim = int(token_input_dim)
        self.hypothesis_indices = tuple(int(index) for index in hypothesis_indices)
        self.hypothesis_count = int(hypothesis_count)
        _validate_hypothesis_indices(self.hypothesis_indices, self.hypothesis_count)
        self.token_key = nn.Linear(self.token_input_dim, self.hidden_dim, bias=False)
        self.hypothesis_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.grounding_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        hypothesis_features: torch.Tensor,
        text_context: EvidenceSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full-token logits/probabilities [B,2,N_full_text] and mask."""

        _validate_hypotheses(hypothesis_features, self.hidden_dim, self.hypothesis_count)
        text_context.validate()
        if text_context.full_features is None:
            raise ValueError("text_context.full_features is mandatory for full-text grounding")
        if text_context.full_word_ids is None:
            raise ValueError("text_context.full_word_ids is mandatory for word-aware grounding")
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
        for hypothesis_index in self.hypothesis_indices:
            query = self.hypothesis_query(hypothesis_features[:, hypothesis_index, :])
            score = torch.sum(keys.float() * query.float().unsqueeze(1), dim=-1) + self.grounding_bias.float()
            logits.append(score)
        stacked_logits = torch.stack(logits, dim=1).masked_fill(~mask.unsqueeze(1), -1e4)
        probabilities = torch.sigmoid(stacked_logits).masked_fill(~mask.unsqueeze(1), 0.0)
        return stacked_logits, probabilities, mask


class GenericImageGroundingHead(nn.Module):
    """Dense bbox ablation without hypothesis-specific grounding queries."""

    def __init__(self, *, hidden_dim: int, dropout: float = 0.0,
                 output_count: int = len(IMAGE_HYPOTHESIS_INDICES),
                 hypothesis_count: int = len(HYPOTHESIS_ORDER)) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_count = int(output_count)
        self.hypothesis_count = int(hypothesis_count)
        if self.output_count < 1:
            raise ValueError("output_count must be positive")
        self.patch_key = LazyFeatureProjection(output_dim=self.hidden_dim, dropout=float(dropout))
        self.patch_scorer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        self.box_regressor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + 4),
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 4),
        )

    def forward(
        self,
        *,
        hypothesis_features: torch.Tensor,
        image_evidence: EvidenceSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_hypotheses(hypothesis_features, self.hidden_dim, self.hypothesis_count)
        image_evidence.validate()
        full_features, mask, normalized_patch_boxes = _dense_image_store(image_evidence)
        if full_features.shape[0] != hypothesis_features.shape[0]:
            raise ValueError("image_evidence must align with hypotheses")
        keys = self.patch_key(full_features)
        generic_logits = self.patch_scorer(keys).squeeze(-1).masked_fill(~mask, -1e4)
        attention = _masked_softmax(generic_logits, mask)
        context = torch.sum(attention.unsqueeze(-1) * keys, dim=1)
        geometric_prior = torch.sum(attention.unsqueeze(-1) * normalized_patch_boxes, dim=1)
        generic_box = _normalized_center_size_box(
            self.box_regressor(torch.cat([context, geometric_prior], dim=-1))
        )
        logits = generic_logits.unsqueeze(1).expand(-1, self.output_count, -1)
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e4)
        probabilities = torch.sigmoid(logits).masked_fill(~mask.unsqueeze(1), 0.0)
        boxes = generic_box.unsqueeze(1).expand(-1, self.output_count, -1)
        return logits, probabilities, mask, boxes


class GenericTextGroundingHead(nn.Module):
    """Ground manipulated tokens without receiving TS/TA hypotheses."""

    def __init__(self, *, hidden_dim: int, token_input_dim: int, dropout: float = 0.0,
                 output_count: int = len(TEXT_HYPOTHESIS_INDICES),
                 hypothesis_count: int = len(HYPOTHESIS_ORDER)) -> None:
        super().__init__()
        if int(token_input_dim) <= 0:
            raise ValueError("token_input_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.token_input_dim = int(token_input_dim)
        self.output_count = int(output_count)
        self.hypothesis_count = int(hypothesis_count)
        if self.output_count < 1:
            raise ValueError("output_count must be positive")
        self.token_scorer = nn.Sequential(
            nn.LayerNorm(self.token_input_dim),
            nn.Linear(self.token_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        hypothesis_features: torch.Tensor,
        text_context: EvidenceSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_hypotheses(hypothesis_features, self.hidden_dim, self.hypothesis_count)
        text_context.validate()
        if text_context.full_features is None or text_context.full_word_ids is None:
            raise ValueError("Generic text grounding requires the full DeBERTa token store")
        full_features = text_context.full_features
        if full_features.shape[0] != hypothesis_features.shape[0]:
            raise ValueError("text_context full store batch size must match hypotheses")
        if full_features.shape[-1] != self.token_input_dim:
            raise ValueError(
                f"text_context.full_features must end in {self.token_input_dim}, "
                f"got {full_features.shape[-1]}"
            )
        full_mask = (
            torch.ones(full_features.shape[:2], dtype=torch.bool, device=full_features.device)
            if text_context.full_mask is None
            else text_context.full_mask.to(torch.bool)
        )
        word_mask = text_context.full_word_ids.to(full_features.device) != PAD_WORD_ID
        mask = full_mask & word_mask
        generic_logits = self.token_scorer(full_features).squeeze(-1)
        logits = generic_logits.unsqueeze(1).expand(-1, self.output_count, -1)
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e4)
        probabilities = torch.sigmoid(logits).masked_fill(~mask.unsqueeze(1), 0.0)
        return logits, probabilities, mask


def _validate_hypotheses(
    hypothesis_features: torch.Tensor, hidden_dim: int, hypothesis_count: int
) -> None:
    if hypothesis_features.ndim != 3:
        raise ValueError(f"hypothesis_features must be [B,K,D], got {tuple(hypothesis_features.shape)}")
    if hypothesis_features.shape[1] != hypothesis_count:
        raise ValueError("hypothesis_features must preserve configured hypothesis order")
    if hypothesis_features.shape[-1] != hidden_dim:
        raise ValueError(f"hypothesis hidden_dim mismatch: expected {hidden_dim}, got {hypothesis_features.shape[-1]}")
    if not torch.isfinite(hypothesis_features).all():
        raise FloatingPointError("hypothesis_features contain non-finite values")


def _validate_hypothesis_indices(indices: tuple[int, ...], hypothesis_count: int) -> None:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("grounding hypothesis indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= hypothesis_count:
        raise ValueError("grounding hypothesis index is out of range")


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Eq.19 softmax over valid positions, with zero for empty rows."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if logits.shape[-1] == 0:
        return logits.float()
    valid_row = mask.any(dim=-1, keepdim=True)
    scores = logits.float().masked_fill(~mask, -torch.inf)
    scores = torch.where(valid_row, scores, torch.zeros_like(scores))
    return torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)


def _dense_image_store(evidence: EvidenceSet) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if evidence.full_features is None:
        raise ValueError("Dense image grounding requires visual_forensic.full_features")
    if evidence.full_normalized_boxes is None:
        raise ValueError("Dense image grounding requires visual_forensic.full_normalized_boxes")
    features = evidence.full_features
    mask = (
        torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if evidence.full_mask is None
        else evidence.full_mask.to(device=features.device, dtype=torch.bool)
    )
    boxes = evidence.full_normalized_boxes.to(device=features.device, dtype=features.dtype)
    if boxes.shape != features.shape[:2] + (4,):
        raise ValueError("Dense image patch boxes must align with full image features")
    return features, mask, boxes


def _normalized_center_size_box(parameters: torch.Tensor) -> torch.Tensor:
    """Map unconstrained parameters to ordered normalized xyxy boxes."""

    if parameters.ndim != 2 or parameters.shape[-1] != 4:
        raise ValueError("bbox parameters must be [B,4]")
    # Eq.20 predicts sigmoid (cx, cy, width, height). Internally preserve
    # unclipped xyxy geometry for SmoothL1/GIoU; inference clips to the image.
    center_size = torch.sigmoid(parameters.float())
    center, size = center_size[:, :2], center_size[:, 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)

"""DeBERTa contextual text evidence extraction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.outputs import DeBERTaOutput

from .budgeting import budget_features
from .outputs import EvidenceSet, PAD_WORD_ID
from .spatial import gather_batched


class TextContextEvidenceExtractor(nn.Module):
    """Project and budget DeBERTa subword contextual evidence."""

    def __init__(self, *, projector: nn.Module, max_nodes: int, budgeting: str) -> None:
        super().__init__()
        self.projector = projector
        self.max_nodes = int(max_nodes)
        self.budgeting = budgeting

    def forward(
        self,
        *,
        deberta: DeBERTaOutput,
        text_word_ids: torch.Tensor | None,
        grounding_valid_mask: torch.Tensor | None,
    ) -> EvidenceSet:
        attention_mask = deberta.attention_mask.to(torch.bool)
        if grounding_valid_mask is None:
            evidence_mask = attention_mask
        else:
            if grounding_valid_mask.shape != attention_mask.shape:
                raise ValueError("grounding_valid_mask must align with DeBERTa attention_mask")
            evidence_mask = attention_mask & grounding_valid_mask.to(torch.bool)
        if text_word_ids is None:
            text_word_ids = torch.full(
                attention_mask.shape,
                PAD_WORD_ID,
                dtype=torch.long,
                device=attention_mask.device,
            )
        else:
            if text_word_ids.shape != attention_mask.shape:
                raise ValueError("text_word_ids must align with DeBERTa attention_mask")
            text_word_ids = text_word_ids.to(device=attention_mask.device, dtype=torch.long)

        budgeted = budget_features(
            features=deberta.token_features,
            valid_mask=evidence_mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
            selection_scores=_context_importance(deberta.token_features, evidence_mask),
        )
        selected_word_ids = gather_batched(text_word_ids.unsqueeze(-1).to(deberta.token_features.dtype), budgeted.source_indices)
        selected_word_ids = selected_word_ids.squeeze(-1).to(torch.long)
        projected = self.projector(budgeted.features)
        return EvidenceSet(
            family="text_context",
            features=projected,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
            selection_scores=budgeted.selection_scores,
            word_ids=selected_word_ids.masked_fill(~budgeted.mask, PAD_WORD_ID),
            full_features=deberta.token_features,
            full_mask=attention_mask,
            full_source_indices=torch.arange(
                deberta.token_features.shape[1],
                device=deberta.token_features.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(deberta.token_features.shape[0], -1),
            full_word_ids=text_word_ids,
            provenance={
                "source": "deberta.token_features",
                "budgeting": self.budgeting,
                "grounding_valid_mask_preserved": grounding_valid_mask is not None,
                "fake_text_pos_used": False,
            },
        )


def _context_importance(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = features.masked_fill(~mask.unsqueeze(-1), 0.0)
    center = masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
    return torch.linalg.vector_norm(features - center, dim=-1).masked_fill(~mask, 0.0)

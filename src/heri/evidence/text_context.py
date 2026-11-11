"""DeBERTa contextual text evidence extraction."""

from __future__ import annotations

import torch
from torch import nn

from heri.models.outputs import DeBERTaOutput

from .budgeting import LearnedEvidenceSelector
from .outputs import EvidenceSet, PAD_WORD_ID
from .spatial import gather_batched


class TextContextEvidenceExtractor(nn.Module):
    """Project and budget DeBERTa subword contextual evidence."""

    def __init__(self, *, projector: nn.Module, max_nodes: int, budgeting: str, selection_eta: float = 0.5) -> None:
        super().__init__()
        self.projector = projector
        self.max_nodes = int(max_nodes)
        self.budgeting = budgeting
        self.selector = LearnedEvidenceSelector(eta=selection_eta)

    def forward(
        self,
        *,
        deberta: DeBERTaOutput,
        text_word_ids: torch.Tensor | None,
        grounding_valid_mask: torch.Tensor | None,
        text_token_spans: torch.Tensor | None = None,
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

        budgeted = self.selector(
            features=self.projector(deberta.token_features),
            mask=evidence_mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
        )
        selected_word_ids = gather_batched(text_word_ids.unsqueeze(-1).to(deberta.token_features.dtype), budgeted.source_indices)
        selected_word_ids = selected_word_ids.squeeze(-1).to(torch.long)
        projected = budgeted.features
        return EvidenceSet(
            family="text_context",
            features=projected,
            reliability_features=budgeted.ungated_features,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
            selection_scores=budgeted.selection_scores,
            word_ids=selected_word_ids.masked_fill(~budgeted.mask, PAD_WORD_ID),
            token_spans=(None if text_token_spans is None else torch.gather(
                text_token_spans, 1, budgeted.source_indices.clamp_min(0).unsqueeze(-1).expand(-1,-1,2)
            ).masked_fill(~budgeted.mask.unsqueeze(-1), -1)),
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

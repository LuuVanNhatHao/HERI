"""SigLIP2 local text-alignment evidence extraction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.outputs import SigLIP2Output

from .budgeting import budget_features
from .outputs import EvidenceSet


class TextAlignmentEvidenceExtractor(nn.Module):
    """Project and budget SigLIP2 text-token semantic evidence."""

    def __init__(self, *, projector: nn.Module, max_nodes: int, budgeting: str) -> None:
        super().__init__()
        self.projector = projector
        self.max_nodes = int(max_nodes)
        self.budgeting = budgeting

    def forward(self, siglip2: SigLIP2Output) -> EvidenceSet:
        text_mask = siglip2.text_mask
        if text_mask is None:
            text_mask = torch.ones(siglip2.text_tokens.shape[:2], dtype=torch.bool, device=siglip2.text_tokens.device)
        word_ids = siglip2.text_word_ids
        if word_ids is None:
            word_ids = torch.arange(
                siglip2.text_tokens.shape[1], device=siglip2.text_tokens.device, dtype=torch.long
            ).unsqueeze(0).expand(siglip2.text_tokens.shape[0], -1)
        else:
            word_ids = word_ids.to(device=siglip2.text_tokens.device, dtype=torch.long)
            if word_ids.shape != text_mask.shape:
                raise ValueError("SigLIP2 text_word_ids must align with text_mask")
            text_mask = text_mask & (word_ids >= 0)
        masked_tokens = siglip2.text_tokens.masked_fill(~text_mask.unsqueeze(-1), 0.0)
        center = masked_tokens.sum(dim=1, keepdim=True) / text_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        token_importance = torch.linalg.vector_norm(siglip2.text_tokens - center, dim=-1).masked_fill(~text_mask, 0.0)
        budgeted = budget_features(
            features=siglip2.text_tokens,
            valid_mask=text_mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
            selection_scores=token_importance,
        )
        projected = self.projector(budgeted.features)
        selected_word_ids = torch.gather(word_ids, 1, budgeted.source_indices.clamp_min(0))
        selected_word_ids = selected_word_ids.masked_fill(~budgeted.mask, -1)
        return EvidenceSet(
            family="text_alignment",
            features=projected,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
            selection_scores=budgeted.selection_scores,
            word_ids=selected_word_ids,
            full_features=siglip2.text_tokens,
            full_mask=text_mask,
            full_source_indices=torch.arange(
                siglip2.text_tokens.shape[1],
                device=siglip2.text_tokens.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(siglip2.text_tokens.shape[0], -1),
            provenance={
                "source": "siglip2.text_tokens",
                "budgeting": self.budgeting,
                "fake_text_pos_used": False,
                "word_correspondence": "siglip_fast_tokenizer_word_ids",
            },
        )

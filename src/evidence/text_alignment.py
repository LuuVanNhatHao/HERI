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
        budgeted = budget_features(
            features=siglip2.text_tokens,
            valid_mask=text_mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
        )
        projected = self.projector(budgeted.features)
        return EvidenceSet(
            family="text_alignment",
            features=projected,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
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
            },
        )

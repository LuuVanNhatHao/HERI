"""SigLIP2 local text-alignment evidence extraction."""

from __future__ import annotations

import torch
from torch import nn

from heri.models.outputs import SigLIP2Output

from .budgeting import LearnedEvidenceSelector
from .outputs import EvidenceSet


class TextAlignmentEvidenceExtractor(nn.Module):
    """Project and budget SigLIP2 text-token semantic evidence."""

    def __init__(self, *, projector: nn.Module, max_nodes: int, budgeting: str, selection_eta: float = 0.5) -> None:
        super().__init__()
        self.projector = projector
        self.max_nodes = int(max_nodes)
        self.budgeting = budgeting
        self.selector = LearnedEvidenceSelector(eta=selection_eta)

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
        budgeted = self.selector(
            features=self.projector(siglip2.text_tokens),
            mask=text_mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
        )
        projected = budgeted.features
        selected_word_ids = torch.gather(word_ids, 1, budgeted.source_indices.clamp_min(0))
        selected_word_ids = selected_word_ids.masked_fill(~budgeted.mask, -1)
        return EvidenceSet(
            family="text_alignment",
            features=projected,
            reliability_features=budgeted.ungated_features,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
            selection_scores=budgeted.selection_scores,
            word_ids=selected_word_ids,
            token_spans=(None if siglip2.text_token_spans is None else torch.gather(
                siglip2.text_token_spans, 1, budgeted.source_indices.clamp_min(0).unsqueeze(-1).expand(-1,-1,2)
            ).masked_fill(~budgeted.mask.unsqueeze(-1), -1)),
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

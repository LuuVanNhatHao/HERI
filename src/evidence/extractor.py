"""High-level Phase-3 heterogeneous evidence extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from src.models.outputs import BackboneOutputs

from .cross_modal import cosine_similarity_metadata
from .outputs import CrossModalMetadata, EvidenceSet, HeterogeneousEvidenceBatch
from .projection import EvidenceProjectionBank
from .text_alignment import TextAlignmentEvidenceExtractor
from .text_context import TextContextEvidenceExtractor
from .visual_forensic import VisualForensicEvidenceExtractor
from .visual_semantic import VisualSemanticEvidenceExtractor


@dataclass(frozen=True)
class EvidenceStructuralMetadata:
    """Target-independent structural metadata consumed by the extractor."""

    text_attention_mask: torch.Tensor | None = None
    text_grounding_mask: torch.Tensor | None = None
    text_word_ids: torch.Tensor | None = None
    image: torch.Tensor | None = None
    image_content_mask: torch.Tensor | None = None

    @classmethod
    def from_batch(cls, batch: Mapping[str, Any]) -> "EvidenceStructuralMetadata":
        """Create metadata from legal structural fields only."""

        return cls(
            text_attention_mask=batch.get("text_attention_mask"),
            text_grounding_mask=batch.get("text_grounding_mask"),
            text_word_ids=batch.get("text_word_ids"),
            image=batch.get("image"),
            image_content_mask=batch.get("image_content_mask"),
        )


class HeterogeneousEvidenceExtractor(nn.Module):
    """Transform Phase-2 backbone outputs into heterogeneous evidence sets."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.hidden_dim = int(config["hidden_dim"])
        self.projections = EvidenceProjectionBank(config)

        visual_semantic = config["visual_semantic"]
        visual_forensic = config["visual_forensic"]
        face = config["face"]
        if bool(face.get("enabled", False)):
            raise ValueError(
                "Final HERA removed annotation-derived MTCNN face evidence; "
                "set evidence.face.enabled=false"
            )
        text_alignment = config["text_alignment"]
        text_context = config["text_context"]
        self.enabled = {
            "visual_semantic": bool(visual_semantic.get("enabled", True)),
            "visual_forensic": bool(visual_forensic.get("enabled", True)),
            "face": bool(face.get("enabled", True)),
            "text_alignment": bool(text_alignment.get("enabled", True)),
            "text_context": bool(text_context.get("enabled", True)),
        }
        self.cross_modal_enabled = bool(config.get("cross_modal", {}).get("enabled", False))
        self.cross_modal_similarity = str(config.get("cross_modal", {}).get("similarity", "cosine"))

        self.visual_semantic_extractor = VisualSemanticEvidenceExtractor(
            projector=self.projections["visual_semantic"],
            max_nodes=int(visual_semantic["max_nodes"]),
            budgeting=str(visual_semantic.get("budgeting", "spatial_balanced")),
        )
        self.visual_forensic_extractor = VisualForensicEvidenceExtractor(
            projector=self.projections["visual_forensic"],
            hidden_dim=self.hidden_dim,
            max_nodes=int(visual_forensic["max_nodes"]),
            budgeting=str(visual_forensic.get("budgeting", "spatial_balanced")),
        )
        self.text_alignment_extractor = TextAlignmentEvidenceExtractor(
            projector=self.projections["text_alignment"],
            max_nodes=int(text_alignment["max_nodes"]),
            budgeting=str(text_alignment.get("budgeting", "sequence_balanced")),
        )
        self.text_context_extractor = TextContextEvidenceExtractor(
            projector=self.projections["text_context"],
            max_nodes=int(text_context["max_nodes"]),
            budgeting=str(text_context.get("budgeting", "sequence_balanced")),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HeterogeneousEvidenceExtractor":
        """Build extractor from the top-level YAML config."""

        return cls(config["evidence"])

    def forward(
        self,
        encoder_outputs: BackboneOutputs,
        structural_metadata: EvidenceStructuralMetadata,
    ) -> HeterogeneousEvidenceBatch:
        """Return Phase-4-ready evidence without reading supervision targets."""

        batch_size = encoder_outputs.dinov3.patch_tokens.shape[0]
        device = encoder_outputs.dinov3.patch_tokens.device
        dtype = encoder_outputs.dinov3.patch_tokens.dtype
        visual_semantic = (
            self.visual_semantic_extractor(encoder_outputs.siglip2)
            if self.enabled["visual_semantic"]
            else _empty_evidence("visual_semantic", batch_size, self.hidden_dim, device=device, dtype=dtype)
        )
        visual_forensic = (
            self.visual_forensic_extractor(
                encoder_outputs.dinov3,
                image=structural_metadata.image,
                image_content_mask=structural_metadata.image_content_mask,
            )
            if self.enabled["visual_forensic"]
            else _empty_evidence("visual_forensic", batch_size, self.hidden_dim, device=device, dtype=dtype)
        )
        face = _empty_evidence("face", batch_size, self.hidden_dim, device=device, dtype=dtype)
        text_alignment = (
            self.text_alignment_extractor(encoder_outputs.siglip2)
            if self.enabled["text_alignment"]
            else _empty_evidence("text_alignment", batch_size, self.hidden_dim, device=device, dtype=dtype)
        )
        text_context = (
            self.text_context_extractor(
                deberta=encoder_outputs.deberta,
                text_word_ids=structural_metadata.text_word_ids,
                grounding_valid_mask=structural_metadata.text_grounding_mask,
            )
            if self.enabled["text_context"]
            else _empty_evidence("text_context", batch_size, self.hidden_dim, device=device, dtype=dtype)
        )
        cross_modal = self._cross_modal(
            visual_semantic=visual_semantic,
            text_alignment=text_alignment,
            native_image_global=encoder_outputs.siglip2.image_global,
            native_text_global=encoder_outputs.siglip2.text_global,
        )
        evidence = HeterogeneousEvidenceBatch(
            visual_semantic=visual_semantic,
            visual_forensic=visual_forensic,
            face=face,
            text_alignment=text_alignment,
            text_context=text_context,
            cross_modal_metadata=cross_modal,
        )
        evidence.validate()
        return evidence

    def _cross_modal(
        self,
        *,
        visual_semantic: EvidenceSet,
        text_alignment: EvidenceSet,
        native_image_global: torch.Tensor | None,
        native_text_global: torch.Tensor | None,
    ) -> CrossModalMetadata:
        if not self.cross_modal_enabled:
            return CrossModalMetadata(method="disabled")
        if self.cross_modal_similarity != "cosine":
            raise ValueError(f"Unsupported cross-modal similarity: {self.cross_modal_similarity!r}")
        return cosine_similarity_metadata(
            visual=visual_semantic,
            text=text_alignment,
            native_image_global=native_image_global,
            native_text_global=native_text_global,
        )


def _empty_evidence(
    family: str,
    batch_size: int,
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> EvidenceSet:
    features = torch.zeros((batch_size, 0, hidden_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
    source_indices = torch.zeros((batch_size, 0), device=device, dtype=torch.long)
    return EvidenceSet(
        family=family,
        features=features,
        mask=mask,
        source_indices=source_indices,
        provenance={"source": "disabled"},
    )

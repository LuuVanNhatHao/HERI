"""Task-aware HERA model for datasets that expose only binary supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from src.graph import HypothesisEvidenceGraphReasoner
from src.models.baselines.base import BaselineBackboneMixin, finite_or_fail
from src.models.baselines.factory import build_backbones
from src.models.heads.binary import IndependentHypothesisBinaryHead
from src.models.outputs import BackboneOutputs
from src.routing import EvidenceReliabilityRouter

from .contracts import SUBTYPE_CLASSES, SignalContract


@dataclass
class BinaryBenchmarkOutput:
    """Outputs whose semantics are legal for the configured binary benchmark."""

    binary_logit: torch.Tensor
    binary_probability: torch.Tensor
    hypothesis_features: torch.Tensor
    cross_modal_alignment_loss: torch.Tensor
    router_regularization_loss: torch.Tensor
    subtype_logits: torch.Tensor | None = None
    text_label_logit: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.binary_logit.ndim != 1:
            raise ValueError("binary_logit must be [B]")
        batch = self.binary_logit.shape[0]
        if self.binary_probability.shape != (batch,):
            raise ValueError("binary_probability must be [B]")
        if self.hypothesis_features.ndim != 3 or self.hypothesis_features.shape[:2] != (
            batch,
            4,
        ):
            raise ValueError("hypothesis_features must be [B,4,D]")
        if not torch.allclose(
            self.binary_probability,
            torch.sigmoid(self.binary_logit),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("binary_probability must equal sigmoid(binary_logit)")
        for name in (
            "binary_logit",
            "binary_probability",
            "hypothesis_features",
            "cross_modal_alignment_loss",
            "router_regularization_loss",
        ):
            if not torch.isfinite(getattr(self, name)).all():
                raise FloatingPointError(f"{name} contains non-finite values")
        if self.subtype_logits is not None:
            if self.subtype_logits.shape != (batch, len(SUBTYPE_CLASSES)):
                raise ValueError("subtype_logits must be [B,4]")
            if not torch.isfinite(self.subtype_logits).all():
                raise FloatingPointError("subtype_logits contains non-finite values")
        if self.text_label_logit is not None:
            if self.text_label_logit.shape != (batch,):
                raise ValueError("text_label_logit must be [B]")
            if not torch.isfinite(self.text_label_logit).all():
                raise FloatingPointError("text_label_logit contains non-finite values")


class BinaryBenchmarkHERA(nn.Module, BaselineBackboneMixin):
    """HERA fusion/reasoning with latent hypotheses and a binary decision head.

    This path deliberately does not instantiate DGM4 attribution or grounding
    heads. The four HGT hypothesis states are latent evidence slots, not FS/FA/TS/TA
    targets. Optional dataset-native auxiliary heads are created only when enabled.
    """

    model_type = "hera"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        siglip2: nn.Module | None = None,
        dinov3: nn.Module | None = None,
        deberta: nn.Module | None = None,
        evidence_extractor: nn.Module | None = None,
        router: nn.Module | None = None,
        graph_reasoner: nn.Module | None = None,
        load_backbones: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.signals = SignalContract.from_config(config)
        backbones = build_backbones(config) if load_backbones else {}
        self.siglip2 = siglip2 if siglip2 is not None else backbones.get("siglip2")
        self.dinov3 = dinov3 if dinov3 is not None else backbones.get("dinov3")
        self.deberta = deberta if deberta is not None else backbones.get("deberta")
        self.evidence_extractor = (
            evidence_extractor
            if evidence_extractor is not None
            else HeterogeneousEvidenceExtractor.from_config(config)
        )
        self.router = (
            router
            if router is not None
            else EvidenceReliabilityRouter.from_config(config)
        )
        if not bool(getattr(self.router, "enabled", True)):
            self.router.requires_grad_(False)
        self.graph_reasoner = (
            graph_reasoner
            if graph_reasoner is not None
            else HypothesisEvidenceGraphReasoner.from_config(config)
        )
        hidden_dim = int(config["model"]["hidden_dim"])
        dropout = float(config["model"]["binary"]["dropout"])
        self.binary_head = IndependentHypothesisBinaryHead(
            hidden_dim=hidden_dim,
            hypotheses=4,
            dropout=dropout,
        )
        self.auxiliary_heads = nn.ModuleDict()
        if self.signals.subtype:
            self.auxiliary_heads["subtype"] = _AuxiliaryHead(
                hidden_dim=hidden_dim,
                output_dim=len(SUBTYPE_CLASSES),
                dropout=dropout,
            )
        if self.signals.text_label:
            self.auxiliary_heads["text_label"] = _AuxiliaryHead(
                hidden_dim=hidden_dim,
                output_dim=1,
                dropout=dropout,
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        load_backbones: bool = True,
    ) -> "BinaryBenchmarkHERA":
        return cls(config, load_backbones=load_backbones)

    @property
    def ablation_variant(self) -> None:
        return None

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        features: BackboneOutputs | None = None,
    ) -> BinaryBenchmarkOutput:
        encoded = self._encode_backbones(batch, features)
        evidence = self.evidence_extractor(
            encoded,
            EvidenceStructuralMetadata.from_batch(batch),
        )
        routed = self.router(evidence)
        graph = self.graph_reasoner(routed.as_graph_evidence())
        hypothesis_features = graph.hypothesis_features
        binary_logit = self.binary_head(hypothesis_features)
        pooled = hypothesis_features.mean(dim=1)
        subtype_logits = (
            self.auxiliary_heads["subtype"](pooled)
            if "subtype" in self.auxiliary_heads
            else None
        )
        text_label_logit = (
            self.auxiliary_heads["text_label"](pooled).squeeze(-1)
            if "text_label" in self.auxiliary_heads
            else None
        )
        output = BinaryBenchmarkOutput(
            binary_logit=binary_logit,
            binary_probability=torch.sigmoid(binary_logit),
            hypothesis_features=hypothesis_features,
            cross_modal_alignment_loss=evidence.cross_modal_metadata.alignment_loss,
            router_regularization_loss=routed.regularization_loss,
            subtype_logits=subtype_logits,
            text_label_logit=text_label_logit,
            metadata={
                "model_type": "hera_binary_benchmark",
                "task_type": "binary_multimodal",
                "hypothesis_semantics": "modality_aligned_latent_evidence_slots",
                "latent_slot_roles": list(
                    self.config["model"]["latent_slots"]["roles"]
                ),
                "signals": self.signals.as_dict(),
                "pipeline": [
                    "SigLIP2",
                    "DINOv3",
                    "TextTransformer",
                    "HeterogeneousEvidenceExtractor",
                    "EvidenceReliabilityRouter",
                    "HGT",
                    "LatentHypothesisBinaryHead",
                ],
            },
        )
        finite_or_fail({"binary_logit": output.binary_logit})
        output.validate()
        return output

    def new_modules(self) -> list[nn.Module]:
        modules = [self.evidence_extractor, self.graph_reasoner, self.binary_head]
        if bool(getattr(self.router, "enabled", True)):
            modules.insert(1, self.router)
        modules.extend(self.auxiliary_heads.values())
        return modules

    def hera_component_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {
            "evidence_extractor": self.evidence_extractor,
            "router": self.router,
            "graph_reasoner": self.graph_reasoner,
            "binary_head": self.binary_head,
        }
        if hasattr(self.graph_reasoner, "graph_encoder_module"):
            modules["hgt"] = self.graph_reasoner.graph_encoder_module()
        if hasattr(self.graph_reasoner, "hypotheses"):
            modules["hypothesis_embeddings"] = self.graph_reasoner.hypotheses
        modules.update(
            {
                f"auxiliary_{name}": module
                for name, module in self.auxiliary_heads.items()
            }
        )
        return modules


class _AuxiliaryHead(nn.Module):
    def __init__(self, *, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

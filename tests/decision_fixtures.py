from __future__ import annotations

from dataclasses import replace

import torch

from graph_fixtures import graph_config, synthetic_evidence
from phase2_fixtures import synthetic_batch
from src.evidence.outputs import HeterogeneousEvidenceBatch
from src.graph.builder import HypothesisEvidenceGraphBuilder
from src.graph.outputs import HypothesisGraphOutput
from src.graph.schema import HYPOTHESIS_ORDER
from src.graph.validation import graph_statistics


def decision_config(hidden_dim: int = 12, token_input_dim: int = 19) -> dict:
    return {
        "model": {
            "hidden_dim": hidden_dim,
            "hypotheses": {"order": list(HYPOTHESIS_ORDER)},
            "attribution": {"hidden_dim": hidden_dim, "dropout": 0.0},
            "grounding": {
                "image": {"scoring": "dot_product", "dropout": 0.0},
                "text": {
                    "scoring": "dot_product",
                    "dropout": 0.0,
                    "full_token_store": True,
                    "token_input_dim": token_input_dim,
                },
            },
            "binary": {"aggregation": "noisy_or"},
        },
        "loss": {
            "lambda_attr": 1.0,
            "lambda_bin": 1.0,
            "lambda_img": 1.0,
            "lambda_txt": 1.0,
        },
    }


def synthetic_graph_output(hidden_dim: int = 12, *, requires_grad: bool = False) -> HypothesisGraphOutput:
    evidence = synthetic_evidence(hidden_dim=hidden_dim)
    return graph_output_from_evidence(evidence, hidden_dim=hidden_dim, requires_grad=requires_grad)


def graph_output_from_evidence(
    evidence: HeterogeneousEvidenceBatch,
    *,
    hidden_dim: int,
    requires_grad: bool = False,
) -> HypothesisGraphOutput:
    torch.manual_seed(29)
    batch_size = evidence.visual_semantic.features.shape[0]
    hypotheses = torch.randn(batch_size, len(HYPOTHESIS_ORDER), hidden_dim, requires_grad=requires_grad)
    graph = HypothesisEvidenceGraphBuilder(graph_config(hidden_dim=hidden_dim)).build(evidence, hypotheses)
    output = HypothesisGraphOutput(
        hypothesis_features=hypotheses,
        hypothesis_mask=torch.ones(batch_size, len(HYPOTHESIS_ORDER), dtype=torch.bool),
        graph_batch=graph,
        graph_statistics=graph_statistics(graph),
        updated_node_features=graph.x_dict,
        refined_visual_semantic=evidence.visual_semantic,
        refined_visual_forensic=evidence.visual_forensic,
        refined_text_alignment=evidence.text_alignment,
        refined_text_context=evidence.text_context,
    )
    output.validate()
    return output


def full_text_graph_output(
    hidden_dim: int = 12,
    *,
    token_input_dim: int = 19,
    full_text_len: int = 20,
    graph_text_nodes: int = 5,
) -> HypothesisGraphOutput:
    evidence = synthetic_evidence(hidden_dim=hidden_dim)
    batch = evidence.text_context.features.shape[0]
    full_features = torch.randn(batch, full_text_len, token_input_dim)
    full_mask = torch.ones(batch, full_text_len, dtype=torch.bool)
    full_word_ids = torch.arange(full_text_len, dtype=torch.long).unsqueeze(0).expand(batch, -1).clone()
    features = torch.randn(batch, graph_text_nodes, hidden_dim)
    mask = torch.ones(batch, graph_text_nodes, dtype=torch.bool)
    source_indices = torch.arange(graph_text_nodes, dtype=torch.long).unsqueeze(0).expand(batch, -1).clone()
    word_ids = torch.arange(graph_text_nodes, dtype=torch.long).unsqueeze(0).expand(batch, -1).clone()
    text_context = replace(
        evidence.text_context,
        features=features,
        mask=mask,
        source_indices=source_indices,
        selection_scores=None,
        word_ids=word_ids,
        full_features=full_features,
        full_mask=full_mask,
        full_source_indices=torch.arange(full_text_len, dtype=torch.long).unsqueeze(0).expand(batch, -1).clone(),
        full_word_ids=full_word_ids,
    )
    evidence = replace(evidence, text_context=text_context)
    return graph_output_from_evidence(evidence, hidden_dim=hidden_dim)


def decision_batch(batch_size: int = 3, full_text_len: int = 5) -> dict:
    batch = synthetic_batch(batch_size=batch_size, text_len=full_text_len)
    batch["text_grounding_target"] = torch.zeros(batch_size, full_text_len, dtype=torch.long)
    batch["text_grounding_mask"] = torch.ones(batch_size, full_text_len, dtype=torch.bool)
    batch["text_word_ids"] = torch.arange(full_text_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).clone()
    if batch_size >= 3 and full_text_len >= 3:
        batch["text_grounding_target"][2, 0] = 1
        batch["text_grounding_target"][2, 2] = 1
    return batch

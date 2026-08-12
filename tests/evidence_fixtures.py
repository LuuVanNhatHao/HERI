from __future__ import annotations


def evidence_config(hidden_dim: int = 12) -> dict:
    return {
        "hidden_dim": hidden_dim,
        "activation": "gelu",
        "dropout": 0.0,
        "layer_norm": True,
        "input_dims": {},
        "visual_semantic": {"enabled": True, "max_nodes": 3, "budgeting": "spatial_balanced"},
        "visual_forensic": {"enabled": True, "max_nodes": 3, "budgeting": "spatial_balanced"},
        "face": {"enabled": True, "truncate": False, "budgeting": "ordered"},
        "text_alignment": {"enabled": True, "max_nodes": 3, "budgeting": "sequence_balanced"},
        "text_context": {"enabled": True, "max_nodes": 3, "budgeting": "sequence_balanced"},
        "cross_modal": {"enabled": True, "similarity": "cosine"},
    }

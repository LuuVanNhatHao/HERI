from __future__ import annotations

import torch

from heri.models.outputs import (
    BackboneOutputs,
    DINOv3Output,
    DeBERTaOutput,
    PatchGrid,
    SigLIP2Output,
)


def synthetic_batch(batch_size: int = 3, faces: int = 3, text_len: int = 5) -> dict:
    candidate_mask = torch.tensor(
        [[True, True, False], [True, False, False], [True, True, True]][:batch_size],
        dtype=torch.bool,
    )
    return {
        "id": list(range(batch_size)),
        "text": ["alpha beta gamma"] * batch_size,
        "fake_cls": ["orig", "face_swap", "face_attribute&text_swap"][:batch_size],
        "generator": [None, "InfoSwap", "HFGI"][:batch_size],
        "image": torch.rand(batch_size, 3, 384, 384),
        "binary_target": torch.tensor([0, 1, 1][:batch_size], dtype=torch.float32),
        "atomic_target": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]][:batch_size],
            dtype=torch.float32,
        ),
        "candidate_face_boxes": torch.tensor(
            [
                [[0, 0, 120, 120], [190, 190, 230, 230], [0, 0, 0, 0]],
                [[0, 0, 10, 10], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 120, 120], [200, 0, 380, 120], [0, 200, 120, 380]],
            ][:batch_size],
            dtype=torch.float32,
        ),
        "candidate_face_mask": candidate_mask,
        "fake_image_box_target": torch.tensor(
            [[0, 0, 0, 0], [0, 0, 10, 10], [200, 0, 380, 120]][:batch_size],
            dtype=torch.float32,
        ),
        "has_image_manipulation": torch.tensor([False, True, True][:batch_size]),
        "manipulated_face_target_index": torch.tensor([-1, 0, 1][:batch_size]),
        "fake_text_pos": torch.tensor(
            [[-1, -1], [-1, -1], [0, 2]][:batch_size],
            dtype=torch.long,
        ),
        "text_input_ids": torch.arange(batch_size * text_len).reshape(batch_size, text_len),
        "text_attention_mask": torch.ones(batch_size, text_len, dtype=torch.bool),
        "text_grounding_target": torch.tensor(
            [[0, 0, 0, -100, -100], [0, 0, 0, -100, -100], [1, 0, 1, 0, -100]][:batch_size],
            dtype=torch.long,
        ),
        "text_grounding_mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
            ][:batch_size],
            dtype=torch.bool,
        ),
        "text_word_ids": torch.tensor(
            [[0, 1, 2, -1, -1], [0, 1, 2, -1, -1], [0, 1, 2, 3, -1]][:batch_size],
            dtype=torch.long,
        ),
        "has_text_manipulation": torch.tensor([False, False, True][:batch_size]),
    }


def synthetic_features(batch_size: int = 3, text_len: int = 5) -> BackboneOutputs:
    torch.manual_seed(7)
    siglip_image_tokens = torch.randn(batch_size, 4, 11)
    siglip_text_tokens = torch.randn(batch_size, text_len, 13)
    dino_patch_tokens = torch.randn(batch_size, 4, 17)
    deberta_tokens = torch.randn(batch_size, text_len, 19)
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    return BackboneOutputs(
        siglip2=SigLIP2Output(
            image_tokens=siglip_image_tokens,
            image_mask=torch.ones(batch_size, 4, dtype=torch.bool),
            image_global=siglip_image_tokens.mean(dim=1),
            text_tokens=siglip_text_tokens,
            text_mask=text_mask,
            text_global=siglip_text_tokens.mean(dim=1),
            image_grid=PatchGrid(height=2, width=2, image_height=384, image_width=384),
        ),
        dinov3=DINOv3Output(
            patch_tokens=dino_patch_tokens,
            patch_mask=torch.ones(batch_size, 4, dtype=torch.bool),
            global_feature=dino_patch_tokens.mean(dim=1),
            patch_grid=PatchGrid(height=2, width=2, image_height=384, image_width=384),
        ),
        deberta=DeBERTaOutput(
            token_features=deberta_tokens,
            attention_mask=text_mask,
            global_feature=deberta_tokens.mean(dim=1),
        ),
    )

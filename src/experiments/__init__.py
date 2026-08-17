"""Production experiment orchestration and controlled-variant contracts."""

from .ablation import (
    HERA_ABLATION_VARIANTS,
    apply_hera_ablation_variant,
    get_hera_ablation_variant,
    validate_hera_checkpoint_variant,
    validate_hera_ablation_contract,
)

__all__ = [
    "HERA_ABLATION_VARIANTS",
    "apply_hera_ablation_variant",
    "get_hera_ablation_variant",
    "validate_hera_checkpoint_variant",
    "validate_hera_ablation_contract",
]

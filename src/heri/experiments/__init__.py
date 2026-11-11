"""Production experiment orchestration and controlled-variant contracts."""

from .ablation import (
    HERA_ABLATION_VARIANTS,
    apply_hera_ablation_variant,
    get_hera_ablation_variant,
    validate_hera_checkpoint_variant,
    validate_hera_ablation_contract,
)
from .cycle_study import (
    HERA_CYCLE_STUDY_VARIANTS,
    apply_hera_cycle_study_variant,
    cycle_study_protocol,
    get_hera_cycle_study_variant,
    validate_hera_cycle_study_contract,
)

__all__ = [
    "HERA_ABLATION_VARIANTS",
    "apply_hera_ablation_variant",
    "get_hera_ablation_variant",
    "validate_hera_checkpoint_variant",
    "validate_hera_ablation_contract",
    "HERA_CYCLE_STUDY_VARIANTS",
    "apply_hera_cycle_study_variant",
    "cycle_study_protocol",
    "get_hera_cycle_study_variant",
    "validate_hera_cycle_study_contract",
]

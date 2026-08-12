"""Validation helpers for Phase-6 routed evidence."""

from __future__ import annotations

from .outputs import RoutedEvidenceBatch


def validate_routed_batch(routed: RoutedEvidenceBatch) -> None:
    """Fail fast on routed evidence contract violations."""

    routed.validate()

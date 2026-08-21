"""Configured text normalization for language-specific binary benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BANGLA_NORMALIZER_PROVIDER = "csebuetnlp/normalizer"


@dataclass(frozen=True)
class BenchmarkTextNormalizer:
    """Pickle-safe callable implementing the configured normalization contract."""

    enabled: bool
    provider: str
    unicode_norm: str = "NFKC"
    punct_replacement: str | None = None
    url_replacement: str | None = None
    emoji_replacement: str | None = None
    apply_unicode_norm_last: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "BenchmarkTextNormalizer":
        """Build an explicit normalizer; omitted config means identity behavior."""

        values = dict(config or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            provider=str(values.get("provider", "none")),
            unicode_norm=str(values.get("unicode_norm", "NFKC")),
            punct_replacement=_optional_string(values.get("punct_replacement")),
            url_replacement=_optional_string(values.get("url_replacement")),
            emoji_replacement=_optional_string(values.get("emoji_replacement")),
            apply_unicode_norm_last=bool(
                values.get("apply_unicode_norm_last", True)
            ),
        )

    def __call__(self, text: str) -> str:
        """Normalize one text with the provider required by its pretrained model."""

        value = str(text)
        if not self.enabled:
            return value
        if self.provider != BANGLA_NORMALIZER_PROVIDER:
            raise ValueError(
                f"Unsupported benchmark text normalizer provider: {self.provider!r}"
            )
        try:
            from normalizer import normalize
        except ImportError as exc:
            raise RuntimeError(
                "BanglaBERT requires the pinned csebuetnlp normalizer. "
                "Install requirements.txt before preparing or training MultiBanFakeDetect."
            ) from exc
        normalized = normalize(
            value,
            unicode_norm=self.unicode_norm,
            punct_replacement=self.punct_replacement,
            url_replacement=self.url_replacement,
            emoji_replacement=self.emoji_replacement,
            apply_unicode_norm_last=self.apply_unicode_norm_last,
        )
        if not isinstance(normalized, str):
            raise TypeError(
                "Configured benchmark text normalizer must return str, "
                f"got {type(normalized).__name__}"
            )
        return normalized

    def signature(self) -> dict[str, Any]:
        """Return the stable preprocessing contract stored in checkpoints."""

        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "unicode_norm": self.unicode_norm,
            "punct_replacement": self.punct_replacement,
            "url_replacement": self.url_replacement,
            "emoji_replacement": self.emoji_replacement,
            "apply_unicode_norm_last": self.apply_unicode_norm_last,
        }


def text_normalization_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a dataset config into a canonical checkpoint signature."""

    text = config.get("dataset", {}).get("text", {})
    normalization = text.get("normalization") if isinstance(text, Mapping) else None
    return BenchmarkTextNormalizer.from_config(normalization).signature()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)

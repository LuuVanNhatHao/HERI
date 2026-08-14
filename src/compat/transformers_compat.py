"""Centralized Hugging Face Transformers compatibility helpers."""

from __future__ import annotations

import platform
import inspect
from dataclasses import dataclass
from typing import Any

import torch
from packaging.version import InvalidVersion, Version


SUPPORTED_TRANSFORMERS_SPEC = ">=4.57.2,<5.0"


class TransformersCompatibilityError(RuntimeError):
    """Raised when a requested Transformers feature is unavailable."""


@dataclass(frozen=True)
class TransformersDiagnostic:
    """Diagnostic details for compatibility failures."""

    python_version: str
    transformers_version: str | None
    requested_feature: str
    reason: str

    def message(self) -> str:
        """Return a readable diagnostic message."""

        return (
            "Transformers compatibility failure: "
            f"python={self.python_version}; "
            f"transformers={self.transformers_version}; "
            f"requested_feature={self.requested_feature}; "
            f"reason={self.reason}"
        )


def _transformers_version() -> str | None:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return None


def raise_transformers_diagnostic(requested_feature: str, reason: str) -> None:
    """Raise a Constitution-compliant compatibility diagnostic."""

    diagnostic = TransformersDiagnostic(
        python_version=platform.python_version(),
        transformers_version=_transformers_version(),
        requested_feature=requested_feature,
        reason=reason,
    )
    raise TransformersCompatibilityError(diagnostic.message())


def _ensure_supported_transformers_version(requested_feature: str) -> None:
    version = _transformers_version()
    if version is None:
        return
    try:
        parsed = Version(version)
    except InvalidVersion:
        return
    if parsed < Version("4.57.2") or parsed >= Version("5.0"):
        raise_transformers_diagnostic(
            requested_feature,
            f"unsupported transformers version {version!r}; install requirements.txt "
            f"with transformers{SUPPORTED_TRANSFORMERS_SPEC}",
        )


def load_fast_tokenizer(
    *,
    name: str,
    use_fast: bool = True,
    local_files_only: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a Hugging Face fast tokenizer with explicit diagnostics."""

    if not use_fast:
        raise_transformers_diagnostic(
            "fast tokenizer word_ids()",
            "tokenizer.use_fast was false; DGM4 grounding requires word_ids()",
        )
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise_transformers_diagnostic("AutoTokenizer", repr(exc))
    _ensure_supported_transformers_version("AutoTokenizer.from_pretrained")
    kwargs: dict[str, Any] = {"use_fast": True, "local_files_only": local_files_only}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    except Exception as exc:
        raise_transformers_diagnostic(
            "AutoTokenizer.from_pretrained",
            f"failed to load {name!r}: {exc!r}",
        )
    if not getattr(tokenizer, "is_fast", False):
        raise_transformers_diagnostic(
            "fast tokenizer word_ids()",
            f"loaded tokenizer {name!r} is not fast",
        )
    return tokenizer


def _from_pretrained_kwargs(
    cls: Any,
    *,
    requested_feature: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Validate kwargs for a Transformers ``from_pretrained`` call.

    The helper refuses unsupported kwargs when the signature exposes enough
    information to decide. If the installed Transformers object accepts
    ``**kwargs``, the kwargs are passed through and any incompatibility is
    surfaced by Transformers itself with a diagnostic wrapper.
    """

    try:
        signature = inspect.signature(cls.from_pretrained)
    except (TypeError, ValueError):
        return kwargs
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    supported = set(parameters)
    unsupported = sorted(key for key in kwargs if key not in supported)
    if unsupported:
        raise_transformers_diagnostic(
            requested_feature,
            f"unsupported from_pretrained kwargs for {cls.__name__}: {unsupported}",
        )
    return kwargs


def precision_to_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    """Map a YAML dtype/precision string to a torch dtype for model loading."""

    if dtype_name is None or dtype_name in {"auto", "none"}:
        return None
    normalized = dtype_name.lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise_transformers_diagnostic("torch dtype", f"unknown dtype value {dtype_name!r}")


def load_auto_model(
    *,
    name: str,
    stage: str,
    revision: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    torch_dtype: str | torch.dtype | None = None,
    attn_implementation: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a Hugging Face AutoModel with centralized diagnostics."""

    try:
        from transformers import AutoModel
    except Exception as exc:
        raise_transformers_diagnostic(f"{stage}: AutoModel", repr(exc))
    _ensure_supported_transformers_version(f"{stage}: AutoModel.from_pretrained")
    dtype = precision_to_torch_dtype(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    _from_pretrained_kwargs(
        AutoModel,
        requested_feature=f"{stage}: AutoModel.from_pretrained",
        kwargs=kwargs,
    )
    try:
        return AutoModel.from_pretrained(name, **kwargs)
    except TypeError as exc:
        raise_transformers_diagnostic(
            f"{stage}: AutoModel.from_pretrained",
            f"failed to load {name!r} with kwargs={sorted(kwargs)}: {exc!r}",
        )
    except Exception as exc:
        raise_transformers_diagnostic(
            f"{stage}: AutoModel.from_pretrained",
            f"failed to load {name!r}: {exc!r}",
        )


def load_auto_tokenizer(
    *,
    name: str,
    stage: str,
    use_fast: bool | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a Hugging Face tokenizer with centralized diagnostics."""

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise_transformers_diagnostic(f"{stage}: AutoTokenizer", repr(exc))
    _ensure_supported_transformers_version(f"{stage}: AutoTokenizer.from_pretrained")
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if use_fast is not None:
        kwargs["use_fast"] = use_fast
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    _from_pretrained_kwargs(
        AutoTokenizer,
        requested_feature=f"{stage}: AutoTokenizer.from_pretrained",
        kwargs=kwargs,
    )
    try:
        return AutoTokenizer.from_pretrained(name, **kwargs)
    except Exception as exc:
        raise_transformers_diagnostic(
            f"{stage}: AutoTokenizer.from_pretrained",
            f"failed to load {name!r}: {exc!r}",
        )


def load_auto_processor(
    *,
    name: str,
    stage: str,
    revision: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a Hugging Face AutoProcessor with centralized diagnostics."""

    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise_transformers_diagnostic(f"{stage}: AutoProcessor", repr(exc))
    _ensure_supported_transformers_version(f"{stage}: AutoProcessor.from_pretrained")
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    _from_pretrained_kwargs(
        AutoProcessor,
        requested_feature=f"{stage}: AutoProcessor.from_pretrained",
        kwargs=kwargs,
    )
    try:
        return AutoProcessor.from_pretrained(name, **kwargs)
    except Exception as exc:
        raise_transformers_diagnostic(
            f"{stage}: AutoProcessor.from_pretrained",
            f"failed to load {name!r}: {exc!r}",
        )


def set_gradient_checkpointing(
    module: Any,
    *,
    enabled: bool,
    stage: str,
    use_reentrant: bool = False,
) -> bool:
    """Enable/disable HF gradient checkpointing through centralized compatibility logic."""

    target = getattr(module, "model", module)
    if not enabled:
        if hasattr(target, "gradient_checkpointing_disable"):
            target.gradient_checkpointing_disable()
            return False
        return False
    if hasattr(target, "gradient_checkpointing_enable"):
        try:
            target.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": bool(use_reentrant)}
            )
        except TypeError as exc:
            raise_transformers_diagnostic(
                f"{stage}: gradient checkpointing",
                "the configured Transformers model cannot select the required "
                f"checkpoint implementation: {exc!r}",
            )
        return True
    if hasattr(target, "enable_gradient_checkpointing"):
        target.enable_gradient_checkpointing()
        return True
    raise_transformers_diagnostic(
        f"{stage}: gradient checkpointing",
        "configured runtime.gradient_checkpointing requested support, but the model exposes no supported API",
    )

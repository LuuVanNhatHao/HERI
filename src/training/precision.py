"""Runtime precision helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


class PrecisionManager:
    """Own AMP/autocast behavior and report actual runtime precision."""

    def __init__(self, *, precision: str, device: torch.device) -> None:
        self.requested = precision.lower()
        self.device = device
        self.actual = self._resolve_actual_precision()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.actual == "fp16")

    def _resolve_actual_precision(self) -> str:
        if self.requested in {"fp32", "float32"}:
            return "fp32"
        if self.device.type != "cuda":
            raise RuntimeError(
                f"Requested precision {self.requested!r} requires CUDA; "
                "local CPU execution must use fp32 in YAML"
            )
        if self.requested in {"bf16", "bfloat16"}:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 was requested but CUDA BF16 support is unavailable")
            return "bf16"
        if self.requested in {"fp16", "float16"}:
            return "fp16"
        raise ValueError(f"Unknown runtime precision: {self.requested!r}")

    def autocast(self) -> ContextManager:
        """Return the correct autocast context."""

        if self.actual == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.actual == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate with scaler only when required."""

        if self.actual == "fp16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply optimizer step under the active precision mode."""

        if self.actual == "fp16":
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

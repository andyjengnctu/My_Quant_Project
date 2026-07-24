"""Shared PyTorch device, precision, and determinism helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

SUPPORTED_TORCH_DEVICES = ("auto", "cpu", "cuda")
SUPPORTED_MIXED_PRECISION_DTYPES = ("auto", "float16", "bfloat16")


@dataclass(frozen=True)
class TorchExecutionPlan:
    requested_device: str
    device_type: str
    device: Any
    mixed_precision_requested: bool
    mixed_precision_enabled: bool
    autocast_dtype_name: str
    deterministic_algorithms: bool
    allow_tf32: bool

    def as_manifest_payload(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.device_type,
            "mixed_precision_requested": bool(self.mixed_precision_requested),
            "mixed_precision_enabled": bool(self.mixed_precision_enabled),
            "autocast_dtype": self.autocast_dtype_name,
            "deterministic_algorithms": bool(self.deterministic_algorithms),
            "allow_tf32": bool(self.allow_tf32),
        }


def _normalize_choice(value: str, *, choices: tuple[str, ...], field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{field_name} 不合法: {value!r}；可用值: {', '.join(choices)}")
    return normalized


def resolve_torch_execution_plan(
    torch: Any,
    *,
    requested_device: str,
    mixed_precision: bool,
    mixed_precision_dtype: str,
    deterministic_algorithms: bool,
    allow_tf32: bool,
) -> TorchExecutionPlan:
    requested = _normalize_choice(
        requested_device,
        choices=SUPPORTED_TORCH_DEVICES,
        field_name="torch device",
    )
    dtype_request = _normalize_choice(
        mixed_precision_dtype,
        choices=SUPPORTED_MIXED_PRECISION_DTYPES,
        field_name="mixed precision dtype",
    )
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("要求 CUDA，但目前 PyTorch 無法使用 CUDA")
    resolved_type = "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"
    device = torch.device(resolved_type)

    enabled = bool(mixed_precision and resolved_type == "cuda")
    dtype_name = "float32"
    if enabled:
        if dtype_request == "auto":
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            dtype_name = "bfloat16" if callable(is_bf16_supported) and bool(is_bf16_supported()) else "float16"
        else:
            dtype_name = dtype_request

    torch.use_deterministic_algorithms(bool(deterministic_algorithms))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)

    return TorchExecutionPlan(
        requested_device=requested,
        device_type=resolved_type,
        device=device,
        mixed_precision_requested=bool(mixed_precision),
        mixed_precision_enabled=enabled,
        autocast_dtype_name=dtype_name,
        deterministic_algorithms=bool(deterministic_algorithms),
        allow_tf32=bool(allow_tf32),
    )


def autocast_context(torch: Any, plan: TorchExecutionPlan):
    if not plan.mixed_precision_enabled:
        return nullcontext()
    dtype = torch.float16 if plan.autocast_dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type=plan.device_type, dtype=dtype, enabled=True)


def build_grad_scaler(torch: Any, plan: TorchExecutionPlan):
    enabled = bool(
        plan.mixed_precision_enabled and plan.autocast_dtype_name == "float16"
    )
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def seed_torch(torch: Any, *, seed: int, plan: TorchExecutionPlan) -> None:
    torch.manual_seed(int(seed))
    if plan.device_type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


__all__ = [
    "SUPPORTED_MIXED_PRECISION_DTYPES",
    "SUPPORTED_TORCH_DEVICES",
    "TorchExecutionPlan",
    "autocast_context",
    "build_grad_scaler",
    "resolve_torch_execution_plan",
    "seed_torch",
]

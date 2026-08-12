"""Shared lightweight runtime helpers for breakout quality model factories."""

from __future__ import annotations


def require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError as exc:
        raise RuntimeError("breakout quality DL 訓練需要 PyTorch；請先安裝 torch") from exc
    return torch, nn


def count_trainable_parameters(model) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["count_trainable_parameters", "require_torch"]

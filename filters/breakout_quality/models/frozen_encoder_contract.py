"""Reusable frozen-encoder training contract; never registers model parameters."""

from __future__ import annotations


def freeze_encoder_parameters(encoder) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()


def train_with_frozen_encoder(*, parent_train, encoder, mode: bool):
    result = parent_train(mode)
    encoder.eval()
    return result


__all__ = ["freeze_encoder_parameters", "train_with_frozen_encoder"]

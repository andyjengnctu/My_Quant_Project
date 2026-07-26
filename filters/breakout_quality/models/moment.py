"""Frozen MOMENT-1-base encoder with a single supervised linear probe."""

from __future__ import annotations

from typing import Mapping

from filters.breakout_quality.moment_contract import (
    MOMENT_INSTALL_HINT,
    MOMENT_PACKAGE_NAME,
    MOMENT_PACKAGE_VERSION,
    moment_model_config,
    require_moment_pipeline_class,
)


MOMENT_ENCODER_FORWARD_BATCH_SIZE = 32


def _build_encoder():
    MOMENTPipeline = require_moment_pipeline_class()
    encoder = MOMENTPipeline(
        moment_model_config(),
        model_kwargs={"task_name": "embedding"},
    )
    encoder.init()
    return encoder


def build_moment_frozen_linear(
    nn,
    torch,
    *,
    feature_count: int,
    context_count: int,
    spec,
    pretrained_encoder_state: Mapping[str, object] | None = None,
):
    del context_count
    if bool(spec.use_dataset_context):
        raise ValueError("MOMENT frozen linear 不得使用 dataset context")
    if int(feature_count) < 1:
        raise ValueError("MOMENT feature_count 必須 >= 1")
    input_length = int(spec.moment_input_length or 0)
    embedding_dim = int(spec.moment_embedding_dim or 0)
    patch_length = int(spec.moment_patch_length or 0)
    if input_length < 1 or patch_length < 1 or input_length % patch_length != 0:
        raise ValueError("MOMENT input_length 必須是 patch_length 的正整數倍")
    if embedding_dim < 1:
        raise ValueError("MOMENT embedding_dim 必須 >= 1")

    encoder = _build_encoder()

    class MomentFrozenLinearClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(int(feature_count) * embedding_dim, 2)
            if pretrained_encoder_state is not None:
                self.encoder.load_state_dict(dict(pretrained_encoder_state), strict=True)
            self.freeze_encoder()

        def freeze_encoder(self) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

        def train(self, mode: bool = True):
            super().train(mode)
            self.encoder.eval()
            return self

        def forward(self, sequence, context):
            del context
            if sequence.ndim != 3:
                raise ValueError(
                    f"MOMENT sequence 必須是 [batch,time,feature]: {sequence.shape}"
                )
            if int(sequence.shape[2]) != int(feature_count):
                raise ValueError(
                    "MOMENT feature_count 不一致: "
                    f"expected={feature_count}, actual={sequence.shape[2]}"
                )
            batch_size = int(sequence.shape[0])
            channel_count = int(sequence.shape[2])
            encoder_input = sequence.transpose(1, 2)
            if int(encoder_input.shape[2]) != input_length:
                encoder_input = torch.nn.functional.interpolate(
                    encoder_input,
                    size=input_length,
                    mode="linear",
                    align_corners=False,
                )
            encoded_chunks = []
            with torch.no_grad():
                for start in range(0, batch_size, MOMENT_ENCODER_FORWARD_BATCH_SIZE):
                    encoder_chunk = encoder_input[
                        start : start + MOMENT_ENCODER_FORWARD_BATCH_SIZE
                    ]
                    outputs = self.encoder.embed(
                        x_enc=encoder_chunk,
                        reduction="none",
                    )
                    encoded_chunk = outputs.embeddings
                    expected_patches = input_length // patch_length
                    expected_shape = (
                        int(encoder_chunk.shape[0]),
                        channel_count,
                        expected_patches,
                        embedding_dim,
                    )
                    if tuple(encoded_chunk.shape) != expected_shape:
                        raise ValueError(
                            "MOMENT encoder output shape 不一致: "
                            f"expected={expected_shape}, actual={tuple(encoded_chunk.shape)}"
                        )
                    encoded_chunks.append(encoded_chunk.mean(dim=2))
            if not encoded_chunks:
                raise ValueError("MOMENT encoder input 不得為空")
            encoded = torch.cat(encoded_chunks, dim=0)
            representation = encoded.reshape(batch_size, channel_count * embedding_dim)
            return self.classifier(representation)

    return MomentFrozenLinearClassifier()


__all__ = [
    "MOMENT_ENCODER_FORWARD_BATCH_SIZE",
    "MOMENT_INSTALL_HINT",
    "MOMENT_PACKAGE_NAME",
    "MOMENT_PACKAGE_VERSION",
    "build_moment_frozen_linear",
    "require_moment_pipeline_class",
]

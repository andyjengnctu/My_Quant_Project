"""Gated recurrent shared Safety/MFE backbone for breakout-quality research."""

from __future__ import annotations


def build_gru_shared_safety_mfe(
    nn,
    torch,
    *,
    feature_count: int,
    context_count: int,
    spec,
):
    """Build a parameter-matched GRU with independent Safety/MFE heads."""

    del context_count
    hidden_size = int(spec.gru_hidden_size or 0)
    num_layers = int(spec.gru_layers or 0)
    bidirectional = bool(spec.gru_bidirectional)
    pooling = str(spec.gru_pooling or "").strip().lower()
    if hidden_size < 1 or num_layers < 1:
        raise ValueError("GRU hidden size/layers 必須 >= 1")
    if bidirectional:
        raise ValueError("MR-13BG GRU control 固定為單向 recurrent state")
    if pooling != "final_state":
        raise ValueError(f"不支援的 GRU pooling: {pooling!r}")

    from filters.breakout_quality.models.architectures import get_architecture_descriptor

    descriptor = get_architecture_descriptor(str(spec.architecture))
    use_outer_autocast = descriptor.has_capability("recurrent_outer_autocast")
    guarded_retry = descriptor.has_capability("same_batch_fp32_nonfinite_retry")
    if guarded_retry and not use_outer_autocast:
        raise ValueError("same-batch FP32 retry 只允許outer-autocast recurrent architecture")

    class GRUSharedSafetyMFE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=int(feature_count),
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
            self.raw_safety_classifier = nn.Linear(hidden_size, 2)
            self.raw_mfe_classifier = nn.Linear(hidden_size, 2)
            self.same_batch_fp32_nonfinite_retry = bool(guarded_retry)
            self.recurrent_outer_autocast = bool(use_outer_autocast)

        def _encode_native(self, x):
            if x.ndim != 3:
                raise ValueError(f"GRU input 必須是 [batch,time,feature]，收到 shape={tuple(x.shape)}")
            _outputs, hidden = self.gru(x)
            return hidden[-1]

        def _encode_fp32(self, x):
            if x.ndim != 3:
                raise ValueError(f"GRU input 必須是 [batch,time,feature]，收到 shape={tuple(x.shape)}")
            _outputs, hidden = self.gru(x.float())
            return hidden[-1]

        def encode(self, x):
            if self.recurrent_outer_autocast:
                return self._encode_native(x)
            # v2 preserves the accepted FP32 recurrent island.  v3 intentionally
            # stays in the outer BF16 autocast and relies on trainer-owned guarded
            # same-batch FP32 retry only when a non-finite step is detected.
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self._encode_fp32(x)

        def forward_safety_mfe_heads(self, x, context):
            del context
            if self.recurrent_outer_autocast:
                shared_encoded = self._encode_native(x)
                return (
                    self.raw_safety_classifier(shared_encoded),
                    self.raw_mfe_classifier(shared_encoded),
                )
            with torch.autocast(device_type=x.device.type, enabled=False):
                shared_encoded = self._encode_fp32(x)
                return (
                    self.raw_safety_classifier(shared_encoded),
                    self.raw_mfe_classifier(shared_encoded),
                )

        def forward_output_head(self, x, context, output_head: str):
            head = str(output_head).strip().lower()
            safety_logits, mfe_logits = self.forward_safety_mfe_heads(x, context)
            if head in {"primary", "mfe", "primary_mfe", "conditional_mfe", "raw_mfe", "final"}:
                return mfe_logits
            if head in {"conditional_both", "both"}:
                return torch.cat([safety_logits, mfe_logits], dim=1)
            if head in {"raw_safety", "safety_condition", "safety"}:
                return safety_logits
            raise ValueError(f"未知 GRU output head: {output_head!r}")

        def forward(self, x, context):
            return self.forward_safety_mfe_heads(x, context)[1]

    return GRUSharedSafetyMFE()


__all__ = ["build_gru_shared_safety_mfe"]

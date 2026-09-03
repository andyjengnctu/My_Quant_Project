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

        def encode(self, x):
            if x.ndim != 3:
                raise ValueError(f"GRU input 必須是 [batch,time,feature]，收到 shape={tuple(x.shape)}")
            _outputs, hidden = self.gru(x)
            return hidden[-1]

        def forward_safety_mfe_heads(self, x, context):
            del context
            shared_encoded = self.encode(x)
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

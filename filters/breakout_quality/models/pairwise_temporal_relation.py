"""Explicit PIT-safe stock-bar relation bias for Safety temporal attention.

This module owns the reusable relation primitive and its execution-only memory
chunking.  Experiment identity and training semantics remain declarative
elsewhere.  The representation is derived on demand from the canonical first
five stock OHLCV channels of the existing 300x10 sequence; no extra persistent
feature artifact is created.
"""

from __future__ import annotations


EXPECTED_RELATION_FEATURES = (
    "delta_log_close",
    "delta_log_high",
    "delta_log_low",
    "delta_normalized_log_volume",
    "normalized_time_distance",
)

# Execution-only cap: bound the temporary [batch,time,time] relation matrices.
# It does not alter relation semantics, model identity, ordering, or training
# samples; only the batch dimension is evaluated in exact independent chunks.
_MAX_PAIR_CELLS_PER_ATTENTION_CHUNK = 12_000_000


def build_pairwise_temporal_relation_bias(
    nn,
    torch,
    *,
    feature_count: int,
    spec,
):
    """Build the fixed 5D -> 8D -> scalar explicit relation-bias primitive."""

    relation_features = tuple(
        str(value) for value in spec.pairwise_temporal_relation_features
    )
    hidden_dim = int(spec.pairwise_temporal_relation_hidden_dim or 0)
    if int(feature_count) < 5:
        raise ValueError("Pairwise temporal relation需要canonical stock OHLCV前5個channels")
    if relation_features != EXPECTED_RELATION_FEATURES:
        raise ValueError(
            "Pairwise temporal relation feature contract不一致: "
            f"expected={EXPECTED_RELATION_FEATURES}, actual={relation_features}"
        )
    if hidden_dim < 1:
        raise ValueError("Pairwise temporal relation hidden dim必須為正整數")

    class PairwiseTemporalRelationBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.relation_mlp = nn.Sequential(
                nn.Linear(len(EXPECTED_RELATION_FEATURES), hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )

        def _bar_features(self, x):
            if x.ndim != 3 or int(x.shape[2]) < 5:
                raise ValueError(
                    "Pairwise temporal relation input必須是[batch,time,feature>=5]"
                )
            time_count = int(x.shape[1])
            if time_count < 2:
                raise ValueError("Pairwise temporal relation至少需要2個time steps")

            with torch.no_grad():
                stock = x[:, :, :5].float()
                # 1 + normalized price equals P/anchor_close and is strictly
                # positive for valid canonical OHLCV.  Clamp only protects the
                # logarithm from impossible numerical underflow/corrupt input.
                tiny = torch.finfo(torch.float32).tiny
                log_high = torch.log(torch.clamp(stock[:, :, 1] + 1.0, min=tiny))
                log_low = torch.log(torch.clamp(stock[:, :, 2] + 1.0, min=tiny))
                log_close = torch.log(torch.clamp(stock[:, :, 3] + 1.0, min=tiny))
                volume = stock[:, :, 4]
                time_axis = torch.linspace(
                    0.0,
                    1.0,
                    steps=time_count,
                    device=x.device,
                    dtype=torch.float32,
                ).unsqueeze(0).expand(int(x.shape[0]), -1)
                bar_features = torch.stack(
                    (log_close, log_high, log_low, volume, time_axis), dim=2
                )
                if not bool(torch.isfinite(bar_features).all()):
                    raise ValueError("Pairwise temporal relation不得含non-finite value")
            return bar_features

        def build_relation_features(self, x, *, dtype=None):
            """Return ``[B,T,T,5]`` signed i-minus-j relation primitives."""

            bar_features = self._bar_features(x)
            relation = bar_features.unsqueeze(2) - bar_features.unsqueeze(1)
            target_dtype = x.dtype if dtype is None else dtype
            return relation.to(dtype=target_dtype)

        def forward(self, x, *, dtype=None):
            """Return scalar ``[B,T,T]`` bias without materializing ``T*T*8``.

            For first-layer weight ``w_h``, ``w_h @ (r_i-r_j)+b_h`` equals
            ``(w_h@r_i) - (w_h@r_j) + b_h``.  Applying the fixed ReLU and output
            projection one hidden unit at a time preserves the declared MLP while
            keeping only one pairwise matrix live at once.
            """

            bar_features = self._bar_features(x)
            first = self.relation_mlp[0]
            output = self.relation_mlp[2]
            # Pointwise FP32 arithmetic keeps the deterministic relation primitive
            # numerically stable under the outer mixed-precision training context.
            projected = None
            for feature_index in range(len(EXPECTED_RELATION_FEATURES)):
                term = bar_features[:, :, feature_index].unsqueeze(2) * first.weight[
                    :, feature_index
                ].view(1, 1, -1)
                projected = term if projected is None else projected + term

            relation_bias = None
            for hidden_index in range(hidden_dim):
                pairwise_hidden = (
                    projected[:, :, hidden_index].unsqueeze(2)
                    - projected[:, :, hidden_index].unsqueeze(1)
                    + first.bias[hidden_index]
                )
                contribution = output.weight[0, hidden_index] * torch.relu(
                    pairwise_hidden
                )
                relation_bias = (
                    contribution
                    if relation_bias is None
                    else relation_bias + contribution
                )
            target_dtype = x.dtype if dtype is None else dtype
            return relation_bias.to(dtype=target_dtype)

        def scaled_dot_product_attention(self, query, key, value, x):
            """Apply relation-biased attention with execution-only batch chunking."""

            if query.ndim != 4 or int(query.shape[1]) != 1:
                raise ValueError("Pairwise relation attention要求single-head [B,1,T,C] query")
            if query.shape[:3] != key.shape[:3] or query.shape[:3] != value.shape[:3]:
                raise ValueError("Pairwise relation attention Q/K/V shape不一致")
            if int(x.shape[0]) != int(query.shape[0]) or int(x.shape[1]) != int(
                query.shape[2]
            ):
                raise ValueError("Pairwise relation attention sequence與Q/K/V shape不一致")

            time_count = int(query.shape[2])
            max_chunk_batch = max(
                1,
                int(_MAX_PAIR_CELLS_PER_ATTENTION_CHUNK) // (time_count * time_count),
            )
            contexts = []
            for start in range(0, int(query.shape[0]), max_chunk_batch):
                end = min(int(query.shape[0]), start + max_chunk_batch)
                attention_bias = self(
                    x[start:end], dtype=query.dtype
                ).unsqueeze(1)
                contexts.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        query[start:end],
                        key[start:end],
                        value[start:end],
                        attn_mask=attention_bias,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                )
            return torch.cat(contexts, dim=0)

    return PairwiseTemporalRelationBias()


__all__ = [
    "EXPECTED_RELATION_FEATURES",
    "build_pairwise_temporal_relation_bias",
]

"""Full-resolution day-token Transformer for breakout-quality Safety/MFE research."""

from __future__ import annotations

from filters.breakout_quality.models.patch_token_joint_min import (
    build_patch_token_encoder_base,
)


def build_day_token_transformer_shared_safety_mfe(
    nn,
    torch,
    *,
    feature_count: int,
    context_count: int,
    spec,
):
    """Build a shared full-day-token Transformer with independent Safety/MFE heads.

    The token size is one trading day, so no temporal patch compression occurs.
    Both heads consume the same mean-pooled globally self-attended latent, matching
    MR-13AO's shared-encoder / independent-head topology.
    """

    del context_count
    encoder_base, embedding_dim, dropout = build_patch_token_encoder_base(
        nn,
        torch,
        feature_count=int(feature_count),
        spec=spec,
        contract_name="Day-Token Transformer",
    )

    class DayTokenTransformerSharedSafetyMFE(encoder_base):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.raw_safety_classifier = nn.Linear(embedding_dim, 2)
            self.raw_mfe_classifier = nn.Linear(embedding_dim, 2)

        def forward_safety_mfe_heads(self, x, context):
            del context
            shared_encoded = self.dropout(self.encode(x))
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
            raise ValueError(f"未知Day-Token Transformer output head: {output_head!r}")

        def forward(self, x, context):
            return self.forward_safety_mfe_heads(x, context)[1]

    return DayTokenTransformerSharedSafetyMFE()


__all__ = ["build_day_token_transformer_shared_safety_mfe"]

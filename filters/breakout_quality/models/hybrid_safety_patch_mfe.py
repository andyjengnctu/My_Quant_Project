
"""Dual-encoder Safety-Patch / Conditional-MFE-InceptionTime model."""

from __future__ import annotations

from filters.breakout_quality.models.architectures import INCEPTION_TIME_SHARED_SAFETY_MFE_V1
from filters.breakout_quality.models.inception_time import build_inception_time
from filters.breakout_quality.models.patch_token_joint_min import build_patch_token_encoder_base
from filters.breakout_quality.models.spec_registry import get_model_spec


def build_safety_patch_mfe_inception(nn, torch, *, feature_count: int, context_count: int, spec):
    """Build an independent Patch Safety encoder and AO-form Inception MFE encoder.

    The AO Inception module is constructed first so same-seed MFE parameters and its
    stochastic path remain directly comparable with the AO control.  Safety uses the
    frozen historical 9F/MR-13Z Patch Transformer recipe declared by ``spec`` and has
    no gradient path into the Inception encoder.  Conversely, Conditional-MFE loss has
    no gradient path into the Patch encoder.
    """

    if bool(spec.use_dataset_context):
        raise ValueError("Safety-Patch/MFE-Inception hybrid 必須停用 Dataset event context")

    ao_spec = get_model_spec(INCEPTION_TIME_SHARED_SAFETY_MFE_V1)
    mfe_model = build_inception_time(
        nn,
        torch,
        feature_count=int(feature_count),
        context_count=int(context_count),
        spec=ao_spec,
    )
    patch_encoder_base, embedding_dim, _dropout = build_patch_token_encoder_base(
        nn,
        torch,
        feature_count=int(feature_count),
        spec=spec,
        contract_name="Safety-Patch/MFE-Inception hybrid",
    )

    class SafetyPatchMFEInception(nn.Module):
        def __init__(self):
            super().__init__()
            self.mfe_model = mfe_model
            self.safety_encoder = patch_encoder_base()
            self.raw_safety_classifier = nn.Linear(int(embedding_dim), 2)

        def forward_safety_mfe_heads(self, x, context):
            # Run MFE first: at equal seed this preserves AO's MFE-side dropout RNG
            # before the independent Patch branch consumes any stochastic draws.
            _unused_ao_safety, mfe_logits = self.mfe_model.forward_safety_mfe_heads(x, context)
            safety_encoded = self.safety_encoder.encode(x)
            safety_logits = self.raw_safety_classifier(safety_encoded)
            return safety_logits, mfe_logits

        def forward_output_head(self, x, context, output_head: str):
            head = str(output_head).strip().lower()
            if head in {"primary", "mfe", "primary_mfe", "conditional_mfe", "raw_mfe", "final"}:
                return self.forward_safety_mfe_heads(x, context)[1]
            if head in {"raw_safety", "safety", "safety_condition"}:
                return self.forward_safety_mfe_heads(x, context)[0]
            if head in {"conditional_both", "both"}:
                safety_logits, mfe_logits = self.forward_safety_mfe_heads(x, context)
                return torch.cat([safety_logits, mfe_logits], dim=1)
            raise ValueError(f"未知Safety-Patch/MFE-Inception output head: {output_head!r}")

        def forward(self, x, context):
            return self.forward_safety_mfe_heads(x, context)[1]

    return SafetyPatchMFEInception()


__all__ = ["build_safety_patch_mfe_inception"]

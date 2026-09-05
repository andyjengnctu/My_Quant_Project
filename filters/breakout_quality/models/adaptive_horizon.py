"""Adaptive future-horizon Safety branch for shared breakout-quality encoders.

The branch predicts a complete 1..H Safety trajectory from the canonical shared
latent and learns a sample-specific soft horizon mixture.  The mixture only adds
an initially-zero residual to the canonical AO Safety logits, so same-seed step-0
behavior remains exact while auxiliary trajectory supervision can reshape the
shared representation.
"""

from __future__ import annotations


def build_adaptive_horizon_safety_branch(nn, torch, *, latent_width: int, spec):
    horizon_bars = int(spec.adaptive_safety_horizon_bars or 0)
    if horizon_bars < 2:
        raise ValueError("adaptive Safety branch需要至少2個future horizons")
    if not bool(spec.adaptive_safety_zero_init_residual):
        raise ValueError("adaptive Safety第一版必須使用zero-init residual")

    class AdaptiveHorizonSafetyBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.horizon_bars = int(horizon_bars)
            self.horizon_classifier = nn.Linear(int(latent_width), 2 * self.horizon_bars)
            self.horizon_gate = nn.Linear(int(latent_width), self.horizon_bars)
            # Uniform horizon weighting is the neutral initial state.  The gate only
            # starts specializing after the zero-init residual gain moves away from 0.
            nn.init.zeros_(self.horizon_gate.weight)
            nn.init.zeros_(self.horizon_gate.bias)
            self.residual_gain = nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward(self, latent):
            if latent.ndim != 2 or int(latent.shape[1]) != int(latent_width):
                raise ValueError("adaptive Safety latent shape不一致")
            horizon_logits = self.horizon_classifier(latent).reshape(
                int(latent.shape[0]), self.horizon_bars, 2
            )
            gate_logits = self.horizon_gate(latent)
            horizon_weights = torch.softmax(gate_logits.float(), dim=1).to(latent.dtype)
            horizon_margin = (
                horizon_logits.float()[:, :, 1] - horizon_logits.float()[:, :, 0]
            )
            weighted_margin = torch.sum(
                horizon_weights.float() * horizon_margin, dim=1
            )
            residual_margin = self.residual_gain * weighted_margin
            residual_logits = torch.stack(
                [-0.5 * residual_margin, 0.5 * residual_margin], dim=1
            ).to(latent.dtype)
            return residual_logits, horizon_logits, horizon_weights

    return AdaptiveHorizonSafetyBranch()


__all__ = ["build_adaptive_horizon_safety_branch"]

"""Adaptive look-back Safety residual for AO-style shared InceptionTime models."""

from __future__ import annotations


def build_adaptive_input_context_safety_branch(nn, torch, *, latent_width: int, spec):
    context_bars = tuple(int(value) for value in (spec.adaptive_input_context_bars or ()))
    if len(context_bars) < 2:
        raise ValueError("adaptive input context至少需要2個候選look-back windows")
    if tuple(sorted(set(context_bars))) != context_bars:
        raise ValueError("adaptive input context bars必須嚴格遞增且不得重複")
    max_window = int(spec.input_window_bars or 0)
    if max_window < 1 or int(context_bars[-1]) != max_window:
        raise ValueError("adaptive input context最後一個window必須等於architecture max input window")
    if any(value < 1 or value > max_window for value in context_bars):
        raise ValueError("adaptive input context bars必須落在合法input window內")
    if not bool(spec.adaptive_input_context_stop_gradient):
        raise ValueError("adaptive input context第一版固定使用stop-gradient auxiliary views")
    if not bool(spec.adaptive_input_context_zero_init_residual):
        raise ValueError("adaptive input context第一版固定使用zero-init Safety residual")

    class AdaptiveInputContextSafetyBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_bars = context_bars
            self.gate = nn.Linear(int(latent_width), len(context_bars), bias=True)
            self.residual_projection = nn.Linear(int(latent_width), 2, bias=False)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            nn.init.zeros_(self.residual_projection.weight)

        def forward(self, full_latent, context_latents):
            if full_latent.ndim != 2 or int(full_latent.shape[1]) != int(latent_width):
                raise ValueError("adaptive input context full latent shape不一致")
            if (
                context_latents.ndim != 3
                or int(context_latents.shape[0]) != int(full_latent.shape[0])
                or int(context_latents.shape[1]) != len(context_bars)
                or int(context_latents.shape[2]) != int(latent_width)
            ):
                raise ValueError("adaptive input context latent stack shape不一致")
            detached_full = full_latent.detach()
            detached_contexts = context_latents.detach()
            gate_logits = self.gate(detached_full)
            weights = torch.softmax(gate_logits.float(), dim=1).to(detached_contexts.dtype)
            adaptive_latent = torch.sum(detached_contexts * weights.unsqueeze(2), dim=1)
            residual_input = adaptive_latent - detached_full
            residual_logits = self.residual_projection(residual_input)
            return residual_logits, gate_logits, weights

    return AdaptiveInputContextSafetyBranch()

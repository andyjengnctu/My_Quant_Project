"""Same-date low-rank dynamic hypergraph primitive for Safety residuals."""

from __future__ import annotations


def build_same_date_dynamic_hypergraph_safety_residual(
    nn,
    torch,
    *,
    latent_width: int,
    spec,
):
    edge_count = int(spec.same_date_hyperedge_count or 0)
    if edge_count < 2:
        raise ValueError("same-date dynamic hypergraph至少需要2個hyperedges")
    stop_gradient = bool(spec.same_date_relation_stop_gradient)
    zero_init = bool(spec.same_date_relation_zero_init_residual)
    if not stop_gradient:
        raise ValueError("首輪same-date dynamic hypergraph scientific contract固定stop-gradient latent")
    if not zero_init:
        raise ValueError("首輪same-date dynamic hypergraph scientific contract固定zero-init residual")

    class SameDateDynamicHypergraphSafetyResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_count = int(edge_count)
            self.latent_width = int(latent_width)
            self.incidence = nn.Linear(self.latent_width, self.edge_count, bias=False)
            self.relation_projection = nn.Sequential(
                nn.Linear(self.latent_width * 2, self.latent_width),
                nn.ReLU(),
            )
            self.gate = nn.Linear(self.latent_width * 2, 1)
            self.residual_logits = nn.Linear(self.latent_width, 2)
            nn.init.zeros_(self.residual_logits.weight)
            nn.init.zeros_(self.residual_logits.bias)

        def forward(self, shared_latent):
            if shared_latent.ndim != 2 or int(shared_latent.shape[1]) != self.latent_width:
                raise ValueError(
                    "same-date dynamic hypergraph latent shape不一致: "
                    f"expected=[N,{self.latent_width}], actual={tuple(shared_latent.shape)}"
                )
            if int(shared_latent.shape[0]) < 1:
                raise ValueError("same-date dynamic hypergraph至少需要1個node")
            detached = shared_latent.detach()
            membership_logits = self.incidence(detached)
            membership = torch.softmax(membership_logits.float(), dim=1).to(detached.dtype)
            edge_mass = membership.sum(dim=0).clamp(min=torch.finfo(detached.dtype).eps)
            edge_state = membership.transpose(0, 1).matmul(detached) / edge_mass.unsqueeze(1)
            relational_state = membership.matmul(edge_state)
            fused = torch.cat([detached, relational_state], dim=1)
            hidden = self.relation_projection(fused)
            gate = torch.sigmoid(self.gate(fused).float()).to(hidden.dtype)
            return self.residual_logits(hidden) * gate

    return SameDateDynamicHypergraphSafetyResidual()


__all__ = ["build_same_date_dynamic_hypergraph_safety_residual"]

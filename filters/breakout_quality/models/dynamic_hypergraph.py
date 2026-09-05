"""Same-date low-rank dynamic hypergraph primitives for Safety residuals."""

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
    history_steps = int(spec.same_date_relation_history_steps or 0)
    if not stop_gradient:
        raise ValueError("same-date dynamic hypergraph scientific contract固定stop-gradient latent")
    if not zero_init:
        raise ValueError("same-date dynamic hypergraph scientific contract固定zero-init residual")
    if history_steps not in {0, 1}:
        raise ValueError("same-date dynamic hypergraph目前只支援0或1個previous-trading-date relation step")

    class SameDateDynamicHypergraphSafetyResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_count = int(edge_count)
            self.latent_width = int(latent_width)
            self.history_steps = int(history_steps)

            # BS primitive. Keep construction order frozen so a same-seed BT model
            # preserves every BS parameter tensor exactly before appending the new
            # relation-change branch.
            self.incidence = nn.Linear(self.latent_width, self.edge_count, bias=False)
            self.relation_projection = nn.Sequential(
                nn.Linear(self.latent_width * 2, self.latent_width),
                nn.ReLU(),
            )
            self.gate = nn.Linear(self.latent_width * 2, 1)
            self.residual_logits = nn.Linear(self.latent_width, 2)
            nn.init.zeros_(self.residual_logits.weight)
            nn.init.zeros_(self.residual_logits.bias)

            # BT adds exactly one previous-trading-date hyperedge-state difference.
            # The shared incidence matrix anchors latent hyperedge identity across
            # dates; no pairwise graph, Hawkes process, sector prior, or edge-count
            # sweep is introduced.
            if self.history_steps == 1:
                self.relation_change_projection = nn.Sequential(
                    nn.Linear(self.latent_width, self.latent_width),
                    nn.ReLU(),
                )
                self.relation_change_gate = nn.Linear(self.latent_width * 2, 1)
                self.relation_change_residual_logits = nn.Linear(self.latent_width, 2)
                nn.init.zeros_(self.relation_change_residual_logits.weight)
                nn.init.zeros_(self.relation_change_residual_logits.bias)
            else:
                self.relation_change_projection = None
                self.relation_change_gate = None
                self.relation_change_residual_logits = None

        def _membership_and_edge_state(self, latent):
            if latent.ndim != 2 or int(latent.shape[1]) != self.latent_width:
                raise ValueError(
                    "same-date dynamic hypergraph latent shape不一致: "
                    f"expected=[N,{self.latent_width}], actual={tuple(latent.shape)}"
                )
            if int(latent.shape[0]) < 1:
                raise ValueError("same-date dynamic hypergraph至少需要1個node")
            detached = latent.detach()
            membership_logits = self.incidence(detached)
            membership = torch.softmax(membership_logits.float(), dim=1).to(detached.dtype)
            edge_mass = membership.sum(dim=0).clamp(min=torch.finfo(detached.dtype).eps)
            edge_state = membership.transpose(0, 1).matmul(detached) / edge_mass.unsqueeze(1)
            return detached, membership, edge_state

        def forward(
            self,
            shared_latent,
            *,
            relation_current_latent=None,
            previous_relation_latent=None,
        ):
            detached, membership, edge_state = self._membership_and_edge_state(shared_latent)
            relational_state = membership.matmul(edge_state)
            fused = torch.cat([detached, relational_state], dim=1)
            hidden = self.relation_projection(fused)
            gate = torch.sigmoid(self.gate(fused).float()).to(hidden.dtype)
            residual = self.residual_logits(hidden) * gate

            if self.history_steps == 0:
                if relation_current_latent is not None or previous_relation_latent is not None:
                    raise ValueError("same-date state-only hypergraph不得收到relation-history latent")
                return residual

            if relation_current_latent is None or previous_relation_latent is None:
                raise ValueError("relation-change hypergraph需要current與previous-trading-date relation latent")
            if self.relation_change_projection is None or self.relation_change_gate is None or self.relation_change_residual_logits is None:
                raise RuntimeError("relation-change hypergraph module未完整初始化")

            current_detached, current_membership, current_edge_state = (
                self._membership_and_edge_state(relation_current_latent)
            )
            previous = previous_relation_latent.detach()
            if previous.ndim != 2 or int(previous.shape[1]) != self.latent_width:
                raise ValueError(
                    "previous-date dynamic hypergraph latent shape不一致: "
                    f"expected=[N,{self.latent_width}], actual={tuple(previous.shape)}"
                )
            if int(previous.shape[0]) == 0:
                # The first score-eligible date may have no prior complete 300-bar
                # universe.  Define no relational change instead of inventing a
                # missing-data signal.
                return residual
            _previous_detached, _previous_membership, previous_edge_state = (
                self._membership_and_edge_state(previous)
            )
            edge_change = current_edge_state - previous_edge_state
            relational_change = current_membership.matmul(edge_change)
            change_hidden = self.relation_change_projection(relational_change)
            change_gate_input = torch.cat(
                [current_detached, relational_change], dim=1
            )
            change_gate = torch.sigmoid(
                self.relation_change_gate(change_gate_input).float()
            ).to(change_hidden.dtype)
            change_residual = self.relation_change_residual_logits(change_hidden) * change_gate
            return residual + change_residual

    return SameDateDynamicHypergraphSafetyResidual()


__all__ = ["build_same_date_dynamic_hypergraph_safety_residual"]

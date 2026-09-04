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
    expected_pooling = "final_state_concat" if bidirectional else "final_state"
    if pooling != expected_pooling:
        raise ValueError(
            f"GRU pooling 與方向性不一致: bidirectional={bidirectional}, "
            f"pooling={pooling!r}, expected={expected_pooling!r}"
        )

    from filters.breakout_quality.models.architectures import get_architecture_descriptor

    descriptor = get_architecture_descriptor(str(spec.architecture))
    use_outer_autocast = descriptor.has_capability("recurrent_outer_autocast")
    guarded_retry = descriptor.has_capability("same_batch_fp32_nonfinite_retry")
    if guarded_retry and not use_outer_autocast:
        raise ValueError("same-batch FP32 retry 只允許outer-autocast recurrent architecture")

    class _SplitBidirectionalGRU(nn.Module):
        """Execution-equivalent 1-layer BiGRU using two unidirectional kernels.

        The scientific topology is unchanged from a native 1-layer BiGRU.  The
        production encoder only needs the two final recurrent states, so its CUDA
        path runs the independent directions on separate streams and does not
        materialize/flip/concatenate the unused full-sequence outputs.  ``forward``
        retains the complete native-like output contract for diagnostics/tests.
        """

        def __init__(self):
            super().__init__()
            if num_layers != 1:
                raise ValueError("split bidirectional GRU目前只支援1-layer等價執行")
            self.bidirectional = True
            self.num_layers = 1
            self.hidden_size = hidden_size
            self.execution_strategy = "split_unidirectional_parallel_cuda"
            # CUDA streams are execution-only state and therefore intentionally
            # absent from state_dict / fitted-model identity.
            self._cuda_stream_pairs = {}
            # Construction order intentionally matches nn.GRU(..., bidirectional=True):
            # forward direction parameters first, then reverse direction parameters.
            self.forward_gru = nn.GRU(
                input_size=int(feature_count),
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
            self.backward_gru = nn.GRU(
                input_size=int(feature_count),
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )

        def _sequential_final_hidden(self, x):
            _forward_output, forward_hidden = self.forward_gru(x)
            reversed_x = torch.flip(x, dims=(1,)).contiguous()
            _reversed_output, backward_hidden = self.backward_gru(reversed_x)
            return torch.cat((forward_hidden, backward_hidden), dim=0)

        def _cuda_stream_pair(self, x):
            device_index = int(
                x.device.index
                if x.device.index is not None
                else torch.cuda.current_device()
            )
            pair = self._cuda_stream_pairs.get(device_index)
            if pair is None:
                pair = (
                    torch.cuda.Stream(device=x.device),
                    torch.cuda.Stream(device=x.device),
                )
                self._cuda_stream_pairs[device_index] = pair
            return pair

        def final_hidden(self, x):
            """Return native-order [forward, backward] final states only.

            Directions are independent by construction.  On CUDA, branch from the
            caller stream after all prior writes (including optimizer.step), execute
            both unidirectional GRUs concurrently, then rejoin before concatenation.
            PyTorch autograd preserves the same branch streams for backward ops.
            """

            if not bool(getattr(x, "is_cuda", False)):
                return self._sequential_final_hidden(x)
            # Avoid introducing custom stream edges inside a CUDA graph capture.
            if bool(torch.cuda.is_current_stream_capturing()):
                return self._sequential_final_hidden(x)

            current_stream = torch.cuda.current_stream(device=x.device)
            forward_stream, backward_stream = self._cuda_stream_pair(x)
            forward_stream.wait_stream(current_stream)
            backward_stream.wait_stream(current_stream)

            with torch.cuda.stream(forward_stream):
                _forward_output, forward_hidden = self.forward_gru(x)
            with torch.cuda.stream(backward_stream):
                reversed_x = torch.flip(x, dims=(1,)).contiguous()
                _reversed_output, backward_hidden = self.backward_gru(reversed_x)

            current_stream.wait_stream(forward_stream)
            current_stream.wait_stream(backward_stream)
            forward_hidden.record_stream(current_stream)
            backward_hidden.record_stream(current_stream)
            return torch.cat((forward_hidden, backward_hidden), dim=0)

        def forward(self, x):
            # Preserve a native-like full-output contract for diagnostics.  The
            # production encoder uses final_hidden() and skips these unused tensors.
            forward_output, forward_hidden = self.forward_gru(x)
            reversed_x = torch.flip(x, dims=(1,)).contiguous()
            reversed_output, backward_hidden = self.backward_gru(reversed_x)
            backward_output = torch.flip(reversed_output, dims=(1,))
            output = torch.cat((forward_output, backward_output), dim=2)
            hidden = torch.cat((forward_hidden, backward_hidden), dim=0)
            return output, hidden

    class GRUSharedSafetyMFE(nn.Module):
        def __init__(self):
            super().__init__()
            if bidirectional:
                self.gru = _SplitBidirectionalGRU()
            else:
                self.gru = nn.GRU(
                    input_size=int(feature_count),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=False,
                )
            latent_width = hidden_size * (2 if bidirectional else 1)
            self.raw_safety_classifier = nn.Linear(latent_width, 2)
            self.raw_mfe_classifier = nn.Linear(latent_width, 2)
            self.same_batch_fp32_nonfinite_retry = bool(guarded_retry)
            self.recurrent_outer_autocast = bool(use_outer_autocast)

        def _final_latent(self, hidden):
            if not bidirectional:
                return hidden[-1]
            # PyTorch orders hidden as [layer0-forward, layer0-backward, ...].
            # The last layer therefore occupies the final two rows.
            return torch.cat((hidden[-2], hidden[-1]), dim=1)

        def _encode_native(self, x):
            if x.ndim != 3:
                raise ValueError(f"GRU input 必須是 [batch,time,feature]，收到 shape={tuple(x.shape)}")
            if bidirectional and hasattr(self.gru, "final_hidden"):
                hidden = self.gru.final_hidden(x)
            else:
                _outputs, hidden = self.gru(x)
            return self._final_latent(hidden)

        def _encode_fp32(self, x):
            if x.ndim != 3:
                raise ValueError(f"GRU input 必須是 [batch,time,feature]，收到 shape={tuple(x.shape)}")
            fp32_x = x.float()
            if bidirectional and hasattr(self.gru, "final_hidden"):
                hidden = self.gru.final_hidden(fp32_x)
            else:
                _outputs, hidden = self.gru(fp32_x)
            return self._final_latent(hidden)

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

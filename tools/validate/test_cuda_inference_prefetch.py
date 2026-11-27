"""Execution-only CUDA input staging preserves inference contracts."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import threading
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

import config.breakout_quality as policy
from filters.breakout_quality import inference
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan


class _Model(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [0.25, 0.5]]))

    def forward(self, features, context):
        return (features.mean(dim=1) + context) @ self.weight

    def forward_output_head(self, features, context, output_head):
        result = self(features, context)
        return torch.cat((result, result + 1), dim=1) if output_head == "both" else result


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "PyTorch CUDA is required for staging equivalence")
class CudaInferencePrefetchTest(unittest.TestCase):
    def setUp(self):
        if torch is None or not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is required for staging equivalence")
        self.plan = resolve_torch_execution_plan(
            torch, requested_device="cuda", mixed_precision=True,
            mixed_precision_dtype="bfloat16", deterministic_algorithms=True, allow_tf32=False,
        )

        rng = np.random.default_rng(812)
        self.features = rng.normal(size=(137, 17, 4)).astype(np.float32)[:, :, ::2]
        self.context = rng.normal(size=(137, 2)).astype(np.float32)
        self.model = _Model().to(self.plan.device)

    def run_inference(self, enabled, **overrides):
        arguments = dict(indices=np.array([5, 1, 5, 136, 0, 33, 80, 11, 20]),
                         batch_size=4, workers=4, execution_plan=self.plan)
        arguments.update(overrides)
        with patch.object(policy, "BREAKOUT_QUALITY_CUDA_INFERENCE_PREFETCH", enabled):
            return inference.strict_parallel_batched_logits(
                torch, self.model, self.features, self.context, **arguments,
            )

    def test_exact_logits_heads_indices_tails_and_state(self):
        original_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        original_rng = torch.cuda.get_rng_state().clone()
        for head in (None, "both"):
            before = self.run_inference(False, output_head=head)
            after = self.run_inference(True, output_head=head)
            self.assertEqual(before.tobytes(), after.tobytes())
        for key, value in original_state.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))
        self.assertTrue(torch.equal(original_rng, torch.cuda.get_rng_state()))

    def test_single_batch_and_same_date_relation_use_existing_path(self):
        with patch.object(inference, "_iter_prefetched_cuda_inputs", side_effect=AssertionError("unexpected prefetch")):
            self.run_inference(True, indices=np.array([1, 2]))
            self.model.requires_same_date_relations = True
            before = self.run_inference(False, same_date_group_labels=np.array([1, 2, 1, 3, 2, 3, 2, 1, 3]))
            after = self.run_inference(True, same_date_group_labels=np.array([1, 2, 1, 3, 2, 3, 2, 1, 3]))
            self.assertEqual(before.tobytes(), after.tobytes())

    def test_float32_and_nondefault_compute_stream_are_exact(self):
        self.plan = resolve_torch_execution_plan(
            torch, requested_device="cuda", mixed_precision=False,
            mixed_precision_dtype="auto", deterministic_algorithms=True, allow_tf32=False,
        )
        stream = torch.cuda.Stream(device=self.plan.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.plan.device))
        with torch.cuda.stream(stream):
            before = self.run_inference(False)
            after = self.run_inference(True)
        self.assertEqual(before.tobytes(), after.tobytes())


    def test_materialization_failure_is_propagated_and_worker_drained(self):
        class BrokenBank:
            def __len__(self):
                return 9

            def __getitem__(self, indices):
                if int(indices[0]) >= 4:
                    raise ValueError("materialization failed")
                return np.ones((len(indices), 17, 2), dtype=np.float32)

        self.features = BrokenBank()
        with self.assertRaisesRegex(ValueError, "materialization failed"):
            self.run_inference(True, indices=None)
        self.assertFalse(any(thread.name.startswith("breakout-quality-cuda-inputs") for thread in threading.enumerate()))

    def test_forward_failure_is_propagated_and_worker_drained(self):
        with patch.object(self.model, "forward", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                self.run_inference(True)
        self.assertFalse(any(thread.name.startswith("breakout-quality-cuda-inputs") for thread in threading.enumerate()))


@unittest.skipUnless(torch is not None, "PyTorch is required for inference equivalence")
class CpuInferencePrefetchFallbackTest(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("PyTorch is required for inference equivalence")

    def test_enabled_flag_preserves_cpu_serial_and_parallel_paths(self):
        rng = np.random.default_rng(311)
        features = rng.normal(size=(37, 7, 2)).astype(np.float32)
        context = rng.normal(size=(37, 2)).astype(np.float32)
        model = _Model()
        plan = resolve_torch_execution_plan(
            torch, requested_device="cpu", mixed_precision=False,
            mixed_precision_dtype="auto", deterministic_algorithms=True, allow_tf32=False,
        )
        with patch.object(inference, "_iter_prefetched_cuda_inputs", side_effect=AssertionError("unexpected prefetch")):
            for workers in (1, 2):
                outputs = []
                for enabled in (False, True):
                    with patch.object(policy, "BREAKOUT_QUALITY_CUDA_INFERENCE_PREFETCH", enabled):
                        outputs.append(inference.strict_parallel_batched_logits(
                            torch, model, features, context, indices=None, batch_size=8,
                            workers=workers, execution_plan=plan,
                        ))
                self.assertEqual(outputs[0].tobytes(), outputs[1].tobytes())


if __name__ == "__main__":
    unittest.main()

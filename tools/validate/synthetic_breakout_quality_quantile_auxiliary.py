"""Scoped synthetic invariants for the reusable Safety conditional-quantile auxiliary.

Run directly, independent of the user's formal suite:
    python -m tools.validate.synthetic_breakout_quality_quantile_auxiliary

No Provider downloads, user OOS rows, checkpoint mutation, or reporting changes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from config.breakout_quality import (
    BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,
    BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
)
from core.breakout_quality_policy import get_continuous_ranker_execution_recipe
from core.breakout_quality_registry import (
    get_breakout_quality_experiment_profile,
    get_continuous_ranker_research_spec,
)
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
)
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from services.breakout_quality.quantile_auxiliary import (
    date_balanced_pinball_loss,
    summarize_date_balanced_quantiles,
)
from services.breakout_quality import train_continuous_ranker as trainer


AO = "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
BZ = "daily_universal_shared_safety_hs_conditional_mfe_adverse_quantile_aux_full_list_ndcg_pairwise"


def _assert_close(actual, expected, *, message):
    actual = actual.detach().item() if isinstance(actual, torch.Tensor) else float(actual)
    expected = expected.detach().item() if isinstance(expected, torch.Tensor) else float(expected)
    if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6):
        raise AssertionError(f"{message}: expected={expected}, actual={actual}")


def main() -> None:
    torch.set_num_threads(1)
    ao_profile = get_breakout_quality_experiment_profile(AO)
    bz_profile = get_breakout_quality_experiment_profile(BZ)
    ao_spec = get_continuous_ranker_research_spec(AO)
    bz_spec = get_continuous_ranker_research_spec(BZ)
    assert ao_spec.model_research_id == "MR-13AO"
    assert bz_spec.model_research_id == "MR-13BZ"
    assert BZ != AO and bz_spec.model_gate_reference_profile_name == AO
    # Dedicated historical BZ test must not require BZ to remain the current model.
    assert BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    assert ("MR-13AO", AO) in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    assert ("MR-13BZ", BZ) in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    assert bz_profile.continuous_target_id == ao_profile.continuous_target_id
    assert bz_profile.epoch_selection_metric == ao_profile.epoch_selection_metric
    ao_policy = get_continuous_ranker_execution_recipe(AO).training_policy
    policy = get_continuous_ranker_execution_recipe(BZ).training_policy
    assert ao_policy.quantile_auxiliary is None
    assert policy.quantile_auxiliary.quantile_levels == (0.2, 0.5, 0.8)
    assert policy.quantile_auxiliary.weight_within_safety == 0.25
    assert policy.score_output_policy == ao_policy.score_output_policy
    assert policy.uses_shared_scoped_head_loss
    assert ao_policy.uses_shared_scoped_head_loss
    assert bz_spec.pairwise_reduction == ao_spec.pairwise_reduction
    assert bz_spec.secondary_pair_scope_threshold == ao_spec.secondary_pair_scope_threshold
    assert bz_spec.selection_pit_authorized is False

    # Same-seed controlled contrast: only two additional aux parameters; all
    # prior AO parameter tensors, buffers and two logits must remain identical.
    torch.manual_seed(42)
    ao = build_model(10, 0, architecture=ao_profile.model_architecture)
    torch.manual_seed(42)
    bz = build_model(10, 0, architecture=bz_profile.model_architecture)
    ao_state, bz_state = ao.state_dict(), bz.state_dict()
    assert set(ao_state).issubset(bz_state)
    assert len(set(bz_state) - set(ao_state)) == 2
    assert all(torch.equal(value, bz_state[key]) for key, value in ao_state.items())

    x = torch.randn((12, 300, 10), generator=torch.Generator().manual_seed(20))
    dates = np.array(["2025-01-06"] * 6 + ["2025-01-07"] * 6, dtype="datetime64[D]")
    table = pd.DataFrame({
        "date": dates,
        "target_favorable_r": [0.2, 0.8, 0.5, 1.2, 0.6, 1.0, 0.1, 0.7, 1.2, 0.8, 0.3, 0.9],
        "target_adverse_r": [0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.3, 0.4, 1.0, 0.5, 0.8, 0.1],
    })
    raw = np.ones((12,), dtype=np.float32)
    percentile = np.ones((12,), dtype=np.float32) * 0.5
    ao_targets = trainer._training_target_for_profile(ao_profile, raw, percentile, table)
    bz_targets = trainer._training_target_for_profile(bz_profile, raw, percentile, table)
    assert ao_targets.shape == (12, 2) and bz_targets.shape == (12, 3)
    np.testing.assert_array_equal(bz_targets[:, :2], ao_targets)
    np.testing.assert_array_equal(bz_targets[:, 2], table["target_adverse_r"].to_numpy(dtype=np.float32))
    assert np.isnan(bz_targets[:, 1]).any(), "LS rows cannot become MFE labels"
    assert np.isfinite(bz_targets[:, 2]).all(), "All mature rows need adverse quantile labels"

    ao.eval()
    bz.eval()
    with torch.no_grad():
        ao_s, ao_m = ao.forward_safety_mfe_heads(x, None)
        bz_s, bz_m, bz_q = bz.forward_safety_mfe_heads(x, None, include_adverse_quantiles=True)
        assert torch.equal(ao_s, bz_s)
        assert torch.equal(ao_m, bz_m)
        assert torch.allclose(bz_q, bz.forward_output_head(x, None, "adverse_quantiles"))
        assert torch.all(bz_q >= 0) and torch.all(bz_q[:, 1:] >= bz_q[:, :-1])
        assert torch.equal(bz.forward_output_head(x, None, "both"), torch.cat([bz_s, bz_m], dim=1))

    # Date equality, not row or date-size weighting, must determine Pinball aggregation.
    pred = torch.tensor([[1.0, 1.2, 1.5], [0.6, 0.7, 1.0], [0.3, 0.6, 0.9]], requires_grad=True)
    true = torch.tensor([0.5, 1.0, 0.8])
    short_dates = np.array(["2025-01-06", "2025-01-06", "2025-01-07"])
    value = date_balanced_pinball_loss(
        torch, pred, true, short_dates, quantile_levels=(0.2, 0.5, 0.8)
    )
    manual = torch.maximum(
        torch.tensor([0.2, 0.5, 0.8]).unsqueeze(0) * (true[:, None] - pred),
        (torch.tensor([0.2, 0.5, 0.8]) - 1).unsqueeze(0) * (true[:, None] - pred),
    ).mean(dim=1)
    _assert_close(value.item(), (manual[:2].mean() + manual[2]) / 2, message="daily pinball")
    value.backward()
    assert pred.grad is not None and torch.any(pred.grad != 0)
    calibration = summarize_date_balanced_quantiles(
        pred.detach().numpy(), true.numpy(), short_dates,
        quantile_levels=(0.2, 0.5, 0.8),
    )
    _assert_close(calibration["mean_date_pinball"], value.item(), message="read-only OOS pinball consistency")
    assert calibration["date_count"] == 2
    assert calibration["row_count"] == 3
    assert len(calibration["mean_date_empirical_coverage_by_quantile"]) == 3

    # Match the complete AO objective plus the fixed auxiliary Safety weight.
    # With model.eval(), BN/dropout cannot mask deterministic formula comparisons.
    target_ao = torch.tensor(ao_targets, dtype=torch.float32)
    target_bz = torch.tensor(bz_targets, dtype=torch.float32)
    ao_loss, ao_n = trainer._compute_shared_safety_scoped_mfe_duo_loss(
        torch, ao, x, None, target_ao, dates,
        training_policy=ao_policy,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.5,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    bz_loss, bz_n = trainer._compute_shared_safety_scoped_mfe_duo_loss(
        torch, bz, x, None, target_bz, dates,
        training_policy=policy,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.5,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    ao_s_loss, _ = trainer._pairwise_logistic_loss(
        torch, (ao_s[:, 1] - ao_s[:, 0]), target_ao[:, 0], dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    ao_m_loss, _ = trainer._pairwise_logistic_loss(
        torch, (ao_m[:, 1] - ao_m[:, 0]), target_ao[:, 1], dates,
        item_eligibility=target_ao[:, 0] >= 0.5,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    quantile_loss = date_balanced_pinball_loss(
        torch, bz_q, target_bz[:, 2], dates, quantile_levels=(0.2, 0.5, 0.8),
    )
    assert ao_loss is not None and bz_loss is not None and ao_n == bz_n and ao_n > 0
    _assert_close(ao_loss.item(), 0.5 * (ao_s_loss + ao_m_loss), message="AO loss unchanged")
    expected = 0.5 * ((0.75 * ao_s_loss + 0.25 * quantile_loss) + ao_m_loss)
    _assert_close(bz_loss.item(), expected.item(), message="fixed nested composition")

    bz.zero_grad(set_to_none=True)
    bz_loss.backward()
    names = dict(bz.named_parameters())
    assert names["adverse_quantile_classifier.weight"].grad is not None
    assert torch.any(names["adverse_quantile_classifier.weight"].grad != 0)
    assert names["raw_safety_classifier.weight"].grad is not None
    assert names["raw_mfe_classifier.weight"].grad is not None
    assert names["inception_modules.0.bottleneck.weight"].grad is not None
    assert torch.any(names["inception_modules.0.bottleneck.weight"].grad != 0)

    # AO invalid/maturing rows must stay excluded; no target extrapolation.
    invalid = table.copy()
    invalid.loc[0, "target_favorable_r"] = np.nan
    bad_raw = raw.copy()
    bad_raw[0] = np.nan
    invalid_target = trainer._training_target_for_profile(bz_profile, bad_raw, percentile, invalid)
    assert np.isnan(invalid_target[0]).all()
    missing = table.drop(columns=["target_adverse_r"])
    try:
        trainer._training_target_for_profile(bz_profile, raw, percentile, missing)
    except ValueError:
        pass
    else:
        raise AssertionError("missing PIT target column must fail")
    bad = table.copy()
    bad.loc[0, "target_adverse_r"] = -0.1
    try:
        trainer._training_target_for_profile(bz_profile, raw, percentile, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("negative adverse R target must fail")
    try:
        trainer._compute_shared_safety_scoped_mfe_duo_loss(
            torch, bz, x, None, target_ao, dates, training_policy=policy,
            secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
            secondary_pair_scope_threshold=0.5,
            pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("legacy AO target width must fail with BZ")

    # Exercise the actual generic date-coherent trainer/optimizer route, not
    # merely the local differentiable-loss helper.
    plan = resolve_torch_execution_plan(
        torch, requested_device="cpu", mixed_precision=False,
        mixed_precision_dtype="auto", deterministic_algorithms=False, allow_tf32=False,
    )
    bz.train()
    optimizer = torch.optim.Adam(bz.parameters(), lr=0.0003, weight_decay=0.0001)
    old_quantiles = bz.adverse_quantile_classifier.weight.detach().clone()
    epoch_loss = trainer._train_epoch(
        torch, bz, optimizer,
        feature_bank=x.detach().numpy(),
        group_context=np.zeros((12, 0), dtype=np.float32),
        group_ids=np.arange(12, dtype=np.int64),
        percentile_target=bz_targets,
        group_dates=dates,
        training_objective=bz_profile.training_objective,
        batch_size=12,
        seed=42,
        gradient_clip_norm=1.0,
        plan=plan,
        grad_scaler=None,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.5,
    )
    assert math.isfinite(epoch_loss) and epoch_loss > 0
    assert not torch.equal(bz.adverse_quantile_classifier.weight, old_quantiles)
    print("PASS: identity, frozen AO init/scores, target-PIT, quantiles, daily Pinball, loss formula, gradients, negative guards, generic optimizer epoch")


if __name__ == "__main__":
    main()

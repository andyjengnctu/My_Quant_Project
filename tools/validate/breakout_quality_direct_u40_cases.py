"""Isolated capability checks for the signed full-horizon MFE minus D40 target.

Not a formal-suite runner. Both the historical CE regression and new CF
ranker consume the same mature full-horizon U40 Producer target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
    DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R,
    CONTINUOUS_RANKER_TARGET_BUILDER_RAW_R,
    CONTINUOUS_RANKER_TARGET_BUILDER_SCALAR_PAIRWISE,
    CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.continuous_target import (
    StrategyAlignedContinuousTargetSpec,
    full_horizon_mae_safety_r_from_minimum_low_return,
)
from filters.breakout_quality.daily_ranker_data import (
    _build_daily_target_contract,
    compute_daily_opportunity_target_batch,
)
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from services.breakout_quality.train_continuous_ranker import (
    _train_epoch, predict_scores, build_daily_percentile_targets,
    _training_target_for_profile, _pairwise_logistic_loss,
)


def _frame(highs: np.ndarray, lows: np.ndarray) -> pd.DataFrame:
    assert highs.shape == lows.shape == (40,)
    seq = np.column_stack((np.r_[100.0, np.full(40, 100.0)],
                           np.r_[100.0, highs], np.r_[100.0, lows]))
    assert np.all(seq[1:, 1] >= seq[1:, 2])
    return pd.DataFrame({"Close": seq[:, 0], "High": seq[:, 1], "Low": seq[:, 2]})


def case() -> None:
    profile = get_breakout_quality_experiment_profile(
        "daily_universal_full_horizon_mfe_minus_d40_huber"
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    assert (spec.model_research_id, profile.continuous_target_id) == (
        "MR-13CE", DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
    )
    assert profile.raw_r_huber_delta_r == 1.0
    assert profile.epoch_selection_metric == "validation_huber_raw_r"
    assert recipe.training_policy.target_builder == CONTINUOUS_RANKER_TARGET_BUILDER_RAW_R
    assert recipe.training_policy.loss_handler == CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R
    assert recipe.training_policy.score_transform == CONTINUOUS_RANKER_SCORE_TRANSFORM_MARGIN_R
    assert recipe.training_policy.epoch_selection_rule == "regression_loss_then_spearman"
    assert recipe.target_policy.contract_kind == "daily_full_horizon_mfe_minus_d40"
    assert not recipe.historical_pit_authorized and not recipe.current_time_validation_authorized
    from filters.breakout_quality.ranker_training_contract import training_semantics
    semantics = training_semantics(profile)["raw_r_regression_contract"]
    assert semantics["target"] == f"{DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID}_raw_r"
    assert semantics["loss"] == "huber_raw_r" and semantics["huber_delta_r"] == 1.0
    historical = get_breakout_quality_experiment_profile("daily_universal_no_time_r_huber")
    assert (training_semantics(historical)["raw_r_regression_contract"]["target"]
            == "daily_opportunity_no_time_r_v1_raw_r")
    contract = _build_daily_target_contract(recipe.target_policy.contract_kind,
                                             spec=StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY),
                                             risk_param_policy=None)
    assert contract["target_id"] == DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID
    assert contract["horizon_bars"] == 40
    assert contract["oos_fitted_parameters"] is False
    assert contract["requires_strategy_candidate_membership"] is False
    policy = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    risk = float(policy.risk_budget_return)

    # The adverse low is *after* the best high. Old adverse-to-peak would miss it.
    highs = np.full(40, 102.0)
    highs[6] = 140.0
    lows = np.full(40, 99.0)
    lows[38] = 70.0
    f = _frame(highs, lows)
    new = compute_daily_opportunity_target_batch(
        f, np.array([0]), spec=policy,
        target_id=DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
    )
    old = compute_daily_opportunity_target_batch(
        f, np.array([0]), spec=policy,
        target_id=DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    )
    assert new.valid_mask.tolist() == old.valid_mask.tolist() == [True]
    np.testing.assert_allclose(new.favorable_return[0], 0.4, atol=1e-6)
    np.testing.assert_allclose(new.minimum_low_return[0], -0.3, atol=1e-6)
    np.testing.assert_allclose(new.target_raw_r[0], (0.4 - 0.3) / risk, atol=1e-5)
    np.testing.assert_allclose(old.target_raw_r[0], (0.4 - 0.01) / risk, atol=1e-5)
    assert new.target_raw_r[0] < old.target_raw_r[0]
    assert int(new.opportunity_bar[0]) == 7

    # If all future lows exceed anchor, adverse risk is exactly zero, not negative.
    f_no_downside = _frame(np.full(40, 120.0), np.full(40, 105.0))
    safe = compute_daily_opportunity_target_batch(
        f_no_downside, np.array([0]), spec=policy,
        target_id=DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
    )
    np.testing.assert_allclose(safe.target_raw_r[0], 0.2 / risk, atol=1e-5)
    np.testing.assert_allclose(
        full_horizon_mae_safety_r_from_minimum_low_return(
            safe.minimum_low_return, risk_budget_return=risk),
        [0.0], atol=1e-6,
    )

    # A lower future high and adverse low must generate signed negative U40.
    f_negative = _frame(np.full(40, 95.0), np.full(40, 91.0))
    negative = compute_daily_opportunity_target_batch(
        f_negative, np.array([0]), spec=policy,
        target_id=DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
    )
    np.testing.assert_allclose(negative.target_raw_r[0], (-0.05 - 0.09) / risk, atol=1e-5)
    assert negative.target_raw_r[0] < 0

    # Immature/invalid paths fail closed; preceding clean rows remain deterministic.
    f_bad = f.copy()
    f_bad.loc[40, "Low"] = 0.0
    invalid = compute_daily_opportunity_target_batch(
        f_bad, np.array([0]), spec=policy,
        target_id=DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID,
    )
    assert not bool(invalid.valid_mask[0])
    assert np.isnan(invalid.target_raw_r[0])

    # Changing a future low cannot alter old Pure-MFE; its historical target is immutable.
    pure = compute_daily_opportunity_target_batch(
        f, np.array([0]), spec=policy, target_id=DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )
    np.testing.assert_allclose(pure.target_raw_r[0], 0.4 / risk, atol=1e-5)

    # CE/CF remain registered historical loss controls regardless of which
    # experiments the mutable official comparison configuration currently selects.
    assert get_continuous_ranker_research_spec(profile.name).model_research_id == "MR-13CE"
    cf_profile = get_breakout_quality_experiment_profile(
        "daily_universal_full_horizon_mfe_minus_d40_full_list_ndcg_pairwise"
    )
    assert get_continuous_ranker_research_spec(cf_profile.name).model_research_id == "MR-13CF"
    assert BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE in BREAKOUT_QUALITY_MODEL_TEST_PROFILES


def cpu_train_case() -> None:
    import torch

    profile = get_breakout_quality_experiment_profile(
        "daily_universal_full_horizon_mfe_minus_d40_huber"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_model(10, 0, architecture=profile.model_architecture)
        features = np.random.default_rng(42).normal(size=(12, 300, 10)).astype(np.float32)
        context = np.empty((12, 0), dtype=np.float32)
        signed_target = np.array([-3.2, -1.2, -0.4, 0.0, 0.7, 1.1] * 2, dtype=np.float32)
        dates = pd.Series(["2020-01-01"] * 6 + ["2020-01-02"] * 6)
        ids = np.arange(12, dtype=np.int64)
        plan = resolve_torch_execution_plan(
            torch, requested_device="cpu", mixed_precision=False,
            mixed_precision_dtype="float16", deterministic_algorithms=True, allow_tf32=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        before = model.classifier.weight.detach().clone()
        loss = _train_epoch(
            torch, model, optimizer, features, context, ids,
            signed_target, dates, training_objective=profile.training_objective,
            batch_size=6, seed=42, gradient_clip_norm=1.0, plan=plan,
            grad_scaler=None, raw_r_loss_name="huber_raw_r", raw_r_huber_delta_r=1.0,
        )
        assert np.isfinite(loss)
        assert not torch.allclose(before, model.classifier.weight.detach(), atol=0.0, rtol=0.0)
        score = predict_scores(
            torch, model, features, context, ids, batch_size=6,
            plan=plan, training_objective=profile.training_objective, group_dates=dates,
        )
        assert score.shape == (12,) and np.isfinite(score).all()
        # Regression score is an unrestricted signed margin, not a softmax probability.
        assert model.classifier.out_features == 2


def pairwise_case() -> None:
    import torch

    ce = get_breakout_quality_experiment_profile(
        "daily_universal_full_horizon_mfe_minus_d40_huber")
    cf = get_breakout_quality_experiment_profile(
        "daily_universal_full_horizon_mfe_minus_d40_full_list_ndcg_pairwise")
    spec = get_continuous_ranker_research_spec(cf.name)
    recipe = get_continuous_ranker_execution_recipe(cf.name)
    assert spec.model_research_id == "MR-13CF"
    assert cf.continuous_target_id == ce.continuous_target_id == DAILY_FULL_HORIZON_MFE_MINUS_D40_TARGET_ID
    assert cf.model_architecture == ce.model_architecture == "inception_time_v1"
    assert cf.training_sampling_mode == ce.training_sampling_mode
    assert cf.training_label_scope == ce.training_label_scope
    assert cf.training_sample_scope == ce.training_sample_scope
    assert cf.optimizer_name == ce.optimizer_name
    assert cf.loss_name == "pairwise_logistic" and cf.epoch_selection_metric == "mean_daily_spearman"
    assert cf.raw_r_huber_delta_r is None
    assert recipe.target_policy.contract_kind == "daily_full_horizon_mfe_minus_d40"
    assert recipe.training_policy.target_builder == CONTINUOUS_RANKER_TARGET_BUILDER_SCALAR_PAIRWISE
    assert recipe.training_policy.loss_handler == CONTINUOUS_RANKER_LOSS_HANDLER_SINGLE_PAIRWISE
    assert recipe.objective_policy.pairwise_reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG
    assert not recipe.historical_pit_authorized and not recipe.current_time_validation_authorized

    raw = np.asarray([-1.3, 2.6, 0.7, -0.2, -0.9, 1.6], dtype=np.float32)
    dates = pd.Series(["2020-06-01"] * 3 + ["2020-06-02"] * 3)
    percentile = build_daily_percentile_targets(raw, np.ones(len(raw), dtype=bool), dates)
    training_target = _training_target_for_profile(
        cf, raw, percentile,
        pd.DataFrame({"date": dates, "group_index": np.arange(len(raw))}),
    )
    assert training_target.shape == raw.shape
    np.testing.assert_allclose(training_target, percentile, rtol=0, atol=0)
    for group in (slice(0, 3), slice(3, 6)):
        assert np.array_equal(np.argsort(raw[group]), np.argsort(training_target[group]))
    assert np.isfinite(training_target).all() and (training_target >= 0).all() and (training_target <= 1).all()
    assert (raw < 0).any() and raw.max() > 1.0   # no signed truth clipping

    truth = torch.tensor(training_target[:3], dtype=torch.float32)
    bad = torch.tensor([0.8, 0.1, 0.4], dtype=torch.float32, requires_grad=True)
    good = -bad.detach()
    wrong_loss, n_wrong = _pairwise_logistic_loss(
        torch, bad, truth, dates.iloc[:3], reduction=recipe.objective_policy.pairwise_reduction,
    )
    good_loss, n_good = _pairwise_logistic_loss(
        torch, good, truth, dates.iloc[:3], reduction=recipe.objective_policy.pairwise_reduction,
    )
    assert n_wrong == n_good == 3
    assert wrong_loss is not None and good_loss is not None and wrong_loss > good_loss
    wrong_loss.backward()
    assert bad.grad is not None and torch.isfinite(bad.grad).all()

    # Actual generic InceptionTime training with the registered pairwise loss.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_model(10, 0, architecture=cf.model_architecture)
        rng = np.random.default_rng(42)
        features = rng.normal(size=(12, 300, 10)).astype(np.float32)
        context = np.empty((12, 0), dtype=np.float32)
        days = pd.Series(["2020-06-01"] * 6 + ["2020-06-02"] * 6)
        raw12 = np.array([-3.2, -1.2, -0.4, 0.0, 0.7, 1.1] * 2, dtype=np.float32)
        rank12 = build_daily_percentile_targets(raw12, np.ones(12, dtype=bool), days)
        plan = resolve_torch_execution_plan(
            torch, requested_device="cpu", mixed_precision=False,
            mixed_precision_dtype="float16", deterministic_algorithms=True, allow_tf32=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        before = model.classifier.weight.detach().clone()
        loss = _train_epoch(
            torch, model, optimizer, features, context, np.arange(12), rank12, days,
            training_objective=cf.training_objective, batch_size=12, seed=42,
            gradient_clip_norm=1.0, plan=plan, grad_scaler=None,
            pairwise_reduction=recipe.objective_policy.pairwise_reduction,
        )
        assert np.isfinite(loss)
        assert not torch.equal(before, model.classifier.weight.detach())
        predicted = predict_scores(
            torch, model, features, context, np.arange(12), batch_size=12,
            plan=plan, training_objective=cf.training_objective, group_dates=days,
        )
        assert predicted.shape == (12,) and np.isfinite(predicted).all()
        assert ((predicted >= 0) & (predicted <= 1)).all()
        assert model.classifier.out_features == 2


if __name__ == "__main__":
    case()
    cpu_train_case()
    pairwise_case()
    print("Direct U40 Huber and registered U40 full-list pairwise scientific controls: PASS")

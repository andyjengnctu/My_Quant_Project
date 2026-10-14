"""Independent regression for the generic complete-horizon MAE and finite-label scope.

This is a directed primitive test, not the Windows-only formal test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_FINITE_SECONDARY_TARGET,
    CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.breakout_quality_registry import (
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_HORIZON_MAE_PROFILE,
)
from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
from filters.breakout_quality.ranker_training_contract import training_semantics
from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from filters.breakout_quality.continuous_target import full_horizon_mae_safety_r_from_minimum_low_return
from services.breakout_quality.full_horizon_mae_comparison import (
    build_paired_full_horizon_mae_rows,
)

from services.breakout_quality import train_continuous_ranker as ranker


def _fixture() -> pd.DataFrame:
    dates = ['2020-01-02'] * 4 + ['2020-01-03'] * 4
    # AO True-HS = tickers 0/1 on each date.  Full-horizon worst low moves
    # ticker 1 outside CC's high safety group WITHOUT changing AO MFE cohort.
    favorable = [3.0, 1.0, 4.0, 2.0] * 2
    old_adverse = [0.1, 0.2, 0.3, 0.4] * 2
    low_returns = [-0.01, -0.08, -0.03, -0.07] * 2
    return pd.DataFrame({
        'date': pd.to_datetime(dates), 'ticker': ['A','B','C','D'] * 2,
        'target_favorable_r': favorable,
        'target_adverse_r': old_adverse,
        'target_adverse_return_to_peak': np.array(old_adverse) * 0.10,
        'target_minimum_low_return': low_returns,
    })


def test_mae_primitive_and_mask_contract():
    source = _fixture()
    valid = np.ones(len(source), dtype=bool)
    ao = build_hs_conditional_mfe_targets(source, valid)
    old_explicit = build_hs_conditional_mfe_targets(source, valid,
        safety_supervision_source='adverse_to_peak')
    cc = build_hs_conditional_mfe_targets(source, valid,
        safety_supervision_source=CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE)
    assert np.array_equal(ao.training_target, old_explicit.training_target, equal_nan=True)
    assert np.array_equal(ao.true_hs_mask, cc.true_hs_mask)
    assert np.array_equal(ao.conditional_mfe_percentile, cc.conditional_mfe_percentile, equal_nan=True)
    assert np.array_equal(ao.primary_mfe_percentile, cc.primary_mfe_percentile, equal_nan=True)
    assert not np.array_equal(ao.low_adverse_safety_percentile, cc.low_adverse_safety_percentile)
    assert np.array_equal(ao.true_hs_mask[:4], np.array([True,True,False,False]))
    assert np.array_equal(cc.low_adverse_safety_percentile[:4]>=0.5, np.array([True,False,True,False]))
    assert np.isnan(cc.training_target[~cc.true_hs_mask,1]).all()
    assert np.isfinite(cc.training_target[cc.true_hs_mask,1]).all()
    risk = full_horizon_mae_safety_r_from_minimum_low_return(
        np.array([-0.05, 0.02, np.nan]), risk_budget_return=0.10)
    assert np.allclose(risk[:2], [-0.5, 0.0]); assert np.isnan(risk[2])
    for bad in (np.array([np.inf]), np.array([-1.0])):
        with np.testing.assert_raises(ValueError):
            full_horizon_mae_safety_r_from_minimum_low_return(bad, risk_budget_return=0.10)
    with np.testing.assert_raises(ValueError):
        build_hs_conditional_mfe_targets(source.drop(columns=['target_minimum_low_return']), valid,
            safety_supervision_source=CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE)
    return source, ao, cc


class _SyntheticDuoHead(torch.nn.Module):
    def forward_safety_mfe_heads(self, xb, cb):
        zero = torch.zeros_like(xb[:,0])
        score = xb[:,0]
        return (torch.stack([zero, score], dim=1),
                torch.stack([zero, torch.flip(score, dims=(0,))],dim=1))


def test_ao_mfe_loss_eligibility_unchanged(ao, cc):
    ao_recipe = get_continuous_ranker_execution_recipe(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE)
    cc_recipe = get_continuous_ranker_execution_recipe(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_HORIZON_MAE_PROFILE)
    assert ao_recipe.model_architecture == cc_recipe.model_architecture
    assert ao_recipe.training_objective == cc_recipe.training_objective
    assert ao_recipe.loss_name == cc_recipe.loss_name
    assert ao_recipe.training_policy is cc_recipe.training_policy
    assert cc_recipe.objective_policy.secondary_pair_scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_FINITE_SECONDARY_TARGET
    assert cc_recipe.objective_policy.safety_supervision_source == CONTINUOUS_RANKER_SAFETY_SUPERVISION_FULL_HORIZON_MAE
    assert cc_recipe.objective_policy.secondary_pair_scope_threshold == 0.5
    assert ao_recipe.objective_policy.safety_supervision_source == 'adverse_to_peak'
    name_ao = DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    name_cc = DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_HORIZON_MAE_PROFILE
    manifest_ao = training_semantics(get_breakout_quality_experiment_profile(name_ao))
    manifest_cc = training_semantics(get_breakout_quality_experiment_profile(name_cc))
    old_label = manifest_ao['shared_safety_hs_conditional_mfe_duo_head_contract']['safety_target']
    cc_contract = manifest_cc['shared_safety_hs_conditional_mfe_duo_head_contract']
    assert old_label == 'same_date_low_adverse_safety_percentile_over_full_universe'
    assert 'full_40bar_anchor_max_adverse' in cc_contract['safety_target']
    assert 'original_adverse_to_earliest_peak' in cc_contract['true_hs_definition']
    assert 'finite_secondary_target' in cc_contract['conditional_mfe_supervision_scope']

    x = torch.tensor([[0.7],[0.2],[0.4],[0.1],[0.6],[0.5],[0.8],[0.3]],dtype=torch.float32)
    date_keys = np.asarray(['2020-01-02']*4+['2020-01-03']*4, dtype=object)
    model = _SyntheticDuoHead()
    orig = ranker._pairwise_logistic_loss
    observed = []
    def monitor(*args, **kwargs):
        if kwargs.get('item_eligibility') is not None:
            observed.append(kwargs['item_eligibility'].detach().cpu().numpy().astype(bool))
        return orig(*args,**kwargs)
    with patch.object(ranker,'_pairwise_logistic_loss',side_effect=monitor):
        loss_ao, n_ao = ranker._compute_shared_safety_scoped_mfe_duo_loss(
            torch,model,x,None,torch.from_numpy(ao.training_target.copy()),date_keys,
            training_policy=ao_recipe.training_policy,
            secondary_pair_scope=ao_recipe.objective_policy.secondary_pair_scope,
            secondary_pair_scope_threshold=ao_recipe.objective_policy.secondary_pair_scope_threshold,
            pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG)
        loss_cc, n_cc = ranker._compute_shared_safety_scoped_mfe_duo_loss(
            torch,model,x,None,torch.from_numpy(cc.training_target.copy()),date_keys,
            training_policy=cc_recipe.training_policy,
            secondary_pair_scope=cc_recipe.objective_policy.secondary_pair_scope,
            secondary_pair_scope_threshold=cc_recipe.objective_policy.secondary_pair_scope_threshold,
            pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG)
    assert loss_ao is not None and loss_cc is not None
    assert torch.isfinite(loss_ao) and torch.isfinite(loss_cc)
    assert n_ao>0 and n_cc>0
    assert len(observed) == 2
    assert np.array_equal(observed[0], observed[1])
    assert np.array_equal(observed[0], ao.true_hs_mask)


def test_paired_common_truth_extension(source):
    sample = source.copy()
    sample["group_index"] = np.arange(len(sample), dtype=np.int64)
    sample["d40_r"] = -full_horizon_mae_safety_r_from_minimum_low_return(
        sample["target_minimum_low_return"].to_numpy(dtype=np.float64),
        risk_budget_return=0.10,
    )
    sample["old_safety_r"] = -sample["target_adverse_r"].to_numpy(dtype=np.float64)
    a = sample[["ticker", "date", "group_index"]].copy()
    a["safety_prediction"] = [0.5, 0.6, 0.7, 0.8] * 2
    c = sample[["ticker", "date", "group_index"]].copy()
    c["safety_prediction"] = [0.8, 0.1, 0.7, 0.2] * 2
    result, meta = build_paired_full_horizon_mae_rows(
        truth=sample,
        predictions_by_model={"MR-13AO": a, "MR-13CC": c},
        split_label="Forward OOS",
    )
    assert meta["paired_stock_days"] == len(sample)
    assert meta["paired_dates"] == 2
    assert result["MR-13AO"]["d40_daily_rho"] is not None
    assert result["MR-13CC"]["d40_daily_rho"] is not None
    assert result["MR-13AO"]["d40_daily_rho"] != result["MR-13CC"]["d40_daily_rho"]
    assert meta["d40_strictly_worse_than_peak_pct"] == 50.0


def main():
    source, ao, cc = test_mae_primitive_and_mask_contract()
    test_ao_mfe_loss_eligibility_unchanged(ao, cc)
    test_paired_common_truth_extension(source)
    print('PASS: full 40D MAE primitive, AO frozen MFE cohort, generic scoped loss, Research [1] paired truth extension')


if __name__ == '__main__':
    main()

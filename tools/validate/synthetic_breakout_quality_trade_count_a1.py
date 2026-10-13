"""Focused AO/transaction-count primitive tests; not the Windows formal suite.

Run from project root: python -m tools.validate.synthetic_breakout_quality_trade_count_a1
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd

from config.breakout_quality import BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
from core.breakout_quality_registry import get_breakout_quality_experiment_profile, get_continuous_ranker_research_spec
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_ranker_data import LazyDailyFeatureBank
from filters.breakout_quality.models.architectures import get_architecture_descriptor
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.trade_activity_features import (
    TRADE_COUNT_FEATURE_NAMES, normalized_trade_count_windows, previous_session_trade_count,
)
from services.breakout_quality.trade_count_source import (
    load_verified_trade_count_observations,
    previous_session_trade_count as legacy_previous_session_trade_count,
    preflight_trade_count_capture_evidence as legacy_trade_count_capture_preflight,
)
from filters.breakout_quality.trade_count_capture_preflight import (
    preflight_trade_count_capture_evidence,
)


def test_trade_count_source() -> None:
    # Compatibility path reuses the exact canonical feature-level PIT alignment.
    assert legacy_previous_session_trade_count is previous_session_trade_count
    assert legacy_trade_count_capture_preflight is preflight_trade_count_capture_evidence
    calendar = pd.bdate_range("2020-12-28", periods=6)
    source = {
        "2330": {
            calendar[0]: (100.0, calendar[0]),
            calendar[1]: (200.0, calendar[2]),  # arrived on signal day: illegal
            calendar[3]: (500.0, calendar[3]),
        }
    }
    counts, mask = previous_session_trade_count(calendar, ticker="2330", calendar=calendar, observations=source)
    assert np.array_equal(mask, [False, True, False, False, True, False])
    assert counts[1] == 100 and counts[4] == 500
    assert counts[5] == 0 and not mask[5], "missing day must not carry-forward older counts"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_verified_trade_count_observations(Path(tmp))
        except FileNotFoundError as exc:
            assert "PIT Data Gate" in str(exc)
        else:
            raise AssertionError("missing archive must fail closed")
        from unittest.mock import patch
        with patch(
            "filters.breakout_quality.trade_count_capture_preflight._snapshot_refs",
            return_value=[(pd.Timestamp("2026-10-12"), "a" * 64)],
        ):
            try:
                load_verified_trade_count_observations(Path(tmp))
            except ValueError as exc:
                assert "Research cutoff" in str(exc)
            else:
                raise AssertionError("post-Research snapshot must never authorize historical OOS")
        with patch(
            "filters.breakout_quality.trade_count_capture_preflight._snapshot_refs",
            return_value=[(pd.Timestamp("2022-01-01"), "b" * 64)],
        ):
            try:
                load_verified_trade_count_observations(Path(tmp))
            except ValueError as exc:
                assert "historical Trading_turnover rows" in str(exc)
            else:
                raise AssertionError("post-OOS snapshot must never authorize historical OOS")


def test_capture_metadata_contract() -> None:
    """READY, fingerprint and capture-day checks survive the layer move."""
    import json
    from core.file_integrity import canonical_json_sha256
    from core.market_data_provider_snapshot import validate_provider_snapshot_payload
    from core.market_data_storage_contract import (
        MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT,
        MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME,
    )
    from services.market_data.provider_snapshot_repository import (
        validate_provider_snapshot_payload as legacy_validate_provider_snapshot_payload,
    )

    assert legacy_validate_provider_snapshot_payload is validate_provider_snapshot_payload
    identity = {
        "schema_version": 1, "provider": "FinMind",
        "snapshot_role": "neutral_provider_bootstrap", "status": "READY",
        "as_of_date": "2020-12-28", "registry_fingerprint": "1" * 64,
        "manifest_fingerprint": "2" * 64,
        "historical_instrument_count": 0, "total_requests": 0,
        "total_rows": 0, "artifacts_fingerprint": canonical_json_sha256([]),
        "datasets": [],
    }
    snapshot = {
        **identity, "snapshot_fingerprint": canonical_json_sha256(identity),
        "finalized_at": "2020-12-28T08:00:00Z",
    }
    assert validate_provider_snapshot_payload(snapshot) is snapshot
    for bad in ({**snapshot, "status": "DUE"}, {**snapshot, "snapshot_fingerprint": "bad"}):
        try:
            validate_provider_snapshot_payload(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid Provider Snapshot metadata was accepted")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT
        first = base / "snapshot_a" / MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME
        first.parent.mkdir(parents=True)
        first.write_text(json.dumps(snapshot), encoding="utf-8")
        evidence = preflight_trade_count_capture_evidence(root)
        assert evidence["earliest_capture_day"] == "2020-12-28"
        assert evidence["verified_manifest_count"] == 1
        assert evidence["eligible_snapshots"][0][1] == snapshot["snapshot_fingerprint"]

        # The metadata-only preflight accepts the capture date, not ledger/parquet
        # provenance. The provider service is still the required full-data gate.
        from services.breakout_quality.trade_count_source import load_verified_trade_count_observations
        try:
            load_verified_trade_count_observations(root)
        except (OSError, ValueError):
            pass
        else:
            raise AssertionError("Metadata-only readiness bypassed full provider proof")

        first.write_text(json.dumps({**snapshot, "snapshot_fingerprint": "bad"}), encoding="utf-8")
        try:
            preflight_trade_count_capture_evidence(root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Tampered snapshot fingerprint was accepted by preflight")
        first.write_text(json.dumps(snapshot), encoding="utf-8")

        second = base / "snapshot_b" / MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME
        second.parent.mkdir(parents=True)
        second.write_text(json.dumps({**snapshot, "finalized_at": "2020-12-29T08:00:00Z"}), encoding="utf-8")
        try:
            preflight_trade_count_capture_evidence(root)
        except ValueError as exc:
            assert "不同capture日" in str(exc)
        else:
            raise AssertionError("Identical snapshot fingerprint accepted multiple capture dates")


def test_upstream_pit_preflight() -> None:
    """The canonical plan blocks a missing non-rebuildable source before fit."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from filters.breakout_quality.artifact_dependency_registry import (
        ARTIFACT_TRADE_COUNT_SOURCE,
        collect_model_upstream_preparation_plan,
        dependency_types_for,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        status = SimpleNamespace(
            ready=True, summary_path=root / "dataset_summary.json",
            summary={"policy": {}, "dataset_artifacts": {}},
            refresh_mode="none", reasons=(),
        )
        profiles = {
            "MR-13AO": "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise",
            "MR-13CB": "daily_universal_shared_safety_hs_conditional_mfe_pit_trade_count_placebo_full_list_ndcg_pairwise",
            "MR-13CA": "daily_universal_shared_safety_hs_conditional_mfe_pit_trade_count_full_list_ndcg_pairwise",
        }
        with patch(
            "filters.breakout_quality.artifact_dependency_registry.collect_dataset_readiness",
            return_value=status,
        ), patch(
            "filters.breakout_quality.trade_count_capture_preflight._snapshot_refs",
            return_value=[(pd.Timestamp("2026-09-15"), "a" * 64)],
        ):
            for model_id, slug in profiles.items():
                profile = get_breakout_quality_experiment_profile(slug)
                plan = collect_model_upstream_preparation_plan(
                    root, filter_id="breakout_quality_v1",
                    model_architecture=profile.model_architecture,
                    experiment_profile=slug, dataset="full", max_tickers=0,
                )
                if model_id == "MR-13AO":
                    assert plan.overall_status == "READY", plan
                    assert all(row.artifact_key != f"model-upstream:{ARTIFACT_TRADE_COUNT_SOURCE}" for row in plan.actions)
                    assert ARTIFACT_TRADE_COUNT_SOURCE not in dependency_types_for(
                        "model_checkpoint", experiment_profile=slug,
                    )
                    assert dependency_types_for("forward_score", experiment_profile=slug) == ("model_checkpoint",)
                else:
                    assert plan.overall_status == "BLOCKED", plan
                    blocked = [row for row in plan.actions if row.action == "BLOCKED"]
                    assert len(blocked) == 1 and blocked[0].artifact_key == f"model-upstream:{ARTIFACT_TRADE_COUNT_SOURCE}"
                    assert "Research cutoff" in blocked[0].description
                    assert ARTIFACT_TRADE_COUNT_SOURCE in dependency_types_for(
                        "model_checkpoint", experiment_profile=slug,
                    )
        with patch(
            "filters.breakout_quality.artifact_dependency_registry.collect_dataset_readiness",
            return_value=status,
        ), patch(
            "filters.breakout_quality.trade_count_capture_preflight._snapshot_refs",
            return_value=[(pd.Timestamp("2020-12-29"), "b" * 64)],
        ):
            slug = profiles["MR-13CA"]
            plan = collect_model_upstream_preparation_plan(
                root, filter_id="breakout_quality_v1",
                model_architecture=get_breakout_quality_experiment_profile(slug).model_architecture,
                experiment_profile=slug, dataset="full", max_tickers=0,
            )
            assert plan.overall_status == "READY", plan
            source = [row for row in plan.actions if row.artifact_key == f"model-upstream:{ARTIFACT_TRADE_COUNT_SOURCE}"][0]
            assert source.action == "REUSE" and "實際讀取仍須驗證" in source.description


def test_comparison_blocks_before_fit() -> None:
    """A later unavailable profile must not trigger earlier model rebuilds."""
    from io import StringIO
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from unittest.mock import patch
    from services.research import breakout_quality_application as app

    models = (
        ("MR-13AO", "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"),
        ("MR-13CB", "daily_universal_shared_safety_hs_conditional_mfe_pit_trade_count_placebo_full_list_ndcg_pairwise"),
    )
    class ModelContract:
        report = {}
    statuses = []
    ready = SimpleNamespace(blocked=False, actions=())
    blocked = SimpleNamespace(blocked=True, actions=(SimpleNamespace(
        action="BLOCKED", description="A1 PIT Data Gate BLOCKED: Research cutoff",
    ),))
    with (
        patch.object(app, "get_breakout_quality_model_test_settings", return_value=SimpleNamespace(model_profiles=models)),
        patch.object(app, "get_breakout_quality_workflow_settings", side_effect=lambda experiment_profile: SimpleNamespace(experiment_profile=experiment_profile)),
        patch.object(app, "_load_reusable_continuous_forward_contract", side_effect=((ModelContract(), None), (None, "model.pt missing"))),
        patch.object(app, "_has_continuous_forward_fitted_candidate", return_value=False),
        patch.object(app, "_collect_continuous_research_input_plan", side_effect=(blocked,)),
        patch.object(app, "_print_model_action_status", side_effect=lambda rows: statuses.extend(rows)),
        patch.object(app, "_ensure_continuous_forward_model_report") as fit,
        patch.object(app, "_write_standard_model_comparison_report") as report,
        redirect_stdout(StringIO()) as output,
    ):
        code = app._run_configured_model_comparison("synthetic", rolling=False)
    assert code == 2
    assert statuses[0][2] == "REUSE" and statuses[1][2] == "BLOCKED"
    assert "A1 PIT Data Gate" in statuses[1][3]
    assert "尚未開始補建任何模型" in output.getvalue()
    fit.assert_not_called()
    report.assert_not_called()


def test_normalization() -> None:
    x=np.array([[0, 10, 20, 100, 5], [0, 0, 0, 0, 0]], dtype=float)
    m=np.array([[0, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=bool)
    with warnings.catch_warnings(record=True) as captured:
        y=normalized_trade_count_windows(x,m)
    assert not captured and y.shape==(2,5)
    assert y[0,0]==0 and y[0,1]!=y[0,3] and np.all(y[1]==0)
    placebo=normalized_trade_count_windows(x,m,placebo=True)
    assert np.array_equal(placebo,np.zeros_like(placebo))
    try:
        normalized_trade_count_windows(np.array([[1,-4]], dtype=float),np.array([[1,1]],bool))
    except ValueError:
        pass
    else:
        raise AssertionError("negative valid counts accepted")


def test_lazy_feature_bank() -> None:
    dates=pd.bdate_range("2020-01-02",periods=330)
    v=np.arange(len(dates),dtype=float)
    prices=pd.DataFrame({
        "Open":100+v*0.1,"High":101+v*0.1,"Low":99+v*0.1,
        "Close":100+v*0.1,"Volume":1000+10*v,
    },index=dates)
    # These are independently timestamped synthetic observations, not historical FinMind proof.
    observations={ticker:{date:(float(50+i*4+offset),date) for i,date in enumerate(dates)} for ticker,offset in (("2330",0),("0050",200))}
    source=dict(rows=observations,calendar=dates,ticker_names=("2330",),benchmark_ticker="0050")
    kwargs=dict(frames=(prices,),benchmark=prices,ticker_ids=np.array([0,0],dtype=int),
                source_positions=np.array([310,315],dtype=int),benchmark_positions=np.array([310,315],dtype=int),
                policy=replace(DEFAULT_LABEL_POLICY, feature_window_bars=300))
    baseline=LazyDailyFeatureBank(**kwargs)
    real=LazyDailyFeatureBank(**kwargs,trade_count_source=source)
    placebo=LazyDailyFeatureBank(**kwargs,trade_count_source=source,trade_count_placebo=True)
    a=baseline[:];b=real[:];c=placebo[:]
    assert a.shape==(2,300,10) and b.shape==c.shape==(2,300,14)
    assert len(real.feature_columns)==14 and tuple(real.feature_columns[10:])==TRADE_COUNT_FEATURE_NAMES
    assert np.array_equal(b[:,:,:10],a) and np.array_equal(c[:,:,:10],a), "AO input changed"
    assert np.array_equal(b[:,:,12:14],c[:,:,12:14]) and np.count_nonzero(b[:,:,10:12])>0
    assert not np.any(c[:,:,10:12]), "placebo must zero count values only"
    assert not np.any(b[:,0,12:14] == 0), "expected synthetic coverage"


def test_model_seed_and_gradients() -> None:
    import torch
    AO="inception_time_shared_safety_mfe_v1"
    CA="inception_time_shared_safety_mfe_pit_trade_count_v1"
    CB="inception_time_shared_safety_mfe_pit_trade_count_placebo_v1"
    models={}
    for arch,n in ((AO,10),(CA,14),(CB,14)):
        torch.manual_seed(42)
        models[arch]=build_model(n,0,architecture=arch)
    ao=models[AO]
    source=torch.randn(2,300,10)
    extra=torch.randn(2,300,4)
    for arch in (CA,CB):
        model=models[arch]
        assert len(model.state_dict())==len(ao.state_dict())+1
        for name,w in ao.state_dict().items():
            assert torch.equal(w,model.state_dict()[name]),name
        assert torch.count_nonzero(model.trade_count_adapter.weight).item()==0
        model.eval();ao.eval()
        with torch.no_grad():
            actual=model(torch.cat([source,extra],dim=-1),None)
            expected=ao(source,None)
        assert torch.allclose(actual,expected,rtol=0,atol=0),"zero adapter changed AO initial logits"
        model.train();model.zero_grad(set_to_none=True)
        loss=model(torch.cat([source,extra],dim=-1),None).sum()
        loss.backward()
        g=model.trade_count_adapter.weight.grad
        assert g is not None and torch.isfinite(g).all() and torch.count_nonzero(g).item()>0
    assert get_architecture_descriptor(CA).has_capability("lagged_trade_count_input")
    assert get_architecture_descriptor(CB).has_capability("lagged_trade_count_placebo")


def test_source_lineage() -> None:
    from types import SimpleNamespace
    from services.breakout_quality.fitted_model_artifacts import source_training_identity

    baseline=source_training_identity(SimpleNamespace(summary={},target_manifest={}))
    assert "input_trade_count_source" not in baseline, "AO legacy artifact identity changed"
    source={"fingerprint":"synthetic_trade_count_source_hash", "source_contract":"t_minus_1"}
    supplement=source_training_identity(SimpleNamespace(
        summary={"input_trade_count_source":source},target_manifest={},
    ))
    assert supplement["input_trade_count_source"]==source


def test_profile_contract() -> None:
    real_id, real_slug=BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    assert real_id=="MR-13CA"
    base=get_breakout_quality_experiment_profile("daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise")
    for m,slug in (("MR-13CA",real_slug),("MR-13CB","daily_universal_shared_safety_hs_conditional_mfe_pit_trade_count_placebo_full_list_ndcg_pairwise")):
        profile=get_breakout_quality_experiment_profile(slug)
        spec=get_continuous_ranker_research_spec(slug)
        assert spec.model_research_id==m
        for field in ("training_objective","continuous_target_id","loss_name","epoch_selection_metric","training_sample_scope","training_label_scope","optimizer_name","training_sampling_mode"):
            assert getattr(profile,field)==getattr(base,field),field
        assert profile.model_architecture!=base.model_architecture
        assert not spec.selection_pit_authorized and not spec.current_time_validation_authorized


def test_train_epoch() -> None:
    """Exercise the same _train_epoch route used by normal AO training."""
    import torch
    from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
    )
    from services.breakout_quality import train_continuous_ranker as trainer

    torch.set_num_threads(1)
    profile=get_breakout_quality_experiment_profile(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[1])
    torch.manual_seed(42)
    model=build_model(14,0,architecture=profile.model_architecture)
    x=np.random.default_rng(42).normal(size=(12,300,14)).astype(np.float32)
    x[:,:,12:14]=1.0
    group_dates=np.array(["2025-01-06"]*6+["2025-01-07"]*6,dtype="datetime64[D]")
    raw=np.ones(12,dtype=np.float32)
    pct=np.full(12,.5,dtype=np.float32)
    table=pd.DataFrame({"date":group_dates,
        "target_favorable_r":[0.2,0.8,0.5,1.2,0.6,1.0,0.1,0.7,1.2,0.8,0.3,0.9],
        "target_adverse_r":[0.7,0.2,0.9,0.4,0.6,0.1,0.3,0.4,1.0,0.5,0.8,0.1],
    })
    target=trainer._training_target_for_profile(profile,raw,pct,table)
    assert target.shape==(12,2)
    plan=resolve_torch_execution_plan(torch,requested_device="cpu",mixed_precision=False,
                mixed_precision_dtype="auto",deterministic_algorithms=False,allow_tf32=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=0.0001)
    old=model.trade_count_adapter.weight.detach().clone()
    loss=trainer._train_epoch(
        torch,model,optimizer,
        feature_bank=x,group_context=np.zeros((12,0),dtype=np.float32),
        group_ids=np.arange(12,dtype=np.int64),percentile_target=target,
        group_dates=group_dates,training_objective=profile.training_objective,
        batch_size=12,seed=42,gradient_clip_norm=1.0,
        plan=plan,grad_scaler=None,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        secondary_pair_scope=CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        secondary_pair_scope_threshold=0.5,
    )
    assert np.isfinite(loss) and float(loss)>0.0
    assert not torch.equal(model.trade_count_adapter.weight,old)



if __name__ == "__main__":
    for name in ("trade_count_source","capture_metadata_contract","upstream_pit_preflight","comparison_blocks_before_fit","normalization","lazy_feature_bank","model_seed_and_gradients","profile_contract","source_lineage","train_epoch"):
        globals()[f"test_{name}"]()
        print(f"PASS trade count A1: {name}")

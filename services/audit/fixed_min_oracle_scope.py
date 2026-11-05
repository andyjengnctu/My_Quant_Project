"""Fixed-Min policy scope for the registered Target Oracle (hindsight only).

The existing P16 Oracle is left untouched.  This source adapter uses the SAME
counterfactual Min path calculator, percentile ranker, canonical score injection,
and Broker-Cash portfolio replay, but the immutable Fixed-Min parameter policy.
It must never read a P16 optimizer schedule or publish a DL training score.
"""
from __future__ import annotations

from hashlib import sha256
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, TARGET_ORACLE_MINTRADE_STOCK_WORKERS
from config.robustness_benchmark import TARGET_ORACLE_ROBUSTNESS_BENCHMARK_ID
from config.breakout_quality import BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
from core.audit_registry import AUDIT_REUSABLE_REPORTS, oracle_target_semantics
from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.breakout_quality_runtime import FIXED_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID
from core.file_integrity import canonical_json_sha256, compute_file_sha256
from core.fixed_min_policy import (
    FIXED_MIN_ID, FIXED_MIN_TARGET, fixed_min_scientific_sha256, fixed_min_signature,
    fixed_min_artifact_path,
    inspect_fixed_min_robustness_parameter_artifact,
    ensure_fixed_min_robustness_parameter_artifact,
)
from core.strategy_compare_policy import get_strategy_comparison_settings
from core.training_policy import get_robustness_benchmark_policy_snapshot
from core.dataset_profiles import get_dataset_dir
from filters.breakout_quality.daily_trade40_target import (
    rank_trade40_outcome_frame, collect_canonical_exchange_sessions,
)
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from filters.breakout_quality.strategy_compare_sources import (
    load_strategy_param_evaluation_view, _validate_requested_param_policy,
    _build_controlled_param_source_pair, OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
)
from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES
from filters.breakout_quality.strategy_compare_contracts import COMPARISON_MODE_SCORE_RANKING
from filters.breakout_quality.runtime import audit_oracle_truth_score_context
from services.breakout_quality.fixed_min_source import fixed_min_source
from services.breakout_quality.trade_path_target_source import registered_trade_path_source
from services.audit.daily_trade40_oracle import (
    _run_ticker_outcomes, truth_producer_fingerprint,
)
from services.audit.oracle_replay_artifacts import (
    price_inventory_sha256, _validate_observation as validate_shared_oracle_observation,
)
from services.audit.oracle_truth_references import verify_truth_score_membership
from services.research.oracle_artifact_contract import (
    dataset_input_contract, portfolio_candidate_execution_sha256,
    replay_implementation_sha256, broker_cash_economics_contract,
)
from services.research.strategy_compare_replay import _run_scenario
from services.audit.target_oracle_romd import _metrics, _aggregate, OracleBlockedError
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY

SOURCE = AUDIT_REUSABLE_REPORTS["breakout_quality"]["target_oracle_romd"]["source"]
OUTPUT_SUBDIR = "breakout_quality/reusable/target_oracle_romd/fixed_min_scope_v1"
CONTRACT_ID = "fixed_min_netpnl_oracle_same_portfolio_v1"
BASELINE = "C139"
COMPARISONS = SOURCE["fixed_min_oracle_comparisons"]


def _rows() -> tuple[dict, ...]:
    """Resolve complete Fixed-Min treatments from the audited Registry."""
    values = tuple(dict(row) for row in COMPARISONS)
    selectors = {str(row.get("selector")) for row in values}
    ids = [row.get("strategy_arm_id") for row in values]
    if (len(values) != 2 or selectors != {"exact", "direct"}
            or len(set(ids)) != len(values) or any(not arm for arm in ids)):
        raise OracleBlockedError("FixedMin Oracle Registry requires one Exact and one Direct arm")
    for row in values:
        expected_target = (FIXED_MIN_TARGET if row["selector"] == "exact"
                           else f"{FIXED_MIN_TARGET}_Direct")
        if (row.get("target") != expected_target or
                oracle_target_semantics(row) != (
                    FIXED_MIN_TARGET, row["selector"].capitalize(), "Breakout")):
            raise OracleBlockedError("FixedMin Oracle Registry target/selector differs")
    return tuple(sorted(values, key=lambda row: (row["selector"] != "exact")))


def _by_selector() -> dict[str, dict]:
    return {row["selector"]: row for row in _rows()}


def _source_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    original = truth_producer_fingerprint("net_pnl")
    files = (
        "services/audit/fixed_min_oracle_scope.py",
        "services/breakout_quality/fixed_min_source.py",
        "core/fixed_min_policy.py", "services/research/strategy_compare_replay.py",
        "filters/breakout_quality/runtime.py",
        "filters/breakout_quality/profile_ranker_data.py",
        "filters/breakout_quality/daily_ranker_data.py",
    )
    cohort_name = str(SOURCE["daily_trade40_profile"])
    cohort = get_breakout_quality_experiment_profile(cohort_name)
    cohort_policy = get_continuous_ranker_execution_recipe(cohort_name).target_policy
    return canonical_json_sha256({
        "original_min_path_source": original,
        "fixed_min_source_files": {f: compute_file_sha256(root / f) for f in files},
        "shared_portfolio_replay": replay_implementation_sha256(),
        "shared_portfolio_candidate_execution": portfolio_candidate_execution_sha256(),
        "feature_cohort_contract": {
            "name": cohort_name,
            "architecture": cohort.model_architecture,
            "sample_scope": cohort.training_sample_scope,
            "target_id": cohort.continuous_target_id,
            "materialization_mode": cohort_policy.materialization_mode,
            "horizon_bars": cohort_policy.horizon_bars,
            "sample_universe_horizon_bars": cohort_policy.sample_universe_horizon_bars,
        },
        "contract": CONTRACT_ID,
    })


def _seeds() -> tuple[int, ...]:
    benchmark = get_robustness_benchmark_policy_snapshot(
        TARGET_ORACLE_ROBUSTNESS_BENCHMARK_ID)
    result = tuple(int(s) for s in benchmark["resolved_seeds"])
    if len(result) != 8 or len(set(result)) != 8:
        raise OracleBlockedError("FixedMin Oracle requires the registered canonical eight seeds")
    return result


def _settings():
    settings = get_strategy_comparison_settings(SOURCE["strategy_profile_id"])
    if not (settings.dataset == "full" and settings.rotation == "off"
            and settings.max_positions == 10 and settings.start_date and settings.end_date):
        raise OracleBlockedError("FixedMin Oracle Research portfolio policy differs")
    by_selector = _by_selector()
    exact_id = by_selector["exact"]["strategy_arm_id"]
    direct_id = by_selector["direct"]["strategy_arm_id"]
    ids = (BASELINE, exact_id, direct_id)
    arms = {key: settings.arms[key] for key in ids}
    if (arms[BASELINE].dl_enabled or
            any(not arms[key].dl_enabled for key in (exact_id, direct_id)) or
            len({arms[key].param_source for key in arms}) != 1 or
            arms[BASELINE].param_source != FIXED_MIN_ID or
            any(arms[key].param_policy != "base-finalist-best" or
                arms[key].rule_policy != "all_off" or
                arms[key].entry_strategy != "Breakout" for key in arms) or
            arms[exact_id].dl_id != arms[direct_id].dl_id or
            arms[exact_id].dl_runtime_mode != "resource-aware-continuous-score-constrained-optimal" or
            arms[direct_id].dl_runtime_mode != "resource-aware-continuous-score-no-k-no-r0"):
        raise OracleBlockedError("FixedMin baseline, ranker or selector is not the formal Strategy Registry")
    if (dict(arms[exact_id].dl_runtime_options or {}).get("preserve_k_r0") is not True or
            dict(arms[exact_id].dl_runtime_options or {}).get("constrained_solver") != "exact_branch_and_bound_v1" or
            dict(arms[direct_id].dl_runtime_options or {}).get("preserve_k") is not False or
            dict(arms[direct_id].dl_runtime_options or {}).get("preserve_r0") is not False):
        raise OracleBlockedError("FixedMin Oracle selectors changed")
    rank_source = settings.dl_sources[arms[exact_id].dl_id]
    profile = get_breakout_quality_experiment_profile(rank_source.experiment_profile)
    if (profile.continuous_target_id != FIXED_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID or
            rank_source.score_source != "selection_point_in_time"):
        raise OracleBlockedError("FixedMin Oracle source is not the MR-13CV FIT Target")
    return settings, arms


def _source_state(root: Path) -> str:
    inspection = inspect_fixed_min_robustness_parameter_artifact(root)
    action = str(inspection["action"])
    if action == "BLOCKED":
        raise OracleBlockedError(f"FixedMin parameter SSOT drift: {inspection['status']}")
    return action


def _root(root: Path) -> Path:
    return Path(root) / AUDIT_OUTPUT_ROOT / OUTPUT_SUBDIR


def _effective_dataset_inputs(root: Path) -> dict[str, Any]:
    try:
        witness = dataset_input_contract(root, str(SOURCE["filter_id"]))
    except (ValueError, OSError, TypeError) as exc:
        raise OracleBlockedError(f"FixedMin Oracle Dataset provenance invalid: {exc}") from exc
    if not isinstance(witness, Mapping) or not witness:
        raise OracleBlockedError("FixedMin Oracle requires canonical Dataset input contract")
    return dict(witness)


def _identity(*, root: Path, seed: int, target: str,
              price_sha: str | None = None,
              source_sha: str | None = None,
              dataset_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings, arms = _settings()
    if target not in {BASELINE, *[row["target"] for row in _rows()]}:
        raise OracleBlockedError("FixedMin Oracle Target unknown")
    treatments = {row["target"]: row for row in _rows()}
    row = treatments.get(target)
    arm = arms[BASELINE if row is None else row["strategy_arm_id"]]
    source_id = arms[_by_selector()["exact"]["strategy_arm_id"]].dl_id
    return {
        "contract": CONTRACT_ID, "target": target, "seed": seed,
        "benchmark": TARGET_ORACLE_ROBUSTNESS_BENCHMARK_ID,
        "dataset": settings.dataset,
        "research_cutoff": str(settings.end_date),
        "evaluation_start": str(settings.start_date),
        "evaluation_end": str(settings.end_date),
        "feature_target_id": FIXED_MINTRADE_FULLCYCLE_NETPNL_TARGET_ID,
        "score_start": str(settings.dl_sources[source_id].point_in_time_score_start_date),
        "source_sha256": source_sha if source_sha is not None else _source_sha(),
        "price_sha256": (price_sha if price_sha is not None else
                         price_inventory_sha256(root, settings.dataset, certify_missing=False)),
        "dataset_inputs": dict(dataset_inputs) if dataset_inputs is not None else _effective_dataset_inputs(root),
        "fixed_min_param_sha256": fixed_min_scientific_sha256(),
        "fixed_min_params_signature": fixed_min_signature(),
        "entry_strategy": "Breakout", "selector": ("baseline" if row is None
            else row["selector"]),
        "ranking_mode": arm.dl_runtime_mode,
        "ranking_options": dict(arm.dl_runtime_options or {}),
        "max_positions": settings.max_positions, "rotation": settings.rotation,
        "broker_economics": broker_cash_economics_contract(),
    }


def _receipt_path(root: Path, identity: Mapping[str, Any]) -> Path:
    return _root(root) / "replay_cache" / str(identity["seed"]) / (
        str(identity["target"]) + "_" + canonical_json_sha256(identity) + ".json")


def _validate_fixed_receipt_metrics(identity: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict:
    """Share the canonical Oracle observation and annual-coverage contracts."""
    try:
        result = validate_shared_oracle_observation(
            metrics, target=str(identity["target"]), seed=int(identity["seed"]),
            baseline_target=BASELINE,
        )
        _aggregate([result], target=str(identity["target"]),
                   period_start=str(identity["evaluation_start"]),
                   period_end=str(identity["evaluation_end"]))
        return result
    except (ValueError, TypeError, KeyError, OverflowError, OracleBlockedError) as exc:
        raise OracleBlockedError(f"FixedMin Oracle scientific receipt invalid: {exc}") from exc


def _read_receipt(root: Path, identity: Mapping[str, Any]) -> dict | None:
    path = _receipt_path(root, identity)
    if not path.is_file():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        if (content.get("identity") != dict(identity) or
                content.get("signature") != canonical_json_sha256({
                    "identity": content.get("identity"), "metrics": content.get("metrics"),
                    "rank_sha256": content.get("rank_sha256"),
                    "cohort_sha256": content.get("cohort_sha256"),
                })):
            raise ValueError("signed receipt integrity mismatch")
        _validate_fixed_receipt_metrics(identity, content["metrics"])
        if (identity["target"] == BASELINE and
                (content.get("rank_sha256"), content.get("cohort_sha256")) !=
                ("baseline_no_score", "baseline_no_score")):
            raise ValueError("FixedMin baseline must not contain an injected future score")
        if identity["target"] != BASELINE and not all(
                isinstance(content.get(key), str) and len(content[key]) == 64 and
                all(c in "0123456789abcdef" for c in content[key])
                for key in ("rank_sha256", "cohort_sha256")):
            raise ValueError("FixedMin scored receipt lacks certified truth SHA256")
        return content
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise OracleBlockedError(f"FixedMin Oracle receipt corrupt: {path.name}: {exc}") from exc


def _write_receipt(root: Path, identity: Mapping[str, Any], *, metrics: Mapping[str, Any],
                   score_sha: str, cohort_sha: str) -> None:
    _validate_fixed_receipt_metrics(identity, metrics)
    path = _receipt_path(root, identity)
    payload = {"identity": dict(identity), "metrics": dict(metrics),
               "rank_sha256": score_sha, "cohort_sha256": cohort_sha}
    content = {**payload, "signature": canonical_json_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Immutable, same-identity output. A conflicting writer must not overwrite.
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
    except FileExistsError:
        previous = _read_receipt(root, identity)
        if previous != content:
            raise OracleBlockedError("FixedMin Oracle concurrent replay differs from signed result")


def preview_fixed_min_oracle(*, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    _rows()
    _settings()
    source_action = _source_state(root)
    seeds = _seeds()
    decisions = {}
    records = {}
    identities = {}
    # One content witness per preview. Recomputing the entire OHLCV inventory
    # once per seed/selector would make a pure REUSE plan needlessly expensive.
    settings, _ = _settings()
    try:
        price_sha = price_inventory_sha256(root, settings.dataset, certify_missing=False)
    except ValueError as exc:
        raise OracleBlockedError(f"FixedMin Oracle price provenance BLOCKED: {exc}") from exc
    source_sha = _source_sha()
    dataset_inputs = _effective_dataset_inputs(root)
    for seed in seeds:
        for target in (BASELINE, *[row["target"] for row in _rows()]):
            identity = _identity(root=root, seed=seed, target=target,
                                 price_sha=price_sha, source_sha=source_sha,
                                 dataset_inputs=dataset_inputs)
            identities[(seed, target)] = identity
            receipt = _read_receipt(root, identity)
            decisions[(seed, target)] = "REUSE" if receipt else "RUN"
            if receipt:
                records[(seed, target)] = receipt
    return {"seed_set": seeds, "preflight": decisions, "completed": records,
            "identities": identities, "source_action": source_action,
            "targets": (BASELINE, *[r["target"] for r in _rows()])}


def _truth_identity(root: Path, *, plan: Mapping[str, Any] | None = None) -> dict:
    exact_target = _by_selector()["exact"]["target"]
    ref = (plan["identities"][(plan["seed_set"][0], exact_target)]
           if plan is not None else
           _identity(root=root, seed=_seeds()[0], target=exact_target))
    return {k: ref[k] for k in (
        "contract", "dataset", "research_cutoff", "evaluation_start",
        "evaluation_end", "feature_target_id", "score_start", "source_sha256",
        "price_sha256", "dataset_inputs", "fixed_min_param_sha256",
        "fixed_min_params_signature")}


def _truth_path(root: Path, identity: Mapping[str, Any]) -> Path:
    return _root(root) / "truth_cache" / (canonical_json_sha256(identity) + ".npz")


def _rank_sha(scores: Mapping[tuple[str, str], float]) -> str:
    digest = sha256()
    for (ticker, day), score in sorted(scores.items()):
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise OracleBlockedError("FixedMin Oracle score out of finite percentile range")
        digest.update(f"{ticker}|{day}|{score:.17g}\n".encode("utf-8"))
    return digest.hexdigest()


def _read_truth(root: Path, identity: Mapping[str, Any]):
    path = _truth_path(root, identity)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "ticker", "day", "score"}:
                raise ValueError("truth fields differ")
            metadata = json.loads(str(archive["metadata"].item()))
            ticker, day, score = (archive[k] for k in ("ticker", "day", "score"))
        if metadata.get("identity") != dict(identity) or not len(score) or len(score) != len(day) or len(score) != len(ticker):
            raise ValueError("truth identity/shape differs")
        scores = dict(zip(zip(ticker.tolist(), day.tolist()), score.tolist()))
        if len(scores) != len(score) or metadata["score_sha256"] != _rank_sha(scores):
            raise ValueError("truth rank SHA differs")
        if metadata.get("signature") != canonical_json_sha256({
                "identity": dict(identity), "score_sha256": metadata["score_sha256"],
                "cohort_sha256": metadata["cohort_sha256"],
                "rows": len(scores)}):
            raise ValueError("truth evidence signature differs")
        eligible = _eligible_group(root, identity)
        certified_keys = _cohort_keys(eligible)
        certified_sha = _cohort_sha(eligible)
        if (metadata["cohort_sha256"] != certified_sha or
                len(scores) != len(certified_keys)):
            raise ValueError("truth feature-eligible cohort provenance differs")
        verify_truth_score_membership(
            scores, {"feature_eligible_keys_sha256": certified_sha})
        return scores, metadata["score_sha256"], metadata["cohort_sha256"]
    except (ValueError, OSError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile) as exc:
        raise OracleBlockedError(f"FixedMin Oracle truth corrupt, cannot reuse: {exc}") from exc


def _eligible_group(root: Path, identity: Mapping[str, Any]) -> pd.DataFrame:
    settings, arms = _settings()
    source_id = arms[_by_selector()["exact"]["strategy_arm_id"]].dl_id
    source = settings.dl_sources[source_id]
    rank_profile = get_breakout_quality_experiment_profile(source.experiment_profile)
    # A real-path FIT profile *must* receive its verified seed-scoped teacher.
    # The hindsight Oracle needs only its score-eligible stock/day population,
    # not a second FIT-label materialization followed by another path replay.
    # The formal Oracle Registry already names the generic daily feature-cohort
    # producer used by the other MinTrade hindsight treatments. It shares the
    # same architecture and information-date eligibility with MR-13CV.
    cohort_profile_name = str(SOURCE["daily_trade40_profile"])
    cohort_profile = get_breakout_quality_experiment_profile(cohort_profile_name)
    if (registered_trade_path_source(cohort_profile_name) is not None or
            cohort_profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS or
            rank_profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS or
            cohort_profile.model_architecture != rank_profile.model_architecture):
        raise OracleBlockedError("FixedMin Oracle feature cohort is not a compatible non-trade-path provider")
    bundle = load_profile_continuous_ranker_data(
        filter_id=str(SOURCE["filter_id"]),
        model_architecture=str(rank_profile.model_architecture),
        experiment_profile=cohort_profile_name, preload_feature_bank=False,
        allow_stale_source=False, project_root=root)
    group = pd.DataFrame(bundle.group_table).reset_index(drop=True)
    summary = dict(bundle.summary or {})
    target_contract = dict((bundle.target_manifest or {}).get("target_contract") or {})
    if (summary.get("dataset") != settings.dataset or
            summary.get("filter_id") != str(SOURCE["filter_id"]) or
            summary.get("sample_scope") != "daily_eligible_stock_days" or
            int(summary.get("score_eligible_sample_count", -1)) != len(group) or
            target_contract.get("target_id") != cohort_profile.continuous_target_id or
            not {"ticker", "date"}.issubset(group.columns)):
        raise OracleBlockedError("FixedMin Oracle feature-eligible source differs from PIT DL")
    group["ticker"] = group["ticker"].map(normalize_ticker)
    group["date"] = pd.to_datetime(group["date"], errors="raise").dt.normalize()
    group = group.loc[
        (group["date"] >= pd.Timestamp(identity["score_start"])) &
        (group["date"] <= pd.Timestamp(identity["evaluation_end"]))
    ].reset_index(drop=True)
    if group.empty or group.duplicated(["ticker", "date"]).any():
        raise OracleBlockedError("FixedMin Oracle missing unique eligible stock-day cohort")
    return group


def _cohort_keys(group: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(group["ticker"].tolist(), group["date"].dt.strftime("%Y-%m-%d").tolist()))


def _cohort_sha(group: pd.DataFrame) -> str:
    # CSV membership is independent of pandas Categorical category ordering.
    ordered = sorted(_cohort_keys(group))
    return sha256(pd.DataFrame.from_records(
        ordered, columns=["ticker", "date"]).to_csv(index=False).encode("utf-8")).hexdigest()


def _build_truth(root: Path, identity: Mapping[str, Any]):
    cached = _read_truth(root, identity)
    if cached is not None:
        return cached
    settings, _ = _settings()
    group = _eligible_group(root, identity)
    eligible_keys = _cohort_keys(group)
    cohort_sha = _cohort_sha(group)
    csv_inputs, duplicates = discover_dataset_csv_inputs(root, settings.dataset)
    if duplicates:
        raise OracleBlockedError("FixedMin Oracle duplicate Research OHLCV source")
    sources = {normalize_ticker(ticker): Path(path) for ticker, path in csv_inputs}
    if len(sources) != len(csv_inputs):
        raise OracleBlockedError("FixedMin Oracle normalized tickers conflict")
    exchange = collect_canonical_exchange_sessions(
        csv_inputs, benchmark_ticker=str(DEFAULT_LABEL_POLICY.benchmark_ticker),
        horizon_sessions=None)
    schedule = fixed_min_source(int(_seeds()[0])).schedule
    if (len(schedule) != 1 or schedule[0].source_id != FIXED_MIN_ID or
            schedule[0].params_signature != fixed_min_signature()):
        raise OracleBlockedError("FixedMin Oracle teacher parameter signature differs")
    jobs = []
    # Score eligibility may leave unused ticker categories after the OOS date
    # slice. Only observed tickers are valid Oracle worker jobs.
    for index, (ticker, values) in enumerate(group.groupby("ticker", sort=True, observed=True)):
        if ticker not in sources:
            raise OracleBlockedError(f"FixedMin Oracle eligible ticker missing OHLCV: {ticker}")
        jobs.append((index, ticker, str(sources[ticker]), pd.DatetimeIndex(values["date"])))
    records, _, _ = _run_ticker_outcomes(
        jobs, schedule=schedule, exchange=exchange, horizon=None,
        cutoff=pd.Timestamp(settings.end_date), metric="net_pnl",
        workers=int(TARGET_ORACLE_MINTRADE_STOCK_WORKERS), progress=None)
    data = pd.concat(records, ignore_index=True).sort_values(["date", "ticker"], kind="stable")
    if not {"ticker", "date", "realized_net_pnl"}.issubset(data.columns):
        raise OracleBlockedError("FixedMin Oracle path worker omitted required truth fields")
    actual_keys = list(zip(data["ticker"].map(normalize_ticker).tolist(),
                           pd.to_datetime(data["date"], errors="raise").dt.strftime("%Y-%m-%d").tolist()))
    if len(actual_keys) != len(eligible_keys) or set(actual_keys) != eligible_keys:
        raise OracleBlockedError("FixedMin Oracle path worker ticker/date membership differs from eligible cohort")
    scores = rank_trade40_outcome_frame(data, value_column="realized_net_pnl")
    if len(scores) != len(eligible_keys):
        raise OracleBlockedError("FixedMin Oracle target truth dropped DL-eligible candidates")
    try:
        verify_truth_score_membership(scores, {"feature_eligible_keys_sha256": cohort_sha})
    except ValueError as exc:
        raise OracleBlockedError(f"FixedMin Oracle ranked membership mismatch: {exc}") from exc
    score_sha = _rank_sha(scores)
    path = _truth_path(root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(scores)
    payload = {"identity": dict(identity), "score_sha256": score_sha,
               "cohort_sha256": cohort_sha, "rows": len(scores)}
    metadata = json.dumps({**payload, "signature": canonical_json_sha256(payload)}, sort_keys=True)
    import os, tempfile
    fd, temporary = tempfile.mkstemp(prefix=".fixed_min.", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, metadata=np.asarray(metadata),
                ticker=np.asarray([ticker for ticker, _ in keys]),
                day=np.asarray([day for _, day in keys]),
                score=np.asarray([scores[key] for key in keys], dtype=np.float64))
            stream.flush(); os.fsync(stream.fileno())
        # Non-overwriting publication; any concurrent different data blocks.
        try:
            os.link(temporary, path)
        except FileExistsError:
            previous = _read_truth(root, identity)
            if previous is None or previous[1:] != (score_sha, cohort_sha):
                raise OracleBlockedError("FixedMin Oracle same-identity truth publication conflict")
    finally:
        os.unlink(temporary)
    return scores, score_sha, cohort_sha


def execute_fixed_min_oracle(*, project_root: Path, approved_plan: Mapping[str, Any]) -> dict:
    root = Path(project_root).resolve()
    before = preview_fixed_min_oracle(project_root=root)
    approved = dict(approved_plan["preflight"])
    approved_identities = dict(approved_plan.get("identities") or {})
    if (tuple(approved_plan["seed_set"]) != before["seed_set"] or
            tuple(approved_plan.get("targets", ())) != before["targets"] or
            set(approved) != set(before["preflight"]) or
            approved_identities != before["identities"] or
            any((old == "REUSE" and before["preflight"][key] != "REUSE") or
                (old == "RUN" and before["preflight"][key] not in {"RUN", "REUSE"}) or
                old not in {"RUN", "REUSE"}
                for key, old in approved.items())):
        raise OracleBlockedError("FixedMin Oracle source/approval identity changed before execution")
    if all(action == "REUSE" for action in before["preflight"].values()):
        return collect_fixed_min_oracle_results(project_root=root)
    # The fixed parameter owner is the only producer; no Optimizer/P16 work.
    ensure_fixed_min_robustness_parameter_artifact(root)
    settings, arms = _settings()
    source, actual_param_sha = load_strategy_param_evaluation_view(
        fixed_min_artifact_path(root), evaluation_mode="rolling",
        start_date=str(settings.start_date), end_date=str(settings.end_date))
    if _source_state(root) != "REUSE":
        raise OracleBlockedError("FixedMin Oracle fixed parameter source changed after authorization")
    _validate_requested_param_policy(source, "base-finalist-best")
    if str(source["kind"]) != "rolling_active_param_ensemble":
        raise OracleBlockedError("FixedMin Oracle params became a non-rolling source")
    kind, base_params, ranked_params, *_ = _build_controlled_param_source_pair(
        source, filter_id=str(SOURCE["filter_id"]),
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD), fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        shared_param_overrides=ALL_RULE_FILTERS_OFF_OVERRIDES)
    truth = None
    data_dir = Path(get_dataset_dir(str(root), settings.dataset)).resolve()
    for seed in before["seed_set"]:
        baseline = None
        for target in before["targets"]:
            identity = before["identities"][(seed, target)]
            previous = _read_receipt(root, identity)
            if previous:
                if target == BASELINE:
                    baseline = previous["metrics"]
                continue
            if target == BASELINE:
                played = _run_scenario(
                    name=f"Fixed Min C139 Oracle baseline S={seed}", data_dir=data_dir,
                    param_source_kind=kind, params=base_params,
                    start_date=str(settings.start_date), end_date=str(settings.end_date),
                    max_positions=settings.max_positions, enable_rotation=False, quiet=True)
                score_sha = "baseline_no_score"
                cohort_sha = "baseline_no_score"
            else:
                if truth is None:
                    truth = _build_truth(root, _truth_identity(root, plan=before))
                scores, score_sha, cohort_sha = truth
                arm_id = next(row["strategy_arm_id"] for row in _rows() if row["target"] == target)
                arm = arms[arm_id]
                with audit_oracle_truth_score_context(
                    scores, ranking_policy=str(arm.dl_runtime_mode),
                    ranking_options=arm.dl_runtime_options, key_mode="information_date",
                    missing_key_policy="unavailable_outside_mature_feature_universe",
                    score_start=str(settings.dl_sources[arms[_by_selector()["exact"]["strategy_arm_id"]].dl_id].point_in_time_score_start_date)):
                    played = _run_scenario(
                        name=f"Fixed Min Oracle {target} S={seed}", data_dir=data_dir,
                        param_source_kind=kind, params=ranked_params,
                        start_date=str(settings.start_date), end_date=str(settings.end_date),
                        max_positions=settings.max_positions, enable_rotation=False, quiet=True)
            metrics = _metrics(played, target=target)
            metrics["seed"] = seed
            if target == BASELINE:
                baseline = metrics
            else:
                if baseline is None:
                    receipt = _read_receipt(root, before["identities"][(seed, BASELINE)])
                    if receipt is None:
                        raise OracleBlockedError("FixedMin Oracle missing same-seed C139 benchmark")
                    baseline = receipt["metrics"]
                metrics["delta_romd_vs_c139"] = metrics["romd"] - baseline["romd"]
            _write_receipt(root, identity, metrics=metrics,
                           score_sha=score_sha, cohort_sha=cohort_sha)
            print(f"[FixedMin Oracle DONE] seed={seed} {target} RoMD={metrics['romd']:.2f}", flush=True)
    return collect_fixed_min_oracle_results(project_root=root)


def collect_fixed_min_oracle_results(*, project_root: Path) -> dict:
    preview = preview_fixed_min_oracle(project_root=project_root)
    root = Path(project_root).resolve()
    result = {}
    for target in preview["targets"]:
        if any(preview["preflight"][(seed, target)] != "REUSE" for seed in preview["seed_set"]):
            continue
        records = [preview["completed"][(seed, target)]["metrics"] for seed in preview["seed_set"]]
        if target != BASELINE:
            for record in records:
                original = preview["completed"].get((record["seed"], BASELINE))
                if original is None or abs(record["delta_romd_vs_c139"] - (
                        record["romd"] - original["metrics"]["romd"])) > 1e-9:
                    raise OracleBlockedError("FixedMin Oracle C139 paired outcome mismatch")
        item = _aggregate(records, target=target,
            period_start=str(_settings()[0].start_date), period_end=str(_settings()[0].end_date))
        result[target] = item
    return {"status": "COMPLETED" if len(result) == len(preview["targets"]) else "PARTIAL",
            "metrics": result, "seeds": preview["seed_set"], "preflight": preview["preflight"]}


def render_fixed_min_oracle_result(report: Mapping[str, Any]) -> str:
    lines = ["Fixed Min Target Oracle（共用交易引擎／固定參數，同 Seed=8）",
             "Target                 Strategy  Selector  Oracle RoMD  Win Rate  Seed"]
    summary = report["metrics"]
    for target in (BASELINE, *[row["target"] for row in _rows()]):
        metric = summary.get(target)
        display = target[:-len("_Direct")] if target.endswith("_Direct") else target
        selector = "-" if target == BASELINE else (
            "Direct" if target.endswith("_Direct") else "Exact")
        lines.append(f"{display:<23}  Breakout  {selector:<8}  " + (
            f"{metric['romd']:>11.2f}  {metric['win_rate_pct']:>7.2f}%  {metric['seeds']}/8"
            if metric else f"{'':>11}  {'':>8}  0/8"))
    return "\n".join(lines)

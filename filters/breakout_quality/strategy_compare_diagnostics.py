"""Post-replay Strategy Compare candidate/selection diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from filters.breakout_quality.ranking_score_store import (
    load_selection_point_in_time_score_table,
    load_selection_point_in_time_score_table_from_path,
)

def _flatten_candidate_replay_rows(replay_counts: dict[str, dict[str, Any]], field: str) -> pd.DataFrame:
    if field not in {"candidate_rows", "orderable_rows"}:
        raise ValueError(f"不支援的candidate replay field: {field}")
    rows: list[dict[str, Any]] = []
    for ticker in sorted(replay_counts):
        bucket = replay_counts.get(ticker) or {}
        for raw in list(bucket.get(field) or []):
            row = dict(raw or {})
            row["ticker"] = str(row.get("ticker") or ticker)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "trade_date", "candidate_date", "signal_date",
            "candidate_type", "entry_source", "is_orderable", "high_len",
            "ensemble_vote_count", "qty", "sort_value", "historical_ev",
            "historical_win_rate", "historical_trade_count",
            "breakout_quality_score", "breakout_quality_score_date",
        ])
    frame = pd.DataFrame(rows)
    for column in ("trade_date", "candidate_date", "signal_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return frame.sort_values(
        ["trade_date", "ticker", "signal_date", "candidate_type"],
        kind="mergesort",
    ).reset_index(drop=True)

def _flatten_selected_buy_rows(trade_history: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(trade_history).copy()
    columns = ["ticker", "trade_date", "signal_date", "type"]
    if frame.empty or not {"Date", "Ticker", "Type"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    buy_mask = frame["Type"].fillna("").astype(str).str.startswith("買進 (")
    out = frame.loc[buy_mask].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({
        "ticker": out["Ticker"].fillna("").astype(str).str.strip(),
        "trade_date": pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "signal_date": pd.to_datetime(
            out.get("買訊日", pd.Series("", index=out.index)), errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "type": out["Type"].fillna("").astype(str),
    })
    return out.dropna(subset=["trade_date"]).sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)

def _selection_target_lookup(
    *, root: Path, filter_id: str, architecture: str, profile: str,
    score_path_override: str | None = None, manifest_path_override: str | None = None,
) -> pd.DataFrame:
    scores = (
        load_selection_point_in_time_score_table_from_path(
            str(score_path_override), manifest_path=str(manifest_path_override)
        )
        if score_path_override not in (None, "")
        else load_selection_point_in_time_score_table(
            str(root), filter_id, architecture, profile
        )
    ).reset_index()
    bundle = load_profile_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=architecture,
        experiment_profile=profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    groups = bundle.group_table[["group_index", "ticker", "date", "label"]].copy()
    groups["ticker"] = groups["ticker"].fillna("").astype(str).str.strip()
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d")
    groups["target_raw_r"] = bundle.raw_target
    groups["target_available"] = bundle.target_valid & pd.Series(bundle.raw_target).map(math.isfinite).to_numpy()
    joined = scores.merge(groups, on="group_index", how="left", validate="one_to_one", suffixes=("", "_dataset"))
    mismatch = (
        joined["ticker"] != joined["ticker_dataset"]
    ) | (
        joined["date"] != joined["date_dataset"]
    )
    if bool(mismatch.any()):
        raise ValueError("Selection PIT Score與Dataset group identity不一致")
    return joined[[
        "ticker", "date", "group_index", "breakout_quality_score", "fold_id",
        "model_information_cutoff", "label", "target_raw_r", "target_available",
    ]].rename(columns={"date": "signal_date"})

def _load_isolated_selection_pit_contract(
    *, score_path: str, manifest_path: str, filter_id: str, model_architecture: str,
    experiment_profile: str, expected_seed: int | None,
):
    score = Path(str(score_path)).resolve()
    manifest_file = Path(str(manifest_path)).resolve()
    for label, path in (("Selection PIT score", score), ("Selection PIT manifest", manifest_file)):
        if not path.is_file():
            raise FileNotFoundError(f"找不到isolated {label}: {path}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("isolated Selection PIT manifest根節點必須是object")
    if str(manifest.get("status") or "") != "BUILT":
        raise ValueError(f"isolated Selection PIT manifest尚未完成: {manifest.get('status')!r}")
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    for field, expected in expected_identity.items():
        actual = str(manifest.get(field) or "")
        if actual != expected:
            raise ValueError(
                f"isolated Selection PIT identity不一致: field={field}, expected={expected}, actual={actual}"
            )
    if expected_seed is not None and int(manifest.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            "isolated Selection PIT seed不一致: "
            f"expected={int(expected_seed)}, actual={manifest.get('seed')!r}"
        )
    lookahead = dict(manifest.get("lookahead_contract") or {})
    required_true = (
        "every_score_uses_model_not_trained_on_scored_event",
        "training_requires_label_eval_end_before_score_start",
    )
    required_false = (
        "oos_rows_or_target_statistics_used_for_training_or_epoch_selection",
        "future_target_in_score_table",
    )
    if any(lookahead.get(key) is not True for key in required_true) or any(
        lookahead.get(key) is not False for key in required_false
    ):
        raise ValueError("isolated Selection PIT lookahead contract不符合策略回放要求")
    score_record = dict((manifest.get("artifacts") or {}).get("scores") or {})
    if str(score_record.get("filename") or "") != score.name:
        raise ValueError("isolated Selection PIT score filename與manifest不一致")
    expected_hash = str(score_record.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(score).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("isolated Selection PIT score SHA256與manifest不一致")
    expected_size = int(score_record.get("size_bytes", -1))
    if expected_size != int(score.stat().st_size):
        raise ValueError("isolated Selection PIT score size與manifest不一致")
    table = load_selection_point_in_time_score_table_from_path(
        str(score), manifest_path=str(manifest_file)
    )
    available_from = str(table.attrs.get("available_from") or "")
    available_through = str(table.attrs.get("available_through") or "")
    if not available_from or not available_through or available_through < available_from:
        raise ValueError("isolated Selection PIT score period不合法")
    return SimpleNamespace(
        score_path=score,
        manifest_path=manifest_file,
        audit_path=None,
        manifest=manifest,
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        seed=int(manifest.get("seed", 0) or 0),
        available_from=available_from,
        available_through=available_through,
        model_validation_gate={
            "status": "ISOLATED_MULTI_SEED_ROBUSTNESS",
            "strategy_metrics_used": False,
            "future_target_used_for_runtime_sort": False,
        },
    )

def _strategy_selection_diagnostics(
    *, orderable: pd.DataFrame, selected: pd.DataFrame, lookup: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    orderable_work = pd.DataFrame(orderable).copy()
    selected_work = pd.DataFrame(selected).copy()
    lookup_work = pd.DataFrame(lookup).copy()

    for frame, columns in (
        (orderable_work, ("trade_date", "signal_date")),
        (selected_work, ("trade_date", "signal_date")),
        (lookup_work, ("signal_date",)),
    ):
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column], errors="coerce"
                ).dt.strftime("%Y-%m-%d").fillna("")

    # Continuation／re-entry 的交易 signal_date 可以晚於目前模型資訊日；
    # runtime Score 與 Future Target 都必須以replay保存的 score_date 對回 PIT 工件。
    raw_score_dates = orderable_work.get(
        "breakout_quality_score_date",
        pd.Series("", index=orderable_work.index, dtype="object"),
    ).fillna("").astype(str).str.strip()
    parsed_score_dates = pd.to_datetime(raw_score_dates, errors="coerce")
    invalid_score_dates = raw_score_dates.ne("") & parsed_score_dates.isna()
    if bool(invalid_score_dates.any()):
        bad = orderable_work.loc[invalid_score_dates].iloc[0]
        bad_score_date = raw_score_dates.loc[invalid_score_dates].iloc[0]
        raise ValueError(
            "策略replay保存的Breakout Quality score_date無法解析: "
            f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
            f"signal_date={bad.get('signal_date')}, score_date={bad_score_date!r}"
        )
    normalized_score_dates = parsed_score_dates.dt.strftime("%Y-%m-%d").fillna("")
    orderable_work["score_event_date"] = normalized_score_dates.where(
        normalized_score_dates.ne(""), orderable_work.get("signal_date", "")
    )

    lookup_work = lookup_work.rename(columns={
        "signal_date": "score_event_date",
        "breakout_quality_score": "pit_breakout_quality_score",
    })
    if bool(lookup_work.duplicated(["ticker", "score_event_date"]).any()):
        raise ValueError("Selection PIT diagnostic lookup同一ticker/score_event_date不唯一")

    orderable_joined = orderable_work.merge(
        lookup_work,
        on=["ticker", "score_event_date"],
        how="left",
        validate="many_to_one",
    )
    score_available = pd.to_numeric(
        orderable_joined.get(
            "pit_breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    runtime_score_available = pd.to_numeric(
        orderable_joined.get(
            "breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    declared_runtime_available = orderable_joined.get(
        "breakout_quality_score_available",
        pd.Series(False, index=orderable_joined.index),
    ).fillna(False).astype(bool)
    runtime_rows = declared_runtime_available | runtime_score_available
    if bool(runtime_rows.any()):
        expected = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "pit_breakout_quality_score"],
            errors="coerce",
        )
        actual = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "breakout_quality_score"],
            errors="coerce",
        )
        mismatch = (
            (~expected.map(math.isfinite))
            | (~actual.map(math.isfinite))
            | ((actual - expected).abs() > 1e-12)
        )
        if bool(mismatch.any()):
            bad_index = mismatch[mismatch].index[0]
            bad = orderable_joined.loc[bad_index]
            raise ValueError(
                "策略replay使用的Breakout Quality Score與PIT score table不一致: "
                f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
                f"signal_date={bad.get('signal_date')}, "
                f"score_event_date={bad.get('score_event_date')}, "
                f"runtime_score={actual.loc[bad_index]}, pit_score={expected.loc[bad_index]}"
            )

    target_available = orderable_joined.get(
        "target_available", pd.Series(False, index=orderable_joined.index)
    ).fillna(False).astype(bool)

    occurrence_keys = ["ticker", "trade_date", "signal_date"]
    occurrence_columns = [
        *occurrence_keys,
        "score_event_date",
        "target_raw_r",
        "target_available",
        "pit_breakout_quality_score",
    ]
    occurrence_score_event_counts = orderable_joined.groupby(
        occurrence_keys, dropna=False
    )["score_event_date"].nunique(dropna=False)
    if bool((occurrence_score_event_counts > 1).any()):
        bad_key = occurrence_score_event_counts[occurrence_score_event_counts > 1].index[0]
        raise ValueError(
            "同一策略候選發生多個Breakout Quality score event date: "
            f"ticker={bad_key[0]}, trade_date={bad_key[1]}, signal_date={bad_key[2]}"
        )
    occurrence_lookup = orderable_joined[occurrence_columns].drop_duplicates(
        occurrence_keys, keep="first"
    )
    selected_joined = selected_work.merge(
        occurrence_lookup,
        on=occurrence_keys,
        how="left",
        validate="many_to_one",
    )

    day_rows = []
    valid_orderable = orderable_joined.loc[target_available].copy()
    for trade_date, day in valid_orderable.groupby("trade_date", sort=True):
        chosen = selected_joined[selected_joined["trade_date"] == trade_date]
        chosen_keys = set(zip(chosen["ticker"], chosen["signal_date"]))
        if not chosen_keys:
            continue
        day = day.drop_duplicates(["ticker", "signal_date"], keep="first").copy()
        day["target_percentile"] = day["target_raw_r"].rank(method="average", pct=True)
        selected_day = day[[
            (ticker, signal_date) in chosen_keys
            for ticker, signal_date in zip(day["ticker"], day["signal_date"])
        ]]
        if selected_day.empty:
            continue
        k = len(selected_day)
        top = day.nlargest(k, "target_raw_r", keep="first")
        top_keys = set(zip(top["ticker"], top["signal_date"]))
        retained = sum(key in top_keys for key in chosen_keys)
        day_rows.append({
            "trade_date": trade_date,
            "selected_count": k,
            "selected_target_mean_r": float(selected_day["target_raw_r"].mean()),
            "selected_target_percentile_mean": float(selected_day["target_percentile"].mean()),
            "target_top_k_retention": float(retained / k),
            "target_opportunity_gap_r": float(
                top["target_raw_r"].mean() - selected_day["target_raw_r"].mean()
            ),
        })
    daily = pd.DataFrame(day_rows)
    metrics = {
        "orderable_occurrences": int(len(orderable_joined)),
        "orderable_score_covered": int(score_available.sum()),
        "orderable_score_coverage_rate": float(score_available.mean()) if len(score_available) else None,
        "runtime_scored_orderable_occurrences": int(runtime_score_available.sum()),
        "runtime_score_identity_match": True,
        "orderable_target_covered": int(target_available.sum()),
        "orderable_target_coverage_rate": float(target_available.mean()) if len(target_available) else None,
        "selected_buy_rows": int(len(selected_joined)),
        "diagnostic_days": int(len(daily)),
        "selected_target_percentile_mean": float(daily["selected_target_percentile_mean"].mean()) if not daily.empty else None,
        "target_top_k_retention_mean": float(daily["target_top_k_retention"].mean()) if not daily.empty else None,
        "target_opportunity_gap_r_mean": float(daily["target_opportunity_gap_r"].mean()) if not daily.empty else None,
        "selected_target_mean_r": float(daily["selected_target_mean_r"].mean()) if not daily.empty else None,
        "future_target_used_for_runtime_sort": False,
    }
    return metrics, orderable_joined, selected_joined


"""Same-seed canonical economic parity gate for aligned Oracle/DL evidence.

This is a research comparison contract. It does not publish results, alter
existing scientific identity, or turn future truth into a PIT prediction.
"""

from __future__ import annotations

from typing import Any, Mapping
import math

import numpy as np
import pandas as pd

def verify_paired_oracle_dl_result(
    *, baseline: Mapping[int, Mapping[str, Any]],
    oracle: Mapping[int, Mapping[str, Any]],
    dl: Mapping[int, Mapping[str, Any]],
    eligible_score_keys: pd.MultiIndex,
    orderable_keys: pd.MultiIndex,
) -> dict[str, Any]:
    """P2/P3 scientific gate for a *new* matched replay, not old summary stitching.

    Each input row must contain the same canonical replay provenance for the given
    seed. Comparison identities include score refresh mode and *cohort* digest.
    """
    seeds = set(baseline)
    if not seeds or seeds != set(oracle) or seeds != set(dl):
        raise ValueError("No-DL／Oracle／DL缺少相同的完整Seed配對")
    if (not isinstance(eligible_score_keys, pd.MultiIndex)
            or not isinstance(orderable_keys, pd.MultiIndex)
            or eligible_score_keys.has_duplicates or orderable_keys.has_duplicates):
        raise ValueError("Score或可掛候選stock-day身份不合法")
    if not len(eligible_score_keys) or not len(orderable_keys):
        raise ValueError("共同合法母體及實際可掛候選都必須有完整證據")
    missing = orderable_keys.difference(eligible_score_keys)
    if len(missing):
        raise ValueError(f"合法可掛候選缺少{len(missing)}筆Score，不得退回原始排序")
    identity_fields = (
        "start_date", "end_date", "dataset_sha256", "param_sha256",
        "seed", "selector", "broker_policy", "max_positions",
        "rotation", "score_refresh", "score_cohort_sha256",
    )
    pairs = []
    for seed in sorted(seeds):
        base, future, predicted = baseline[seed], oracle[seed], dl[seed]
        for record in (base, future, predicted):
            if int(record.get("seed", -1)) != int(seed):
                raise ValueError("Seed身份不一致")
            if not math.isfinite(float(record.get("romd", float("nan")))):
                raise ValueError("三方必須有完整且有限的Broker-Cash RoMD")
            for field in identity_fields:
                if record.get(field) in (None, ""):
                    raise ValueError(f"配對結果缺少不可省略的契約欄位：{field}")
        for field in identity_fields:
            if base[field] != future[field] or base[field] != predicted[field]:
                raise ValueError(f"No-DL／Oracle／DL不具共同契約：seed={seed} field={field}")
        if str(future.get("score_origin")) != "future_truth":
            raise ValueError("Oracle結果必須標示hindsight真值")
        if str(predicted.get("score_origin")) != "pit_prediction":
            raise ValueError("DL結果必須標示合法PIT預測")
        pairs.append({
            "seed": seed, "no_dl_romd": float(base["romd"]),
            "oracle_romd": float(future["romd"]), "dl_romd": float(predicted["romd"]),
            "oracle_minus_baseline": float(future["romd"])-float(base["romd"]),
            "dl_minus_baseline": float(predicted["romd"])-float(base["romd"]),
            "oracle_minus_dl": float(future["romd"])-float(predicted["romd"]),
        })
    return {
        "status": "COMPARABLE", "matched_seeds": len(pairs),
        "orderable_score_coverage": 1.0,
        "same_seed": pairs,
        "oracle_minus_dl_romd_mean": float(np.mean([p["oracle_minus_dl"] for p in pairs])),
        "non_pit_oracle": True,
    }


def _observed_orderable_score_keys(
    replay_counts: Mapping[str, Mapping[str, Any]], *, seed: int, source: str,
) -> tuple[pd.MultiIndex, int]:
    """Use the existing portfolio replay's orderable trace, not an invented proxy.

    The decision information-date has already been fixed by canonical ranking
    runtime. Reusing it is mandatory, including pending/re-entry candidates.
    """
    keys: list[tuple[str, str]] = []
    seen_rows = 0
    for ticker, bucket in replay_counts.items():
        for row in bucket.get("orderable_rows", []):
            seen_rows += 1
            if str(row.get("ticker")) != str(ticker):
                raise ValueError("Replay orderable ticker identity drift")
            day = str(row.get("breakout_quality_score_date") or "")
            if len(day) != 10 or not day[:4].isdigit() or day[4] != "-" or day[7] != "-":
                raise ValueError(
                    f"seed={seed} {source}合法候選缺少可稽核的information date；不得宣稱score coverage"
                )
            keys.append((str(ticker), day))
    if not seen_rows:
        raise ValueError(f"seed={seed} {source}沒有實際orderable trace，無法宣稱候選覆蓋")
    return pd.MultiIndex.from_tuples(sorted(set(keys)), names=["ticker", "date"]), seen_rows


def run_matched_oracle_dl_replays(
    *, seed_panels: Mapping[int, Mapping[str, Any]],
    start_date: str, end_date: str,
    data_dir: str, dataset_sha256: str, broker_policy: str,
    selector_policy: str, ranking_options: Mapping[str, Any],
    max_positions: int, rotation: bool = False, quiet: bool = True,
    score_start_date: str | None = None,
) -> dict[str, Any]:
    """Execute the *same* canonical replay for three research-only score sources.

    `seed_panels` must be resolved by the canonical upstream producer before this
    consumer is called.  Each per-seed row requires independently verified PIT
    score and future-truth tables, an ex-ante eligible stock-day MultiIndex,
    P16 parameter provenance, and the canonical baseline/scored parameter views.
    This never publishes a fitted model, official Score, or Standard SOP report.
    """
    import hashlib
    from filters.breakout_quality.score_comparison_alignment import (
        rank_daily_stock_days, common_stock_day_cohort,
    )
    from filters.breakout_quality.runtime import audit_oracle_truth_score_context
    from services.research.strategy_compare_replay import _run_scenario

    score_start_date = score_start_date or start_date
    if not seed_panels or not start_date or not end_date or end_date < start_date or score_start_date > start_date:
        raise ValueError("matched Oracle/DL必須有合法期間與Seed")
    if not dataset_sha256 or not broker_policy or not selector_policy or max_positions < 1:
        raise ValueError("matched Oracle/DL缺少交易身分")
    expected = {"oracle", "dl", "eligible", "param_sha256", "kind", "baseline_params",
                "scored_params", "pit_score_sha256", "oracle_truth_sha256",
                "eligibility_source", "eligibility_manifest_sha256"}
    prepared = {}
    for seed, entry in sorted(seed_panels.items()):
        missing = expected.difference(entry)
        if missing:
            raise ValueError(f"seed={seed}缺少正式上游證據：{sorted(missing)}")
        if not str(entry["pit_score_sha256"] or "") or not str(entry["oracle_truth_sha256"] or ""):
            raise ValueError("不能在未驗證PIT／Oracle工件身分下執行")
        if (entry["eligibility_source"] != "feature_history_only"
                or not str(entry["eligibility_manifest_sha256"] or "")):
            raise ValueError("共同排名母體必須由獨立PIT feature-history-only的正式證據解析，不能用Oracle成熟樣本反推")
        cohort = common_stock_day_cohort(
            entry["oracle"], entry["dl"], start=score_start_date, end=end_date,
            declared_eligible=entry["eligible"],
        )
        oracle_pct = rank_daily_stock_days(
            entry["oracle"], score_column="score", start=score_start_date, end=end_date,
            cohort=cohort,
        )
        dl_pct = rank_daily_stock_days(
            entry["dl"], score_column="score", start=score_start_date, end=end_date,
            cohort=cohort,
        )
        if not oracle_pct.index.equals(dl_pct.index):
            raise RuntimeError("共同分數母體未對齊")
        digest = hashlib.sha256()
        for ticker, date in cohort.sort_values():
            digest.update(f"{ticker}\0{date}\n".encode("utf-8"))
        prepared[int(seed)] = (entry, cohort, oracle_pct, dl_pct, digest.hexdigest())

    # Reject cohort/score identity conflicts before starting expensive Replay.
    first_cohort = prepared[next(iter(sorted(prepared)))][1].sort_values()
    for _entry, cohort, _oracle, _dl, _digest in prepared.values():
        if not cohort.sort_values().equals(first_cohort):
            raise ValueError("四Seed評估母體不同，不得執行或發布一份共同母體結果")

    controls: dict[int, dict[str, Any]] = {}
    truths: dict[int, dict[str, Any]] = {}
    models: dict[int, dict[str, Any]] = {}
    observed: dict[int, dict[str, Any]] = {}
    observed_all: set[tuple[str, str]] = set()
    # The *same* policy and price/parameter configuration is used for all arms.
    # The audit-only delivery adapter is intentional, and cannot enter PIT/Trading.
    for seed, (entry, cohort, oracle_pct, dl_pct, cohort_hash) in sorted(prepared.items()):
        replay_args = {
            "data_dir": data_dir, "param_source_kind": str(entry["kind"]),
            "start_date": start_date, "end_date": end_date,
            "max_positions": int(max_positions), "enable_rotation": bool(rotation),
            "quiet": bool(quiet),
        }
        common = {
            "start_date": start_date, "end_date": end_date,
            "dataset_sha256": dataset_sha256,
            "param_sha256": str(entry["param_sha256"]), "seed": seed,
            "selector": selector_policy, "broker_policy": broker_policy,
            "max_positions": int(max_positions), "rotation": bool(rotation),
            "score_refresh": "daily_information_date",
            "score_cohort_sha256": cohort_hash,
        }
        def romd_of(replay):
            value = float(replay.get("return_over_max_drawdown", float("nan")))
            if not math.isfinite(value):
                raise ValueError("matched Oracle/DL Replay未產生有效RoMD")
            return value

        base = _run_scenario(name=f"No-DL seed={seed}", params=entry["baseline_params"], **replay_args)
        controls[seed] = {**common, "score_origin": "none", "romd": romd_of(base)}
        observed[seed] = {}
        for source, series, destination in (
            ("future_truth", oracle_pct, truths),
            ("pit_prediction", dl_pct, models),
        ):
            score_map = {key: float(value) for key, value in series.items()}
            replay_counts: dict[str, dict[str, Any]] = {}
            with audit_oracle_truth_score_context(
                score_map, ranking_policy=selector_policy,
                ranking_options=dict(ranking_options), key_mode="information_date",
            ):
                result = _run_scenario(
                    name=f"{source} seed={seed}", params=entry["scored_params"],
                    replay_counts=replay_counts, **replay_args,
                )
            destination[seed] = {**common, "score_origin": source, "romd": romd_of(result)}
            keys, observed_rows = _observed_orderable_score_keys(
                replay_counts, seed=seed, source=source,
            )
            missing = keys.difference(cohort)
            if len(missing):
                raise ValueError(
                    f"seed={seed} {source}合法orderable候選有{len(missing)}筆不在共同Score母體；不得fallback"
                )
            observed_all.update(keys.tolist())
            observed[seed][source] = {
                "orderable_rows": observed_rows,
                "unique_information_date_keys": len(keys),
                "score_coverage": 1.0,
            }
    # Strict audit lookup also fails for any actually requested ticker/day not
    # covered by the cohort. The observed trace provides independent coverage
    # evidence rather than using the declared cohort as a stand-in.
    combined_cohort = first_cohort
    all_orderable = pd.MultiIndex.from_tuples(
        sorted(observed_all), names=["ticker", "date"],
    )
    result = verify_paired_oracle_dl_result(
        baseline=controls, oracle=truths, dl=models,
        eligible_score_keys=combined_cohort, orderable_keys=all_orderable,
    )
    result["eligible_cohort_score_coverage"] = 1.0
    result["observed_orderable_score_coverage"] = result["orderable_score_coverage"]
    result["orderable_trace_by_seed"] = observed
    result.update({
        "evaluation_period": {"start": start_date, "end": end_date},
        "score_window": {"start": score_start_date, "end": end_date},
        "alignment_method": "daily_average_zero_based_v1",
        "replay_delivery": "isolated_hindsight_audit_adapter_for_both_sources",
        "pit_lineage_required": True,
        "standard_sop_report": False,
        "source_identity_note": (
            "Only canonical PIT-manifest-validated feature eligibility and score "
            "artifacts may be supplied. Oracle is hindsight and not promotable."
        ),
    })
    return result

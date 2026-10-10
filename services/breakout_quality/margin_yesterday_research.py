"""Canonical service adapter for one approved, isolated current-vintage DL experiment.

Invoked by the existing Research model menu; it uses the canonical AO fitting and
metric primitives. It cannot modify formal Research, PIT or Trading artifacts.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256, compute_file_sha256
from core.breakout_quality_registry import (
    get_continuous_ranker_research_spec, get_exploratory_model_research_study,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from services.breakout_quality import ranker_training as ranker_api
from services.breakout_quality.continuous_ranker_pipeline import (
    predict_score_output_payload, resolve_ranker_execution_plan,
)
from services.breakout_quality.fitted_model_artifacts import (
    canonical_fitting_settings, source_training_identity, fitting_split_identity,
    scientific_torch_execution,
)
from services.breakout_quality.margin_yesterday_exploratory import (
    SCIENTIFIC_ID, EXPERIMENT_ARCHITECTURE, FEATURE_CONTRACT, MarginAugmentedDailyFeatureBank,
    load_margin_source_for_ao_bundle,
)
from services.breakout_quality.train_continuous_ranker import parse_args as parse_canonical_training_args
from services.breakout_quality.price_input_source import load_daily_universal_ranker_data_with_price_source
from filters.breakout_quality.daily_ranker_data import build_daily_ranker_split
from filters.breakout_quality.ranking_score_store import load_continuous_ranker_oos_contract

EXPERIMENT_RELATIVE_OUTPUT = Path("outputs/filters/breakout_quality/experiments/mr13by_margin_yesterday_current_vintage")
EXPERIMENT_RELATIVE_MODEL = Path("models/filters/breakout_quality/experiments/mr13by_margin_yesterday_current_vintage")
AO_ARCHITECTURE = "inception_time_shared_safety_mfe_v1"
_STUDY = get_exploratory_model_research_study(SCIENTIFIC_ID)


def _summary(quality: dict) -> dict:
    return {
        "raw_safety": _metric_summary(quality["raw_safety"]),
        "conditional_mfe_true_hs": _metric_summary(quality["conditional_mfe_true_hs"]),
        "lexicographic_model_gate": dict(quality["lexicographic_model_gate"]),
    }


def _ao_reference_quality(bundle, ids, hs, *, filter_id: str) -> dict | None:
    """Evaluate frozen canonical AO scores on exactly BY's OOS groups (no fit)."""
    try:
        contract = load_continuous_ranker_oos_contract(
            str(ROOT), str(filter_id),
            AO_ARCHITECTURE, _STUDY.reference_profile,
        )
    except FileNotFoundError:
        return None
    score_path = Path(contract.score_path)
    header = list(pd.read_csv(score_path, nrows=0).columns)
    primary = next((name for name in ("conditional_mfe_score", "model_score") if name in header), None)
    required = {"group_index", "ticker", "date", "raw_safety_score"}
    if primary is None or not required.issubset(header):
        raise ValueError("canonical AO Forward score缺少群組／雙頭欄位")
    columns = ["group_index", "ticker", "date", "raw_safety_score", primary]
    reference = pd.read_csv(score_path, usecols=columns, dtype={"ticker": "string"}, low_memory=False)
    reference["group_index"] = pd.to_numeric(reference["group_index"], errors="raise").astype("int64")
    if reference["group_index"].duplicated().any():
        raise ValueError("canonical AO Forward score的group_index重複")
    index = reference.set_index("group_index")
    group_ids = np.asarray(ids, dtype=np.int64)
    if not index.index.is_unique or not np.isin(group_ids, index.index.values).all():
        raise ValueError("AO/BY Forward OOS sample無法完整對齊")
    selected = index.loc[group_ids]
    expected = bundle.group_table.iloc[group_ids]
    if not np.array_equal(selected["ticker"].astype(str).to_numpy(), expected["ticker"].astype(str).to_numpy()):
        raise ValueError("AO/BY OOS ticker錯配")
    if not np.array_equal(pd.to_datetime(selected["date"]).dt.normalize().to_numpy(),
                          pd.to_datetime(expected["date"]).dt.normalize().to_numpy()):
        raise ValueError("AO/BY OOS date錯配")
    heads = {
        "raw_safety": pd.to_numeric(selected["raw_safety_score"], errors="raise").to_numpy(dtype=np.float32),
        "conditional_mfe": pd.to_numeric(selected[primary], errors="raise").to_numpy(dtype=np.float32),
    }
    if not all(np.isfinite(value).all() for value in heads.values()):
        raise ValueError("AO歷史OOS score包含非有限數值")
    return {
        "metrics": _summary(ranker_api.hs_conditional_mfe_metrics(
            group_ids, bundle.group_table, hs, heads, include_top_k_quality=False,
        )),
        "canonical_score_sha256": compute_file_sha256(score_path),
        "reference_groups": int(len(group_ids)),
        "reference_profile": _STUDY.reference_profile,
    }


def _metric_summary(metrics: dict) -> dict:
    items = ("group_count", "mean_daily_spearman", "global_spearman_vs_raw_target", "pairwise_concordance")
    return {name: metrics.get(name) for name in items}


def _markdown(result: dict) -> str:
    m = result["metrics"]
    lines = [
        "# MR-13BY｜FinMind YesterdayBalance current-vintage 探索性 DL（Forward）", "",
        "- **非正式 PIT**：歷史首次發布／修訂 Vintage 未證明；此實驗的 Forward/OOS 只能用作探索性證據。",
        "- **正式模型 Input 授權：NO；不得進入 Trading、策略升格或正式 AO Comparison。**",
        "- 對照：MR-13AO；AO 調整價 300×10 + 融資／融券餘額 2 欄＋獨立缺值 mask 2 欄 = 300×14。",
        "- 日 T 輸入只選 T 前一正式交易日的 FinMind YesterdayBalance（表示 T−2 確認餘額）。",
        "- 特徵：單一 300 日視窗內每欄 log1p(餘額) 相對有效觀測 log1p 中位數；缺值中性 0、mask=0；真實零餘額 mask=1。",
        f"- AO Target、full-universe 樣本、Forward 切分與 Loss 不變；Seed={result['fitting_identity']['seed']}，僅增加輸入欄位。", "",
        f"- 工件身分：`{result['fitting_fingerprint']}`",
        f"- Epoch：`{result['selected_epoch']}`；OOS 樣本：`{result['oos_groups']}`", "",
        "| OOS 專題指標 | Raw Safety | Conditional-MFE（true HS） |",
        "|---|---:|---:|",
    ]
    keys = (("Daily rho", "mean_daily_spearman"),
            ("Global rho", "global_spearman_vs_raw_target"),
            ("Pair accuracy", "pairwise_concordance"))
    for label, key in keys:
        left = m["raw_safety"].get(key)
        right = m["conditional_mfe_true_hs"].get(key)
        fmt = lambda value: "—" if value is None else f"{float(value):.4f}"
        lines.append(f"| {label} | {fmt(left)} | {fmt(right)} |")
    lex = m["lexicographic_model_gate"]
    lines.extend(["", "| Pred-HS / Top-tail | Result |", "|---|---:|",
                  f"| Pred-HS true-LS | {lex.get('predicted_hs_true_ls_pct')}% |",
                  f"| Pred-HS HM/HS | {lex.get('predicted_hs_hmhs_pct')}% |",
                  f"| Selected Top10 HM/HS | {lex.get('selected_hmhs_pct')}% |", "",
                  "**不能僅因 OOS 數字較高就把這個模型認定 PIT 合法。**", "",
                  "所有後續比較／Robustness僅屬隔離探索，不能當成已完成嚴格PIT驗證。", ""])
    return "\n".join(lines)


def _fit_exploratory_segment(*, args, features, bundle, split, torch, plan, recipe, phase_label: str) -> dict:
    """Reuse the canonical AO fit/evaluate primitives for any legal time split.

    Rolling training never reads evaluation rows during target percentile fitting,
    epoch selection, normalization or final fitting. This is an exploratory
    current-vintage study, not a claim of historical point-in-time source legality.
    """
    selection_mask = np.zeros(len(bundle.raw_target), dtype=bool)
    selection_mask[split.selection_ids] = True
    percentile_targets = ranker_api.build_daily_percentile_targets(
        bundle.raw_target, selection_mask, bundle.group_table["date"]
    )
    epoch = ranker_api.select_epoch(
        torch, features, bundle.group_context, bundle.group_table,
        bundle.raw_target, percentile_targets, split.inner_train_ids,
        split.validation_ids, args=args, plan=plan, evaluate_train_metrics=False,
    )
    selected_epoch = int(epoch["best_epoch"])
    model, history = ranker_api.fit_final(
        torch, features, bundle.group_context, bundle.raw_target,
        percentile_targets, bundle.group_table, split.selection_ids,
        epochs=selected_epoch, args=args, plan=plan,
        phase_label=phase_label,
    )
    ids = np.asarray(split.oos_ids, dtype=np.int64)
    conditional_scores, sidecars = predict_score_output_payload(
        torch, model, bundle, ids, batch_size=int(args.evaluation_batch_size), plan=plan
    )
    if "raw_safety_score" not in sidecars:
        raise ValueError("探索模型需要AO雙Head OOS score")
    heads = {"raw_safety": sidecars["raw_safety_score"],
             "conditional_mfe": conditional_scores}
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    hs = build_hs_conditional_mfe_targets(
        bundle.group_table, np.isfinite(bundle.raw_target),
        true_hs_percentile_cutoff=float(recipe.objective_policy.secondary_pair_scope_threshold),
    )
    quality = ranker_api.hs_conditional_mfe_metrics(
        ids, bundle.group_table, hs, heads, include_top_k_quality=False,
    )
    scores = bundle.group_table.iloc[ids][["ticker", "date"]].copy()
    scores["date"] = pd.to_datetime(scores["date"]).dt.strftime("%Y-%m-%d")
    scores["raw_safety_score"] = np.asarray(heads["raw_safety"], dtype=np.float32)
    scores["conditional_mfe_score"] = np.asarray(heads["conditional_mfe"], dtype=np.float32)
    return {
        "selected_epoch": selected_epoch,
        "selection_history": epoch.get("history", []),
        "final_history": history,
        "group_ids": ids,
        "hs": hs,
        "quality": quality,
        "scores": scores,
        "model": model,
    }


def run(
    *, snapshot_fingerprint: str | None = None, seed: int = 42,
    benchmark: bool = False,
) -> tuple[Path, Path]:
    args = parse_canonical_training_args([
        "--model-architecture", AO_ARCHITECTURE,
        "--experiment-profile", _STUDY.reference_profile,
        "--seed", str(int(seed)),
    ])
    if (not benchmark and int(args.seed) != 42) or bool(args.allow_stale_source):
        raise ValueError("單次探索實驗沿用AO Seed42；多Seed僅允許Robustness Benchmark")
    spec = get_continuous_ranker_research_spec(args.experiment_profile)
    recipe = get_continuous_ranker_execution_recipe(args.experiment_profile)
    if spec.profile_name != _STUDY.reference_profile:
        raise ValueError("AO supervision recipe drift")
    print("[MR-13BY] CURRENT-VINTAGE EXPLORATION ONLY; FORMAL PIT=BLOCKED; TRAINING AO BASE", flush=True)
    original = load_daily_universal_ranker_data_with_price_source(
        filter_id=str(args.filter_id), model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile), preload_feature_bank=False,
        allow_stale_source=False,
    )
    margin = load_margin_source_for_ao_bundle(ROOT, original, snapshot_fingerprint=snapshot_fingerprint)
    features = MarginAugmentedDailyFeatureBank(original.feature_bank, margin.arrays)
    if features.shape[2] != 14 or len(features) != len(original.group_table):
        raise RuntimeError("MR-13BY input shape/sample identity不合法")
    bundle = replace(original, feature_bank=features)
    split = build_daily_ranker_split(bundle, inner_validation_months=int(args.inner_validation_months))
    torch, plan = resolve_ranker_execution_plan(args)
    # Identical canonical fitting recipe as AO; only the read-only external
    # feature input and its scientific identity differ.
    identity = {
        "scientific_id": SCIENTIFIC_ID,
        "scientific_architecture": EXPERIMENT_ARCHITECTURE,
        "input_contract": FEATURE_CONTRACT,
        "research_authorization": "exploration_only_current_vintage_unverified",
        "formal_pit_input_authorized": False,
        "ao_profile": str(args.experiment_profile),
        "ao_model_architecture": str(args.model_architecture),
        "ao_training_semantics": ranker_api.training_semantics(bundle.profile),
        "ao_profile_manifest": bundle.profile.as_manifest_payload(),
        "ao_model_spec": bundle.model_spec.as_manifest_payload(),
        "training_settings": canonical_fitting_settings(args),
        "torch_execution": scientific_torch_execution(plan),
        "seed": int(args.seed),
        "source": source_training_identity(original),
        "margin_source": margin.source_identity,
        "split": fitting_split_identity(split.report),
        "feature_columns": features.feature_columns,
        "feature_shape": [int(features.shape[1]), int(features.shape[2])],
    }
    fingerprint = canonical_json_sha256(identity)
    subpath = Path("benchmark_forward") / f"seed_{int(seed)}" if benchmark else Path()
    output = ROOT / EXPERIMENT_RELATIVE_OUTPUT / subpath / fingerprint
    model_dir = ROOT / EXPERIMENT_RELATIVE_MODEL / subpath / fingerprint
    if output.exists() or model_dir.exists():
        raise FileExistsError("MR-13BY此fitting identity已有工件；不得覆寫或冒稱正式REUSE: " + str(output))
    fitted = _fit_exploratory_segment(
        args=args, features=features, bundle=bundle, split=split,
        torch=torch, plan=plan, recipe=recipe,
        phase_label="MR-13BY experimental Forward refit",
    )
    ids = fitted["group_ids"]
    selected_epoch = fitted["selected_epoch"]
    history = fitted["final_history"]
    quality = fitted["quality"]
    scores = fitted["scores"]
    model = fitted["model"]
    hs = fitted["hs"]
    # Canonical AO reference is evaluated only after BY fitting and prediction.
    # Neither reference OOS scores nor any comparator metric may influence fitting.
    ao_reference = _ao_reference_quality(bundle, ids, hs, filter_id=str(args.filter_id))
    result = {
        "scientific_id": SCIENTIFIC_ID,
        "scientific_architecture": EXPERIMENT_ARCHITECTURE,
        "status": "EXPLORATORY_FORWARD_RESULT_AVAILABLE_PIT_UNPROVEN",
        "formal_model_input_authorized": False,
        "historical_contemporaneous_vintage_proven": False,
        "fitting_fingerprint": fingerprint,
        "fitting_identity": identity,
        "selected_epoch": selected_epoch,
        "selection_history": fitted["selection_history"],
        "final_history": history,
        "oos_groups": int(len(ids)),
        "metrics": _summary(quality),
        "reference_ao_same_oos": ao_reference,
        "coverage": margin.coverage,
    }
    model_dir.mkdir(parents=True, exist_ok=False)
    import os
    temp = model_dir / "model.incomplete.pt"
    torch.save({
        "scientific_id": SCIENTIFIC_ID,
        "scientific_architecture": EXPERIMENT_ARCHITECTURE,
        "fitting_fingerprint": fingerprint,
        "fitting_identity": identity,
        "model_spec": bundle.model_spec.as_manifest_payload(),
        "feature_count": features.shape[2],
        "input_feature_contract": FEATURE_CONTRACT,
        "feature_columns": features.feature_columns,
        "selected_epoch": selected_epoch,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "promotion_authorized": False,
    }, temp)
    os.replace(temp, model_dir / "model.pt")
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / "experimental_forward_oos_scores.csv.gz", index=False, compression="gzip")
    # The presence of JSON marks a complete result with human-readable report.
    md_path = atomic_write_text(output / "experimental_report.md", _markdown(result))
    json_path = atomic_write_json(output / "experimental_report.json", result)
    print("MR-13BY EXPLORATORY RESULT | formal PIT=NO | output=" + str(md_path), flush=True)
    return json_path, md_path



def _validate_current_exploratory_source(result: dict) -> None:
    from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
    current = load_ready_provider_snapshot_archive(ROOT)
    stored = dict(dict(result.get("fitting_identity") or {}).get("margin_source") or {})
    if not stored.get("snapshot_fingerprint") or str(stored["snapshot_fingerprint"]) != str(current.snapshot_fingerprint):
        raise RuntimeError("MR-13BY目前FinMind READY Snapshot與歷史exploratory fitting來源不同；不得靜默REUSE")


def _read_experimental_report() -> tuple[Path | None, dict | None]:
    """Resolve one immutable completed experiment; never pick a best/latest OOS."""
    output_root = ROOT / EXPERIMENT_RELATIVE_OUTPUT
    candidates = sorted(output_root.glob("*/experimental_report.json"))
    if not candidates:
        return None, None
    if len(candidates) > 1:
        raise RuntimeError(
            "隔離實驗已有多個不同fitting指紋，不得自動挑選較佳結果；"
            "請先使用受控配置指定正確的來源版本／工件"
        )
    path = candidates[0]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (raw.get("scientific_id") != SCIENTIFIC_ID
            or raw.get("fitting_fingerprint") != path.parent.name
            or raw.get("formal_model_input_authorized") is not False
            or raw.get("historical_contemporaneous_vintage_proven") is not False):
        raise ValueError("MR-13BY隔離結果身分或PIT禁止條件不合法")
    checkpoint = ROOT / EXPERIMENT_RELATIVE_MODEL / path.parent.name / "model.pt"
    score = path.parent / "experimental_forward_oos_scores.csv.gz"
    if not checkpoint.is_file() or not score.is_file():
        raise ValueError("MR-13BY模型／OOS artifacts不完整，不得宣稱READY")
    return path, raw


def _fmt(value, *, signed=False):
    if value is None:
        return "—"
    return f"{float(value):+.4f}" if signed else f"{float(value):.4f}"


def _comparison_payload(result: dict) -> dict:
    source = dict(result.get("reference_ao_same_oos") or {})
    if source.get("reference_profile") != _STUDY.reference_profile:
        raise ValueError("MR-13BY報告沒有已驗證同樣本AO參照；不得替代用不同時期績效")
    if int(source.get("reference_groups") or -1) != int(result.get("oos_groups") or 0):
        raise ValueError("AO/BY參照樣本數不一致")
    ao = dict(source.get("metrics") or {})
    by = dict(result.get("metrics") or {})
    selectors = (
        ("raw_safety", "mean_daily_spearman", "Raw Safety Daily rho", True),
        ("raw_safety", "global_spearman_vs_raw_target", "Raw Safety Global rho", True),
        ("raw_safety", "pairwise_concordance", "Raw Safety Pair", True),
        ("conditional_mfe_true_hs", "mean_daily_spearman", "Conditional-MFE Daily rho", True),
        ("conditional_mfe_true_hs", "global_spearman_vs_raw_target", "Conditional-MFE Global rho", True),
        ("conditional_mfe_true_hs", "pairwise_concordance", "Conditional-MFE Pair", True),
        ("lexicographic_model_gate", "predicted_hs_true_ls_pct", "Pred-HS true-LS (%)", False),
        ("lexicographic_model_gate", "selected_hmhs_pct", "Selected HM/HS (%)", True),
        ("lexicographic_model_gate", "selected_high_safety_pct", "Selected High-Safety (%)", True),
    )
    rows = []
    for section, field, label, higher in selectors:
        a = dict(ao.get(section) or {}).get(field)
        b = dict(by.get(section) or {}).get(field)
        rows.append({"label": label, "ao": a, "by": b,
                     "delta_by_minus_ao": None if a is None or b is None else float(b) - float(a),
                     "higher_is_better": bool(higher)})
    return {
        "report_kind": "one_off_current_vintage_exploratory_same_oos_comparison",
        "scientific_id": SCIENTIFIC_ID,
        "reference_model_id": _STUDY.reference_model_id,
        "fitting_fingerprint": result["fitting_fingerprint"],
        "same_oos_group_count": int(result["oos_groups"]),
        "canonical_reference_score_sha256": source["canonical_score_sha256"],
        "formal_pit_input_authorized": False,
        "trading_promotion_authorized": False,
        "historical_vintage_proven": False,
        "rows": rows,
    }


def run_forward_menu() -> int:
    existing, result = _read_experimental_report()
    print("\n=== 探索性 Forward OOS｜歷史 Vintage 未證明｜正式 PIT/Trading 禁止 ===")
    if existing is not None:
        _validate_current_exploratory_source(result)
        print("REUSE｜" + str(existing.with_suffix(".md")))
        print("模型身份：" + str(result["scientific_id"]) + "｜Forward OOS：" + str(result["oos_groups"]))
        return 0
    _json_path, md_path = run()
    print("DONE｜" + str(md_path))
    return 0


def _resolve_forward_comparison_reference(result: dict) -> dict | None:
    """Build missing *reference* evidence via AO's owner; never mutate BY fit."""
    if result.get("reference_ao_same_oos") is not None:
        return dict(result["reference_ao_same_oos"])
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings
    from services.research.breakout_quality_application import _run_continuous_forward_model_gate
    settings = get_breakout_quality_workflow_settings(experiment_profile=_STUDY.reference_profile)
    print("[BUILD] 缺少AO同樣本Forward參照；委派正式AO Forward producer。")
    code = _run_continuous_forward_model_gate("apps/research.py model", settings)
    if code != 0:
        return None
    args = parse_canonical_training_args([
        "--model-architecture", AO_ARCHITECTURE,
        "--experiment-profile", _STUDY.reference_profile,
        "--seed", str(int(dict(result.get("fitting_identity") or {}).get("seed", 42))),
    ])
    original = load_daily_universal_ranker_data_with_price_source(
        filter_id=str(args.filter_id), model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile), preload_feature_bank=False,
        allow_stale_source=False,
    )
    identity = dict(result.get("fitting_identity") or {})
    if source_training_identity(original) != identity.get("source"):
        raise RuntimeError("BY歷史訓練來源與目前AO資料來源不同，不可重新推測group_index")
    split = build_daily_ranker_split(original, inner_validation_months=int(args.inner_validation_months))
    if (fitting_split_identity(split.report) != identity.get("split")
            or len(split.oos_ids) != int(result.get("oos_groups", -1))):
        raise RuntimeError("AO/BY Forward歷史OOS切分身份不一致")
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    recipe = get_continuous_ranker_execution_recipe(_STUDY.reference_profile)
    hs = build_hs_conditional_mfe_targets(
        original.group_table, np.isfinite(original.raw_target),
        true_hs_percentile_cutoff=float(recipe.objective_policy.secondary_pair_scope_threshold),
    )
    reference = _ao_reference_quality(original, split.oos_ids, hs, filter_id=str(args.filter_id))
    if reference is None:
        raise RuntimeError("AO Producer已結束但canonical Forward OOS scores仍缺失")
    return reference


def run_forward_comparison() -> int:
    path, result = _read_experimental_report()
    if path is None:
        # Missing evidence is a BUILD dependency, not an authorization BLOCK.
        # Delegate to the one canonical experimental fitting producer.
        print("BUILD｜Forward OOS 比較所需模型工件尚未存在，先委派 Forward Producer。")
        code = run_forward_menu()
        if code != 0:
            return int(code)
        path, result = _read_experimental_report()
        if path is None:
            raise RuntimeError("Forward producer未產生可驗證的BY研究工件")
    _validate_current_exploratory_source(result)
    reference = _resolve_forward_comparison_reference(result)
    if reference is None:
        raise RuntimeError("AO Forward參照 Producer失敗；不產生假比較")
    payload = _comparison_payload({**result, "reference_ao_same_oos": reference})
    # Official reference remains a separate canonical artifact; if its bytes
    # changed, the comparison requires a new fixed-source scientific decision.
    from filters.breakout_quality.ranking_score_store import resolve_continuous_ranker_oos_score_path
    score_path = resolve_continuous_ranker_oos_score_path(
        ROOT, "breakout_quality_v1", AO_ARCHITECTURE, _STUDY.reference_profile
    )
    if compute_file_sha256(score_path) != payload["canonical_reference_score_sha256"]:
        raise RuntimeError("AO參照OOS工件已變動，不得重用原AO/BY比較")
    lines = [
        "# AO vs BY｜同 Forward OOS 樣本的探索性模型比較", "",
        "- **禁止正式使用**：歷史 Vintage 未驗證；非正式 PIT／Standard SOP／Trading 升格。",
        "- **共同樣本**：同一批 AO OOS group_index／ticker／date；兩組指標均由 canonical 評估函式計算。",
        f"- Scientific fitting：`{payload['fitting_fingerprint']}`",
        f"- OOS Groups：`{payload['same_oos_group_count']}`", "",
        "| 指標 | AO | BY | BY－AO |", "|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(f"| {row['label']} | {_fmt(row['ao'])} | {_fmt(row['by'])} | {_fmt(row['delta_by_minus_ao'], signed=True)} |")
    lines += ["", "**觀察到的差異不構成歷史版本合法性證據；不自動選模、升格或更改任何正式參數。**", ""]
    json_path = path.parent / "experimental_comparison.json"
    md_path = path.parent / "experimental_comparison.md"
    from core.file_integrity import canonical_json_sha256
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(existing) != canonical_json_sha256(payload):
            raise RuntimeError("同一fitting身分已有不同的比較證據，不得覆寫")
    else:
        atomic_write_json(json_path, payload)
    content = "\n".join(lines)
    if md_path.exists() and md_path.read_text(encoding="utf-8") != content:
        raise RuntimeError("同一fitting身分已有不同的探索性Markdown比較，不得覆寫")
    if not md_path.exists():
        atomic_write_text(md_path, content)
    print("\n" + content)
    print("REUSE/DONE｜" + str(md_path))
    return 0


def _read_isolated_result(root: Path, *, expected_seed: int, mode: str) -> tuple[Path | None, dict | None]:
    """Resolve immutable completed evidence; never select a latest/best model."""
    if mode == "forward_robustness":
        candidate_root = root / "benchmark_forward" / f"seed_{int(expected_seed)}"
        glob_pattern = "*/experimental_report.json"
    elif mode == "rolling":
        candidate_root = root / "rolling" / f"seed_{int(expected_seed)}"
        glob_pattern = "report_*/experimental_report.json"
    else:
        raise ValueError(f"未知隔離工件模式：{mode}")
    candidates = sorted(candidate_root.glob(glob_pattern))
    if not candidates:
        return None, None
    if len(candidates) != 1:
        raise RuntimeError(f"隔離研究有多個fitting身份；請在config明確指定來源，禁止自動擇優: {candidate_root}")
    path = candidates[0]
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("scientific_id") != SCIENTIFIC_ID
            or result.get("formal_model_input_authorized") is not False
            or result.get("historical_contemporaneous_vintage_proven") is not False
            or int(dict(result.get("fitting_identity") or {}).get("seed", -1)) != int(expected_seed)):
        raise ValueError("隔離研究工件身份／禁止正式使用標籤不合法")
    _validate_current_exploratory_source(result)
    if mode == "forward_robustness":
        if (result.get("fitting_fingerprint") != path.parent.name
                or not (path.parent / "experimental_forward_oos_scores.csv.gz").is_file()
                or not (ROOT / EXPERIMENT_RELATIVE_MODEL / "benchmark_forward"
                        / f"seed_{expected_seed}" / path.parent.name / "model.pt").is_file()):
            raise ValueError("Forward robustness工件不完整")
    else:
        if result.get("fitting_fingerprint") != path.parent.name.removeprefix("report_"):
            raise ValueError("Rolling aggregate工件身份不一致")
        score_path = path.parent / "experimental_rolling_oos_scores.csv.gz"
        if not score_path.is_file() or compute_file_sha256(score_path) != result.get("scores_sha256"):
            raise ValueError("Rolling scores工件缺失或SHA不符")
        for fold in result.get("folds") or []:
            checkpoint = ROOT / EXPERIMENT_RELATIVE_MODEL / "rolling" / f"seed_{expected_seed}" / str(fold["fitting_fingerprint"]) / "model.pt"
            if not checkpoint.is_file():
                raise ValueError("Rolling fold模型不完整: " + str(fold["fold_id"]))
    return path, result


def _rolling_fold_plan(bundle, *, args):
    """Use the same canonical score calendar and strict label maturity as PIT.

    A fold is a causal *fitting split*; this experiment still is not legally PIT
    because the historical source-vintage provenance has not been established.
    """
    from core.breakout_quality_policy import get_breakout_quality_rolling_test_mode, get_breakout_quality_workflow_settings
    from filters.breakout_quality.point_in_time_schedule import build_point_in_time_fold_periods
    from services.breakout_quality.point_in_time_scores import _fold_group_ids
    from types import SimpleNamespace

    mode = get_breakout_quality_rolling_test_mode("rolling")
    settings = get_breakout_quality_workflow_settings(experiment_profile=_STUDY.reference_profile)
    start = str(mode.score_start_date)
    if start.lower() == "auto":
        raise ValueError("隔離探索Rolling起點必須是config明確日期，不可自行推估")
    end = str(mode.score_end_date or bundle.outer_policy["effective_oos_end_date"])
    if end.lower() == "auto":
        end = str(bundle.outer_policy["effective_oos_end_date"])
    end = min(pd.Timestamp(end), pd.Timestamp(str(bundle.outer_policy["effective_oos_end_date"])))
    folds = build_point_in_time_fold_periods(
        start, end, fold_months=int(mode.fold_months),
        fold_anchor=mode.fold_anchor_date,
        single_score_block=bool(mode.single_score_block),
    )
    target_valid = np.asarray(bundle.target_valid, dtype=bool)
    result = []
    for fold in folds:
        ids = _fold_group_ids(
            bundle, fold,
            validation_months=int(settings.point_in_time_inner_validation_months),
            train_window_months=settings.point_in_time_train_window_months,
        )
        # Follow the canonical PIT scoring calendar.  Fold-end is the *last
        # prediction date*, not the outcome-maturity cutoff for retrospective
        # OOS evaluation.  Training labels remain strictly known before the
        # prediction cutoff in _fold_group_ids; OOS outcomes are used only
        # after scoring and never enter fitting or selection.
        score = ids["score_ids"]
        score = score[target_valid[score]]
        if min(len(ids["train_ids"]), len(ids["validation_ids"]), len(ids["final_ids"]), len(score)) < 20:
            raise ValueError(f"{fold['fold_id']}合法train/validation/final/OOS sample不足")
        if len(np.intersect1d(ids["final_ids"], score)):
            raise ValueError(f"{fold['fold_id']}有未隔離OOS樣本")
        report = {
            "fold_id": fold["fold_id"],
            "selection_start_date": str(pd.Timestamp(ids["history_start"]).date()) if ids["history_start"] is not None else str(bundle.summary["training_universe_start_date"]),
            "inner_validation_start_date": str(pd.Timestamp(ids["validation_start"]).date()),
            "oos_start_date": str(pd.Timestamp(fold["score_start"]).date()),
            "oos_end_date": str(pd.Timestamp(fold["score_end"]).date()),
            "counts": {
                "inner_train": len(ids["train_ids"]), "validation": len(ids["validation_ids"]),
                "selection": len(ids["final_ids"]), "oos": len(score),
            },
            "no_lookahead": True,
            "selection_label_end_before_oos": True,
            "inner_train_label_end_before_validation": True,
            "oos_outcomes_excluded_from_fit": True,
            "target_valid_filter_applied": True,
        }
        split = SimpleNamespace(
            inner_train_ids=ids["train_ids"], validation_ids=ids["validation_ids"],
            selection_ids=ids["final_ids"], oos_ids=score, report=report,
        )
        result.append((fold, split))
    return result


def _build_rolling_exploration(*, seed: int) -> tuple[Path, dict]:
    """Real isolated rolling fits; never publishes to the formal PIT namespace."""
    import os
    args = parse_canonical_training_args([
        "--model-architecture", AO_ARCHITECTURE,
        "--experiment-profile", _STUDY.reference_profile,
        "--seed", str(int(seed)),
    ])
    if bool(args.allow_stale_source):
        raise ValueError("隔離研究不得消費stale來源")
    recipe = get_continuous_ranker_execution_recipe(args.experiment_profile)
    original = load_daily_universal_ranker_data_with_price_source(
        filter_id=str(args.filter_id), model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile), preload_feature_bank=False,
        allow_stale_source=False,
    )
    margin = load_margin_source_for_ao_bundle(ROOT, original)
    features = MarginAugmentedDailyFeatureBank(original.feature_bank, margin.arrays)
    if features.shape[2] != len(features.feature_columns) or len(features) != len(original.group_table):
        raise ValueError("隔離Rolling特徵Shape／樣本身份不合法")
    bundle = replace(original, feature_bank=features)
    fold_plans = _rolling_fold_plan(bundle, args=args)
    torch, plan = resolve_ranker_execution_plan(args)
    identity = {
        "scientific_id": SCIENTIFIC_ID,
        "scientific_architecture": EXPERIMENT_ARCHITECTURE,
        "input_contract": FEATURE_CONTRACT,
        "mode": "rolling_oos_current_vintage_exploration",
        "research_authorization": "exploration_only_current_vintage_unverified",
        "formal_pit_input_authorized": False,
        "ao_profile": str(args.experiment_profile),
        "ao_training_semantics": ranker_api.training_semantics(bundle.profile),
        "ao_profile_manifest": bundle.profile.as_manifest_payload(),
        "ao_model_spec": bundle.model_spec.as_manifest_payload(),
        "training_settings": canonical_fitting_settings(args),
        "torch_execution": scientific_torch_execution(plan),
        "seed": int(seed),
        "source": source_training_identity(original),
        "margin_source": margin.source_identity,
        "folds": [fitting_split_identity(split.report) for _period, split in fold_plans],
        "feature_columns": features.feature_columns,
        "feature_shape": [int(features.shape[1]), int(features.shape[2])],
    }
    fingerprint = canonical_json_sha256(identity)
    base = ROOT / EXPERIMENT_RELATIVE_OUTPUT / "rolling" / f"seed_{int(seed)}"
    aggregate = base / f"report_{fingerprint}"
    if aggregate.exists():
        # A directory without a committed report is an interrupted aggregate;
        # completed folds are separate immutable artifacts and will be reused.
        if (aggregate / "experimental_report.json").exists():
            raise FileExistsError("已存在Rolling提交工件；必須REUSE驗證而非覆寫")
        import shutil
        shutil.rmtree(aggregate)
        print("[RESUME] 回收未完成Rolling彙總，保留已完成fold", flush=True)
    rows = []
    reports = []
    for index, (period, split) in enumerate(fold_plans, start=1):
        fold_identity = {
            "aggregate_fingerprint": fingerprint,
            "seed": int(seed),
            "fold_id": str(period["fold_id"]),
            "split": fitting_split_identity(split.report),
        }
        fold_fp = canonical_json_sha256(fold_identity)
        fold_output = base / "folds" / fold_fp
        model_dir = ROOT / EXPERIMENT_RELATIVE_MODEL / "rolling" / f"seed_{int(seed)}" / fold_fp
        report_file = fold_output / "fold.json"
        score_file = fold_output / "scores.csv.gz"
        model_file = model_dir / "model.pt"
        if report_file.is_file() and score_file.is_file() and model_file.is_file():
            record = json.loads(report_file.read_text(encoding="utf-8"))
            if (record.get("fitting_identity") != fold_identity
                    or compute_file_sha256(score_file) != record.get("scores_sha256")):
                raise RuntimeError(f"{period['fold_id']} fold工件身份／SHA不合法")
            frame = pd.read_csv(score_file, dtype={"ticker": "string"})
            print(f"[REUSE] Rolling {index}/{len(fold_plans)} | {period['fold_id']}", flush=True)
        else:
            if any(path.exists() for path in (fold_output, model_dir)):
                # Uncommitted output has no valid result.  Refit only this fold,
                # never overwrite a committed report or a completed sibling.
                if report_file.exists():
                    raise RuntimeError(f"{period['fold_id']}已提交fold但工件缺失；拒絕覆寫")
                import shutil
                for incomplete in (fold_output, model_dir):
                    if incomplete.is_dir():
                        shutil.rmtree(incomplete)
                    elif incomplete.exists():
                        raise RuntimeError(f"{period['fold_id']}異常路徑，拒絕刪除")
                print(f"[RESUME] Rolling {period['fold_id']}從未完成fold重訓", flush=True)
            fitted = _fit_exploratory_segment(
                args=args, features=features, bundle=bundle, split=split,
                torch=torch, plan=plan, recipe=recipe,
                phase_label=f"{SCIENTIFIC_ID} exploratory Rolling {period['fold_id']}",
            )
            frame = fitted["scores"].copy()
            frame.insert(0, "group_index", fitted["group_ids"])
            record = {
                "fitting_fingerprint": fold_fp,
                "fitting_identity": fold_identity,
                "selected_epoch": fitted["selected_epoch"],
                "selection_history": fitted["selection_history"],
                "final_history": fitted["final_history"],
                "oos_groups": len(fitted["group_ids"]),
                "metrics": _summary(fitted["quality"]),
            }
            model_dir.mkdir(parents=True, exist_ok=False)
            tmp_model = model_dir / "model.incomplete.pt"
            torch.save({
                "scientific_id": SCIENTIFIC_ID,
                "fitting_fingerprint": fold_fp,
                "fitting_identity": fold_identity,
                "model_spec": bundle.model_spec.as_manifest_payload(),
                "input_feature_contract": FEATURE_CONTRACT,
                "feature_count": int(features.shape[2]),
                "feature_columns": features.feature_columns,
                "model_state_dict": {name: value.detach().cpu() for name, value in fitted["model"].state_dict().items()},
                "promotion_authorized": False,
            }, tmp_model)
            os.replace(tmp_model, model_file)
            fold_output.mkdir(parents=True, exist_ok=False)
            frame.to_csv(score_file, index=False, compression="gzip")
            record["scores_sha256"] = compute_file_sha256(score_file)
            atomic_write_json(report_file, record)
            print(f"[DONE] Rolling {index}/{len(fold_plans)} | {period['fold_id']}", flush=True)
        reports.append({
            "fold_id": str(period["fold_id"]),
            "fitting_fingerprint": fold_fp,
            "selected_epoch": int(record["selected_epoch"]),
            "oos_groups": int(record["oos_groups"]),
            "scores_sha256": record["scores_sha256"],
        })
        rows.append(frame)
    if not rows:
        raise RuntimeError("Rolling必須至少有一個合法fold")
    combined = pd.concat(rows, ignore_index=True)
    if combined["group_index"].duplicated().any():
        raise ValueError("Rolling OOS score重複group_index")
    combined = combined.sort_values("group_index", kind="stable").reset_index(drop=True)
    group_ids = combined["group_index"].to_numpy(dtype=np.int64)
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    hs = build_hs_conditional_mfe_targets(
        bundle.group_table, np.isfinite(bundle.raw_target),
        true_hs_percentile_cutoff=float(recipe.objective_policy.secondary_pair_scope_threshold),
    )
    quality = ranker_api.hs_conditional_mfe_metrics(
        group_ids, bundle.group_table, hs,
        {"raw_safety": combined["raw_safety_score"].to_numpy(dtype=np.float32),
         "conditional_mfe": combined["conditional_mfe_score"].to_numpy(dtype=np.float32)},
        include_top_k_quality=False,
    )
    result = {
        "scientific_id": SCIENTIFIC_ID,
        "scientific_architecture": EXPERIMENT_ARCHITECTURE,
        "status": "EXPLORATORY_ROLLING_RESULT_AVAILABLE_PIT_UNPROVEN",
        "formal_model_input_authorized": False,
        "historical_contemporaneous_vintage_proven": False,
        "fitting_fingerprint": fingerprint,
        "fitting_identity": identity,
        "oos_groups": len(combined),
        "metrics": _summary(quality),
        "folds": reports,
        "coverage": margin.coverage,
    }
    aggregate.mkdir(parents=True, exist_ok=False)
    scores_file = aggregate / "experimental_rolling_oos_scores.csv.gz"
    combined.to_csv(scores_file, index=False, compression="gzip")
    result["scores_sha256"] = compute_file_sha256(scores_file)
    # Commit JSON last; a resumable partial aggregate has no published report.
    summary = [
        f"# {SCIENTIFIC_ID}｜Rolling OOS current-vintage 探索（非正式PIT）", "",
        "歷史首次發布及修訂Vintage未證明；不可宣稱正式PIT、選模或升格Trading。", "",
        f"- Seed：{int(seed)}",
        f"- Folds：{len(reports)}",
        f"- OOS groups：{len(combined)}",
        f"- Identity：`{fingerprint}`", "",
        "| Fold | OOS groups | Epoch |", "|---|---:|---:|",
    ]
    summary.extend(f"| {row['fold_id']} | {row['oos_groups']} | {row['selected_epoch']} |" for row in reports)
    summary += ["", "| Metric | Raw Safety | Conditional-MFE true HS |", "|---|---:|---:|"]
    for label, key in (("Daily rho", "mean_daily_spearman"), ("Global rho", "global_spearman_vs_raw_target"), ("Pair", "pairwise_concordance")):
        summary.append(f"| {label} | {_fmt(result['metrics']['raw_safety'].get(key))} | {_fmt(result['metrics']['conditional_mfe_true_hs'].get(key))} |")
    atomic_write_text(aggregate / "experimental_report.md", "\n".join(summary) + "\n")
    report_file = atomic_write_json(aggregate / "experimental_report.json", result)
    return report_file, result


def run_rolling_menu() -> int:
    """User-facing Rolling action: REUSE or build all missing folds in one entry."""
    path, result = _read_isolated_result(ROOT / EXPERIMENT_RELATIVE_OUTPUT, expected_seed=42, mode="rolling")
    if path is not None:
        print("[REUSE] 探索性Rolling OOS｜" + str(path.with_suffix(".md")))
        return 0
    path, result = _build_rolling_exploration(seed=42)
    print("[DONE] 探索性Rolling OOS｜" + str(path.with_suffix(".md")))
    return 0


def _rolling_reference_quality(scores: pd.DataFrame, *, filter_id: str, hs, group_table) -> dict:
    """Validate canonical AO Rolling score identity and compare the exact BY sample."""
    from filters.breakout_quality.ranking_score_store import load_selection_point_in_time_ranking_contract
    contract = load_selection_point_in_time_ranking_contract(
        str(ROOT), str(filter_id), AO_ARCHITECTURE, _STUDY.reference_profile,
        require_model_validation_pass=False,
    )
    score_path = Path(contract.score_path)
    header = list(pd.read_csv(score_path, nrows=0).columns)
    preferred = next((column for column in ("conditional_mfe_score", "model_score", "breakout_quality_score") if column in header), None)
    if preferred is None or not {"ticker", "date", "raw_safety_score"}.issubset(header):
        raise ValueError("AO Rolling canonical score缺少雙Head／日期欄位")
    ao = pd.read_csv(score_path, usecols=["ticker", "date", "raw_safety_score", preferred], dtype={"ticker": "string"})
    ao["date"] = pd.to_datetime(ao["date"], errors="raise").dt.strftime("%Y-%m-%d")
    by = scores.copy()
    by["date"] = pd.to_datetime(by["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if ao.duplicated(["ticker", "date"]).any() or by.duplicated(["ticker", "date"]).any():
        raise ValueError("AO/BY Rolling score ticker/date存在重複")
    aligned = by[["ticker", "date"]].merge(
        ao, on=["ticker", "date"], how="left", validate="one_to_one", indicator=True,
    )
    if (aligned["_merge"] != "both").any():
        raise ValueError("AO Rolling canonical score與BY實際OOS樣本無法精確對齊")
    raw = pd.to_numeric(aligned["raw_safety_score"], errors="raise").to_numpy(dtype=np.float32)
    primary = pd.to_numeric(aligned[preferred], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all() or not np.isfinite(primary).all():
        raise ValueError("AO Rolling OOS score存在非有限值")
    ids = by["group_index"].to_numpy(dtype=np.int64)
    expected = group_table.iloc[ids]
    if not np.array_equal(by["ticker"].astype(str).to_numpy(), expected["ticker"].astype(str).to_numpy()):
        raise ValueError("BY Rolling group_index/ticker錯配")
    if not np.array_equal(by["date"].to_numpy(), pd.to_datetime(expected["date"]).dt.strftime("%Y-%m-%d").to_numpy()):
        raise ValueError("BY Rolling group_index/date錯配")
    return {
        "metrics": _summary(ranker_api.hs_conditional_mfe_metrics(
            ids, group_table, hs, {"raw_safety": raw, "conditional_mfe": primary},
            include_top_k_quality=False,
        )),
        "canonical_score_sha256": compute_file_sha256(score_path),
        "reference_groups": len(ids),
        "reference_profile": _STUDY.reference_profile,
    }


def run_rolling_comparison() -> int:
    """Compare completed exploratory Rolling against verified same-sample AO PIT."""
    path, result = _read_isolated_result(ROOT / EXPERIMENT_RELATIVE_OUTPUT, expected_seed=42, mode="rolling")
    if path is None:
        print("[BUILD] Rolling比較缺少BY前置訓練；委派同一canonical producer。")
        code = run_rolling_menu()
        if code:
            return int(code)
        path, result = _read_isolated_result(ROOT / EXPERIMENT_RELATIVE_OUTPUT, expected_seed=42, mode="rolling")
        if path is None:
            raise RuntimeError("Rolling producer未產生可驗證的BY研究工件")
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings
    from services.research.breakout_quality_application import _run_continuous_rolling_mode_direct
    from filters.breakout_quality.ranking_score_store import load_selection_point_in_time_ranking_contract
    reference_settings = get_breakout_quality_workflow_settings(experiment_profile=_STUDY.reference_profile)
    try:
        load_selection_point_in_time_ranking_contract(
            str(ROOT), reference_settings.filter_id, AO_ARCHITECTURE,
            _STUDY.reference_profile, require_model_validation_pass=False,
        )
    except FileNotFoundError:
        print("[BUILD] Rolling比較缺少AO參照，委派正式AO Rolling producer。")
        code = _run_continuous_rolling_mode_direct(
            "apps/research.py model", reference_settings
        )
        if code:
            return int(code)
    original = load_daily_universal_ranker_data_with_price_source(
        filter_id=str(reference_settings.filter_id), model_architecture=AO_ARCHITECTURE,
        experiment_profile=_STUDY.reference_profile, preload_feature_bank=False,
        allow_stale_source=False,
    )
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    recipe = get_continuous_ranker_execution_recipe(_STUDY.reference_profile)
    hs = build_hs_conditional_mfe_targets(
        original.group_table, np.isfinite(original.raw_target),
        true_hs_percentile_cutoff=float(recipe.objective_policy.secondary_pair_scope_threshold),
    )
    score = pd.read_csv(path.parent / "experimental_rolling_oos_scores.csv.gz", dtype={"ticker": "string"})
    reference = _rolling_reference_quality(score, filter_id=str(reference_settings.filter_id), hs=hs, group_table=original.group_table)
    comparison = _comparison_payload({
        "reference_ao_same_oos": reference,
        "oos_groups": result["oos_groups"],
        "metrics": result["metrics"],
        "fitting_fingerprint": result["fitting_fingerprint"],
    })
    comparison["report_kind"] = "one_off_current_vintage_exploratory_rolling_same_oos_comparison"
    output = path.parent
    comparison_json = output / "experimental_comparison.json"
    comparison_md = output / "experimental_comparison.md"
    if comparison_json.is_file():
        if canonical_json_sha256(json.loads(comparison_json.read_text(encoding="utf-8"))) != canonical_json_sha256(comparison):
            raise RuntimeError("Rolling comparison evidence身份不同，不得覆寫")
    else:
        atomic_write_json(comparison_json, comparison)
    lines = [
        "# AO vs BY｜Rolling OOS current-vintage探索性同樣本比較", "",
        "**非正式PIT：BY歷史Vintage未驗證；不可作為Trading升格／模型選擇依據。**", "",
        f"- Fold count：{len(result['folds'])}；OOS group count：{result['oos_groups']}",
        f"- BY fitting fingerprint：`{result['fitting_fingerprint']}`", "",
        "| 指標 | AO | BY | BY－AO |", "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {row['label']} | {_fmt(row['ao'])} | {_fmt(row['by'])} | {_fmt(row['delta_by_minus_ao'], signed=True)} |" for row in comparison["rows"])
    content = "\n".join(lines) + "\n"
    if comparison_md.is_file() and comparison_md.read_text(encoding="utf-8") != content:
        raise RuntimeError("Rolling Markdown evidence身份不同，不得覆寫")
    if not comparison_md.exists():
        atomic_write_text(comparison_md, content)
    print("\n" + content)
    print("[DONE/REUSE] " + str(comparison_md))
    return 0


def _run_isolated_robustness(*, rolling: bool) -> int:
    """Same configured benchmark-seed set; each fit is independent and immutable."""
    from core.training_policy import resolve_robustness_benchmark_seeds
    seeds = tuple(int(value) for value in resolve_robustness_benchmark_seeds())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Robustness必須有至少兩個互異benchmark seeds")
    mode = "rolling" if rolling else "forward_robustness"
    output_root = ROOT / EXPERIMENT_RELATIVE_OUTPUT
    found = [_read_isolated_result(output_root, expected_seed=seed, mode=mode) for seed in seeds]
    complete = []
    for index, seed in enumerate(seeds, start=1):
        path, result = found[index-1]
        if path is None:
            if rolling:
                path, result = _build_rolling_exploration(seed=seed)
            else:
                path, _report_md = run(seed=seed, benchmark=True)
                result = json.loads(path.read_text(encoding="utf-8"))
        print(f"[{'REUSE' if found[index-1][0] is not None else 'DONE'}] {mode} seed {index}/{len(seeds)} | {seed}")
        complete.append({"seed": seed, "path": str(path.relative_to(ROOT)), "fingerprint": result["fitting_fingerprint"], "metrics": result["metrics"]})
    from core.file_integrity import canonical_json_sha256
    identity = {"scientific_id": SCIENTIFIC_ID, "mode": mode, "seeds": list(seeds),
                "fitting_fingerprints": [row["fingerprint"] for row in complete],
                "formal_model_input_authorized": False}
    fingerprint = canonical_json_sha256(identity)
    output = output_root / "robustness" / mode / fingerprint
    payload = {
        "scientific_id": SCIENTIFIC_ID,
        "fitting_fingerprint": fingerprint,
        "fitting_identity": identity,
        "formal_model_input_authorized": False,
        "historical_contemporaneous_vintage_proven": False,
        "benchmark_seeds": list(seeds),
        "seeds": complete,
    }
    if output.exists():
        if not (output / "experimental_report.json").is_file():
            import shutil
            shutil.rmtree(output)  # Only the uncommitted aggregate; seed artifacts are immutable.
        else:
            stored = json.loads((output / "experimental_report.json").read_text(encoding="utf-8"))
            if canonical_json_sha256(stored) != canonical_json_sha256(payload):
                raise RuntimeError("同Robustness身份的result不能被改寫")
            if not (output / "experimental_report.md").is_file():
                raise RuntimeError("Robustness已提交但缺少人讀工件，拒絕假裝REUSE")
            print("[REUSE] " + str(output / "experimental_report.md"))
            return 0
    output.mkdir(parents=True, exist_ok=False)
    sections = (("Raw Safety Daily rho", "raw_safety", "mean_daily_spearman"),
                ("Conditional MFE Daily rho", "conditional_mfe_true_hs", "mean_daily_spearman"),
                ("Pred-HS true-LS %", "lexicographic_model_gate", "predicted_hs_true_ls_pct"))
    lines = [f"# {SCIENTIFIC_ID}｜{mode} current-vintage探索性Robustness", "",
             "**非正式PIT、無自動挑Best seed、無Trading升格。**", "",
             f"Seeds：{' / '.join(map(str, seeds))}", "",
             "| 指標 | 各Seed平均 | 最小 | 最大 |", "|---|---:|---:|---:|"]
    for label, section, key in sections:
        values = [dict(dict(item["metrics"]).get(section) or {}).get(key) for item in complete]
        observed = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
        if len(observed) != len(seeds):
            raise ValueError(f"Robustness缺少一致的canonical metric: {section}.{key}")
        lines.append(f"| {label} | {_fmt(observed.mean())} | {_fmt(observed.min())} | {_fmt(observed.max())} |")
    lines += ["", "此彙整僅做seed敏感度診斷，沒有被視為未觀察Holdout。", ""]
    report = atomic_write_text(output / "experimental_report.md", "\n".join(lines))
    atomic_write_json(output / "experimental_report.json", payload)
    print("\n" + report.read_text(encoding="utf-8"))
    print("[DONE] " + str(report))
    return 0


def run_forward_robustness() -> int:
    return _run_isolated_robustness(rolling=False)


def run_rolling_robustness() -> int:
    return _run_isolated_robustness(rolling=True)


def run_timing_menu() -> int:
    """Timing is an engineering work type, owned by the shared Timing producer.

    The engineering benchmark profile is selected in config and shown here;
    it is never silently replaced by the current-vintage experimental model.
    """
    from core.breakout_quality_policy import (
        get_breakout_quality_rolling_timing_settings, get_breakout_quality_workflow_settings,
    )
    from services.research.breakout_quality_application import _interactive_rolling_timing_mode
    settings = get_breakout_quality_rolling_timing_settings()
    print("[工程Timing] Benchmark training profile = " + str(settings.experiment_profile))
    return int(_interactive_rolling_timing_mode(
        "apps/research.py model",
        get_breakout_quality_workflow_settings(experiment_profile=str(settings.experiment_profile)),
    ) or 0)

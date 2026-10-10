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
        "- AO Target、full-universe 樣本、Forward 切分、Loss 與 Seed42 不變；僅增加輸入欄位。", "",
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
                  "下一步若無明確 Safety/Pred-HS 純度共同改善且 Conditional-MFE 不退化，停止本次外部資料 Input 探索；不做多 Seed／超參數掃描。", ""])
    return "\n".join(lines)


def run(*, snapshot_fingerprint: str | None = None) -> tuple[Path, Path]:
    args = parse_canonical_training_args([
        "--model-architecture", AO_ARCHITECTURE,
        "--experiment-profile", _STUDY.reference_profile,
        "--seed", "42",
    ])
    if int(args.seed) != 42 or bool(args.allow_stale_source):
        raise ValueError("探索實驗只允許 AO Seed42 與當前來源")
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
    output = ROOT / EXPERIMENT_RELATIVE_OUTPUT / fingerprint
    model_dir = ROOT / EXPERIMENT_RELATIVE_MODEL / fingerprint
    if output.exists() or model_dir.exists():
        raise FileExistsError("MR-13BY此fitting identity已有工件；不得覆寫或冒稱正式REUSE: " + str(output))
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
        phase_label="MR-13BY experimental Forward refit",
    )
    # OOS truth must never influence the train, early stop, feature normalization,
    # or any other computational fitting decision.
    ids = np.asarray(split.oos_ids, dtype=np.int64)
    conditional_scores, sidecars = predict_score_output_payload(
        torch, model, bundle, ids, batch_size=int(args.evaluation_batch_size), plan=plan
    )
    if "raw_safety_score" not in sidecars:
        raise ValueError("MR-13BY需要AO完整雙Head OOS score")
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
        "selection_history": epoch.get("history", []),
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
    json_path = atomic_write_json(output / "experimental_report.json", result)
    md_path = atomic_write_text(output / "experimental_report.md", _markdown(result))
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
    answer = input("僅允許 current-vintage 探索性擬合，非正式模型；輸入 EXPLORE 確認：").strip()
    if answer != "EXPLORE":
        print("已取消探索性訓練。")
        return 0
    _json_path, md_path = run()
    print("DONE｜" + str(md_path))
    return 0


def run_forward_comparison() -> int:
    path, result = _read_experimental_report()
    if path is None:
        print("[BLOCKED] 探索性 Forward 工件尚未產生；請先執行[1]。")
        return 1
    _validate_current_exploratory_source(result)
    payload = _comparison_payload(result)
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

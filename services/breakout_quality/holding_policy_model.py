"""Whole-trade stopping producer with a mandatory pre-2021 development gate."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from core.dataset_profiles import build_dataset_generation_identity
from core.file_integrity import canonical_json_sha256, compute_observed_file_sha256, exclusive_file_lock
from core.position_exit_policy import bind_position_exit_provider, PositionExitDecision
from filters.breakout_quality.holding_value import HoldingFeatures, position_context
from filters.breakout_quality.holding_policy import (
    CONTRACT, DEVELOPMENT, MODEL_ID, build_policy_network, development_gate, episode_outcomes, outcome_metrics,
)
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from services.breakout_quality.holding_value_model import configuration, fold_ranges, _write_json
from services.breakout_quality.holding_episode_data import build_episode_rows, CachedHoldingFeatures
from services.breakout_quality.holding_policy_training import _fit_fold, predict_policy, configure_torch
from services.breakout_quality.formal_mintrade40_training_source import load_formal_mintrade40_training_source
from services.breakout_quality.formal_mintrade40_pit_source import slice_formal_mintrade40_pit_source_for_fold


def model_base(root, seed):
    return Path(root) / "models/research/breakout_quality/holding_value" / MODEL_ID / f"seed_{seed}"


def development_directory(root):
    return Path(root) / "outputs/research/breakout_quality/holding_policy" / MODEL_ID / "development_seed42_2018_2020"


def implementation_identity(root, *, samples=False):
    from core.execution_source_dependencies import execution_source_inventory
    names = {*execution_source_inventory("mintrade_truth", "daily_prices"),
             "filters/breakout_quality/holding_value.py", "services/breakout_quality/holding_episode_data.py",
             "services/breakout_quality/holding_value_model.py",
             "services/breakout_quality/formal_mintrade40_training_source.py",
             "services/breakout_quality/formal_mintrade40_pit_source.py"}
    if not samples:
        names.update(("filters/breakout_quality/holding_policy.py", "services/breakout_quality/holding_policy_training.py",
                      "services/breakout_quality/holding_policy_model.py", "core/position_exit_policy.py",
                      "core/position_model_registry.py", "services/breakout_quality/position_models.py"))
    return canonical_json_sha256({name: compute_observed_file_sha256(Path(root) / name) for name in sorted(names)})


def sources(root, *, seed, dataset, build=False):
    inputs, duplicates = discover_dataset_csv_inputs(Path(root), dataset)
    if duplicates or "0050" not in dict(inputs):
        raise ValueError("Ambiguous or missing canonical holding price inputs")
    teacher = load_formal_mintrade40_training_source(project_root=root, seed=int(seed), allow_optimizer_build=build)
    identity = {"contract": CONTRACT, "implementation": implementation_identity(root), "seed": int(seed),
                "dataset": build_dataset_generation_identity(root, dataset), "teacher_sha256": teacher.param_sha256,
                "input_files": {ticker: compute_observed_file_sha256(path) for ticker, path in inputs}}
    return inputs, teacher, identity


def load_episode_data(root, *, seed, inputs, teacher, cutoff):
    rows, evidence = build_episode_rows(root, seed=seed, teacher=teacher, cutoff=cutoff,
        inputs=inputs, implementation=implementation_identity(root, samples=True))
    needed = set(rows.ticker) | {"0050"}
    frames = {ticker: load_dataset_frame(path, ticker, min_rows=1).loc[:pd.Timestamp(cutoff)]
              for ticker, path in inputs if ticker in needed}
    features = CachedHoldingFeatures(rows, frames)
    evidence["unscorable_states"] = int(rows.feature_index.eq(0).sum())
    return rows, features, evidence


def fit_for_cutoff(root, *, seed, start, rows, features, teacher):
    mature = rows.loc[pd.to_datetime(rows.maturity_date) < start].drop(columns=["feature_index"])
    _, prefix = slice_formal_mintrade40_pit_source_for_fold(teacher, fit_cutoff=start - pd.Timedelta(days=1))
    digest = hashlib.sha256()
    for ticker in sorted(set(mature.ticker) | {"0050"}):
        frame = features.frames[ticker].loc[:start - pd.Timedelta(days=1)]
        digest.update(ticker.encode()); digest.update(frame.index.to_numpy().tobytes())
        digest.update(frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float).tobytes())
    identity = canonical_json_sha256({"contract": CONTRACT, "implementation": implementation_identity(root),
        "seed": int(seed), "cutoff": str(start.date()), "teacher_prefix": prefix,
        "prices_before_cutoff": digest.hexdigest(), "complete_mature_rows": mature.to_json(orient="records")})
    # Mode/end date are consumption properties, never a second physical FIT.
    directory = model_base(root, seed) / "fits" / f"fold_{start:%Y%m%d}_{identity[:16]}"
    with exclusive_file_lock(directory / "writer.lock", conflict_message="相同持倉 FIT 正在建立"):
        result = _fit_fold(rows, features, start=start, seed=seed, directory=directory, fit_identity=identity)
    return {**result, "fit_path": (directory / "fold.json").relative_to(root).as_posix(),
            "fit_sha256": compute_observed_file_sha256(directory / "fold.json")}


def _verify_fit(root, fold, seed):
    path = Path(root) / fold["fit_path"]
    if not path.resolve().is_relative_to(model_base(root, seed).resolve()):
        raise ValueError("Holding FIT path escapes its model/seed")
    if compute_observed_file_sha256(path) != fold["fit_sha256"]:
        raise ValueError("Holding FIT metric hash mismatch")
    if not pd.Timestamp(fold["fit_max_maturity"]) < pd.Timestamp(fold["cutoff"]):
        raise ValueError("Holding FIT maturity is not strictly before cutoff")
    for item in fold["models"].values():
        weight = path.parent / item["weights"]
        if weight.resolve().parent != path.parent.resolve() or compute_observed_file_sha256(weight) != item["weights_sha256"]:
            raise ValueError("Holding policy weight hash mismatch")


def inspect_development(root, *, dataset="full", identity=None):
    path = development_directory(root) / "result.json"
    if not path.exists():
        return {"complete": False, "passed": False, "reason": "尚未完成 2018–2020 Seed42 前置驗證"}
    try:
        if identity is None:
            _, _, identity = sources(root, seed=DEVELOPMENT["seed"], dataset=dataset)
        result = json.loads(path.read_text("utf-8"))
        if result["identity"] != {"sources": identity, "development": DEVELOPMENT} or result["status"] != "COMPLETED":
            raise ValueError("早期驗證的資料／程式／契約已變更")
        if development_gate(result["years"]) != result["gate"]:
            raise ValueError("早期驗證 gate 與逐年數值不符")
        for fold in result["folds"]:
            _verify_fit(root, fold, DEVELOPMENT["seed"])
        for name, sha in result["evidence_files"].items():
            if Path(name).name != name or compute_observed_file_sha256(path.parent / name) != sha:
                raise ValueError("早期逐筆交易證據不符")
        return {"complete": True, "passed": result["gate"]["passed"], "result": result,
                "sha256": compute_observed_file_sha256(path),
                "reason": "早期驗證 PASS" if result["gate"]["passed"] else "早期驗證未通過；禁止正式 FIT／Replay"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"complete": False, "passed": False, "reason": str(exc)}


def render_development(result):
    lines = ["HV2：2018–2020 Seed42 完整交易收益驗證", "", "年度 | Episodes | DL ΔR | Linear ΔR | DL 提前出場 | 持倉中位數（日）", "---|---:|---:|---:|---:|---:"]
    for year, values in result["years"].items():
        dl, linear = values["dl"], values["linear"]
        lines.append(f"{year} | {dl['episodes']} | {dl['mean_gain_r']:+.6f} | {linear['mean_gain_r']:+.6f} | "
                     f"{dl['early_exit_fraction']:.2%} | {dl['holding_days_median']:.1f}")
    gate = result["gate"]
    lines += ["", f"判定：{'PASS' if gate['passed'] else 'REJECT'}；年度等權 ΔR={gate['equal_year_mean_gain_r']:+.6f}；"
              f"勝原 P16={gate['positive_years']}/3；勝線性平均 ΔR={gate['equal_year_mean_gain_over_linear_r']:+.6f}",
              "", "原 P16／DL／線性使用相同完整交易；未成熟交易整筆排除並另報，不期末強平。",
              "此為整筆出場收益驗證；資金重新投入的效果須由通過後的 Research 正式回放評估。",
              "排除筆數：" + json.dumps(result["samples"]["excluded"], ensure_ascii=False),
              "", "原大贏家前十分位（逐年，以原 P16 完整 Net R 定義；僅診斷，不進 FIT）："]
    for year, values in result["years"].items():
        d = values["dl"]
        lines.append(f"{year}：原 {d['top_decile_natural_r_sum']:.4f}R → DL {d['top_decile_policy_r_sum']:.4f}R；"
                     f"被截短的正損失 {d['top_decile_lost_positive_r']:.4f}R")
    return "\n".join(lines) + "\n"


def run_development(root, *, dataset="full"):
    import torch
    root = Path(root)
    folder = development_directory(root); folder.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(folder / "writer.lock", conflict_message="早期出場驗證正在執行"):
        inputs, teacher, identity = sources(root, seed=DEVELOPMENT["seed"], dataset=dataset, build=True)
        old = inspect_development(root, dataset=dataset, identity=identity)
        if old["complete"]:
            print(render_development(old["result"]), flush=True)
            return old["result"]
        rows, features, samples = load_episode_data(root, seed=DEVELOPMENT["seed"], inputs=inputs,
                                                  teacher=teacher, cutoff=DEVELOPMENT["truth_cutoff"])
        fits = [fit_for_cutoff(root, seed=DEVELOPMENT["seed"], start=pd.Timestamp(f"{year}-01-01"),
                rows=rows, features=features, teacher=teacher) for year in DEVELOPMENT["years"]]
        evaluation = rows.loc[pd.to_datetime(rows.entry_date).dt.year.isin(DEVELOPMENT["years"])].reset_index(drop=True)
        device = configure_torch()
        outcomes = {}
        for kind in ("dl", "linear"):
            prediction = np.full(len(evaluation), np.nan)
            for fold in fits:
                ids = np.flatnonzero(pd.to_datetime(evaluation.decision_execution_date).dt.year == pd.Timestamp(fold["cutoff"]).year)
                weights = root / fold["fit_path"]
                payload = torch.load(weights.parent / fold["models"][kind]["weights"], map_location="cpu", weights_only=True)
                model = build_policy_network(kind).to(device); model.load_state_dict(payload["state_dict"])
                prediction[ids] = predict_policy(model, evaluation.iloc[ids], features,
                    (payload["state_mean"].numpy(), payload["state_scale"].numpy()), device)
            outcomes[kind] = episode_outcomes(evaluation, prediction)
            outcomes[kind].to_csv(folder / (kind + "_episodes.csv"), index=False, encoding="utf-8-sig")
        years = {str(year): {kind: outcome_metrics(data.loc[pd.to_datetime(data.entry_date).dt.year == year])
                            for kind, data in outcomes.items()} for year in DEVELOPMENT["years"]}
        result = {"status": "COMPLETED", "identity": {"sources": identity, "development": DEVELOPMENT},
                  "folds": fits, "samples": samples, "years": years, "gate": development_gate(years),
                  "evidence_files": {kind + "_episodes.csv": compute_observed_file_sha256(folder / (kind + "_episodes.csv"))
                                     for kind in outcomes}}
        (folder / "result.md").write_text(render_development(result), encoding="utf-8")
        _write_json(folder / "result.json", result)
        if not inspect_development(root, dataset=dataset, identity=identity)["complete"]:
            raise RuntimeError("Development publication failed verification")
        print(render_development(result), flush=True)
        return result


def inspect_holding_model(root, *, seed, config):
    path = model_base(root, seed) / config["mode"] / "manifest.json"
    gate = inspect_development(root, dataset=config["dataset"])
    if not gate["complete"] or not gate["passed"]:
        return {"ready": False, "path": str(path), "reason": gate["reason"],
                "build_allowed": not gate["complete"]}
    try:
        _, _, source = sources(root, seed=seed, dataset=config["dataset"])
        expected = {"sources": source, "configuration": config, "development_sha256": gate["sha256"]}
        manifest = json.loads(path.read_text("utf-8"))
        if manifest.get("status") != "COMPLETED" or manifest["identity"] != expected:
            raise ValueError("Whole-trade policy contract/source mismatch")
        if [(f["start"], f["end"]) for f in manifest["folds"]] != [(str(a.date()), str(b.date())) for a, b in fold_ranges(config)]:
            raise ValueError("Whole-trade policy annual coverage incomplete")
        for fold in manifest["folds"]:
            _verify_fit(root, fold, seed)
        return {"ready": True, "path": str(path), "sha256": compute_observed_file_sha256(path),
                "manifest": manifest, "reason": "整筆交易出場模型 PIT／前置驗證完整，可重用"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"ready": False, "path": str(path), "reason": str(exc)}


def ensure_holding_model(root, *, seed, config):
    root = Path(root)
    gate = inspect_development(root, dataset=config["dataset"])
    if not gate["complete"]:
        run_development(root, dataset=config["dataset"])
        gate = inspect_development(root, dataset=config["dataset"])
    if not gate["complete"] or not gate["passed"]:
        raise ValueError(gate["reason"])
    directory = model_base(root, seed) / config["mode"]; directory.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(directory / "writer.lock", conflict_message="整筆交易出場模型正在準備"):
        status = inspect_holding_model(root, seed=seed, config=config)
        if status["ready"]:
            return status
        inputs, teacher, source = sources(root, seed=seed, dataset=config["dataset"], build=True)
        ranges = fold_ranges(config)
        rows, features, samples = load_episode_data(root, seed=seed, inputs=inputs, teacher=teacher,
                                                    cutoff=ranges[-1][0] - pd.Timedelta(days=1))
        folds = [{**fit_for_cutoff(root, seed=seed, start=start, rows=rows, features=features, teacher=teacher),
                  "start": str(start.date()), "end": str(end.date())} for start, end in ranges]
        _write_json(directory / "manifest.json", {"status": "COMPLETED", "identity": {
            "sources": source, "configuration": config, "development_sha256": gate["sha256"]},
            "folds": folds, "samples": samples, "evaluation_label": "迭代研究 OOS", "oracle_status": "CONTRACT_UNRESOLVED"})
        result = inspect_holding_model(root, seed=seed, config=config)
        if not result["ready"]:
            raise ValueError(result["reason"])
        return result


@contextmanager
def holding_model_runtime(root, *, seed, config, evidence_path=None):
    import torch
    root = Path(root)
    ready = inspect_holding_model(root, seed=seed, config=config)
    if not ready["ready"]:
        raise ValueError("Whole-trade policy not READY: " + ready["reason"])
    inputs, _ = discover_dataset_csv_inputs(root, config["dataset"])
    paths = dict(inputs); market = load_dataset_frame(paths["0050"], "0050", min_rows=1)
    frames, features, loaded, records = {}, {}, {}, []
    torch.set_num_threads(2)
    def predict(*, ticker, position, information_date, execution_date, completed_atr, completed_high, params):
        if ticker not in paths:
            return None
        if ticker not in frames:
            frames[ticker] = load_dataset_frame(paths[ticker], ticker, min_rows=1)
            features[ticker] = HoldingFeatures({ticker: frames[ticker]}, market)
        coordinate = features[ticker].coordinates(ticker, pd.Timestamp(information_date))
        if coordinate is None:
            return None
        fits = [f for f in ready["manifest"]["folds"] if pd.Timestamp(f["start"]) <= execution_date <= pd.Timestamp(f["end"])]
        if len(fits) != 1:
            raise ValueError("No unique PIT stopping fit for executable session")
        fold = fits[0]
        if fold["fit_identity"] not in loaded:
            path = root / fold["fit_path"]
            payload = torch.load(path.parent / fold["models"]["dl"]["weights"], map_location="cpu", weights_only=True)
            if payload["seed"] != seed or payload["fit_identity"] != fold["fit_identity"] or payload["kind"] != "dl":
                raise ValueError("Wrong position-model seed, FIT or head")
            model = build_policy_network(); model.load_state_dict(payload["state_dict"]); model.eval()
            loaded[fold["fit_identity"]] = model, payload
        model, payload = loaded[fold["fit_identity"]]
        state = position_context(position, frame=frames[ticker], information_pos=coordinate[0],
                                 completed_atr=completed_atr, completed_high=completed_high, params=params)
        sequence, context = features[ticker].batch(pd.DataFrame([{"ticker": ticker, "information_date": information_date, "state": state}]))
        with torch.inference_mode():
            value = float(torch.sigmoid(model(torch.from_numpy(sequence),
                (torch.from_numpy(context) - payload["state_mean"]) / payload["state_scale"])).item())
        requested = value >= CONTRACT["threshold_probability"]
        records.append({"ticker": ticker, "entry_date": str(position["entry_trade_date"]),
            "information_date": str(information_date), "execution_date": str(execution_date),
            "exit_probability": value, "exit_requested": requested, "fit_identity": fold["fit_identity"]})
        return PositionExitDecision(requested, value, "exit_probability")
    with bind_position_exit_provider(predict):
        yield ready
    if evidence_path is not None:
        _write_json(evidence_path, {"model_manifest_sha256": ready["sha256"], "seed": seed,
                                   "contract": CONTRACT, "decisions": records})

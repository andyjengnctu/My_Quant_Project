"""Storage proof for failed robustness work replaced by completed observations.

Age is diagnostic only. Scientific manifests, observations, model weights and
unique scores are never rewritten or removed by this adapter.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import re

from core.artifact_lifecycle import relative_path
from core.compare_csv_storage import clean_path, file_metadata, plain
from core.file_integrity import (canonical_json_sha256, exclusive_file_lock,
                                 file_change_signature, sibling_publication_lock_path)
from core.strategy_compare_policy import (get_strategy_multi_seed_robustness_profiles,
    get_strategy_multi_seed_robustness_settings, get_strategy_comparison_settings)
from services.research import strategy_multi_seed_robustness as owner
from services.research.strategy_robustness_units import scientific_unit_signature

_FINGERPRINT = re.compile(r"[0-9a-f]{16}\Z")
_BASELINE_FILES = frozenset({"no_filter_orderable_candidates.csv", "no_filter_daily_capacity.csv",
    "no_filter_equity.csv", "no_filter_trades.csv", "no_filter_selected_buys.csv",
    "yearly_returns_comparison.csv", "strategy_comparison.md"})
_SCORE_NAMES = frozenset({"scores.csv", "validation_scores.csv", "selection_point_in_time_scores.csv"})


def _remember(path, revisions):
    if not clean_path(path) or not plain(path.lstat()):
        raise ValueError("取代證據不是一般實體檔案")
    stamp = file_change_signature(path)
    if revisions.setdefault(path, stamp) != stamp:
        raise ValueError("取代證據在核對期間變動")
    return stamp


def _read(path, revisions):
    before = _remember(path, revisions)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or file_change_signature(path) != before:
        raise ValueError("取代證據格式無效或讀取期間變動")
    return data


def _unchanged(revisions):
    if any(not clean_path(path) or file_change_signature(path) != stamp
           for path, stamp in revisions.items()):
        raise ValueError("取代成果、舊工作狀態或引用證據已變動")


def _snapshot_science(root, run, revisions):
    """Pin durable evidence and the exact external files its receipts name."""
    for path in run.rglob("*"):
        if not path.is_file() or path.is_relative_to(run / "work"):
            continue
        _remember(path, revisions)
        if path.suffix != ".json":
            continue
        payload = _read(path, revisions)
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                recorded = value.get("path")
                if isinstance(recorded, str) and recorded:
                    raw = Path(recorded)
                    target = raw if raw.is_absolute() else root / relative_path(recorded)
                    if not target.is_relative_to(root):
                        raise ValueError("科學收據引用專案外來源，不能清理")
                    if target.is_file():
                        _remember(target, revisions)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)


def _units(contract):
    arms = contract.get("stochastic_arms")
    seeds = contract.get("resolved_seeds")
    if not isinstance(arms, list) or not arms or not isinstance(seeds, list) or not seeds:
        raise ValueError("舊工作缺少完整 arm／seed 契約")
    result = {}
    for arm in arms:
        for seed in seeds:
            key = str(arm["arm_id"]), int(seed)
            signature = scientific_unit_signature(contract=contract, arm_id=key[0], seed=key[1])
            if signature is None or key in result:
                raise ValueError("舊工作單元身分缺少或重複")
            result[key] = signature
    return result


def _observation(validated, unit):
    arm, seed = unit
    rows, years = validated["seed_frame"], validated["seed_yearly_frame"]
    row = rows[(rows["arm_id"].astype(str) == arm) & (rows["seed"].astype(int) == seed)]
    yearly = years[(years["arm_id"].astype(str) == arm) & (years["seed"].astype(int) == seed)]
    if len(row) != 1 or yearly.empty:
        raise ValueError("正式替代成果缺少唯一觀察或年度資料")
    return {"row": row.iloc[0].to_dict(), "yearly": yearly.to_dict("records")}


class SupersededRobustnessCleanup:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.candidates = {}
        self.groups = {}
        self.sources = {}
        self.errors = {}
        self.evidence = {}
        self.leases = set()
        self.ownership_references = {}
        self.retained_roots = set()

    def candidate(self, relative):
        return relative in self.candidates

    @property
    def candidate_paths(self):
        return set(self.candidates)

    @property
    def reference_targets(self):
        result = set(self.candidates)
        for relative, group in self.candidates.items():
            boundary = group["model_root"] if relative.is_relative_to(group["model_root"]) else group["run"] / "work"
            result.update(p for p in relative.parents if p.is_relative_to(boundary))
        return result

    @property
    def proof_revisions(self):
        """Return only captured publication/config authority, without re-reading."""
        revisions = {}
        for group in self.candidates.values():
            for path, stamp in group["revisions"].items():
                if revisions.setdefault(path, stamp) != stamp:
                    raise ValueError("取代證明引用了同一檔案的不同版本")
        return revisions

    @property
    def proof_scopes(self):
        """Keep old late results, replacements and model publication trees strict."""
        scopes = set()
        for group in self.candidates.values():
            scopes.update((group["run"], group["model_root"], *group["replacements"]))
            scopes.update(path.relative_to(self.root) for path in group.get("pit_roots", ()))
        return scopes

    def prepare(self, inventory):
        for table in (self.candidates, self.groups, self.sources, self.errors, self.evidence,
                      self.ownership_references):
            table.clear()
        self.retained_roots.clear()
        # Unrelated projects and scopes need no robustness configuration. The
        # configured producer is consulted only after identifying its namespace.
        if not any(p.name == "manifest.json" and _FINGERPRINT.fullmatch(p.parent.name)
                   and p.parts[:3] == ("outputs", "strategy_compare", "robustness")
                   for p in inventory):
            return
        for menu in get_strategy_multi_seed_robustness_profiles():
            cfg = get_strategy_multi_seed_robustness_settings(menu["robustness_id"])
            settings = get_strategy_comparison_settings(cfg.profile_id)
            base = relative_path(cfg.output_root)
            model_base = relative_path(cfg.model_work_root)
            if base.parts[:3] != ("outputs", "strategy_compare", "robustness") or model_base.parts[:1] != ("models",):
                continue
            markers = sorted(p for p in inventory if p.name == "manifest.json" and p.parent.parent == base
                             and _FINGERPRINT.fullmatch(p.parent.name))
            records = {}
            model_owners = {}
            for marker in markers:
                try:
                    revisions = {}
                    payload = _read(self.root / marker, revisions)
                    contract = payload.get("contract", {})
                    if not isinstance(contract, dict):
                        raise ValueError("Robustness manifest 契約格式無效")
                    model_owners[marker] = (contract.get("model_artifact_fingerprint"), revisions)
                    if (contract.get("robustness_id") != cfg.robustness_id
                            or contract.get("fingerprint") != marker.parent.name):
                        continue
                    records[marker.parent] = (payload, revisions)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self.errors[marker] = str(exc)
            if records and owner.PROJECT_ROOT.resolve() != self.root:
                for run in records:
                    self.errors[run] = "Robustness producer 與清理專案根目錄不同"
                continue
            completed = {}
            for run, (payload, _) in records.items():
                if payload.get("status") != "COMPLETED":
                    continue
                revisions = {}
                try:
                    _snapshot_science(self.root, self.root / run, revisions)
                    inspection = owner.inspect_completed_robustness_run(self.root / run)
                    if inspection.ready:
                        result = inspection.value
                        _unchanged(revisions)
                        completed[run] = result, revisions, _units(result["contract"])
                    else:
                        self.errors[run] = f"{inspection.status.value}: {inspection.reason}"
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self.errors[run] = str(exc)
            for run, (payload, revisions) in records.items():
                if payload.get("status") != "FAILED":
                    continue
                try:
                    group = self._group(run, payload, revisions, completed, cfg, settings)
                    self.groups[run] = group
                    for relative in inventory:
                        local = relative.relative_to(run / "work") if relative.is_relative_to(run / "work") else None
                        if (local is not None and len(local.parts) == 3 and local.parts[0] == "benchmark_baselines"
                                and local.name in _BASELINE_FILES
                                and local.parts[1] in {f"{a}__seed_{s}" for a, s in group["units"]}):
                            marker = relative.parent / "strategy_comparison.json"
                            pair = _read(self.root / marker, group["revisions"])
                            arm, seed = local.parts[1].rsplit("__seed_", 1)
                            observed = group["observations"][(arm, int(seed))]
                            meta = pair.get("metadata", {})
                            if (meta.get("output_dir") != relative.parent.as_posix()
                                    or meta.get("output_scope") != "caller_override"
                                    or meta.get("score_source") != "dl_off_baseline_only"
                                    or meta.get("comparison_design") != "standalone_dl_off_active_param_replay"
                                    or meta.get("comparison_period") != group["contract"]["comparison_period"]
                                    or meta.get("dataset") != group["contract"].get("dataset")
                                    or meta.get("params_file_sha256") != observed["row"].get("strategy_param_sha256")):
                                continue
                            yearly = [{"year": row["year"], "return_pct": row["no_filter_return_pct"],
                                       "is_complete_year": row["is_full_year"]} for row in pair.get("yearly", [])]
                            owner._validate_rebuilt_observation(existing_row=observed["row"],
                                rebuilt={**pair["no_filter"], "selected_epoch": 0, "yearly": yearly},
                                existing_yearly=owner.pd.DataFrame(observed["yearly"]))
                            self.candidates[relative] = group
                            self.ownership_references.setdefault(marker, {})[("metadata", "output_dir")] = meta["output_dir"]
                    # Bare model fingerprints are producer references too. An
                    # unknown contract must not escape protection just because
                    # it cannot establish this adapter's deletion authority.
                    shared = [marker for marker, (fingerprint, _) in model_owners.items()
                              if marker.parent != run and (not _FINGERPRINT.fullmatch(str(fingerprint))
                                  or fingerprint == payload["contract"].get("model_artifact_fingerprint"))]
                    if not shared and len(model_owners) == len(markers):
                        for _, owner_revisions in model_owners.values():
                            for path, stamp in owner_revisions.items():
                                if group["revisions"].setdefault(path, stamp) != stamp:
                                    raise ValueError("共享模型 owner 在核對期間變動")
                        group["owner_markers"] = frozenset(markers)
                        group["owner_base"] = base
                        self._model_candidates(group, inventory, settings, cfg)
                    self.retained_roots.update(group["replacements"])
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                    self.errors[run] = str(exc)
        # A validated receipt may itself depend on an old work file. Such a
        # backing file is evidence, even when its basename otherwise is transient.
        dependencies = {path for group in self.candidates.values() for path in group["revisions"]}
        for relative in list(self.candidates):
            if self.root / relative in dependencies:
                self.candidates.pop(relative)
                self.sources.pop(relative, None)

    def _group(self, run, payload, revisions, completed, cfg, settings):
        contract = payload["contract"]
        retention = contract.get("retention", {})
        if any(retention.get(k) is not False for k in ("keep_checkpoints", "keep_scores", "keep_replay_details")):
            raise ValueError("原 producer retention 未放行暫存")
        if not _FINGERPRINT.fullmatch(str(contract.get("model_artifact_fingerprint", ""))):
            raise ValueError("舊模型暫存身分缺少")
        units = _units(contract)
        # Pin before reading rows. A replacement between reading and the first
        # snapshot must never acquire the previously verified observation's authority.
        _snapshot_science(self.root, self.root / run, revisions)
        matched = {}
        replacements = set()
        # A single validated completed superset is sufficient; do not merge
        # conflicting score revisions just because aggregate fingerprints differ.
        for source, (result, source_revisions, signatures) in completed.items():
            if not all(signatures.get(unit) == sig for unit, sig in units.items()):
                continue
            matched = {unit: _observation(result, unit) for unit in units}
            old_rows = owner._load_seed_results(self.root / run / owner.SEED_RESULTS_FILENAME)
            old_years = owner._load_seed_yearly_results(self.root / run / owner.SEED_YEARLY_RESULTS_FILENAME)
            old = {"seed_frame": old_rows, "seed_yearly_frame": old_years}
            try:
                for row in old_rows.to_dict("records"):
                    unit = str(row["arm_id"]), int(row["seed"])
                    owner._observation_equivalence(_observation(old, unit), matched[unit])
                for path in (self.root / run / "work/results").glob("*.json"):
                    row = _read(path, revisions)
                    unit = str(row["arm_id"]), int(row["seed"])
                    owner._observation_equivalence({"row": row, "yearly": row["yearly"]}, matched[unit])
            except (ValueError, KeyError):
                matched = {}
                continue
            for path, stamp in source_revisions.items():
                if revisions.setdefault(path, stamp) != stamp:
                    raise ValueError("舊工作與替代成果引用同一檔案的版本不同")
            replacements.add(source)
            break
        if not matched:
            raise ValueError("舊工作所有 arm／seed 尚無同身分且觀察一致的完整替代成果")
        _snapshot_science(self.root, self.root / run, revisions)
        for file in (self.root / "config").rglob("*.py"):
            _remember(file, revisions)
        group = {"run": run, "contract": contract, "revisions": revisions, "units": units,
                 "model_root": relative_path(cfg.model_work_root) / contract["model_artifact_fingerprint"],
                 "replacements": replacements, "observations": matched,
                 "config": canonical_json_sha256({"cfg": asdict(cfg), "settings": asdict(settings)})}
        _unchanged(revisions)
        return group

    def _model_candidates(self, group, inventory, settings, cfg):
        # Filled by the model owner's complete publication validator below.
        # Baseline-work cleanup does not authorize deleting a model or a score.
        from services.research.strategy_compare_training import (
            resolve_shared_model_rolling_robustness_dirs, validate_strategy_compare_training_artifacts)
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        old_root = self.root / group["model_root"]
        for child in old_root.iterdir() if old_root.is_dir() else ():
            if not child.is_dir() or "__seed_" not in child.name:
                continue
            dl_id, raw_seed = child.name.rsplit("__seed_", 1)
            if dl_id not in settings.dl_sources or not raw_seed.isdecimal():
                continue
            source = settings.dl_sources[dl_id]
            workflow = get_breakout_quality_workflow_settings(experiment_profile=source.experiment_profile)
            period = group["contract"]["comparison_period"]
            dirs = resolve_shared_model_rolling_robustness_dirs(project_root=self.root, source=source,
                workflow=workflow, seed=int(raw_seed), benchmark_id=cfg.benchmark_id, comparison_start=period["start"])
            if dirs is None:
                continue
            canonical, research = dirs
            if canonical.is_relative_to(old_root) or not clean_path(canonical):
                continue
            revisions = dict(group["revisions"])
            pending = {}
            exclusions = {}
            try:
                _snapshot_science(self.root, canonical, revisions)
                validate_strategy_compare_training_artifacts(project_root=self.root, source=source,
                    workflow=workflow, seed=int(raw_seed), model_dir=canonical, research_dir=research,
                    comparison_start=period["start"], comparison_end=period["end"],
                    point_in_time_dir_override=canonical)
                for relative in inventory:
                    path = self.root / relative
                    if not path.is_relative_to(child / "model") or path.name not in _SCORE_NAMES:
                        continue
                    local = path.relative_to(child / "model")
                    if local.name == "selection_point_in_time_scores.csv" and len(local.parts) == 1:
                        marker = "selection_point_in_time_manifest.json"
                        old = _read(child / "model" / marker, revisions)
                        new = _read(canonical / marker, revisions)
                        identity = ("filter_id", "experiment_profile", "model_architecture", "seed", "score_period",
                            "selection_period", "source_dataset", "source_continuous_target", "fold_count", "fold_identity")
                        if any(old.get(k) != new.get(k) for k in identity):
                            continue
                        key = "scores"
                    elif len(local.parts) == 3 and local.parts[0] == "folds" and local.name in {"scores.csv", "validation_scores.csv"}:
                        old = _read(path.parent / "manifest.json", revisions)
                        new = _read((canonical / local).parent / "manifest.json", revisions)
                        # Timestamp and import location are execution provenance;
                        # every FIT, seed, period, source and artifact field stays.
                        omit = {"generated_at_utc", "migration"}
                        if {k:v for k,v in old.items() if k not in omit} != {k:v for k,v in new.items() if k not in omit}:
                            continue
                        key = "scores" if local.name == "scores.csv" else "validation_scores"
                    else:
                        continue
                    old_receipt = old.get("artifacts", {}).get(key)
                    new_receipt = new.get("artifacts", {}).get(key)
                    if not old_receipt or old_receipt != new_receipt or old_receipt.get("filename") != path.name:
                        continue
                    retained = canonical / local
                    if not retained.is_file() or retained.stat().st_size != path.stat().st_size:
                        continue
                    pending[relative] = retained.relative_to(self.root), old_receipt["sha256"]
                    marker = (child / "model" / "selection_point_in_time_manifest.json"
                              if len(local.parts) == 1 else path.parent / "manifest.json")
                    exclusions.setdefault(marker.relative_to(self.root), {})[("artifacts", key, "filename")] = path.name
                _unchanged(revisions)
                for relative, (retained, digest) in pending.items():
                    self.candidates[relative] = {**group, "revisions": dict(revisions), "expected_sha256": digest,
                        "pit_roots": (child / "model", canonical)}
                    self.sources[relative] = retained
                if pending:
                    self.retained_roots.add(canonical.relative_to(self.root))
                    for marker, fields in exclusions.items():
                        self.ownership_references.setdefault(marker, {}).update(fields)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                self.errors[child.relative_to(self.root)] = str(exc)

    @contextmanager
    def deletion_lease(self, worker, relative):
        group = self.candidates[relative]
        roots = {group["run"], *group["replacements"]}
        # A baseline-only attempt may never have produced a model workspace.
        # Its run lease already excludes that attempt's writer; do not create
        # an unrelated models tree merely to remove known baseline work files.
        if relative.is_relative_to(group["model_root"]) or (self.root / group["model_root"]).exists():
            roots.add(group["model_root"])
        acquired = []
        try:
            with ExitStack() as stack:
                if worker.apply:
                    from services.breakout_quality.fit_repair_transaction import selection_pit_build_lock_path
                    locks = {sibling_publication_lock_path(self.root / run) for run in roots}
                    locks.update(selection_pit_build_lock_path(pit) for pit in group.get("pit_roots", ()))
                    for lock in sorted(locks):
                        if not clean_path(lock):
                            raise ValueError("Robustness publication lease 路徑不安全")
                        stack.enter_context(exclusive_file_lock(lock, conflict_message="Robustness工作或替代成果仍在使用"))
                        acquired.append(lock)
                self.leases.add(relative)
                try:
                    yield
                finally:
                    self.leases.discard(relative)
        finally:
            for lock in acquired:
                worker.lifecycle.removed(lock.relative_to(self.root))

    def proof(self, worker, relative):
        if relative not in self.leases:
            raise ValueError("尚未取得Robustness原工作與替代成果lease")
        group = self.candidates[relative]
        revisions = dict(group["revisions"])
        path = self.root / relative
        _remember(path, revisions)
        if relative in self.sources:
            source = self.root / self.sources[relative]
            _remember(source, revisions)
            if (file_metadata(path, path.lstat()) != file_metadata(source, source.lstat())
                    or worker.hash_file(path) != worker.hash_file(source)
                    or worker.hash_file(path) != group["expected_sha256"]):
                return ""
        self.evidence[relative] = revisions
        self.verify_before_delete(worker, relative)
        return "舊失敗工作所有計算單元已有同身分且觀察一致的完整成果；原保留政策允許清理暫存"

    def verify_before_delete(self, worker, relative):
        if relative not in self.leases or relative not in self.evidence:
            raise ValueError("缺少本次Robustness刪除證據或lease")
        _unchanged(self.evidence[relative])
        group = self.candidates[relative]
        if relative in self.sources:
            current_owners = {p.relative_to(self.root) for p in (self.root / group["owner_base"]).glob("*/manifest.json")
                              if _FINGERPRINT.fullmatch(p.parent.name)}
            if current_owners != group["owner_markers"]:
                raise ValueError("共享模型的 owner 清單在核對後變動")
        cfg = get_strategy_multi_seed_robustness_settings(group["contract"]["robustness_id"])
        settings = get_strategy_comparison_settings(cfg.profile_id)
        if canonical_json_sha256({"cfg": asdict(cfg), "settings": asdict(settings)}) != group["config"]:
            raise ValueError("Robustness current config在核對後變動")

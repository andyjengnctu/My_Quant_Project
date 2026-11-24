"""Remove result-free Optuna scratch only after its producer replaced the work identity."""
from contextlib import closing, contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

from core.compare_csv_storage import clean_path, plain
from core.file_integrity import file_change_signature, is_sha256_hex
from services.optimizer.runtime import _OptimizerStudyWriteLock


_NAME = re.compile(r"outer_oos_runtime_([0-9a-f]{20})_(\d{4}(?:\d{2})?)(?:_seed\d+_\d+)?\.db\Z")
_SIDECARS = ("-wal", "-shm", "-journal")
_TABLES = {
    "studies", "version_info", "study_directions", "study_user_attributes",
    "study_system_attributes", "trials", "trial_user_attributes", "trial_system_attributes",
    "trial_params", "trial_values", "trial_intermediate_values", "trial_heartbeats", "alembic_version",
}
_INPUT_STUDY_ATTRIBUTES = {
    "optimizer_runtime_cache_identity", "optimizer_fixed_strategy_param_overrides",
    "optimizer_effective_policy_fingerprint_sha256", "optimizer_effective_policy_snapshot",
}


def _workspace(relative):
    relative = Path(relative)
    if (relative.is_absolute() or ".." in relative.parts
            or relative.parts[:3] != ("outputs", "optimizer", "strategy_param_schedule")
            or not _NAME.fullmatch(relative.name) or len(relative.parents) < 4
            or tuple(parent.name for parent in relative.parents[:3])
            != ("db", "outer_rolling_oos", "optimizer_runtime")):
        return None
    return relative.parents[3]


def _observe(path, revisions):
    if (not clean_path(path) or not plain(path.lstat()) or path.stat().st_nlink != 1):
        raise ValueError("Optimizer scratch evidence is not a plain single-link file")
    revisions[path] = file_change_signature(path)


def _database(path, revisions, *, result_free):
    _observe(path, revisions)
    for suffix in _SIDECARS:
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("Optimizer SQLite journal/WAL/SHM is present")
        revisions[sidecar] = ("missing",)
    # AI: immutable + read-only must never create a SQLite sidecar during preview.
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != _TABLES:
            raise ValueError("Unsupported Optuna storage schema")
        if db.execute("SELECT count(*) FROM studies").fetchone()[0] != 1:
            return None
        attrs = db.execute("SELECT key, value_json FROM study_user_attributes").fetchall()
        if len({key for key, _ in attrs}) != len(attrs):
            raise ValueError("Ambiguous Optuna study attributes")
        attrs = {key: json.loads(value) for key, value in attrs}
        identity = attrs.get("optimizer_runtime_cache_identity")
        if not is_sha256_hex(identity) or not path.name.startswith("outer_oos_runtime_" + identity[:20] + "_"):
            raise ValueError("Optuna database filename and producer identity disagree")
        states = dict(db.execute("SELECT state, count(*) FROM trials GROUP BY state"))
        if result_free:
            if set(states) != {"RUNNING"} or not attrs.keys() <= _INPUT_STUDY_ATTRIBUTES:
                return None
            for table in ("trial_values", "trial_intermediate_values", "trial_heartbeats", "trial_system_attributes"):
                if db.execute("SELECT count(*) FROM " + table).fetchone()[0]:
                    return None
            # Only the original pre-computation TP setting is admitted. Unknown
            # diagnostics, failure details and sampler checkpoints are evidence.
            for key, value in db.execute("SELECT key, value_json FROM trial_user_attributes"):
                value = json.loads(value)
                if (key != "fixed_tp_percent" or type(value) not in (int, float)
                        or not math.isfinite(value)):
                    return None
            if db.execute("SELECT count(*) FROM trials WHERE datetime_complete IS NOT NULL").fetchone()[0]:
                return None
            if db.execute("SELECT count(*) FROM study_system_attributes").fetchone()[0]:
                return None
        elif not states.get("COMPLETE"):
            return None
    if file_change_signature(path) != revisions[path]:
        raise ValueError("Optimizer study changed during inspection")
    return identity


def _record(root, relative):
    workspace = _workspace(relative)
    if workspace is None:
        return None
    path = root / relative
    revisions = {}
    old_identity = _database(path, revisions, result_free=True)
    if old_identity is None:
        return None
    preflight_path = root / workspace / "rolling_preflight.json"
    _observe(preflight_path, revisions)
    from services.optimizer.optimizer_workspace_recovery import verify_optimizer_runtime_preflight
    from services.optimizer.strategy_param_training import (
        SCHEMA_VERSION, _build_selection_full_roos_schedule_contract,
        _selection_full_roos_runtime_contract, _selection_full_roos_optimizer_runtime_cache_identity,
    )
    raw = json.loads(preflight_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Optimizer signed preflight must be an object")
    preflight = verify_optimizer_runtime_preflight(
        preflight_path, expected_runtime_identity_sha256=raw.get("runtime_identity_sha256", ""),
    )
    if (preflight.get("schema_version") != SCHEMA_VERSION
            or preflight.get("mode") != "selection_full_roos_parameter_training"
            or preflight.get("status") != "P4_HISTORY_PREFLIGHT_PASS"
            or preflight.get("training_dl_enabled") is not False):
        return None
    created = datetime.fromisoformat(preflight.get("created_at", ""))
    if (created.tzinfo is None or created.timestamp() <= path.stat().st_mtime
            or preflight_path.stat().st_mtime_ns <= path.stat().st_mtime_ns):
        return None
    contract = {key: value for key, value in preflight.items() if key not in {"created_at", "status"}}
    rolling = preflight["rolling_policy"]
    schedule = _build_selection_full_roos_schedule_contract(
        first_oos_date=rolling["first_oos_date"], last_oos_date=rolling["last_oos_date"],
        train_window_months=rolling["train_window_months"], oos_months=rolling["oos_horizon_months"],
    )
    args = SimpleNamespace(**{key: preflight[key] for key in (
        "dataset", "trials_per_fold", "fixed_risk", "max_position_cap_pct", "max_positions", "rotation",
    )}, seed_ensemble_policy_override=preflight.get("seed_ensemble_policy"),
        score_policy_overrides=preflight.get("score_policy_overrides"))
    if contract != _selection_full_roos_runtime_contract(
        root=root, args=args, schedule_contract=schedule,
        dataset_identity_override=preflight["dataset_identity"],
    ):
        return None
    # The same producer can address a direct study or a fold-local study when
    # extending a schedule. Reject both current forms; never guess a protocol.
    current = {
        _selection_full_roos_optimizer_runtime_cache_identity(contract, optimizer_search_protocol=None),
        _selection_full_roos_optimizer_runtime_cache_identity(contract, optimizer_search_protocol="fixed_multiseed_v1"),
    }
    if old_identity in current:
        return None
    replacements = [path.with_name(path.name.replace(old_identity[:20], identity[:20], 1))
                    for identity in current]
    replacements = [candidate for candidate in replacements if candidate.is_file()]
    if len(replacements) != 1:
        return None
    replacement = replacements[0]
    if _database(replacement, revisions, result_free=False) not in current:
        return None
    for evidence, revision in revisions.items():
        if file_change_signature(evidence) != revision:
            raise ValueError("Optimizer replacement proof changed during inspection")
    return replacement.relative_to(root), revisions


def _checked_record(root, relative):
    try:
        return _record(root, relative)
    except (sqlite3.Error, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError("Optimizer replacement evidence is malformed or unreadable") from exc


class OptimizerScratchCleanup:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records, self.errors, self.evidence = {}, {}, {}
        self.sources = {}  # These are result-free leftovers, never equivalent copies.
        self.leases = set()

    def prepare(self, inventory):
        self.records.clear()
        self.errors.clear()
        self.evidence.clear()
        for item in inventory:
            relative = Path(item)
            if _workspace(relative) is None:
                continue
            try:
                record = _checked_record(self.root, relative)
                if record is not None:
                    self.records[relative] = record
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                self.errors[relative] = str(exc)

    def candidate(self, relative):
        return relative in self.records

    @property
    def reference_targets(self):
        return {path for relative in self.records
                for path in (relative, *relative.parents)
                if path.is_relative_to(_workspace(relative))}

    @property
    def retained_roots(self):
        return {path for relative, (replacement, _) in self.records.items()
                for path in (replacement, _workspace(relative) / "rolling_preflight.json")}

    @property
    def proof_revisions(self):
        """Expose only the evidence observed at prepare; never renew a proof."""
        observed = {}
        for _, revisions in self.records.values():
            for path, revision in revisions.items():
                if path in observed and observed[path] != revision:
                    raise ValueError("Optimizer proof inputs changed between candidates")
                observed[path] = revision
        return observed

    @property
    def proof_scopes(self):
        return {_workspace(relative) for relative in self.records}

    @contextmanager
    def deletion_lease(self, worker, relative):
        locks = []
        try:
            if worker.apply:
                for path in sorted((self.root / relative, self.root / self.records[relative][0])):
                    lock_path = Path(str(path) + ".writer.lock")
                    if not clean_path(lock_path):
                        raise ValueError("Optimizer writer lock path is unsafe")
                    if lock_path.exists():
                        _observe(lock_path, {})
                    locks.append((_OptimizerStudyWriteLock(str(path)), lock_path))
            self.leases.add(relative)
            yield
        finally:
            self.leases.discard(relative)
            for lock, lock_path in reversed(locks):
                lock.release()
                worker.lifecycle.removed(lock_path.relative_to(self.root))

    def proof(self, worker, relative):
        self.evidence.pop(relative, None)
        if relative not in self.leases or relative not in self.records:
            raise ValueError("Missing Optimizer scratch lease")
        if worker.lifecycle.protected(relative):
            raise ValueError("Optimizer scratch is retained by an external reference")
        record = _checked_record(self.root, relative)
        if record is None or record != self.records[relative]:
            raise ValueError("Optimizer scratch or its replacement changed after inventory")
        self.evidence[relative] = record
        return "產生者已更換工作身分，舊 Optuna 僅有未完成抽樣、沒有試驗結果；保留現行工作庫：" + record[0].as_posix()

    def verify_before_delete(self, worker, relative):
        if relative not in self.leases or relative not in self.evidence:
            raise ValueError("Missing final Optimizer scratch deletion proof")
        if worker.lifecycle.protected(relative) or _checked_record(self.root, relative) != self.evidence[relative]:
            raise ValueError("Optimizer scratch, reference or replacement changed before deletion")

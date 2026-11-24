"""Proofs supplied by the producers of disposable files. All other paths stay intact."""
from contextlib import contextmanager
from importlib.util import MAGIC_NUMBER, source_from_cache
import os
from pathlib import Path
import re
import subprocess
import time

from core.compare_csv_storage import clean_path
from core.file_integrity import compute_file_sha256, exclusive_file_lock, file_change_signature, sibling_publication_lock_path
from core.regression_fixture_access import fixture_access

STAGING = Path("outputs/local_regression/_staging")
ROBUST = Path("outputs/strategy_compare/robustness")
DIAGNOSTICS = Path("outputs/filters/breakout_quality/real_path_diagnostic_cache")
BYTECODE_COLD_GRACE_NS = 24 * 60 * 60 * 1_000_000_000
PROBE = re.compile(r"quick_gate_(missing_run_best_params|broken_run_best_params|broken_optimizer_db|optimizer_study_db)_[a-z0-9_]{8}\Z")


def tracking_snapshot(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True,
                                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
        return result.stdout
    try:
        git_root = git("rev-parse", "--show-toplevel").rstrip(b"\r\n")
    except subprocess.CalledProcessError as exc:
        if (b"not a git repository" in (exc.stderr or b"")
                and not (root / ".git").exists() and not os.getenv("GIT_DIR")):
            return set(), root / ".git/index", ("missing",)
        raise
    if not root.is_relative_to(Path(os.fsdecode(git_root)).resolve()):
        raise ValueError("Git 索引與專案位置不符")
    index = Path(os.fsdecode(git("rev-parse", "--git-path", "index").rstrip(b"\r\n")))
    index = (root / index).absolute()
    revision = file_change_signature(index)
    tracked = {Path(os.fsdecode(p)) for p in git("ls-files", "--cached", "-z").split(b"\0") if p}
    if file_change_signature(index) != revision:
        raise RuntimeError("讀取期間 Git 索引變動")
    return tracked, index, revision


def generated_bytecode(root, rel, tracked):
    from core.compare_csv_storage import plain

    if (rel.is_absolute() or ".." in rel.parts or rel.parent.name != "__pycache__"
            or rel.suffix != ".pyc" or rel in tracked):
        return False
    try:
        source = Path(source_from_cache(str(rel)))
    except ValueError:
        return False
    # A local source is enough to rebuild its cache, whether or not Git tracks
    # that source. Both files must still be ordinary files inside this project.
    path = root / rel
    source = root / source
    if not clean_path(path) or not clean_path(source):
        return False
    try:
        if not plain(source.lstat()) or not plain(path.lstat()):
            return False
        with path.open("rb") as stream:
            header = stream.read(16)
    except OSError:
        return False
    return (len(header) == 16 and header[:4] == MAGIC_NUMBER
            and int.from_bytes(header[4:8], "little") in {0, 1, 3})


def bytecode_cleanup_ready(root, rel, tracked, *, now_ns=None):
    """Ownership plus a cold-cache veto; age alone never authorizes removal.

    NTFS ChangeTime detects recent rewrites even if mtime was restored. The
    cleaner calls this proof again for apply and checks the file revision at
    unlink, so a prior preview cannot authorize deleting a regenerated cache.
    """
    if not generated_bytecode(root, rel, tracked):
        return False
    try:
        stamp = file_change_signature(root / rel)
        if len(stamp) != 5:
            return False
        now = time.time_ns() if now_ns is None else now_ns
        modified, changed = stamp[3], stamp[4]
        return (type(now) is int and now > 0
                and all(type(value) is int and 0 < value <= now - BYTECODE_COLD_GRACE_NS
                        for value in (modified, changed)))
    except OSError:
        return False


def active_artifact_subjects(root):
    """Read the configured suites and their transitive canonical dependencies."""
    from config import breakout_quality, strategy_compare, trading
    from core.research_summary import get_research_summary_model_profiles
    from core.strategy_compare_policy import (get_strategy_comparison_menu_profiles,
                                             get_strategy_comparison_settings)
    from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_required_artifact_sources

    def strings(item):
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for value in item.values():
                yield from strings(value)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for value in item:
                yield from strings(value)

    live = {profile for _, profile in get_research_summary_model_profiles()}
    for module in (breakout_quality, strategy_compare, trading):
        for name, value in vars(module).items():
            if name.isupper():
                live.update(strings(value))
    # A catalog also contains retired/historical choices. Resolve only the
    # arms actually enabled by the configured suites and the preparation owner's
    # complete dependency graph, including indirect parameters and their teachers.
    for menu in get_strategy_comparison_menu_profiles():
        settings = get_strategy_comparison_settings(menu["profile_id"])
        parameters, models, _ = resolve_required_artifact_sources(settings)
        live.update(parameters)
        live.update(models)
        for arm in settings.enabled_arms:
            live.add(arm.arm_id)
            live.update(strings(arm.as_dict()))
        for source_id in parameters:
            live.update(strings(settings.parameter_sources[source_id].as_dict()))
        for dl_id in models:
            live.update(strings(settings.dl_sources[dl_id].as_dict()))
    return live


def disposable_cache(rel):
    """Cache ownership and proof come from producers, never the model roster."""
    if rel.suffix != ".npz":
        return None
    from services.breakout_quality.real_path_diagnostic_cache import real_path_diagnostic_cache_candidate
    from services import cleanup_outcome_cache
    if real_path_diagnostic_cache_candidate(rel):
        return "diagnostic_cache", DIAGNOSTICS
    if cleanup_outcome_cache.candidate(rel):
        return "outcome_cache", cleanup_outcome_cache.CACHE_ROOT
    return None


def disposable_cache_proof(root, kind, rel):
    if kind == "diagnostic_cache":
        from services.breakout_quality.real_path_diagnostic_cache import real_path_diagnostic_cache_proof
        return real_path_diagnostic_cache_proof(root, rel)
    if kind == "outcome_cache":
        from services.cleanup_outcome_cache import proof
        return proof(root, rel)
    return ""


def legacy_parameters(root):
    from services.optimizer.strategy_param_service import collect_legacy_root_strategy_parameter_cleanup_plan
    plan = collect_legacy_root_strategy_parameter_cleanup_plan(root)
    if plan["status"] != "READY":
        return {}
    # Migration proves a replacement exists, not that every current consumer
    # has switched to it. Resolve active overrides/templates through their owners.
    from core import model_paths
    from core.strategy_compare_policy import get_strategy_comparison_menu_profiles, get_strategy_comparison_settings
    from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_required_artifact_sources
    from filters.breakout_quality.strategy_compare_sources import resolve_param_source_path
    from services.research.strategy_compare_preparation_status import resolve_required_param_policies_for_source
    live = {Path(resolver(str(root))).resolve() for resolver in (
        model_paths.resolve_run_best_params_path, model_paths.resolve_candidate_best_params_path,
        model_paths.resolve_candidate_retention_best_params_path, model_paths.resolve_candidate_val_score_best_params_path,
    )}
    for menu in get_strategy_comparison_menu_profiles():
        settings = get_strategy_comparison_settings(menu["profile_id"])
        parameters, _, _ = resolve_required_artifact_sources(settings)
        for source_id in parameters:
            for policy in resolve_required_param_policies_for_source(settings, source_id):
                live.add(resolve_param_source_path(root, settings, source_id, param_policy=policy).resolve())
    return {Path(r["source"]): r["evidence"] for r in plan["removable"]
            if Path(r["source"]).parts[:1] == ("models",) and (root / r["source"]).resolve() not in live}


def fixture_reason(root, rel):
    parts = rel.parts
    if len(parts) <= len(STAGING.parts) or not rel.is_relative_to(STAGING):
        return ""
    local = rel.relative_to(STAGING)
    match = PROBE.fullmatch(local.parts[0])
    if not match:
        return ""
    remainder = Path(*local.parts[1:]).as_posix()
    path = root / rel
    if match[1] == "broken_run_best_params" and remainder == "models/run_best_params.json":
        if path.stat().st_size == 15 and path.read_bytes() == b"{not-valid-json":
            return "已結束的故意損壞參數測試"
    if match[1] == "optimizer_study_db":
        if remainder == "broken_optimizer_db/ml_optimizer/study_db/optimizer_study_reduced_full.db":
            if path.stat().st_size == 15 and path.read_bytes() == b"not-a-sqlite-db":
                return "已結束的故意損壞資料庫測試"
        if remainder == "export_only_empty_db/ml_optimizer/study_db/optimizer_study_reduced_full.db" and not path.stat().st_size:
            return "已結束的空資料庫測試"
    if match[1] == "broken_optimizer_db" and remainder == "models/run_best_params.json":
        from core.model_paths import resolve_default_primary_param_source_path
        canonical = Path(resolve_default_primary_param_source_path(root))
        if canonical.is_file() and clean_path(canonical) and compute_file_sha256(path) == compute_file_sha256(canonical):
            return "已結束的測試用參數副本"
    return ""


def robust_unit(rel):
    # Registered historical suites have different directory depths. This only
    # identifies a candidate; the producer validates the completed run under lease.
    if rel.is_relative_to(ROBUST):
        p = rel.relative_to(ROBUST).parts
        if "work" in p:
            index = p.index("work")
            if index:
                return ROBUST.joinpath(*p[:index])
    return None


def robust_disposable(run):
    from services.research.strategy_multi_seed_robustness import completed_robustness_cleanup_scopes
    return completed_robustness_cleanup_scopes(run)


@contextmanager
def deletion_lease(root, key, *, preview=False):
    from services import cleanup_outcome_cache
    if key == DIAGNOSTICS:
        from services.breakout_quality.real_path_diagnostic_cache import real_path_diagnostic_cache_access
        with real_path_diagnostic_cache_access(root, preview=preview):
            yield
    elif key == cleanup_outcome_cache.CACHE_ROOT:
        with cleanup_outcome_cache.deletion_lease(root):
            yield
    elif key == STAGING:
        with fixture_access(root / STAGING, cleanup=True):
            yield
    elif key and key.is_relative_to(ROBUST):
        lock = sibling_publication_lock_path(root / key)
        if not clean_path(lock) or (lock.exists() and lock.stat().st_nlink != 1):
            raise ValueError("Robustness 鎖路徑異常")
        # An estimate must not create the writer's lock file in a research run.
        # Apply always takes the real lease and validates the publication again.
        if preview and (not lock.exists() or not lock.stat().st_size):
            yield
            return
        with exclusive_file_lock(lock, conflict_message="Robustness 正在執行"):
            yield
    else:
        yield


def abandoned_pid(path, rel):
    from services.breakout_quality.formal_mintrade_outcome_cache import _CACHE_SUBDIR as outcomes
    if rel.parent.parent not in (outcomes, DIAGNOSTICS):
        return False
    match = re.fullmatch(r"\.([0-9a-f]{64})\.([1-9][0-9]{0,9})\.[0-9a-f]{32}\.npz", path.name)
    if not match or match[1][:2] != rel.parent.name:
        return False
    pid = int(match[2])
    from core.artifact_lifecycle import process_is_absent
    return process_is_absent(pid)

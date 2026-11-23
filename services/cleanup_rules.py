"""Proofs supplied by the producers of disposable files. All other paths stay intact."""
from contextlib import contextmanager
from importlib.util import MAGIC_NUMBER, source_from_cache
import os
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile, BadZipFile

from core.compare_csv_storage import clean_path
from core.file_integrity import compute_file_sha256, exclusive_file_lock, file_change_signature, sibling_publication_lock_path
from core.regression_fixture_access import fixture_access

STAGING = Path("outputs/local_regression/_staging")
ROBUST = Path("outputs/strategy_compare/robustness")
DIAGNOSTICS = Path("outputs/filters/breakout_quality/real_path_diagnostic_cache")
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
    if rel.parent.name != "__pycache__" or rel.suffix != ".pyc":
        return False
    try:
        source = Path(source_from_cache(str(rel)))
    except ValueError:
        return False
    if ((tracked and source not in tracked) or rel in tracked or not clean_path(root / source)
            or not (root / source).is_file()):
        return False
    with (root / rel).open("rb") as stream:
        header = stream.read(16)
    return (len(header) == 16 and header[:4] == MAGIC_NUMBER
            and int.from_bytes(header[4:8], "little") in {0, 1, 3})


def retired_diagnostics(root):
    """Map only recognized historical model+seed slots of this disposable projection."""
    from config import breakout_quality, strategy_compare, trading
    from core.research_summary import get_research_summary_model_profiles
    from core.strategy_compare_policy import (_compatibility_catalogs, get_strategy_comparison_menu_profiles,
                                             get_strategy_comparison_settings)
    from core.training_policy import resolve_robustness_benchmark_seeds
    from services.breakout_quality.real_path_diagnostic_cache import _slot

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
    # arms actually enabled by the configured suites, plus their parameter teachers.
    for menu in get_strategy_comparison_menu_profiles():
        settings = get_strategy_comparison_settings(menu["profile_id"])
        for arm in settings.enabled_arms:
            names = set(strings(arm.as_dict()))
            names.update(strings(settings.parameter_sources[arm.param_source].as_dict()))
            live.update(names)
            for dl_id in names.intersection(settings.dl_sources):
                live.update(strings(settings.dl_sources[dl_id].as_dict()))
    seeds = {*resolve_robustness_benchmark_seeds(), breakout_quality.BREAKOUT_QUALITY_RANDOM_SEED}
    historical = _compatibility_catalogs()[1]
    slots = {}
    for data in historical.values():
        profile = data.get("experiment_profile")
        architecture = data.get("model_architecture")
        filter_id = data.get("filter_id")
        if profile and architecture and filter_id and profile not in live:
            for seed in seeds:
                slot = _slot(root, filter_id, architecture, profile, seed)
                slots[slot.relative_to(root)] = profile
    return slots


def is_projection(path):
    # This producer stores a disposable projection only; its FIT/PIT inputs live elsewhere.
    expected = {"identity", "ticker", "dates", "labels", "raw_r", "valid", "payload_sha256"}
    try:
        from numpy.lib.format import read_array
        with ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != len(expected) or set(members) != {name + ".npy" for name in expected}:
                return False
            # Both fingerprints are short scalar strings emitted by the producer.
            for name in ("identity.npy", "payload_sha256.npy"):
                if archive.getinfo(name).file_size > 1024:
                    return False
                with archive.open(name) as entry:
                    token = read_array(entry, allow_pickle=False)
                if token.shape != () or not re.fullmatch("[0-9a-f]{64}", str(token.item())):
                    return False
        return True
    except (BadZipFile, ValueError, EOFError):
        return False


def legacy_parameters(root):
    from services.optimizer.strategy_param_service import collect_legacy_root_strategy_parameter_cleanup_plan
    plan = collect_legacy_root_strategy_parameter_cleanup_plan(root)
    if plan["status"] != "READY":
        return {}
    # Migration proves a replacement exists, not that every current consumer
    # has switched to it. Resolve active overrides/templates through their owners.
    from core import model_paths
    from core.strategy_compare_policy import get_strategy_comparison_menu_profiles, get_strategy_comparison_settings
    from core.strategy_comparison import resolve_strategy_comparison_arm_param_policy
    from filters.breakout_quality.strategy_compare_sources import resolve_param_source_path
    live = {Path(resolver(str(root))).resolve() for resolver in (
        model_paths.resolve_run_best_params_path, model_paths.resolve_candidate_best_params_path,
        model_paths.resolve_candidate_retention_best_params_path, model_paths.resolve_candidate_val_score_best_params_path,
    )}
    for menu in get_strategy_comparison_menu_profiles():
        settings = get_strategy_comparison_settings(menu["profile_id"])
        for arm in settings.enabled_arms:
            live.add(resolve_param_source_path(root, settings, arm.param_source,
                     param_policy=resolve_strategy_comparison_arm_param_policy(settings, arm)).resolve())
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
    if rel.is_relative_to(ROBUST):
        p = rel.relative_to(ROBUST).parts
        if len(p) >= 6 and p[3] == "work":
            return ROBUST.joinpath(*p[:3])
    return None


def robust_disposable(run):
    from services.research.strategy_multi_seed_robustness import completed_robustness_cleanup_scopes
    return completed_robustness_cleanup_scopes(run)


@contextmanager
def deletion_lease(root, key, *, preview=False):
    if key == STAGING:
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
    if pid > 0x7fffffff:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        lib = ctypes.WinDLL("kernel32", use_last_error=True)
        lib.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        lib.OpenProcess.restype = wintypes.HANDLE
        lib.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = lib.OpenProcess(0x1000, False, pid)
        if handle:
            lib.CloseHandle(handle)
            return False
        return ctypes.get_last_error() == 87  # ERROR_INVALID_PARAMETER for an absent PID
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False

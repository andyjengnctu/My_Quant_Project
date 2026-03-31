from __future__ import annotations

import ast
import os
import py_compile
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import parse_no_arg_cli, run_cli_entrypoint
from core.config import V16StrategyParams
from core.dataset_profiles import DATASET_PROFILE_SPECS, DEFAULT_VALIDATE_DATASET_PROFILE, normalize_dataset_profile_key
from core.output_paths import build_output_dir
from core.log_utils import append_issue_log, build_timestamped_log_path, resolve_log_dir
from tools.local_regression.common import MANIFEST_DEFAULTS, build_bundle_zip, ensure_reduced_dataset, load_manifest, resolve_run_dir, run_command, summarize_result, write_json, write_text

PYTHON_FILES_EXCLUDE_PARTS = {".git", "__pycache__", "outputs", ".venv", "venv"}
HELP_TARGETS = [
    ([sys.executable, "apps/ml_optimizer.py", "--help"], "python apps/ml_optimizer.py"),
    ([sys.executable, "apps/portfolio_sim.py", "--help"], "python apps/portfolio_sim.py"),
    ([sys.executable, "apps/smart_downloader.py", "--help"], "python apps/smart_downloader.py"),
    ([sys.executable, "apps/test_suite.py", "--help"], "python apps/test_suite.py"),
    ([sys.executable, "apps/vip_scanner.py", "--help"], "python apps/vip_scanner.py"),
    ([sys.executable, "requirements/export_requirements_lock.py", "--help"], "python requirements/export_requirements_lock.py"),
    ([sys.executable, "tools/debug/trade_log.py", "--help"], "python tools/debug/trade_log.py"),
    ([sys.executable, "tools/downloader/main.py", "--help"], "python tools/downloader/main.py"),
    ([sys.executable, "tools/local_regression/run_all.py", "--help"], "python tools/local_regression/run_all.py"),
    ([sys.executable, "tools/local_regression/run_chain_checks.py", "--help"], "python tools/local_regression/run_chain_checks.py"),
    ([sys.executable, "tools/local_regression/run_ml_smoke.py", "--help"], "python tools/local_regression/run_ml_smoke.py"),
    ([sys.executable, "tools/local_regression/run_quick_gate.py", "--help"], "python tools/local_regression/run_quick_gate.py"),
    ([sys.executable, "tools/optimizer/main.py", "--help"], "python tools/optimizer/main.py"),
    ([sys.executable, "tools/portfolio_sim/main.py", "--help"], "python tools/portfolio_sim/main.py"),
    ([sys.executable, "tools/scanner/main.py", "--help"], "python tools/scanner/main.py"),
    ([sys.executable, "tools/validate/cli.py", "--help"], "python tools/validate/cli.py"),
    ([sys.executable, "tools/validate/main.py", "--help"], "python tools/validate/main.py"),
    ([sys.executable, "tools/validate/preflight_env.py", "--help"], "python tools/validate/preflight_env.py"),
]
NO_ARG_CLI_TARGETS = [
    "requirements/export_requirements_lock.py",
    "tools/local_regression/run_chain_checks.py",
    "tools/local_regression/run_ml_smoke.py",
    "tools/local_regression/run_quick_gate.py",
]
RUN_ALL_CLI_CASES = [
    (["--only"], "--only 缺少值"),
    (["--only="], "--only 不可為空"),
    (["--only", "bad"], "--only 只接受"),
    (["--bad"], "不支援的參數"),
]


def _outcome_detail(outcome: Dict[str, Any], fallback_text: str) -> str:
    if outcome.get("timed_out"):
        return f"timeout after {outcome.get('timeout_sec', 0)}s"
    combined = "\n".join(part for part in [outcome.get("stdout", "").strip(), outcome.get("stderr", "").strip()] if part).strip()
    if combined:
        return combined.splitlines()[0]
    return fallback_text


def iter_python_files() -> List[Path]:
    files: List[Path] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in PYTHON_FILES_EXCLUDE_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def run_static_checks() -> List[Dict[str, Any]]:
    py_files = iter_python_files()
    results: List[Dict[str, Any]] = []

    py_compile_errors = []
    for path in py_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            py_compile_errors.append(f"{path.relative_to(PROJECT_ROOT)}:{exc.lineno}: {exc.msg}")
    results.append(summarize_result("py_compile", not py_compile_errors, detail=f"檢查 {len(py_files)} 個 Python 檔案", extra={"errors": py_compile_errors}))

    pyc_compile_errors = []
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            pyc_compile_errors.append(f"{path.relative_to(PROJECT_ROOT)}: {exc.msg}")
    results.append(summarize_result("compileall", not pyc_compile_errors, detail=f"compileall 等價檢查 {len(py_files)} 個 Python 檔案", extra={"errors": pyc_compile_errors}))

    bare_except_hits = []
    bare_except_scan_errors = []
    for path in py_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            bare_except_scan_errors.append(f"{path.relative_to(PROJECT_ROOT)}:{exc.lineno}: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bare_except_hits.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    results.append(
        summarize_result(
            "bare_except_scan",
            (not bare_except_hits) and (not bare_except_scan_errors),
            detail=f"裸 except 命中 {len(bare_except_hits)} 筆；AST 解析失敗 {len(bare_except_scan_errors)} 筆",
            extra={"hits": bare_except_hits, "ast_parse_errors": bare_except_scan_errors},
        )
    )
    return results


def check_output_path_contract() -> List[Dict[str, Any]]:
    results = []
    invalid_cases = [
        ("output_path_contract::empty_category", "", "category 必填"),
        ("output_path_contract::nested_category", "local_regression/archive", "單一工具分類資料夾名稱"),
        ("output_path_contract::backslash_nested_category", r"local_regression\archive", "單一工具分類資料夾名稱"),
        ("output_path_contract::dot_category", ".", "單一工具分類資料夾名稱"),
        ("output_path_contract::parent_category", "..", "單一工具分類資料夾名稱"),
        ("output_path_contract::absolute_category", str((PROJECT_ROOT / "outputs").resolve()), "不可為絕對路徑"),
    ]

    for name, category, expected_text in invalid_cases:
        try:
            build_output_dir(PROJECT_ROOT, category)
            ok = False
            detail = "應拒絕不合法 category，但函式未拋出例外"
        except ValueError as exc:
            ok = expected_text in str(exc)
            detail = str(exc)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result(name, ok, detail=detail))

    valid_path = str((PROJECT_ROOT / "outputs" / "local_regression").resolve())
    try:
        resolved = str(Path(build_output_dir(PROJECT_ROOT, "local_regression")).resolve())
        ok = resolved == valid_path
        detail = resolved
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    results.append(summarize_result("output_path_contract::valid_category_resolves_under_outputs", ok, detail=detail))

    return results


def check_outputs_root_layout() -> List[Dict[str, Any]]:
    results = []
    outputs_root = PROJECT_ROOT / "outputs"
    stray_entries = []

    if outputs_root.exists():
        for child in sorted(outputs_root.iterdir()):
            if child.is_symlink():
                stray_entries.append(f"symlink:{child.name}")
                continue
            if child.is_dir():
                continue
            stray_entries.append(f"file:{child.name}")

    detail = "outputs/ 根目錄無散落檔案或 symlink" if not stray_entries else ", ".join(stray_entries)
    results.append(
        summarize_result(
            "outputs_root_layout::root_has_only_category_dirs",
            not stray_entries,
            detail=detail,
            extra={"stray_entries": stray_entries},
        )
    )
    return results


def check_dataset_profile_contract() -> List[Dict[str, Any]]:
    results = []

    supported_profiles = sorted(DATASET_PROFILE_SPECS.keys())
    results.append(
        summarize_result(
            "dataset_profile_contract::supported_profiles",
            supported_profiles == ["full", "reduced"],
            detail=f"supported={supported_profiles}",
            extra={"supported_profiles": supported_profiles},
        )
    )

    results.append(
        summarize_result(
            "dataset_profile_contract::validate_default_is_reduced",
            DEFAULT_VALIDATE_DATASET_PROFILE == "reduced",
            detail=f"DEFAULT_VALIDATE_DATASET_PROFILE={DEFAULT_VALIDATE_DATASET_PROFILE}",
        )
    )

    try:
        normalize_dataset_profile_key("raw")
        ok = False
        detail = 'normalize_dataset_profile_key("raw") 不應通過'
    except ValueError as exc:
        ok = "不支援的資料集模式" in str(exc)
        detail = str(exc)
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    results.append(summarize_result("dataset_profile_contract::raw_profile_rejected", ok, detail=detail))

    return results


def check_log_path_contract() -> List[Dict[str, Any]]:
    results = []
    invalid_dir_cases = [
        ("log_path_contract::resolve_log_dir_parent_escape_rejected", "../outputs/debug_trade_log", "不可包含 . 或 .."),
        ("log_path_contract::resolve_log_dir_absolute_outside_project_rejected", "/tmp/outside_logs", "必須落在專案目錄內"),
    ]

    for name, log_dir, expected_text in invalid_dir_cases:
        try:
            resolve_log_dir(log_dir)
            ok = False
            detail = "應拒絕不合法 log_dir，但函式未拋出例外"
        except ValueError as exc:
            ok = expected_text in str(exc)
            detail = str(exc)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result(name, ok, detail=detail))

    invalid_path_cases = [
        ("log_path_contract::append_issue_log_parent_escape_rejected", "../outside.log", "不可包含 . 或 .."),
        ("log_path_contract::append_issue_log_absolute_outside_project_rejected", "/tmp/outside.log", "必須落在專案目錄內"),
        ("log_path_contract::append_issue_log_root_file_rejected", "outside.log", "必須包含目錄"),
        ("log_path_contract::append_issue_log_outputs_root_file_rejected", "outputs/outside.log", "不可直接輸出到 outputs/ 根目錄"),
    ]

    for name, log_path, expected_text in invalid_path_cases:
        try:
            append_issue_log(log_path, ["probe"])
            ok = False
            detail = "應拒絕不合法 log_path，但函式未拋出例外"
        except ValueError as exc:
            ok = expected_text in str(exc)
            detail = str(exc)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result(name, ok, detail=detail))

    invalid_prefix_cases = [
        ("log_path_contract::build_timestamped_log_path_parent_escape_prefix_rejected", "../outside", "不可包含路徑分隔或 . / .."),
        ("log_path_contract::build_timestamped_log_path_nested_prefix_rejected", "nested/prefix", "不可包含路徑分隔或 . / .."),
        ("log_path_contract::build_timestamped_log_path_backslash_prefix_rejected", r"nested\prefix", "不可包含路徑分隔或 . / .."),
    ]

    for name, prefix, expected_text in invalid_prefix_cases:
        try:
            build_timestamped_log_path(prefix, log_dir=str(PROJECT_ROOT / "outputs" / "smart_downloader"), timestamp="20260331_000000")
            ok = False
            detail = "應拒絕不合法 prefix，但函式未拋出例外"
        except ValueError as exc:
            ok = expected_text in str(exc)
            detail = str(exc)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result(name, ok, detail=detail))

    return results


def check_local_regression_contract() -> List[Dict[str, Any]]:
    results = []
    invalid_bundle_cases = [
        ("local_regression_contract::manifest_bundle_name_parent_escape_rejected", "../escape.zip", "不可包含路徑分隔"),
        ("local_regression_contract::manifest_bundle_name_nested_rejected", "nested/escape.zip", "不可包含路徑分隔"),
        ("local_regression_contract::manifest_bundle_name_backslash_nested_rejected", r"nested\escape.zip", "不可包含路徑分隔"),
        ("local_regression_contract::manifest_bundle_name_requires_zip_suffix", "escape_bundle", ".zip 副檔名"),
    ]

    temp_root = Path(tempfile.mkdtemp(prefix="quick_gate_manifest_contract_"))
    try:
        for name, bundle_name, expected_text in invalid_bundle_cases:
            manifest_path = temp_root / f"{name.replace('::', '_')}.json"
            write_json(manifest_path, {**MANIFEST_DEFAULTS, "bundle_name": bundle_name})
            try:
                load_manifest(manifest_path)
                ok = False
                detail = "應拒絕不合法 bundle_name，但 manifest 驗證未拋出例外"
            except Exception as exc:
                ok = expected_text in str(exc)
                detail = f"{type(exc).__name__}: {exc}"
            results.append(summarize_result(name, ok, detail=detail))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    bundle_run_dir = Path(tempfile.mkdtemp(prefix="quick_gate_bundle_zip_"))
    try:
        probe_file = bundle_run_dir / "probe.txt"
        probe_file.write_text("probe", encoding="utf-8")
        try:
            build_bundle_zip(bundle_run_dir, "../escape.zip", include_paths=[probe_file])
            ok = False
            detail = "應拒絕不合法 bundle_name，但 build_bundle_zip 未拋出例外"
        except Exception as exc:
            ok = "不可包含路徑分隔" in str(exc)
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result("local_regression_contract::build_bundle_zip_parent_escape_rejected", ok, detail=detail))

        try:
            bundle_path = build_bundle_zip(bundle_run_dir, "safe_bundle.zip", include_paths=[probe_file])
            ok = bundle_path.parent.resolve() == bundle_run_dir.resolve() and bundle_path.name == "safe_bundle.zip" and bundle_path.exists()
            detail = str(bundle_path)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result("local_regression_contract::build_bundle_zip_valid_name_stays_in_run_dir", ok, detail=detail))
    finally:
        shutil.rmtree(bundle_run_dir, ignore_errors=True)

    env_var = "V16_LOCAL_REGRESSION_RUN_DIR"
    original_env = os.environ.get(env_var)
    env_cases = [
        ("local_regression_contract::run_dir_env_outside_project_rejected", "/tmp/outside_local_regression", "必須落在"),
        ("local_regression_contract::run_dir_env_parent_escape_rejected", "../outside_local_regression", "不可包含 . 或 .."),
    ]
    try:
        for name, env_value, expected_text in env_cases:
            os.environ[env_var] = env_value
            try:
                resolve_run_dir("quick_gate")
                ok = False
                detail = "應拒絕不合法 V16_LOCAL_REGRESSION_RUN_DIR，但函式未拋出例外"
            except Exception as exc:
                ok = expected_text in str(exc)
                detail = f"{type(exc).__name__}: {exc}"
            results.append(summarize_result(name, ok, detail=detail))

        valid_run_dir = PROJECT_ROOT / "outputs" / "local_regression" / "_staging" / "quick_gate_env_probe"
        shutil.rmtree(valid_run_dir, ignore_errors=True)
        os.environ[env_var] = str(valid_run_dir)
        try:
            resolved = resolve_run_dir("quick_gate")
            ok = Path(resolved).resolve() == valid_run_dir.resolve() and valid_run_dir.exists()
            detail = str(resolved)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result("local_regression_contract::run_dir_env_valid_under_staging", ok, detail=detail))
    finally:
        if original_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = original_env
        shutil.rmtree(PROJECT_ROOT / "outputs" / "local_regression" / "_staging" / "quick_gate_env_probe", ignore_errors=True)

    return results


def check_help(timeout: int) -> List[Dict[str, Any]]:
    results = []
    for args, expected_usage in HELP_TARGETS:
        outcome = run_command(args, timeout=timeout)
        first_line = outcome["stdout"].splitlines()[0] if outcome["stdout"].strip() else _outcome_detail(outcome, expected_usage)
        ok = (
            (not outcome.get("timed_out"))
            and outcome["returncode"] == 0
            and "用法:" in outcome["stdout"]
            and expected_usage in outcome["stdout"]
        )
        results.append(summarize_result(f"help::{Path(args[1]).name}", ok, detail=first_line, extra={"expected_usage": expected_usage}))
    return results


def check_dataset_cli_errors(timeout: int) -> List[Dict[str, Any]]:
    results = []
    targets = [
        "tools/validate/cli.py",
        "apps/portfolio_sim.py",
        "apps/vip_scanner.py",
        "apps/ml_optimizer.py",
        "tools/debug/trade_log.py",
    ]
    cases = [
        (["--dataset", "bad"], "不支援的資料集模式"),
        (["--dataset"], "缺少值"),
        (["--dataset="], "不能為空"),
    ]
    for target in targets:
        for suffix_args, expected in cases:
            outcome = run_command([sys.executable, target, *suffix_args], timeout=timeout)
            combined = f"{outcome['stdout']}\n{outcome['stderr']}"
            ok = (not outcome.get("timed_out")) and outcome["returncode"] != 0 and expected in combined
            results.append(summarize_result(f"dataset_cli::{Path(target).name}::{' '.join(suffix_args)}", ok, detail=_outcome_detail(outcome, expected)))
    return results


def check_generic_cli_errors(timeout: int) -> List[Dict[str, Any]]:
    results = []
    no_arg_cases = [
        (["--bad"], "不支援的參數"),
        (["bad"], "不支援的位置參數"),
    ]
    for target in NO_ARG_CLI_TARGETS:
        for suffix_args, expected in no_arg_cases:
            outcome = run_command([sys.executable, target, *suffix_args], timeout=timeout)
            combined = f"{outcome['stdout']}\n{outcome['stderr']}"
            ok = (not outcome.get("timed_out")) and outcome["returncode"] != 0 and expected in combined
            results.append(summarize_result(f"generic_cli::{Path(target).name}::{' '.join(suffix_args)}", ok, detail=_outcome_detail(outcome, expected)))

    for suffix_args, expected in RUN_ALL_CLI_CASES:
        outcome = run_command([sys.executable, "tools/local_regression/run_all.py", *suffix_args], timeout=timeout)
        combined = f"{outcome['stdout']}\n{outcome['stderr']}"
        ok = (not outcome.get("timed_out")) and outcome["returncode"] != 0 and expected in combined
        results.append(summarize_result(f"generic_cli::run_all.py::{' '.join(suffix_args)}", ok, detail=_outcome_detail(outcome, expected)))
    return results


@contextmanager
def temporary_missing_file(target_path: Path):
    backup_path = target_path.with_suffix(target_path.suffix + ".bak_local_regression")
    if backup_path.exists():
        raise RuntimeError(f"暫存備份已存在，疑似上次中斷殘留: {backup_path}")
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    shutil.move(target_path, backup_path)
    try:
        yield
    finally:
        if backup_path.exists():
            shutil.move(backup_path, target_path)


@contextmanager
def temporary_file_content(target_path: Path, replacement_text: str):
    backup_path = target_path.with_suffix(target_path.suffix + ".bak_local_regression")
    if backup_path.exists():
        raise RuntimeError(f"暫存備份已存在，疑似上次中斷殘留: {backup_path}")
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    shutil.move(target_path, backup_path)
    try:
        target_path.write_text(replacement_text, encoding="utf-8")
        yield
    finally:
        if target_path.exists():
            target_path.unlink()
        if backup_path.exists():
            shutil.move(backup_path, target_path)


def check_error_paths(timeout: int) -> List[Dict[str, Any]]:
    results = []
    params_path = PROJECT_ROOT / "models" / "best_params.json"
    db_path = PROJECT_ROOT / "models" / "portfolio_ai_10pos_overnight_reduced.db"
    db_backup = db_path.with_suffix(db_path.suffix + ".bak_local_regression")

    try:
        with temporary_missing_file(params_path):
            outcome = run_command([sys.executable, "apps/portfolio_sim.py", "--dataset", "reduced"], input_text="\n\n\n\n", timeout=timeout)
    except (FileNotFoundError, RuntimeError) as exc:
        results.append(summarize_result("error_path::missing_best_params", False, detail=f"前置條件不足，無法模擬缺參數檔: {type(exc).__name__}: {exc}"))
    else:
        results.append(summarize_result("error_path::missing_best_params", outcome["returncode"] != 0 and "找不到參數檔" in f"{outcome['stdout']}\n{outcome['stderr']}", detail="缺參數檔應 fail-fast"))

    try:
        with temporary_file_content(params_path, "{not-valid-json"):
            outcome = run_command([sys.executable, "apps/portfolio_sim.py", "--dataset", "reduced"], input_text="\n\n\n\n", timeout=timeout)
    except (FileNotFoundError, RuntimeError) as exc:
        results.append(summarize_result("error_path::broken_best_params", False, detail=f"前置條件不足，無法模擬壞參數檔: {type(exc).__name__}: {exc}"))
    else:
        results.append(summarize_result("error_path::broken_best_params", outcome["returncode"] != 0 and "JSONDecodeError" in f"{outcome['stdout']}\n{outcome['stderr']}", detail="壞參數檔應 fail-fast"))

    original_db_exists = db_path.exists()
    if db_backup.exists():
        results.append(summarize_result("error_path::optimizer_db_backup_slot", False, detail=f"暫存備份已存在，疑似上次中斷殘留: {db_backup}"))
        return results
    if original_db_exists:
        shutil.move(db_path, db_backup)
    try:
        db_path.write_text("not-a-sqlite-db", encoding="utf-8")
        outcome = run_command([sys.executable, "apps/ml_optimizer.py", "--dataset", "reduced"], timeout=timeout, env={"V16_OPTIMIZER_TRIALS": "0"})
        results.append(summarize_result("error_path::broken_optimizer_db", (not outcome.get("timed_out")) and outcome["returncode"] != 0 and "Optimizer 記憶庫檔案損壞或不可讀" in f"{outcome['stdout']}\n{outcome['stderr']}", detail=_outcome_detail(outcome, "壞 DB 應 fail-fast")))

        db_path.unlink(missing_ok=True)
        outcome = run_command([sys.executable, "apps/ml_optimizer.py", "--dataset", "reduced"], timeout=timeout, env={"V16_OPTIMIZER_TRIALS": "0"})
        results.append(summarize_result("error_path::export_only_missing_db", (not outcome.get("timed_out")) and outcome["returncode"] != 0 and "記憶庫不存在，無法匯出" in f"{outcome['stdout']}\n{outcome['stderr']}", detail=_outcome_detail(outcome, "無 DB export-only 應 fail-fast")))

        db_path.unlink(missing_ok=True)
        import sqlite3
        sqlite3.connect(db_path).close()
        outcome = run_command([sys.executable, "apps/ml_optimizer.py", "--dataset", "reduced"], timeout=timeout, env={"V16_OPTIMIZER_TRIALS": "0"})
        results.append(summarize_result("error_path::export_only_empty_db", (not outcome.get("timed_out")) and outcome["returncode"] != 0 and "記憶庫為空，無法匯出" in f"{outcome['stdout']}\n{outcome['stderr']}", detail=_outcome_detail(outcome, "空 DB export-only 應 fail-fast")))
    finally:
        db_path.unlink(missing_ok=True)
        if original_db_exists and db_backup.exists():
            shutil.move(db_backup, db_path)
    return results


def check_dataset_runtime_error_paths() -> List[Dict[str, Any]]:
    results = []
    params = V16StrategyParams()

    from tools.optimizer.raw_cache import load_all_raw_data
    from tools.portfolio_sim.simulation_runner import run_portfolio_simulation
    from tools.scanner.scan_runner import run_daily_scanner

    runtime_cases = [
        (
            "runtime_error_path::portfolio_sim_missing_data_dir",
            lambda: run_portfolio_simulation("", params, verbose=False),
        ),
        (
            "runtime_error_path::scanner_missing_data_dir",
            lambda: run_daily_scanner("", params),
        ),
        (
            "runtime_error_path::optimizer_missing_data_dir",
            lambda: load_all_raw_data("", required_min_rows=50, output_dir=str(PROJECT_ROOT / "outputs" / "ml_optimizer")),
        ),
    ]

    for name, func in runtime_cases:
        try:
            func()
            ok = False
            detail = "應 fail-fast，但函式未拋出例外"
        except FileNotFoundError as exc:
            ok = "找不到資料夾" in str(exc)
            detail = str(exc)
        except Exception as exc:  # 非裸 except，明確保留型別，避免把錯誤路徑測試誤判為通過
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append(summarize_result(name, ok, detail=detail))

    return results


def main(argv=None) -> int:
    parsed = parse_no_arg_cli(argv, "tools/local_regression/run_quick_gate.py", description="執行 reduced quick gate 靜態與錯誤路徑檢查；不接受額外參數。")
    if parsed["help"]:
        return 0

    manifest = load_manifest()
    run_dir = resolve_run_dir("quick_gate")
    timeout = int(manifest["subprocess_timeout_sec"])
    dataset_info = ensure_reduced_dataset()

    steps: List[Dict[str, Any]] = []
    steps.extend(run_static_checks())
    steps.extend(check_output_path_contract())
    steps.extend(check_outputs_root_layout())
    steps.extend(check_dataset_profile_contract())
    steps.extend(check_log_path_contract())
    steps.extend(check_local_regression_contract())
    steps.extend(check_help(timeout))
    steps.extend(check_dataset_cli_errors(timeout))
    steps.extend(check_generic_cli_errors(timeout))
    steps.extend(check_error_paths(timeout))
    steps.extend(check_dataset_runtime_error_paths())

    failed = [step for step in steps if step["status"] != "PASS"]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "dataset": manifest["dataset"],
        "dataset_info": dataset_info,
        "step_count": len(steps),
        "failed_count": len(failed),
        "failed_steps": [step["name"] for step in failed],
        "steps": steps,
    }
    write_json(run_dir / "quick_gate_summary.json", summary)
    write_text(run_dir / "quick_gate_console.log", str({"status": summary["status"], "failed_steps": summary["failed_steps"]}))
    print({"status": summary["status"], "failed_steps": summary["failed_steps"]})
    return 0 if not failed else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)

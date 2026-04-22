from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from core.output_paths import output_dir_path

from tools.optimizer.raw_cache import load_all_raw_data
from tools.scanner.stock_processor import process_single_stock
from tools.validate.scanner_expectations import normalize_scanner_result

from .checks import add_check


def _build_repeatable_ohlcv_df() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        base = 100.0 + idx
        rows.append(
            {
                "Date": f"2026-01-{idx + 2:02d}",
                "Open": base,
                "High": base + 1.0,
                "Low": base - 1.0,
                "Close": base + 0.5,
                "Volume": 1000 + idx * 10,
            }
        )
    return pd.DataFrame(rows)


def _normalize_cache_payload(raw_cache: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for ticker, df in sorted(raw_cache.items()):
        normalized = df.reset_index().copy()
        normalized[normalized.columns[0]] = normalized[normalized.columns[0]].dt.strftime("%Y-%m-%d")
        records = []
        for row in normalized.to_dict(orient="records"):
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, float):
                    clean_row[key] = round(value, 8)
                else:
                    clean_row[key] = value
            records.append(clean_row)
        payload[ticker] = {
            "columns": list(df.columns),
            "records": records,
        }
    return payload


def _payload_digest(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_scanner_worker_repeatability_case(base_params):
    case_id = "SCANNER_WORKER_REPEATABILITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "2330.csv"
        dummy_df = _build_repeatable_ohlcv_df()
        dummy_df.to_csv(file_path, index=False)
        sanitize_stats = {
            "invalid_row_count": 0,
            "duplicate_date_count": 0,
            "dropped_row_count": 0,
            "negative_volume_corrected_count": 0,
        }
        repeated_stats = {
            "is_candidate": True,
            "is_setup_today": True,
            "buy_limit": 105.0,
            "stop_loss": 99.0,
            "expected_value": 1.1,
            "win_rate": 56.0,
            "trade_count": 9,
            "max_drawdown": 7.5,
            "extended_candidate_today": None,
        }
        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "tools.scanner.stock_processor.run_v16_backtest", return_value=repeated_stats
        ):
            first_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", base_params))
            second_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", base_params))

    add_check(results, "synthetic_regression", case_id, "scanner_worker_repeatable_payload", first_result, second_result)
    add_check(results, "synthetic_regression", case_id, "scanner_worker_repeatable_status", first_result["status"], second_result["status"])
    add_check(results, "synthetic_regression", case_id, "scanner_worker_repeatable_sort_value", first_result["sort_value"], second_result["sort_value"])
    summary["status"] = first_result.get("status")
    return results, summary


def validate_optimizer_raw_cache_rerun_consistency_case(_base_params):
    case_id = "OPTIMIZER_RAW_CACHE_RERUN"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    rows_2330 = [
        {"Date": "2026-01-05", "Open": 13, "High": 14, "Low": 12, "Close": 13.5, "Volume": 130},
        {"Date": "2026-01-03", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 110},
        {"Date": "2026-01-02", "Open": 10, "High": 11, "Low": 9, "Close": 10.5, "Volume": 100},
        {"Date": "2026-01-03", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 111},
        {"Date": "2026-01-04", "Open": 12, "High": 13, "Low": 11, "Close": 12.5, "Volume": -5},
    ]
    rows_2317 = [
        {"Date": "2026-01-02", "Open": 20, "High": 21, "Low": 19, "Close": 20.5, "Volume": 200},
        {"Date": "2026-01-03", "Open": 21, "High": 22, "Low": 20, "Close": 21.5, "Volume": 210},
        {"Date": "2026-01-04", "Open": 22, "High": 23, "Low": 21, "Close": 22.5, "Volume": 220},
    ]

    project_tmp_root = output_dir_path(PROJECT_ROOT, "local_regression") / "_staging" / "validate_runtime" / "_tmp_raw_cache"
    project_tmp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="v16_raw_cache_", dir=str(project_tmp_root)) as tmp_dir:
        tmp_root = Path(tmp_dir)
        data_dir = tmp_root / "data"
        output_dir = tmp_root / "outputs"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_2330).to_csv(data_dir / "2330.csv", index=False)
        pd.DataFrame(rows_2317).to_csv(data_dir / "2317.csv", index=False)

        with contextlib.redirect_stdout(io.StringIO()):
            first_cache = load_all_raw_data(str(data_dir), required_min_rows=3, output_dir=str(output_dir))
        first_payload = _normalize_cache_payload(first_cache)
        first_digest = _payload_digest(first_payload)

        first_cache["2330"].iat[0, 0] = 999.0
        mutated_digest = _payload_digest(_normalize_cache_payload(first_cache))

        with contextlib.redirect_stdout(io.StringIO()):
            second_cache = load_all_raw_data(str(data_dir), required_min_rows=3, output_dir=str(output_dir))
        second_payload = _normalize_cache_payload(second_cache)
        second_digest = _payload_digest(second_payload)

        with contextlib.redirect_stdout(io.StringIO()):
            third_cache = load_all_raw_data(str(data_dir), required_min_rows=3, output_dir=str(output_dir))
        third_payload = _normalize_cache_payload(third_cache)
        third_digest = _payload_digest(third_payload)

        source_rows = pd.read_csv(data_dir / "2330.csv").to_dict(orient="records")

    add_check(results, "synthetic_regression", case_id, "raw_cache_first_and_second_digest_match", first_digest, second_digest)
    add_check(results, "synthetic_regression", case_id, "raw_cache_second_and_third_digest_match", second_digest, third_digest)
    add_check(results, "synthetic_regression", case_id, "raw_cache_mutation_does_not_persist", True, mutated_digest != second_digest)
    add_check(results, "synthetic_regression", case_id, "raw_cache_source_csv_not_mutated", 13.0, float(source_rows[0]["Open"]))
    add_check(results, "synthetic_regression", case_id, "raw_cache_ticker_keys_stable", ["2317", "2330"], sorted(second_cache.keys()))
    add_check(results, "synthetic_regression", case_id, "raw_cache_negative_volume_corrected", 0.0, float(second_cache["2330"].loc[pd.Timestamp("2026-01-04"), "Volume"]))

    summary["ticker_count"] = len(second_cache)
    return results, summary



def _normalize_scan_summary_payload(payload: dict[str, object]) -> dict[str, object]:
    candidate_rows = []
    for item in payload.get("candidate_rows", []):
        candidate_rows.append({
            "kind": str(item.get("kind", "")),
            "ticker": str(item.get("ticker", "")),
            "text": str(item.get("text", "")),
            "proj_cost": None if item.get("proj_cost") is None else round(float(item.get("proj_cost")), 6),
            "ev": None if item.get("ev") is None else round(float(item.get("ev")), 6),
            "sort_value": None if item.get("sort_value") is None else round(float(item.get("sort_value")), 6),
        })
    return {
        "count_scanned": int(payload.get("count_scanned", 0) or 0),
        "count_history_qualified": int(payload.get("count_history_qualified", 0) or 0),
        "count_skipped_insufficient": int(payload.get("count_skipped_insufficient", 0) or 0),
        "count_sanitized_candidates": int(payload.get("count_sanitized_candidates", 0) or 0),
        "max_workers": int(payload.get("max_workers", 0) or 0),
        "pool_start_method": str(payload.get("pool_start_method", "")),
        "candidate_rows": candidate_rows,
        "scanner_issue_log_name": Path(str(payload.get("scanner_issue_log_path", "") or "")).name,
    }


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeExecutor:
    def __init__(self, *args, **kwargs):
        self._futures = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        future = _FakeFuture(fn(*args, **kwargs))
        self._futures.append(future)
        return future


def validate_scan_runner_repeatability_case(base_params):
    case_id = "SCAN_RUNNER_REPEATABILITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.scanner import scan_runner

    class _FakeParams:
        def __init__(self, max_workers=2):
            self.max_workers = max_workers

    with TemporaryDirectory() as tmp_dir:
        file_a = Path(tmp_dir) / '1101.csv'
        file_b = Path(tmp_dir) / '2330.csv'
        file_a.write_text('Date,Open,High,Low,Close,Volume\n2026-01-02,1,2,1,2,10\n', encoding='utf-8')
        file_b.write_text('Date,Open,High,Low,Close,Volume\n2026-01-02,1,2,1,2,10\n', encoding='utf-8')
        csv_inputs = [('1101', str(file_a)), ('2330', str(file_b))]
        deterministic_results = {
            '1101': ('buy', 1000.0, 1.25, 2.5, 'BUY-1101', '1101', None),
            '2330': ('extended', 800.0, 1.1, 1.8, 'EXT-2330', '2330', 'sanitize-2330'),
        }
        collected = []

        def _fake_process_single_stock(file_path, ticker, params):
            return deterministic_results[str(ticker)]

        def _fake_summary(**kwargs):
            collected.append(_normalize_scan_summary_payload(kwargs))

        with patch('core.data_utils.discover_unique_csv_inputs', return_value=(csv_inputs, ['dup.csv'])), \
             patch('tools.scanner.scan_runner.ensure_runtime_dirs', return_value=None), \
             patch('tools.scanner.scan_runner.resolve_scanner_max_workers', return_value=2), \
             patch('tools.scanner.scan_runner.get_process_pool_executor_kwargs', return_value=({}, 'spawn')), \
             patch('tools.scanner.scan_runner.ProcessPoolExecutor', _FakeExecutor), \
             patch('tools.scanner.scan_runner.as_completed', side_effect=lambda futures: list(reversed(list(futures)))), \
             patch('tools.scanner.scan_runner.print_scanner_start_banner', return_value=None), \
             patch('tools.scanner.scan_runner.print_scanner_header', return_value=None), \
             patch('tools.scanner.scan_runner.write_issue_log', return_value=str(Path(tmp_dir) / 'scanner_issues.log')), \
             patch('tools.scanner.scan_runner.print_scanner_summary', side_effect=_fake_summary), \
             patch('tools.scanner.scan_runner.time.time', side_effect=[100.0, 103.5, 200.0, 203.5]), \
             patch('tools.scanner.stock_processor.process_single_stock', side_effect=_fake_process_single_stock):
            with contextlib.redirect_stdout(io.StringIO()):
                scan_runner.run_daily_scanner(tmp_dir, _FakeParams())
                scan_runner.run_daily_scanner(tmp_dir, _FakeParams())

    add_check(results, 'synthetic_regression', case_id, 'scan_runner_repeatable_summary_payload', collected[0], collected[1])
    add_check(results, 'synthetic_regression', case_id, 'scan_runner_repeatable_candidate_order', [r['ticker'] for r in collected[0]['candidate_rows']], [r['ticker'] for r in collected[1]['candidate_rows']])
    add_check(results, 'synthetic_regression', case_id, 'scan_runner_repeatable_issue_log_name', collected[0]['scanner_issue_log_name'], collected[1]['scanner_issue_log_name'])
    summary['candidate_count'] = len(collected[0]['candidate_rows'])
    return results, summary


def _canonicalize_master_summary(payload: dict[str, object]) -> dict[str, object]:
    scripts = []
    for item in payload.get('scripts', []):
        scripts.append({
            'name': str(item.get('name', '')),
            'status': str(item.get('status', '')),
            'reported_status': str(item.get('reported_status', '')),
            'summary_file': str(item.get('summary_file', '')),
            'failure_reasons': list(item.get('failure_reasons', [])),
            'timed_out': bool(item.get('timed_out', False)),
        })
    return {
        'overall_status': str(payload.get('overall_status', '')),
        'dataset': str(payload.get('dataset', '')),
        'selected_steps': list(payload.get('selected_steps', [])),
        'failures': int(payload.get('failures', 0) or 0),
        'failed_step_names': list(payload.get('failed_step_names', [])),
        'not_run_step_names': list(payload.get('not_run_step_names', [])),
        'bundle_mode': str(payload.get('bundle_mode', '')),
        'bundle_entries': list(payload.get('bundle_entries', [])),
        'scripts': scripts,
        'preflight_status': str((payload.get('preflight') or {}).get('status', '')),
        'dataset_prepare_status': str((payload.get('dataset_prepare') or {}).get('status', '')),
    }


def validate_run_all_repeatability_case(_base_params):
    case_id = 'RUN_ALL_REPEATABILITY'
    results = []
    summary = {'ticker': case_id, 'synthetic': True}

    from tools.local_regression import run_all as run_all_module

    project_tmp_root = output_dir_path(PROJECT_ROOT, 'local_regression') / '_staging' / 'validate_runtime' / '_tmp_raw_cache'
    project_tmp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix='v16_run_all_', dir=str(project_tmp_root)) as tmp_dir:
        run_dir = Path(tmp_dir) / 'staging'
        run_dir.mkdir(parents=True, exist_ok=True)
        fixed_now = datetime(2026, 4, 2, 8, 0, 0)
        manifest = dict(run_all_module.MANIFEST_DEFAULTS)
        manifest.update({'dataset': 'reduced', 'bundle_name': 'to_chatgpt_bundle.zip'})
        summary_name_by_script = {script: summary_name for _name, script, summary_name in run_all_module.SCRIPT_ORDER}

        def _fake_run_preflight(run_dir_arg, *, selected_step_names):
            payload = {
                'status': 'PASS',
                'duration_sec': 0.1,
                'failed_packages': [],
                'python_executable': sys.executable,
                'summary_file': 'preflight_summary.json',
                'summary_text_file': 'preflight_summary.txt',
            }
            (run_dir_arg / 'preflight_summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            (run_dir_arg / 'preflight_summary.txt').write_text('status: PASS\n', encoding='utf-8')
            return payload

        def _fake_dataset_info():
            return {'dataset_dir': str(PROJECT_ROOT / 'data' / 'tw_stock_data_vip_reduced'), 'source': 'repo_data_dir', 'csv_count': 24, 'reused_existing': True}

        def _fake_run_script(*, name, relative_script, timeout_sec, env, log_path, progress_callback, major_index, major_total, execution_mode="serial"):
            payload = {'status': 'PASS', 'failures': [], 'failed_steps': [], 'failed_count': 0, 'fail_count': 0}
            summary_name = summary_name_by_script[relative_script]
            (run_dir / summary_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            log_path.write_text(f'{name}: PASS\n', encoding='utf-8')
            return {
                'returncode': 0,
                'duration_sec': 0.2,
                'timed_out': False,
                'error_type': '',
                'error_message': '',
                'stdout': '',
                'stderr': '',
            }

        def _fake_archive(bundle_path, **kwargs):
            return bundle_path

        def _fake_publish(bundle_path, **kwargs):
            return bundle_path

        with patch('tools.local_regression.run_all.create_staging_run_dir', return_value=run_dir), \
             patch('tools.local_regression.run_all.load_manifest', return_value=manifest), \
             patch('tools.local_regression.run_all._run_preflight', side_effect=_fake_run_preflight), \
             patch('tools.local_regression.run_all.ensure_reduced_dataset', side_effect=_fake_dataset_info), \
             patch('tools.local_regression.run_all.build_shared_prep_cache', return_value={'prepared_count': 0, 'skipped_count': 0, 'duplicate_issue_count': 0}), \
             patch('tools.local_regression.run_all._run_script', side_effect=_fake_run_script), \
             patch('tools.local_regression.run_all.archive_bundle_history', side_effect=_fake_archive), \
             patch('tools.local_regression.run_all.publish_root_bundle_copy', side_effect=_fake_publish), \
             patch('tools.local_regression.run_all._apply_output_retention', return_value={'removed_count': 0, 'removed_bytes': 0, 'removed_entries': []}), \
             patch('tools.local_regression.run_all.cleanup_staging_dir', return_value=None), \
             patch('tools.local_regression.run_all.resolve_git_commit', return_value='deadbeef'), \
             patch('tools.local_regression.run_all.taipei_now', return_value=fixed_now), \
             patch('tools.local_regression.run_all.gather_recent_console_tail', return_value='tail text'):
            with contextlib.redirect_stdout(io.StringIO()):
                rc1 = run_all_module.main(['run_all.py'])
            first_master = json.loads((run_dir / 'master_summary.json').read_text(encoding='utf-8'))
            first_zip_members = sorted(zipfile.ZipFile(run_dir / 'to_chatgpt_bundle.zip').namelist())
            first_digest = _payload_digest(_canonicalize_master_summary(first_master))

            (run_dir / 'master_summary.json').write_text('{"tampered": true}\n', encoding='utf-8')
            (run_dir / 'artifacts_manifest.json').write_text('{"tampered": true}\n', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                rc2 = run_all_module.main(['run_all.py'])
            second_master = json.loads((run_dir / 'master_summary.json').read_text(encoding='utf-8'))
            second_zip_members = sorted(zipfile.ZipFile(run_dir / 'to_chatgpt_bundle.zip').namelist())
            second_digest = _payload_digest(_canonicalize_master_summary(second_master))

    add_check(results, 'synthetic_regression', case_id, 'run_all_repeatable_return_code_first', 0, rc1)
    add_check(results, 'synthetic_regression', case_id, 'run_all_repeatable_return_code_second', 0, rc2)
    add_check(results, 'synthetic_regression', case_id, 'run_all_repeatable_master_summary_digest', first_digest, second_digest)
    add_check(results, 'synthetic_regression', case_id, 'run_all_repeatable_bundle_members', first_zip_members, second_zip_members)
    add_check(results, 'synthetic_regression', case_id, 'run_all_rerun_replaces_tampered_master_summary', False, bool(second_master.get('tampered', False)))
    add_check(results, 'synthetic_regression', case_id, 'run_all_repeatable_not_run_step_names', [], second_master.get('not_run_step_names', []))
    summary['bundle_member_count'] = len(second_zip_members)
    return results, summary

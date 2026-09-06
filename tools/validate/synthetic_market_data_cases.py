from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .checks import add_check


def validate_market_data_v2_preflight_planner_contract_case(_base_params):
    """Round-1 primitives remain registry-driven, quota-visible and fail closed."""

    from core.market_data_bootstrap_planner import DatasetProbeEvidence, build_bootstrap_request_plan
    from core.market_data_dataset_registry import (
        BOOTSTRAP_BULK_REFERENCE_DATES,
        BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE,
        BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
        BOOTSTRAP_SINGLE_FULL_RANGE,
        BOOTSTRAP_SINGLE_NO_DATES,
        get_market_dataset_spec,
        get_market_dataset_specs,
        validate_market_dataset_registry,
    )
    from services.downloader.finmind_http import FinMindHttpClient, FinMindHttpError
    from services.downloader.market_data_preflight import _historical_instruments

    case_id = "MARKET_DATA_V2_PREFLIGHT_PLANNER"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    registry_summary = validate_market_dataset_registry()
    specs = get_market_dataset_specs(included_only=True)
    excluded = tuple(spec for spec in get_market_dataset_specs() if not spec.included)
    datasets = {spec.dataset for spec in specs}
    excluded_ids = {spec.dataset for spec in excluded}
    add_check(results, "market_data", case_id, "registry_has_included_datasets", True, bool(specs))
    add_check(results, "market_data", case_id, "registry_has_explicit_exclusions", True, bool(excluded))
    add_check(results, "market_data", case_id, "raw_price_is_archived", True, "TaiwanStockPrice" in datasets)
    add_check(results, "market_data", case_id, "adjusted_price_is_archived", True, "TaiwanStockPriceAdj" in datasets)
    add_check(results, "market_data", case_id, "sponsor_margin_maintenance_is_not_backer_required", True, "TaiwanStockMarginMaintenance" in excluded_ids)
    add_check(results, "market_data", case_id, "industry_chain_classification_is_archived", True, "TaiwanStockIndustryChain" in datasets)
    add_check(results, "market_data", case_id, "derivative_product_master_is_archived", True, "TaiwanFutOptDailyInfo" in datasets)
    add_check(results, "market_data", case_id, "price_limit_bootstrap_is_per_instrument", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, get_market_dataset_spec("TaiwanStockPriceLimit").bootstrap_mode)
    add_check(results, "market_data", case_id, "futures_large_trader_uses_dataset_product_code", ("TXF",), get_market_dataset_spec("TaiwanFuturesOpenInterestLargeTraders").fixed_data_ids)

    stock_info = pd.DataFrame(
        [
            {"stock_id": "2330", "type": "twse"},
            {"stock_id": "6488", "type": "tpex"},
            {"stock_id": "7777", "type": "emerging"},
        ]
    )
    delisting = pd.DataFrame([{"stock_id": "1204"}])
    instruments = _historical_instruments(stock_info, delisting)
    add_check(results, "market_data", case_id, "historical_universe_keeps_twse_tpex_and_delisting", ("1204", "2330", "6488"), instruments)

    evidence = {}
    for spec in specs:
        observed_dates = ("2020-01-31", "2020-02-29") if spec.bootstrap_mode == BOOTSTRAP_BULK_REFERENCE_DATES else ()
        evidence[spec.dataset] = DatasetProbeEvidence(
            dataset=spec.dataset,
            status="PASS",
            request_count=1,
            row_count=1,
            columns=("date",),
            observed_dates=observed_dates,
            earliest_date=observed_dates[0] if observed_dates else None,
            latest_date=observed_dates[-1] if observed_dates else None,
        )

    plan3 = build_bootstrap_request_plan(
        specs=specs,
        historical_instruments=("1101", "2330", "0050"),
        evidence_by_dataset=evidence,
        as_of_date="2020-12-31",
        quota_limit=1600,
    )
    expected_total = 0
    for spec in specs:
        if spec.bootstrap_mode in {BOOTSTRAP_SINGLE_NO_DATES, BOOTSTRAP_SINGLE_FULL_RANGE}:
            expected_total += 1
        elif spec.bootstrap_mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE:
            expected_total += 3
        elif spec.bootstrap_mode == BOOTSTRAP_BULK_REFERENCE_DATES:
            expected_total += 2
        elif spec.bootstrap_mode == BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE:
            expected_total += len(spec.fixed_data_ids)
    add_check(results, "market_data", case_id, "planner_total_is_registry_derived", expected_total, plan3.total_requests)
    add_check(results, "market_data", case_id, "planner_uses_live_quota_limit", 1600, plan3.quota_limit)

    plan4 = build_bootstrap_request_plan(
        specs=specs,
        historical_instruments=("1101", "2330", "0050", "2317"),
        evidence_by_dataset=evidence,
        as_of_date="2020-12-31",
        quota_limit=1600,
    )
    per_instrument_count = sum(spec.bootstrap_mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE for spec in specs)
    add_check(results, "market_data", case_id, "planner_has_no_hardcoded_instrument_count", per_instrument_count, plan4.total_requests - plan3.total_requests)

    first = specs[0]
    failed = dict(evidence)
    failed[first.dataset] = replace(failed[first.dataset], status="FAIL", error="synthetic denial")
    try:
        build_bootstrap_request_plan(
            specs=specs,
            historical_instruments=("2330",),
            evidence_by_dataset=failed,
            as_of_date="2020-12-31",
            quota_limit=1600,
        )
    except ValueError:
        failed_probe_blocks = True
    else:
        failed_probe_blocks = False
    add_check(results, "market_data", case_id, "failed_probe_blocks_exact_plan", True, failed_probe_blocks)

    class _Response:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

    class _Session:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return self.responses.pop(0)

    session = _Session(
        [
            _Response({"user_count": 7, "api_request_limit": 1600}),
            _Response({"status": 200, "data": [{"date": "2020-01-02", "stock_id": "2330"}]}),
            _Response({"status": 403, "msg": "denied"}, status_code=403),
        ]
    )
    client = FinMindHttpClient(token="synthetic-token", session=session)
    usage = client.get_usage()
    frame = client.get_data(dataset="TaiwanStockPrice", data_id="2330", start_date="2020-01-02", end_date="2020-01-02")
    try:
        client.get_data(dataset="DeniedDataset", start_date="2020-01-02")
    except FinMindHttpError:
        denied_raised = True
    else:
        denied_raised = False
    add_check(results, "market_data", case_id, "http_usage_reads_live_limit", 1600, usage.api_request_limit)
    add_check(results, "market_data", case_id, "http_success_decodes_rows", 1, len(frame))
    add_check(results, "market_data", case_id, "http_client_counts_data_attempts", 2, client.data_request_count)
    add_check(results, "market_data", case_id, "http_4xx_is_not_hidden_retry", 3, len(session.calls))
    add_check(results, "market_data", case_id, "http_4xx_fails_closed", True, denied_raised)

    summary.update(
        {
            "included_dataset_count": registry_summary["included"],
            "excluded_dataset_count": registry_summary["excluded"],
            "per_instrument_dataset_count": plan3.per_instrument_dataset_count,
            "synthetic_total_requests": plan3.total_requests,
        }
    )
    return results, summary



def validate_market_data_v2_resumable_executor_contract_case(_base_params):
    """Round-2 request identity, ledger resume, quota wait and retry semantics."""

    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.market_data_bootstrap_planner import DatasetProbeEvidence, build_bootstrap_request_plan
    from core.market_data_bootstrap_requests import build_bootstrap_request_manifest
    from core.market_data_dataset_registry import get_market_dataset_specs
    from core.market_data_execution_policy import MarketDataExecutionPolicy
    from services.downloader.finmind_http import FinMindHttpError, FinMindUsage
    from services.downloader.market_data_executor import MarketDataBootstrapExecutor, MarketDataCommitReceipt
    from services.downloader.market_data_ledger import JOB_PENDING, WORKLOAD_BLOCKED, WORKLOAD_DONE, MarketDataJobLedger

    case_id = "MARKET_DATA_V2_RESUMABLE_EXECUTOR"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    specs_all = get_market_dataset_specs(included_only=True)
    # One single request + one per-instrument dataset gives a compact manifest
    # while exercising different registry expansion modes.
    selected = tuple(
        spec
        for spec in specs_all
        if spec.dataset in {"TaiwanStockTradingDate", "TaiwanStockPrice"}
    )
    evidence = {
        spec.dataset: DatasetProbeEvidence(
            dataset=spec.dataset,
            status="PASS",
            request_count=1,
            row_count=1,
            columns=("date", "stock_id"),
        )
        for spec in selected
    }
    instruments = ("1101", "2330")
    manifest = build_bootstrap_request_manifest(
        specs=selected,
        historical_instruments=instruments,
        evidence_by_dataset=evidence,
        as_of_date="2026-09-04",
    )
    plan = build_bootstrap_request_plan(
        specs=selected,
        historical_instruments=instruments,
        evidence_by_dataset=evidence,
        as_of_date="2026-09-04",
        quota_limit=1600,
    )
    add_check(results, "market_data", case_id, "planner_and_executor_manifest_count_same", plan.total_requests, manifest.total_requests)
    add_check(results, "market_data", case_id, "planner_and_executor_manifest_identity_same", plan.manifest_fingerprint, manifest.manifest_fingerprint)
    add_check(results, "market_data", case_id, "logical_request_ids_unique", manifest.total_requests, len({item.request_id for item in manifest.requests}))

    class _Clock:
        def __init__(self):
            self.value = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
            self.sleeps = []

        def now(self):
            return self.value

        def sleep(self, seconds):
            seconds = float(seconds)
            self.sleeps.append(seconds)
            self.value += timedelta(seconds=seconds)

    policy = MarketDataExecutionPolicy(
        quota_reserve_requests=1,
        quota_refresh_every_requests=99,
        quota_poll_seconds=2.0,
        max_retryable_attempts=3,
        retry_backoff_seconds=(1.0, 2.0),
        job_lease_seconds=5.0,
        executor_lock_seconds=4.0,
        progress_every_committed_requests=2,
    )

    with TemporaryDirectory() as td:
        ledger = MarketDataJobLedger(Path(td) / "resume.sqlite3")
        workload_id = ledger.seed_manifest(manifest, now=datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc))
        first = ledger.claim_next_job(
            workload_id,
            owner_id="crashed-worker",
            now=datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc),
            lease_until=datetime(2026, 9, 6, 10, 0, 5, tzinfo=timezone.utc),
        )
        ledger.record_http_attempt(workload_id, first.request_id, now=datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc))
        resumed = ledger.claim_next_job(
            workload_id,
            owner_id="replacement-worker",
            now=datetime(2026, 9, 6, 10, 0, 6, tzinfo=timezone.utc),
            lease_until=datetime(2026, 9, 6, 10, 0, 11, tzinfo=timezone.utc),
        )
        add_check(results, "market_data", case_id, "expired_running_job_is_reclaimed", first.request_id, resumed.request_id)
        add_check(results, "market_data", case_id, "resume_preserves_http_attempt_history", 1, resumed.http_attempt_count)
        add_check(results, "market_data", case_id, "resume_increments_execution_attempt", 2, resumed.attempt_count)

    with TemporaryDirectory() as td:
        lock_ledger = MarketDataJobLedger(Path(td) / "lock.sqlite3")
        lock_workload = lock_ledger.seed_manifest(manifest, now=datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc))
        lock_ledger.acquire_executor_lock(
            lock_workload,
            owner_id="owner-a",
            now=datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc),
            lease_until=datetime(2026, 9, 6, 10, 1, tzinfo=timezone.utc),
        )
        try:
            lock_ledger.acquire_executor_lock(
                lock_workload,
                owner_id="owner-b",
                now=datetime(2026, 9, 6, 10, 0, 30, tzinfo=timezone.utc),
                lease_until=datetime(2026, 9, 6, 10, 1, 30, tzinfo=timezone.utc),
            )
        except RuntimeError:
            concurrent_blocked = True
        else:
            concurrent_blocked = False
        add_check(results, "market_data", case_id, "active_executor_lock_blocks_concurrent_executor", True, concurrent_blocked)
        lock_ledger.acquire_executor_lock(
            lock_workload,
            owner_id="owner-b",
            now=datetime(2026, 9, 6, 10, 1, 1, tzinfo=timezone.utc),
            lease_until=datetime(2026, 9, 6, 10, 2, 1, tzinfo=timezone.utc),
        )
        add_check(results, "market_data", case_id, "expired_executor_lock_can_be_taken_over", True, True)

    class _QuotaThenSuccessClient:
        def __init__(self):
            self.usage_request_count = 0
            self.data_request_count = 0
            self.usage_values = [
                FinMindUsage(user_count=0, api_request_limit=4),
                FinMindUsage(user_count=4, api_request_limit=4),
                FinMindUsage(user_count=0, api_request_limit=4),
            ]
            self.quota_raised = False

        def get_usage(self):
            self.usage_request_count += 1
            if self.usage_values:
                return self.usage_values.pop(0)
            return FinMindUsage(user_count=0, api_request_limit=4)

        def get_data(self, *, dataset, data_id=None, start_date=None, end_date=None):
            self.data_request_count += 1
            if not self.quota_raised:
                self.quota_raised = True
                raise FinMindHttpError(
                    "synthetic quota",
                    http_status=402,
                    api_status=402,
                    retryable=True,
                    quota_exhausted=True,
                )
            return pd.DataFrame([{"dataset": dataset, "data_id": data_id, "date": end_date or start_date}])

    clock = _Clock()
    quota_client = _QuotaThenSuccessClient()
    committed = []

    def _sink(request, frame):
        committed.append(request.request_id)
        return MarketDataCommitReceipt(committed=True, row_count=len(frame), content_sha256=f"sha-{request.request_id[:8]}")

    with TemporaryDirectory() as td:
        ledger = MarketDataJobLedger(Path(td) / "quota.sqlite3")
        executor = MarketDataBootstrapExecutor(
            ledger=ledger,
            client=quota_client,
            policy=policy,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
            owner_id="quota-worker",
        )
        execution_summary = executor.run(manifest=manifest, sink=_sink)
        add_check(results, "market_data", case_id, "quota_402_waits_and_resumes_to_done", WORKLOAD_DONE, execution_summary.workload_status)
        add_check(results, "market_data", case_id, "quota_resume_commits_every_logical_request", manifest.total_requests, execution_summary.done)
        add_check(results, "market_data", case_id, "quota_402_is_counted_as_actual_http_attempt", manifest.total_requests + 1, execution_summary.http_attempts)
        add_check(results, "market_data", case_id, "quota_wait_uses_injected_polling", True, bool(clock.sleeps))

        class _NoCallClient:
            def __init__(self):
                self.usage_request_count = 0
                self.data_request_count = 0

            def get_usage(self):
                self.usage_request_count += 1
                raise AssertionError("DONE workload 不應再次讀 quota")

            def get_data(self, **kwargs):
                self.data_request_count += 1
                raise AssertionError("DONE workload 不應重抓 HTTP data")

        no_call = _NoCallClient()
        second_executor = MarketDataBootstrapExecutor(
            ledger=ledger,
            client=no_call,
            policy=policy,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
            owner_id="quota-worker-second-run",
        )
        second_summary = second_executor.run(manifest=manifest, sink=_sink)
        add_check(results, "market_data", case_id, "restart_reuses_all_done_jobs_without_http", 0, no_call.data_request_count)
        add_check(results, "market_data", case_id, "restart_done_summary_stays_done", WORKLOAD_DONE, second_summary.workload_status)

    class _TransientClient:
        def __init__(self):
            self.usage_request_count = 0
            self.data_request_count = 0
            self.failed_once = False

        def get_usage(self):
            self.usage_request_count += 1
            return FinMindUsage(user_count=0, api_request_limit=100)

        def get_data(self, *, dataset, data_id=None, start_date=None, end_date=None):
            self.data_request_count += 1
            if not self.failed_once:
                self.failed_once = True
                raise FinMindHttpError("synthetic 503", http_status=503, retryable=True)
            return pd.DataFrame([{"dataset": dataset, "data_id": data_id}])

    transient_clock = _Clock()
    with TemporaryDirectory() as td:
        ledger = MarketDataJobLedger(Path(td) / "transient.sqlite3")
        executor = MarketDataBootstrapExecutor(
            ledger=ledger,
            client=_TransientClient(),
            policy=policy,
            now_fn=transient_clock.now,
            sleep_fn=transient_clock.sleep,
            owner_id="retry-worker",
        )
        retry_summary = executor.run(manifest=manifest, sink=_sink)
        add_check(results, "market_data", case_id, "transient_error_retries_to_done", WORKLOAD_DONE, retry_summary.workload_status)
        add_check(results, "market_data", case_id, "transient_retry_adds_one_http_attempt", manifest.total_requests + 1, retry_summary.http_attempts)

    one_spec = tuple(spec for spec in selected if spec.dataset == "TaiwanStockTradingDate")
    one_evidence = {"TaiwanStockTradingDate": evidence["TaiwanStockTradingDate"]}
    one_manifest = build_bootstrap_request_manifest(
        specs=one_spec,
        historical_instruments=instruments,
        evidence_by_dataset=one_evidence,
        as_of_date="2026-09-04",
    )

    class _SuccessClient:
        def __init__(self):
            self.usage_request_count = 0
            self.data_request_count = 0

        def get_usage(self):
            self.usage_request_count += 1
            return FinMindUsage(user_count=0, api_request_limit=100)

        def get_data(self, *, dataset, data_id=None, start_date=None, end_date=None):
            self.data_request_count += 1
            return pd.DataFrame([{"dataset": dataset}])

    commit_clock = _Clock()
    with TemporaryDirectory() as td:
        ledger = MarketDataJobLedger(Path(td) / "commit_gate.sqlite3")
        executor = MarketDataBootstrapExecutor(
            ledger=ledger,
            client=_SuccessClient(),
            policy=policy,
            now_fn=commit_clock.now,
            sleep_fn=commit_clock.sleep,
            owner_id="commit-gate-worker",
        )
        commit_blocked = executor.run(
            manifest=one_manifest,
            sink=lambda request, frame: MarketDataCommitReceipt(committed=False, row_count=len(frame)),
        )
        add_check(results, "market_data", case_id, "http_success_without_commit_does_not_mark_done", 0, commit_blocked.done)
        add_check(results, "market_data", case_id, "uncommitted_sink_blocks_workload", WORKLOAD_BLOCKED, commit_blocked.workload_status)

    class _PermanentClient:
        def __init__(self):
            self.usage_request_count = 0
            self.data_request_count = 0

        def get_usage(self):
            self.usage_request_count += 1
            return FinMindUsage(user_count=0, api_request_limit=100)

        def get_data(self, *, dataset, data_id=None, start_date=None, end_date=None):
            self.data_request_count += 1
            raise FinMindHttpError("synthetic 403", http_status=403, retryable=False)

    class _AlwaysTransientClient:
        def __init__(self):
            self.usage_request_count = 0
            self.data_request_count = 0

        def get_usage(self):
            self.usage_request_count += 1
            return FinMindUsage(user_count=0, api_request_limit=100)

        def get_data(self, *, dataset, data_id=None, start_date=None, end_date=None):
            self.data_request_count += 1
            raise FinMindHttpError("synthetic persistent 503", http_status=503, retryable=True)

    exhausted_clock = _Clock()
    with TemporaryDirectory() as td:
        ledger = MarketDataJobLedger(Path(td) / "retry_exhausted.sqlite3")
        executor = MarketDataBootstrapExecutor(
            ledger=ledger,
            client=_AlwaysTransientClient(),
            policy=policy,
            now_fn=exhausted_clock.now,
            sleep_fn=exhausted_clock.sleep,
            owner_id="retry-exhausted-worker",
        )
        exhausted_summary = executor.run(manifest=one_manifest, sink=_sink)
        add_check(results, "market_data", case_id, "retry_budget_exhaustion_blocks_workload", WORKLOAD_BLOCKED, exhausted_summary.workload_status)
        add_check(results, "market_data", case_id, "retry_budget_is_bounded_by_policy", policy.max_retryable_attempts, exhausted_summary.http_attempts)

    permanent_clock = _Clock()
    with TemporaryDirectory() as td:
        ledger = MarketDataJobLedger(Path(td) / "permanent.sqlite3")
        executor = MarketDataBootstrapExecutor(
            ledger=ledger,
            client=_PermanentClient(),
            policy=policy,
            now_fn=permanent_clock.now,
            sleep_fn=permanent_clock.sleep,
            owner_id="permanent-worker",
        )
        blocked_summary = executor.run(manifest=manifest, sink=_sink)
        add_check(results, "market_data", case_id, "permanent_4xx_blocks_workload", WORKLOAD_BLOCKED, blocked_summary.workload_status)
        add_check(results, "market_data", case_id, "permanent_4xx_no_hidden_retry", 1, blocked_summary.http_attempts)
        add_check(results, "market_data", case_id, "permanent_block_leaves_unstarted_jobs_pending", True, blocked_summary.pending > 0)

    summary.update(
        {
            "logical_requests": manifest.total_requests,
            "manifest_fingerprint": manifest.manifest_fingerprint,
            "quota_wait_sleep_calls": len(clock.sleeps),
        }
    )
    return results, summary




def validate_market_data_v2_parquet_storage_contract_case(_base_params):
    """Round-3 atomic storage, schema drift, disk gate and commit recovery."""

    from collections import namedtuple
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.file_integrity import compute_file_sha256, load_json_strict
    from core.market_data_bootstrap_requests import build_bootstrap_request_manifest
    from core.market_data_dataset_registry import get_market_dataset_specs
    from core.market_data_execution_policy import MarketDataExecutionPolicy
    from core.market_data_storage_contract import (
        MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME,
        MARKET_DATA_DATASET_SCHEMA_FILENAME,
        MarketDataCommitError,
        resolve_market_data_bootstrap_archive_dir,
        resolve_market_data_request_parquet_path,
    )
    from core.market_data_storage_policy import MarketDataStoragePolicy, get_market_data_storage_policy
    from services.downloader.finmind_http import FinMindUsage
    from services.downloader.market_data_executor import MarketDataBootstrapExecutor
    from services.downloader.market_data_ledger import WORKLOAD_DONE, MarketDataJobLedger
    from services.downloader.market_data_storage import (
        MarketDataBootstrapStorageSink,
        ParquetArtifactInspection,
    )

    case_id = "MARKET_DATA_V2_PARQUET_STORAGE"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    trading_date_specs = tuple(
        spec for spec in get_market_dataset_specs(included_only=True)
        if spec.dataset == "TaiwanStockTradingDate"
    )
    trading_date_evidence = {
        "TaiwanStockTradingDate": {"status": "PASS", "observed_dates": ("2026-09-04",)}
    }
    recovery_manifest = build_bootstrap_request_manifest(
        specs=trading_date_specs,
        historical_instruments=("2330",),
        evidence_by_dataset=trading_date_evidence,
        as_of_date="2026-09-04",
    )
    recovery_request = recovery_manifest.requests[0]

    price_specs = tuple(
        spec for spec in get_market_dataset_specs(included_only=True)
        if spec.dataset == "TaiwanStockPrice"
    )
    price_evidence = {"TaiwanStockPrice": {"status": "PASS"}}
    manifest = build_bootstrap_request_manifest(
        specs=price_specs,
        historical_instruments=("1101", "2330"),
        evidence_by_dataset=price_evidence,
        as_of_date="2026-09-04",
    )
    request, second_request = manifest.requests

    class _JsonParquetCodec:
        def __init__(self):
            self.write_count = 0

        def write(self, frame, path, *, metadata, compression):
            self.write_count += 1
            payload = {
                "compression": compression,
                "row_count": len(frame),
                "columns": list(frame.columns),
                "metadata": metadata,
            }
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")

        def inspect(self, path):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return ParquetArtifactInspection(
                row_count=int(payload["row_count"]),
                columns=tuple(payload["columns"]),
                metadata=dict(payload["metadata"]),
            )

    tiny_policy = MarketDataStoragePolicy(
        format="parquet",
        compression="zstd",
        minimum_free_bytes=0,
        staging_headroom_multiplier=1.0,
        minimum_staging_headroom_bytes=0,
    )
    DiskUsage = namedtuple("DiskUsage", "total used free")
    enough_disk = lambda _path: DiskUsage(10**9, 0, 10**9)

    with TemporaryDirectory() as td:
        root = Path(td)
        codec = _JsonParquetCodec()
        sink = MarketDataBootstrapStorageSink(
            project_root=root,
            manifest=manifest,
            policy=tiny_policy,
            codec=codec,
            disk_usage_fn=enough_disk,
        )
        archive_dir = resolve_market_data_bootstrap_archive_dir(root, manifest.manifest_fingerprint)
        add_check(results, "market_data", case_id, "archive_root_is_neutral_market_data_v2_namespace", True, "data/market_data_v2/bootstrap" in archive_dir.as_posix())
        bootstrap_payload = load_json_strict(archive_dir / MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME)
        add_check(results, "market_data", case_id, "bootstrap_manifest_binds_request_manifest_fingerprint", manifest.manifest_fingerprint, bootstrap_payload["manifest_fingerprint"])
        add_check(results, "market_data", case_id, "bootstrap_storage_format_is_parquet", "parquet", bootstrap_payload["format"])
        add_check(results, "market_data", case_id, "bootstrap_storage_compression_is_zstd", "zstd", bootstrap_payload["compression"])

        frame = pd.DataFrame([{"date": "2026-09-03", "stock_id": "2330"}, {"date": "2026-09-04", "stock_id": "2330"}])
        receipt = sink(request, frame)
        parquet_path = resolve_market_data_request_parquet_path(root, manifest.manifest_fingerprint, request)
        add_check(results, "market_data", case_id, "commit_receipt_true_after_atomic_publication", True, receipt.committed)
        add_check(results, "market_data", case_id, "commit_receipt_row_count_matches_frame", len(frame), receipt.row_count)
        add_check(results, "market_data", case_id, "commit_receipt_hash_matches_published_file", compute_file_sha256(parquet_path), receipt.content_sha256)
        add_check(results, "market_data", case_id, "published_request_artifact_exists", True, parquet_path.is_file())
        temp_files = list(parquet_path.parent.glob("*.parquet.tmp")) + list(parquet_path.parent.glob(".*.parquet.tmp"))
        add_check(results, "market_data", case_id, "atomic_commit_leaves_no_temp_file", 0, len(temp_files))
        schema_payload = load_json_strict(parquet_path.parent / MARKET_DATA_DATASET_SCHEMA_FILENAME)
        add_check(results, "market_data", case_id, "dataset_schema_preserves_provider_column_order", list(frame.columns), schema_payload["columns"])

        recovered = sink.recover_committed(request)
        add_check(results, "market_data", case_id, "committed_artifact_is_recoverable", True, recovered is not None and recovered.committed)
        add_check(results, "market_data", case_id, "recovery_hash_matches_original_receipt", receipt.content_sha256, recovered.content_sha256)
        add_check(results, "market_data", case_id, "recovery_does_not_rewrite_artifact", 1, codec.write_count)

        try:
            sink(second_request, pd.DataFrame([{"date": "2026-09-04", "new_column": 1}]))
        except MarketDataCommitError:
            drift_blocked = True
        else:
            drift_blocked = False
        add_check(results, "market_data", case_id, "provider_schema_drift_fails_closed", True, drift_blocked)

    with TemporaryDirectory() as td:
        low_disk = lambda _path: DiskUsage(100, 99, 1)
        policy = MarketDataStoragePolicy(
            format="parquet",
            compression="zstd",
            minimum_free_bytes=100,
            staging_headroom_multiplier=1.0,
            minimum_staging_headroom_bytes=10,
        )
        sink = MarketDataBootstrapStorageSink(
            project_root=Path(td),
            manifest=manifest,
            policy=policy,
            codec=_JsonParquetCodec(),
            disk_usage_fn=low_disk,
        )
        try:
            sink(request, pd.DataFrame([{"date": "2026-09-04"}]))
        except MarketDataCommitError as exc:
            low_disk_blocked = "free-space gate" in str(exc)
        else:
            low_disk_blocked = False
        add_check(results, "market_data", case_id, "low_disk_space_fails_before_publication", True, low_disk_blocked)

    # Crash window regression: file is already atomically published but ledger
    # was not marked DONE.  Executor must recover storage before issuing data HTTP.
    class _RecoveryClient:
        def __init__(self):
            self.usage_request_count = 0
            self.data_request_count = 0

        def get_usage(self):
            self.usage_request_count += 1
            return FinMindUsage(user_count=0, api_request_limit=10)

        def get_data(self, **_kwargs):
            self.data_request_count += 1
            raise AssertionError("committed artifact recovery 不應重新下載 FinMind data")

    execution_policy = MarketDataExecutionPolicy(
        quota_reserve_requests=1,
        quota_refresh_every_requests=99,
        quota_poll_seconds=1.0,
        max_retryable_attempts=2,
        retry_backoff_seconds=(1.0,),
        job_lease_seconds=5.0,
        executor_lock_seconds=5.0,
        progress_every_committed_requests=1,
    )
    with TemporaryDirectory() as td:
        root = Path(td)
        sink = MarketDataBootstrapStorageSink(
            project_root=root,
            manifest=recovery_manifest,
            policy=tiny_policy,
            codec=_JsonParquetCodec(),
            disk_usage_fn=enough_disk,
        )
        sink(recovery_request, pd.DataFrame([{"date": "2026-09-04"}]))
        client = _RecoveryClient()
        ledger = MarketDataJobLedger(root / "ledger.sqlite3")
        executor = MarketDataBootstrapExecutor(
            ledger=ledger,
            client=client,
            policy=execution_policy,
            now_fn=lambda: datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc),
            sleep_fn=lambda _seconds: None,
            owner_id="recovery-worker",
        )
        execution_summary = executor.run(manifest=recovery_manifest, sink=sink)
        add_check(results, "market_data", case_id, "post_commit_pre_ledger_crash_recovers_to_done", WORKLOAD_DONE, execution_summary.workload_status)
        add_check(results, "market_data", case_id, "post_commit_recovery_uses_zero_data_http", 0, client.data_request_count)
        add_check(results, "market_data", case_id, "post_commit_recovery_ledger_http_attempts_zero", 0, execution_summary.http_attempts)

    project_root = Path(__file__).resolve().parents[2]
    requirements = (project_root / "requirements" / "requirements.txt").read_text(encoding="utf-8")
    lock = (project_root / "requirements" / "requirements-lock.txt").read_text(encoding="utf-8")
    add_check(results, "market_data", case_id, "pyarrow_declared_in_requirements", True, any(line.strip() == "pyarrow" for line in requirements.splitlines()))
    add_check(results, "market_data", case_id, "pyarrow_exact_version_locked", True, any(line.strip().startswith("pyarrow==") for line in lock.splitlines()))
    add_check(results, "market_data", case_id, "default_storage_policy_is_parquet_zstd", ("parquet", "zstd"), (get_market_data_storage_policy().format, get_market_data_storage_policy().compression))

    summary.update({"checks": len(results), "manifest_fingerprint": manifest.manifest_fingerprint})
    return results, summary


def validate_market_data_v2_bootstrap_activation_contract_case(_base_params):
    """Round-4 activation requires newest READY evidence and exact identity parity."""

    from dataclasses import asdict
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.market_data_bootstrap_planner import DatasetProbeEvidence, build_bootstrap_request_plan
    from core.market_data_dataset_registry import (
        BOOTSTRAP_BULK_REFERENCE_DATES,
        BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
        get_market_dataset_spec,
        get_market_dataset_specs,
    )
    from core.market_data_execution_policy import MarketDataExecutionPolicy
    from core.market_data_storage_contract import MarketDataCommitError, MarketDataCommitReceipt
    from services.downloader.finmind_http import FinMindUsage
    from services.downloader.market_data_bootstrap_activation import (
        MarketDataBootstrapActivationError,
        execute_market_data_v2_bootstrap,
        get_existing_bootstrap_summary,
        prepare_market_data_v2_bootstrap_activation,
    )
    from services.downloader.market_data_ledger import WORKLOAD_DONE

    case_id = "MARKET_DATA_V2_BOOTSTRAP_ACTIVATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    # Full-history completeness may not depend on one reference stock's observed
    # dates. These historically sparse/periodic datasets therefore use the
    # correctness-first per-instrument mode before activation is allowed.
    correctness_first = (
        "TaiwanStockHoldingSharesPer",
        "TaiwanStockFinancialStatements",
        "TaiwanStockBalanceSheet",
        "TaiwanStockCashFlowsStatement",
        "TaiwanStockMonthRevenue",
        "TaiwanStockMarketValueWeight",
    )
    add_check(
        results,
        "market_data",
        case_id,
        "periodic_full_history_does_not_depend_on_probe_stock_date_union",
        True,
        all(
            get_market_dataset_spec(dataset).bootstrap_mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE
            for dataset in correctness_first
        ),
    )
    add_check(
        results,
        "market_data",
        case_id,
        "current_registry_has_no_probe_date_union_bootstrap_dependency",
        0,
        sum(
            spec.bootstrap_mode == BOOTSTRAP_BULK_REFERENCE_DATES
            for spec in get_market_dataset_specs(included_only=True)
        ),
    )

    specs = get_market_dataset_specs(included_only=True)
    instruments = ("1101", "2330")
    evidence = {
        spec.dataset: DatasetProbeEvidence(
            dataset=spec.dataset,
            status="PASS",
            request_count=1,
            row_count=1,
            columns=("date", "stock_id"),
            observed_dates=(),
            earliest_date="2026-09-04",
            latest_date="2026-09-04",
        )
        for spec in specs
    }
    plan = build_bootstrap_request_plan(
        specs=specs,
        historical_instruments=instruments,
        evidence_by_dataset=evidence,
        as_of_date="2026-09-04",
        quota_limit=1600,
    )

    def _payload(*, status="READY", manifest_fingerprint=None):
        payload = {
            "schema_version": 2,
            "status": status,
            "generated_at": "2026-09-06T23:00:00+08:00",
            "as_of_date": "2026-09-04",
            "historical_instrument_count": len(instruments),
            "historical_instruments": list(instruments),
            "included_dataset_count": len(specs),
            "excluded_dataset_count": 0,
            "quota": {
                "user_count_before": 0,
                "user_count_after": 1,
                "api_request_limit": 1600,
                "observed_usage_delta": 1,
                "accounting_warning": None,
            },
            "probes": [asdict(evidence[spec.dataset]) for spec in specs],
            "probe_failures": [] if status == "READY" else [{"dataset": specs[0].dataset, "error": "synthetic"}],
            "plan_error": None,
            "plan": asdict(plan) if status == "READY" else None,
        }
        if manifest_fingerprint is not None and payload["plan"] is not None:
            payload["plan"]["manifest_fingerprint"] = manifest_fingerprint
        return payload

    with TemporaryDirectory() as td:
        root = Path(td)
        output_dir = root / "outputs" / "trading" / "smart_downloader" / "market_data_v2"
        output_dir.mkdir(parents=True)
        older = output_dir / "market_data_v2_preflight_plan_20260906_220000.json"
        newer = output_dir / "market_data_v2_preflight_plan_20260906_230000.json"
        older.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
        newer.write_text(json.dumps(_payload(status="BLOCKED"), ensure_ascii=False), encoding="utf-8")
        try:
            prepare_market_data_v2_bootstrap_activation(output_dir=output_dir)
        except MarketDataBootstrapActivationError:
            newest_blocked_stops = True
        else:
            newest_blocked_stops = False
        add_check(results, "market_data", case_id, "newest_blocked_preflight_cannot_fall_back_to_older_ready", True, newest_blocked_stops)

        newer.write_text(json.dumps(_payload(manifest_fingerprint="0" * 64), ensure_ascii=False), encoding="utf-8")
        try:
            prepare_market_data_v2_bootstrap_activation(output_dir=output_dir)
        except MarketDataBootstrapActivationError:
            drift_blocks = True
        else:
            drift_blocks = False
        add_check(results, "market_data", case_id, "manifest_fingerprint_drift_blocks_activation", True, drift_blocks)

        newer.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
        activation = prepare_market_data_v2_bootstrap_activation(output_dir=output_dir)
        add_check(results, "market_data", case_id, "ready_preflight_rederives_exact_manifest", plan.manifest_fingerprint, activation.manifest.manifest_fingerprint)
        add_check(results, "market_data", case_id, "ready_preflight_request_count_matches_plan", plan.total_requests, activation.planned_total_requests)
        add_check(results, "market_data", case_id, "no_ledger_is_created_by_read_only_activation_prepare", None, get_existing_bootstrap_summary(project_root=root, activation=activation))

        class _Client:
            def __init__(self):
                self.data_request_count = 0
                self.usage_request_count = 0

            def get_usage(self):
                self.usage_request_count += 1
                return FinMindUsage(user_count=0, api_request_limit=10000)

            def get_data(self, *, dataset, data_id=None, start_date=None, end_date=None):
                self.data_request_count += 1
                return pd.DataFrame(
                    [{"date": end_date or start_date or "2026-09-04", "stock_id": data_id or "ALL"}]
                )

        class _Sink:
            def recover_committed(self, _request):
                return None

            def __call__(self, _request, frame):
                return MarketDataCommitReceipt(committed=True, row_count=len(frame), content_sha256="a" * 64)

        policy = MarketDataExecutionPolicy(
            quota_reserve_requests=1,
            quota_refresh_every_requests=9999,
            quota_poll_seconds=1.0,
            max_retryable_attempts=2,
            retry_backoff_seconds=(1.0,),
            job_lease_seconds=5.0,
            executor_lock_seconds=5.0,
            progress_every_committed_requests=10,
        )
        fixed_now = lambda: datetime(2026, 9, 6, 23, 30, tzinfo=timezone.utc)
        class _RuntimeBlockedSink:
            def validate_activation_readiness(self):
                raise MarketDataCommitError("synthetic missing parquet runtime")

        blocked_client = _Client()
        try:
            execute_market_data_v2_bootstrap(
                activation=activation,
                token="synthetic",
                project_root=root / "runtime_blocked",
                output_dir=output_dir,
                client=blocked_client,
                sink=_RuntimeBlockedSink(),
                policy=policy,
                now_fn=fixed_now,
                sleep_fn=lambda _seconds: None,
            )
        except MarketDataCommitError:
            startup_blocked = True
        else:
            startup_blocked = False
        add_check(results, "market_data", case_id, "storage_runtime_gate_blocks_before_any_data_http", True, startup_blocked and blocked_client.data_request_count == 0)

        progress = []
        client = _Client()
        result = execute_market_data_v2_bootstrap(
            activation=activation,
            token="synthetic",
            project_root=root,
            output_dir=output_dir,
            client=client,
            sink=_Sink(),
            policy=policy,
            now_fn=fixed_now,
            sleep_fn=lambda _seconds: None,
            progress_fn=progress.append,
        )
        add_check(results, "market_data", case_id, "explicit_activation_executes_manifest_to_done", WORKLOAD_DONE, result["status"])
        add_check(results, "market_data", case_id, "activation_done_count_equals_exact_plan", plan.total_requests, result["done"])
        add_check(results, "market_data", case_id, "activation_first_run_data_http_equals_logical_requests", plan.total_requests, client.data_request_count)
        add_check(results, "market_data", case_id, "activation_emits_bounded_progress_events", True, 0 < len(progress) <= (plan.total_requests // 10 + 1))
        add_check(results, "market_data", case_id, "activation_status_paths_are_project_relative", True, not str(result["archive_dir"]).startswith(str(root)))

        resumed = get_existing_bootstrap_summary(project_root=root, activation=activation)
        add_check(results, "market_data", case_id, "completed_ledger_is_visible_for_resume_preview", plan.total_requests, resumed.done if resumed else -1)
        second_client = _Client()
        second = execute_market_data_v2_bootstrap(
            activation=activation,
            token="synthetic",
            project_root=root,
            output_dir=output_dir,
            client=second_client,
            sink=_Sink(),
            policy=policy,
            now_fn=fixed_now,
            sleep_fn=lambda _seconds: None,
        )
        add_check(results, "market_data", case_id, "completed_resume_stays_done", WORKLOAD_DONE, second["status"])
        add_check(results, "market_data", case_id, "completed_resume_uses_zero_data_http", 0, second_client.data_request_count)

    summary.update(
        {
            "checks": len(results),
            "synthetic_exact_requests": plan.total_requests,
            "per_instrument_dataset_count": plan.per_instrument_dataset_count,
        }
    )
    return results, summary


__all__ = [
    "validate_market_data_v2_preflight_planner_contract_case",
    "validate_market_data_v2_resumable_executor_contract_case",
    "validate_market_data_v2_parquet_storage_contract_case",
    "validate_market_data_v2_bootstrap_activation_contract_case",
]

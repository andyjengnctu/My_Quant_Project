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


__all__ = [
    "validate_market_data_v2_preflight_planner_contract_case",
    "validate_market_data_v2_resumable_executor_contract_case",
]

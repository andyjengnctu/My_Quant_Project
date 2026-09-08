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
    from core.market_data_instrument_universe import historical_stock_etf_universe_contract_fingerprint
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
            {"stock_id": "2330", "type": "twse", "industry_category": "半導體業"},
            {"stock_id": "0050", "type": "twse", "industry_category": "ETF"},
            {"stock_id": "6488", "type": "tpex", "industry_category": "半導體業"},
            {"stock_id": "7777", "type": "emerging", "industry_category": "其他"},
            {"stock_id": "TAIEX", "type": "twse", "industry_category": "大盤"},
            {"stock_id": "TPEx", "type": "tpex", "industry_category": "Index"},
            {"stock_id": "02001L", "type": "twse", "industry_category": "ETN"},
            {"stock_id": "ALL", "type": "twse", "industry_category": "所有證券"},
        ]
    )
    delisting = pd.DataFrame([{"stock_id": "1204"}])
    instruments = _historical_instruments(stock_info, delisting)
    add_check(results, "market_data", case_id, "historical_universe_keeps_stock_etf_and_delisting", ("0050", "1204", "2330", "6488"), instruments)
    add_check(results, "market_data", case_id, "historical_universe_excludes_index_aggregate_and_etn_rows", True, all(value not in instruments for value in ("TAIEX", "TPEx", "02001L", "ALL")))
    add_check(results, "market_data", case_id, "historical_universe_contract_has_stable_fingerprint", 64, len(historical_stock_etf_universe_contract_fingerprint()))

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
            if spec.bootstrap_chunk_months > 0:
                start = max("1900-01-01", str(spec.bootstrap_start_date or "1900-01-01"))
                sy, sm = int(start[:4]), int(start[5:7])
                ey, em = 2020, 12
                total_months = (ey - sy) * 12 + (em - sm) + 1
                chunk_count = (total_months + spec.bootstrap_chunk_months - 1) // spec.bootstrap_chunk_months
                expected_total += len(spec.fixed_data_ids) * chunk_count
            elif spec.bootstrap_chunk_years > 0:
                start_year = int(max("1900-01-01", str(spec.bootstrap_start_date or "1900-01-01"))[:4])
                end_year = 2020
                chunk_count = (end_year - start_year) // spec.bootstrap_chunk_years + 1
                expected_total += len(spec.fixed_data_ids) * chunk_count
            else:
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

    from datetime import date, datetime, timedelta, timezone
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
        quota_resume_headroom_requests=1,
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
    quota_wait_events = []
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
            quota_wait_observer=quota_wait_events.append,
        )
        execution_summary = executor.run(manifest=manifest, sink=_sink)
        add_check(results, "market_data", case_id, "quota_402_waits_and_resumes_to_done", WORKLOAD_DONE, execution_summary.workload_status)
        add_check(results, "market_data", case_id, "quota_resume_commits_every_logical_request", manifest.total_requests, execution_summary.done)
        add_check(results, "market_data", case_id, "quota_402_is_counted_as_actual_http_attempt", manifest.total_requests + 1, execution_summary.http_attempts)
        add_check(results, "market_data", case_id, "quota_wait_uses_injected_polling", True, bool(clock.sleeps))
        add_check(results, "market_data", case_id, "quota_wait_emits_heartbeat_each_poll", True, len(quota_wait_events) >= 2)
        add_check(results, "market_data", case_id, "quota_wait_heartbeat_does_not_add_usage_refresh", 4, quota_client.usage_request_count)
        last_wait = quota_wait_events[-1]
        add_check(results, "market_data", case_id, "quota_wait_heartbeat_uses_live_limit", 4, last_wait.get("quota_limit"))
        add_check(results, "market_data", case_id, "quota_wait_heartbeat_reports_stop_threshold", 2, last_wait.get("quota_safe_used_max"))
        add_check(results, "market_data", case_id, "quota_wait_heartbeat_reports_resume_headroom", 1, last_wait.get("quota_resume_headroom"))
        add_check(results, "market_data", case_id, "quota_wait_heartbeat_reports_resume_threshold", 2, last_wait.get("quota_resume_used_max"))
        add_check(results, "market_data", case_id, "quota_wait_heartbeat_reports_required_drop", 2, last_wait.get("quota_needed_drop"))
        add_check(results, "market_data", case_id, "quota_wait_heartbeat_accumulates_wait_time", True, float(last_wait.get("waited_seconds") or 0.0) >= policy.quota_poll_seconds)

        sponsor_policy = MarketDataExecutionPolicy(
            quota_reserve_requests=50,
            quota_resume_headroom_requests=500,
            quota_refresh_every_requests=25,
            quota_poll_seconds=30.0,
            max_retryable_attempts=4,
            retry_backoff_seconds=(5.0, 30.0, 120.0),
            job_lease_seconds=300.0,
            executor_lock_seconds=300.0,
            progress_every_committed_requests=100,
        )
        with TemporaryDirectory() as sponsor_td:
            sponsor_ledger = MarketDataJobLedger(Path(sponsor_td) / "sponsor_hysteresis.sqlite3")
            sponsor_executor = MarketDataBootstrapExecutor(
                ledger=sponsor_ledger,
                client=_SuccessClient() if "_SuccessClient" in locals() else quota_client,
                policy=sponsor_policy,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
                owner_id="sponsor-hysteresis-worker",
            )
            sponsor_executor._quota.usage = FinMindUsage(user_count=5968, api_request_limit=6000)
            add_check(results, "market_data", case_id, "quota_hysteresis_stop_threshold_still_uses_reserve", False, sponsor_executor._quota_has_capacity())
            add_check(results, "market_data", case_id, "quota_hysteresis_waits_until_500_safe_headroom", False, sponsor_executor._quota_has_resume_capacity())
            sponsor_executor._quota.usage = FinMindUsage(user_count=5450, api_request_limit=6000)
            add_check(results, "market_data", case_id, "quota_hysteresis_resumes_at_500_safe_headroom", True, sponsor_executor._quota_has_resume_capacity())

        from services.downloader.main import _estimate_bootstrap_eta_seconds

        early_resume_eta = _estimate_bootstrap_eta_seconds(
            done=5700,
            initial_done=5665,
            total=69360,
            elapsed_seconds=402.0,
            quota_wait_seconds=390.0,
            quota_limit=6000,
            quota_reserve=50,
            observed_sample_floor=100,
        )
        rolling_wait_eta = _estimate_bootstrap_eta_seconds(
            done=5800,
            initial_done=5665,
            total=69360,
            elapsed_seconds=886.0,
            quota_wait_seconds=826.0,
            quota_limit=6000,
            quota_reserve=50,
            observed_sample_floor=100,
        )
        add_check(results, "market_data", case_id, "eta_early_resume_uses_sustainable_quota_not_tiny_sample", True, early_resume_eta is not None and early_resume_eta < 12 * 3600)
        add_check(results, "market_data", case_id, "eta_excludes_quota_wait_from_active_throughput", True, rolling_wait_eta is not None and rolling_wait_eta < 12 * 3600)

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
        quota_snapshot = executor.quota_progress_snapshot()
        add_check(results, "market_data", case_id, "progress_quota_snapshot_uses_live_limit", 100, quota_snapshot["quota_limit"])
        add_check(results, "market_data", case_id, "progress_quota_snapshot_reports_effective_used", 1, quota_snapshot["quota_user_count"])
        add_check(results, "market_data", case_id, "progress_quota_snapshot_deducts_local_attempt", 99, quota_snapshot["quota_remaining"])
        add_check(results, "market_data", case_id, "progress_quota_snapshot_applies_reserve", 98, quota_snapshot["quota_usable_remaining"])

        from services.downloader.main import _estimate_bootstrap_eta_seconds, _format_bootstrap_duration
        eta_seconds = _estimate_bootstrap_eta_seconds(
            done=3900,
            initial_done=0,
            total=69360,
            elapsed_seconds=15 * 60,
            quota_limit=6000,
            quota_reserve=50,
        )
        add_check(results, "market_data", case_id, "bootstrap_eta_is_capped_by_safe_quota_rate", "11:00:06", _format_bootstrap_duration(eta_seconds))

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
        quota_resume_headroom_requests=1,
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
    from core.market_data_instrument_universe import historical_stock_etf_universe_contract_fingerprint
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
            "schema_version": 3,
            "status": status,
            "historical_universe_contract_fingerprint": historical_stock_etf_universe_contract_fingerprint(),
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

        stale_universe = _payload()
        stale_universe["historical_universe_contract_fingerprint"] = "0" * 64
        newer.write_text(json.dumps(stale_universe, ensure_ascii=False), encoding="utf-8")
        try:
            prepare_market_data_v2_bootstrap_activation(output_dir=output_dir)
        except MarketDataBootstrapActivationError:
            universe_drift_blocks = True
        else:
            universe_drift_blocks = False
        add_check(results, "market_data", case_id, "historical_universe_contract_drift_blocks_activation", True, universe_drift_blocks)

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
            quota_resume_headroom_requests=1,
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


def validate_market_data_v2_adjusted_price_representation_invariance_contract_case(_base_params):
    """Round-13 proves scoped price-representation invariance without authorizing raw levels or dataset scope."""

    import numpy as np
    import pandas as pd

    from core.market_data_adjusted_price_invariance import (
        ADJUSTED_PRICE_PROOF_STATUS,
        ADJUSTED_PRICE_REPRESENTATION_SCHEMA_VERSION,
        ADJUSTED_PRICE_VENDOR_CORRECTION_STATUS,
        PRICE_REPRESENTATION_STATUS_BLOCKED,
        PRICE_REPRESENTATION_STATUS_DEFERRED,
        PRICE_REPRESENTATION_STATUS_INVARIANT,
        adjusted_price_representation_contract_fingerprint,
        adjusted_price_representation_contract_payload,
        get_adjusted_price_representation_contract,
        validate_adjusted_price_representation_contract,
    )
    from core.market_data_research_pit_contract import (
        PIT_LEGALITY_STATUS_CURRENT_VINTAGE_BLOCKED,
        build_research_v2_pit_review_contracts,
        validate_research_v2_pit_review_contracts,
    )
    from core.market_data_research_v2 import (
        RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS,
        RESEARCH_V2_CANDIDATE_SCHEMA_VERSION,
    )
    from filters.breakout_quality.contract import BreakoutQualityLabelPolicy
    from filters.breakout_quality.features import (
        build_breakout_quality_context,
        build_candidate_event_positions,
        label_from_cached_path,
        normalize_ohlcv_array_window,
    )
    from core.market_data_research_scope import (
        RESEARCH_V2_ADJUSTED_PRICE_DATASET as SCOPE_ADJUSTED_PRICE_DATASET,
        build_research_v2_dataset_scope_contracts,
    )

    case_id = "MARKET_DATA_V2_ADJUSTED_PRICE_REPRESENTATION_INVARIANCE"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    contract = get_adjusted_price_representation_contract()
    stats = validate_adjusted_price_representation_contract(contract)
    payload = adjusted_price_representation_contract_payload(contract)
    rules = {row.representation_id: row for row in contract.rules}

    add_check(results, "market_data", case_id, "round13_adjusted_price_representation_contract_is_versioned", True, ADJUSTED_PRICE_REPRESENTATION_SCHEMA_VERSION >= 1)
    add_check(results, "market_data", case_id, "round13_corporate_action_restatement_proof_is_explicit", ADJUSTED_PRICE_PROOF_STATUS, contract.proof_status)
    add_check(results, "market_data", case_id, "round13_arbitrary_vendor_corrections_are_not_claimed_invariant", ADJUSTED_PRICE_VENDOR_CORRECTION_STATUS, contract.vendor_correction_status)
    add_check(results, "market_data", case_id, "round13_representation_fingerprint_is_deterministic", adjusted_price_representation_contract_fingerprint(contract), adjusted_price_representation_contract_fingerprint())
    add_check(results, "market_data", case_id, "round13_raw_absolute_price_levels_remain_blocked", PRICE_REPRESENTATION_STATUS_BLOCKED, rules["absolute_adjusted_price_level_v1"].status)
    add_check(results, "market_data", case_id, "round13_volume_field_is_outside_price_restatement_proof", PRICE_REPRESENTATION_STATUS_DEFERRED, rules["adjusted_dataset_volume_field_v1"].status)
    add_check(results, "market_data", case_id, "round13_relative_price_primitives_are_proven_invariant", True, all(rules[key].status == PRICE_REPRESENTATION_STATUS_INVARIANT for key in ("predecision_price_order_relations_v1", "anchor_relative_ohlc_v1", "relative_price_context_v1", "matured_anchor_relative_future_path_v1")))
    add_check(results, "market_data", case_id, "round13_never_authorizes_dataset_scope", False, bool(stats["scientific_input_authorized"]))
    add_check(results, "market_data", case_id, "round13_payload_keeps_proof_and_vendor_revision_scope_separate", True, payload["proof_status"] == ADJUSTED_PRICE_PROOF_STATUS and payload["vendor_correction_status"] == ADJUSTED_PRICE_VENDOR_CORRECTION_STATUS)

    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    base = pd.DataFrame(
        {
            "Open": [10.0, 10.5, 11.0, 11.0, 11.5, 12.0, 13.0, 13.0],
            "High": [10.5, 11.0, 11.5, 11.2, 12.0, 12.2, 13.5, 13.2],
            "Low": [9.8, 10.2, 10.8, 10.7, 11.2, 11.8, 12.5, 12.7],
            "Close": [10.2, 10.8, 11.2, 11.0, 11.8, 12.1, 13.2, 13.0],
            "Volume": [100, 110, 120, 130, 140, 150, 160, 170],
        },
        index=dates,
    )
    scale = 0.25
    restated = base.copy()
    restated.loc[:, ["Open", "High", "Low", "Close"]] *= scale
    base_events = build_candidate_event_positions(base, (3,))
    restated_events = build_candidate_event_positions(restated, (3,))
    event_signature = [(int(row["pos"]), int(row["high_len"])) for row in base_events]
    restated_signature = [(int(row["pos"]), int(row["high_len"])) for row in restated_events]
    add_check(results, "market_data", case_id, "predecision_candidate_membership_is_scale_invariant", event_signature, restated_signature)

    event = base_events[0]
    event_pos = int(event["pos"])
    feature_window = base.iloc[event_pos - 3:event_pos + 1][["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)
    restated_window = restated.iloc[event_pos - 3:event_pos + 1][["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)
    normalized = normalize_ohlcv_array_window(feature_window, float(base["Close"].iloc[event_pos]))
    normalized_restated = normalize_ohlcv_array_window(restated_window, float(restated["Close"].iloc[event_pos]))
    add_check(results, "market_data", case_id, "anchor_relative_ohlcv_feature_is_scale_invariant", True, bool(np.allclose(normalized, normalized_restated, rtol=0.0, atol=1e-7)))

    benchmark_scale = 0.4
    benchmark = feature_window.copy()
    benchmark[:, :4] *= 2.0
    benchmark_restated = benchmark.copy()
    benchmark_restated[:, :4] *= benchmark_scale
    benchmark_norm = normalize_ohlcv_array_window(benchmark, float(benchmark[-1, 3]))
    benchmark_norm_restated = normalize_ohlcv_array_window(benchmark_restated, float(benchmark_restated[-1, 3]))
    add_check(results, "market_data", case_id, "benchmark_price_representation_allows_independent_positive_scale", True, bool(np.allclose(benchmark_norm, benchmark_norm_restated, rtol=0.0, atol=1e-7)))

    policy = BreakoutQualityLabelPolicy(
        feature_window_bars=4,
        label_horizon_bars=3,
        label_path_cache_bars=3,
        high_len_values=(3, 4),
        min_mfe_return=0.05,
        min_reward_risk_ratio=2.0,
        max_adverse_return=-0.10,
    )
    context = build_breakout_quality_context(base, event_pos=event_pos, high_len=int(event["high_len"]), breakout_level=float(event["breakout_level"]), policy=policy)
    restated_context = build_breakout_quality_context(restated, event_pos=event_pos, high_len=int(restated_events[0]["high_len"]), breakout_level=float(restated_events[0]["breakout_level"]), policy=policy)
    add_check(results, "market_data", case_id, "relative_price_context_is_scale_invariant", True, bool(np.allclose(context, restated_context, rtol=0.0, atol=1e-7)))

    anchor = 100.0
    highs = np.asarray([106.0, 108.0, 109.0], dtype=np.float64)
    lows = np.asarray([99.0, 98.0, 97.0], dtype=np.float64)
    label = label_from_cached_path(highs, lows, anchor_price=anchor, available_bars=3, policy=policy)
    later_restatement_scale = 0.37
    label_restated = label_from_cached_path(highs * later_restatement_scale, lows * later_restatement_scale, anchor_price=anchor * later_restatement_scale, available_bars=3, policy=policy)
    label_signature = (label.label, label.reason, label.max_upside_return, label.max_downside_return, label.decision_mfe_return, label.decision_mae_return, label.decision_reward_risk_ratio, label.first_hit_bar)
    restated_label_signature = (label_restated.label, label_restated.reason, label_restated.max_upside_return, label_restated.max_downside_return, label_restated.decision_mfe_return, label_restated.decision_mae_return, label_restated.decision_reward_risk_ratio, label_restated.first_hit_bar)
    add_check(results, "market_data", case_id, "matured_ratio_target_and_hit_order_are_scale_invariant", True, bool(np.allclose(np.asarray(label_signature[2:7], dtype=float), np.asarray(restated_label_signature[2:7], dtype=float), equal_nan=True, rtol=0.0, atol=1e-12) and label_signature[:2] == restated_label_signature[:2] and label_signature[7] == restated_label_signature[7]))
    add_check(results, "market_data", case_id, "raw_absolute_anchor_level_is_not_invariant", True, anchor != anchor * later_restatement_scale)

    pit_contracts = build_research_v2_pit_review_contracts()
    pit_stats = validate_research_v2_pit_review_contracts(pit_contracts)
    adjusted = next(row for row in pit_contracts if row.dataset == "TaiwanStockPriceAdj")
    add_check(results, "market_data", case_id, "pit_review_binds_adjusted_price_representation_proof_identity", adjusted_price_representation_contract_fingerprint(), adjusted.representation_contract_fingerprint)
    add_check(results, "market_data", case_id, "pit_review_keeps_raw_current_vintage_level_fail_closed", PIT_LEGALITY_STATUS_CURRENT_VINTAGE_BLOCKED, adjusted.pit_legality_status)
    add_check(results, "market_data", case_id, "pit_review_exposes_proven_representation_status", ADJUSTED_PRICE_PROOF_STATUS, pit_stats["adjusted_price_representation_proof_status"])
    add_check(results, "market_data", case_id, "candidate_identity_pins_adjusted_price_representation_proof", True, RESEARCH_V2_CANDIDATE_SCHEMA_VERSION >= 5 and "adjusted_price_representation_contract_fingerprint" in RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS)

    scope_adjusted = next(
        row for row in build_research_v2_dataset_scope_contracts()
        if row.dataset == SCOPE_ADJUSTED_PRICE_DATASET
    )
    scope_direct_fields = {
        field
        for rule in scope_adjusted.field_authorizations
        if rule.direct_scientific_use_authorized
        for field in rule.fields
    }
    invariant_ids = {
        rule.representation_id for rule in contract.rules
        if rule.status == PRICE_REPRESENTATION_STATUS_INVARIANT
    }
    add_check(results, "market_data", case_id, "round13_proof_owner_remains_separate_from_later_scope_authorization", False, bool(stats["scientific_input_authorized"]))
    add_check(results, "market_data", case_id, "round14_scope_never_authorizes_raw_adjusted_price_fields_directly", set(), scope_direct_fields)
    add_check(results, "market_data", case_id, "round14_scope_uses_exactly_round13_invariant_representations", invariant_ids, set(scope_adjusted.authorized_representation_ids))

    summary.update({
        "checks": len(results),
        "proof_status": contract.proof_status,
        "contract_fingerprint": stats["contract_fingerprint"],
    })
    return results, summary


__all__ = [
    "validate_market_data_v2_preflight_planner_contract_case",
    "validate_market_data_v2_resumable_executor_contract_case",
    "validate_market_data_v2_parquet_storage_contract_case",
    "validate_market_data_v2_bootstrap_activation_contract_case",
    "validate_market_data_v2_provider_snapshot_completion_contract_case",
    "validate_market_data_v2_trading_workbench_sidecar_contract_case",
    "validate_market_data_v2_research_candidate_contract_case",
    "validate_market_data_v2_research_pit_review_contract_case",
    "validate_market_data_v2_research_required_cutoff_isolation_contract_case",
    "validate_market_data_v2_research_non_daily_pit_legality_contract_case",
    "validate_market_data_v2_adjusted_price_representation_invariance_contract_case",
    "validate_market_data_v2_research_scope_field_authorization_contract_case",
]

def validate_market_data_v2_provider_snapshot_completion_contract_case(_base_params):
    """Round-5 finalization publishes READY only after full local artifact verification."""

    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.file_integrity import canonical_json_sha256, load_json_strict
    from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest
    from core.market_data_storage_contract import MarketDataCommitReceipt, resolve_market_data_provider_snapshot_path
    from services.downloader.market_data_bootstrap_activation import MarketDataBootstrapActivation
    from services.downloader.market_data_bootstrap_completion import (
        MarketDataBootstrapCompletionError,
        finalize_market_data_v2_provider_snapshot,
    )
    from services.downloader.market_data_ledger import MarketDataJobLedger, WORKLOAD_DONE

    case_id = "MARKET_DATA_V2_PROVIDER_SNAPSHOT_COMPLETION"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    requests = (
        BootstrapHttpRequest("TaiwanStockPriceAdj", "per_instrument_full_range", "2330", "1900-01-01", "2026-09-07"),
        BootstrapHttpRequest("TaiwanStockPrice", "per_instrument_full_range", "2330", "1900-01-01", "2026-09-07"),
    )
    manifest = BootstrapRequestManifest(
        as_of_date="2026-09-07",
        full_range_start="1900-01-01",
        registry_fingerprint=canonical_json_sha256({"registry": "synthetic"}),
        manifest_fingerprint=canonical_json_sha256({"manifest": [request.request_id for request in requests]}),
        historical_instrument_count=1,
        requests=requests,
    )
    activation = MarketDataBootstrapActivation(
        preflight_path=Path("outputs/trading/smart_downloader/market_data_v2/synthetic_preflight.json"),
        manifest=manifest,
        preflight_quota_limit=1600,
        planned_total_requests=manifest.total_requests,
    )
    now = datetime(2026, 9, 7, 0, 30, tzinfo=timezone.utc)
    receipts = {
        requests[0].request_id: MarketDataCommitReceipt(True, 100, "a" * 64),
        requests[1].request_id: MarketDataCommitReceipt(True, 120, "b" * 64),
    }

    class _Storage:
        def __init__(self, values):
            self.values = values

        def recover_committed(self, request):
            return self.values.get(request.request_id)

    def _write_storage_manifest(root: Path):
        from core.file_integrity import atomic_write_json
        from core.market_data_storage_contract import (
            MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME,
            build_bootstrap_storage_manifest_payload,
            resolve_market_data_bootstrap_archive_dir,
        )
        from core.market_data_storage_policy import get_market_data_storage_policy

        archive_dir = resolve_market_data_bootstrap_archive_dir(root, manifest.manifest_fingerprint)
        archive_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            archive_dir / MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME,
            build_bootstrap_storage_manifest_payload(manifest, get_market_data_storage_policy()),
        )

    def _seed_done(root: Path):
        from core.market_data_storage_contract import resolve_market_data_bootstrap_ledger_path

        _write_storage_manifest(root)
        ledger = MarketDataJobLedger(
            resolve_market_data_bootstrap_ledger_path(root, manifest.manifest_fingerprint)
        )
        workload_id = ledger.seed_manifest(manifest, now=now)
        owner = "synthetic-finalizer"
        ledger.acquire_executor_lock(
            workload_id,
            owner_id=owner,
            now=now,
            lease_until=now + timedelta(minutes=5),
        )
        for request in requests:
            job = ledger.claim_next_job(
                workload_id,
                owner_id=owner,
                now=now,
                lease_until=now + timedelta(minutes=5),
            )
            if job is None or job.request_id != request.request_id:
                raise AssertionError("synthetic ledger claim order mismatch")
            receipt = receipts[request.request_id]
            ledger.mark_done(
                workload_id,
                request.request_id,
                row_count=receipt.row_count,
                content_sha256=receipt.content_sha256,
                now=now,
            )
        ledger.set_workload_status(workload_id, status=WORKLOAD_DONE, now=now)
        return ledger, workload_id

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _ledger, _workload_id = _seed_done(root)
        output_dir = root / "outputs" / "trading" / "smart_downloader" / "market_data_v2"
        progress = []
        result = finalize_market_data_v2_provider_snapshot(
            activation=activation,
            project_root=root,
            output_dir=output_dir,
            now_fn=lambda: now,
            storage=_Storage(receipts),
            progress_fn=lambda event: progress.append(dict(event)),
            progress_every=1,
        )
        snapshot_path = resolve_market_data_provider_snapshot_path(root, manifest.manifest_fingerprint)
        payload = load_json_strict(snapshot_path)
        add_check(results, "market_data", case_id, "done_archive_finalizes_ready_snapshot", "READY", result["status"])
        add_check(results, "market_data", case_id, "all_requests_verified_before_ready", manifest.total_requests, result["verified_requests"])
        add_check(results, "market_data", case_id, "provider_snapshot_total_rows_are_ledger_derived", 220, result["total_rows"])
        add_check(results, "market_data", case_id, "provider_snapshot_manifest_is_persisted", True, snapshot_path.is_file())
        add_check(results, "market_data", case_id, "provider_snapshot_identity_matches_report", result["snapshot_fingerprint"], payload["snapshot_fingerprint"])
        add_check(results, "market_data", case_id, "provider_snapshot_path_is_project_relative", False, str(result["provider_snapshot_path"]).startswith(str(root)))
        add_check(results, "market_data", case_id, "verification_progress_reaches_total", manifest.total_requests, progress[-1]["verified"])

        from core.market_data_storage_contract import (
            MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME,
            resolve_market_data_bootstrap_archive_dir,
        )
        bootstrap_manifest_path = resolve_market_data_bootstrap_archive_dir(
            root, manifest.manifest_fingerprint
        ) / MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME
        bootstrap_manifest_path.unlink()
        try:
            finalize_market_data_v2_provider_snapshot(
                activation=activation,
                project_root=root,
                output_dir=output_dir,
                now_fn=lambda: now,
                storage=_Storage(receipts),
            )
        except MarketDataBootstrapCompletionError:
            missing_storage_manifest_blocks = True
        else:
            missing_storage_manifest_blocks = False
        add_check(results, "market_data", case_id, "missing_storage_manifest_blocks_snapshot", True, missing_storage_manifest_blocks)
        _write_storage_manifest(root)

        reused = finalize_market_data_v2_provider_snapshot(
            activation=activation,
            project_root=root,
            output_dir=output_dir,
            now_fn=lambda: now + timedelta(minutes=1),
            storage=_Storage(receipts),
            progress_every=10,
        )
        add_check(results, "market_data", case_id, "immutable_snapshot_is_reused_after_reverification", True, reused["reused_existing_snapshot"])
        add_check(results, "market_data", case_id, "reverify_keeps_same_snapshot_fingerprint", result["snapshot_fingerprint"], reused["snapshot_fingerprint"])

        bad_receipts = dict(receipts)
        bad_receipts[requests[0].request_id] = MarketDataCommitReceipt(True, 100, "c" * 64)
        try:
            finalize_market_data_v2_provider_snapshot(
                activation=activation,
                project_root=root,
                output_dir=output_dir,
                now_fn=lambda: now,
                storage=_Storage(bad_receipts),
            )
        except MarketDataBootstrapCompletionError:
            sha_mismatch_blocks = True
        else:
            sha_mismatch_blocks = False
        add_check(results, "market_data", case_id, "artifact_sha_mismatch_blocks_snapshot", True, sha_mismatch_blocks)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        from core.market_data_storage_contract import resolve_market_data_bootstrap_ledger_path

        _write_storage_manifest(root)
        ledger = MarketDataJobLedger(
            resolve_market_data_bootstrap_ledger_path(root, manifest.manifest_fingerprint)
        )
        ledger.seed_manifest(manifest, now=now)
        try:
            finalize_market_data_v2_provider_snapshot(
                activation=activation,
                project_root=root,
                output_dir=root / "outputs",
                now_fn=lambda: now,
                storage=_Storage(receipts),
            )
        except MarketDataBootstrapCompletionError:
            incomplete_blocks = True
        else:
            incomplete_blocks = False
        add_check(results, "market_data", case_id, "unfinished_ledger_cannot_publish_provider_snapshot", True, incomplete_blocks)

    summary.update({"checks": len(results), "manifest_fingerprint": manifest.manifest_fingerprint})
    return results, summary


def validate_market_data_v2_trading_workbench_sidecar_contract_case(_base_params):
    """Round-6 Trading integration stays isolated, resumable and non-blocking for current rule-based execution."""

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.file_integrity import atomic_write_json, canonical_json_sha256
    from core.market_data_bootstrap_requests import build_registry_fingerprint
    from core.market_data_dataset_registry import (
        DAILY_PERIODIC_REPAIR,
        TRADING_QUERY_AUTO,
        get_market_dataset_specs,
        validate_market_dataset_registry,
    )
    from core.market_data_provider_snapshot import (
        MARKET_DATA_PROVIDER_NAME,
        MARKET_DATA_PROVIDER_SNAPSHOT_ROLE,
        MARKET_DATA_PROVIDER_SNAPSHOT_SCHEMA_VERSION,
    )
    from core.market_data_storage_contract import resolve_market_data_provider_snapshot_path
    from core.market_data_trading_storage_contract import (
        resolve_trading_market_data_v2_root,
        resolve_trading_market_data_v2_state_path,
    )
    from core.market_data_trading_sync import build_trading_sync_request_manifest
    from core.market_data_trading_sync_policy import get_market_data_trading_sync_policy
    from services.downloader.finmind_http import FinMindUsage
    from services.downloader.market_data_trading_sync import sync_market_data_v2_trading_archive

    case_id = "MARKET_DATA_V2_TRADING_WORKBENCH_SIDECAR"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    registry = validate_market_dataset_registry()
    specs = get_market_dataset_specs(included_only=True)
    from services.downloader.market_data_trading_storage import MarketDataTradingStorageSink
    observed_bounds = MarketDataTradingStorageSink._frame_date_bounds(
        pd.DataFrame({"date": ["2026-09-05", "2026-09-07", "bad"]})
    )
    add_check(results, "market_data", case_id, "trading_storage_observes_actual_frame_date_bounds", ("2026-09-05", "2026-09-07"), observed_bounds)
    periodic = tuple(spec for spec in specs if spec.daily_mode == DAILY_PERIODIC_REPAIR)
    add_check(results, "market_data", case_id, "registry_remains_valid_after_trading_query_contract", True, registry["included"] > 0)
    add_check(results, "market_data", case_id, "periodic_datasets_have_explicit_query_policy", True, all(spec.trading_query_mode != TRADING_QUERY_AUTO for spec in periodic))
    add_check(results, "market_data", case_id, "periodic_datasets_have_positive_lookback", True, all(spec.trading_lookback_periods > 0 for spec in periodic))

    registry_fp = build_registry_fingerprint(specs)
    provider = {
        "status": "READY",
        "snapshot_fingerprint": "a" * 64,
        "manifest_fingerprint": "b" * 64,
        "registry_fingerprint": registry_fp,
        "as_of_date": "2026-09-04",
        "historical_instrument_count": 3210,
    }
    policy = get_market_data_trading_sync_policy()
    manifest_a = build_trading_sync_request_manifest(
        specs=specs,
        provider_snapshot=provider,
        target_date="2026-09-07",
        previous_sync_date=None,
        policy=policy,
    )
    manifest_b = build_trading_sync_request_manifest(
        specs=specs,
        provider_snapshot=provider,
        target_date="2026-09-07",
        previous_sync_date=None,
        policy=policy,
    )
    add_check(results, "market_data", case_id, "trading_manifest_is_deterministic", manifest_a.manifest_fingerprint, manifest_b.manifest_fingerprint)
    add_check(results, "market_data", case_id, "trading_manifest_covers_every_included_dataset", {spec.dataset for spec in specs}, {request.dataset for request in manifest_a.requests})
    add_check(results, "market_data", case_id, "trading_manifest_never_expands_to_historical_stock_universe", True, all(request.data_id is None or request.data_id in next(spec for spec in specs if spec.dataset == request.dataset).fixed_data_ids for request in manifest_a.requests))
    add_check(results, "market_data", case_id, "trading_policy_is_non_blocking_for_current_execution", False, policy.execution_fail_closed)
    selective_manifest = build_trading_sync_request_manifest(
        specs=specs,
        provider_snapshot=provider,
        target_date="2026-09-07",
        previous_sync_date=None,
        policy=policy,
        selected_datasets={"TaiwanStockPrice", "TaiwanStockPER"},
        previous_ready_dates_by_dataset={"TaiwanStockPrice": "2026-09-04", "TaiwanStockPER": "2026-09-04"},
    )
    add_check(results, "market_data", case_id, "selective_due_manifest_keeps_full_registry_identity", manifest_a.registry_fingerprint, selective_manifest.registry_fingerprint)
    add_check(results, "market_data", case_id, "selective_due_manifest_only_contains_due_datasets", {"TaiwanStockPrice", "TaiwanStockPER"}, {request.dataset for request in selective_manifest.requests})
    add_check(results, "market_data", case_id, "selective_due_manifest_is_smaller_than_full_daily_manifest", True, 0 < selective_manifest.total_requests < manifest_a.total_requests)

    # Round 6 canonical provider fetch geometry.  Full-history current-vintage
    # PriceAdj semantics stay intact while one process-local cache is shared by
    # the execution CSV producer and the V2 sidecar.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from unittest.mock import patch
    from services.downloader.finmind_http import FinMindUsage
    from services.downloader.finmind_shared_client import SharedFinMindRequestClient
    from services.downloader.trading_price_refresh import (
        build_price_history_ranges,
        probe_latest_adjusted_price_market_date,
        refresh_trading_adjusted_price_dataset,
    )

    full_price_ranges = build_price_history_ranges(
        start_date="1990-01-01",
        end_date="2026-09-07",
        chunk_months=6,
    )
    add_check(results, "market_data", case_id, "canonical_price_history_uses_74_six_month_ranges", 74, len(full_price_ranges))
    add_check(results, "market_data", case_id, "canonical_price_dedup_normal_day_design_is_about_409_data_calls", 409, len(full_price_ranges) + manifest_a.total_requests - 7)

    class _SharedBaseClient:
        def __init__(self):
            self.data_request_count = 0
            self.usage_request_count = 0
            self.calls = []
        def get_usage(self):
            self.usage_request_count += 1
            return FinMindUsage(user_count=0, api_request_limit=6000)
        def get_data(self, **kwargs):
            self.data_request_count += 1
            self.calls.append(dict(kwargs))
            return pd.DataFrame({"date": ["2026-09-07"], "stock_id": ["2330"], "open": [1.0]})

    shared_base = _SharedBaseClient()
    shared = SharedFinMindRequestClient(shared_base)
    first = shared.get_data(dataset="Synthetic", start_date="2026-09-07", end_date="2026-09-07")
    second = shared.get_data(dataset="Synthetic", start_date="2026-09-07", end_date="2026-09-07")
    add_check(results, "market_data", case_id, "shared_finmind_identical_request_hits_provider_once", 1, shared_base.data_request_count)
    add_check(results, "market_data", case_id, "shared_finmind_identical_request_returns_same_payload", True, first.equals(second))
    seeded_base = _SharedBaseClient()
    seeded = SharedFinMindRequestClient(seeded_base)
    seeded.seed_data(dataset="SyntheticSeed", start_date="2026-09-07", end_date="2026-09-07", frame=first)
    seeded_result = seeded.get_data(dataset="SyntheticSeed", start_date="2026-09-07", end_date="2026-09-07")
    add_check(results, "market_data", case_id, "shared_finmind_seeded_exact_request_consumes_zero_provider_calls", 0, seeded_base.data_request_count)
    add_check(results, "market_data", case_id, "shared_finmind_seeded_exact_request_preserves_provider_payload", True, first.equals(seeded_result))

    from datetime import timezone
    from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest
    from core.market_data_execution_policy import get_market_data_execution_policy
    from core.market_data_storage_contract import MarketDataCommitReceipt
    from services.downloader.market_data_executor import MarketDataBootstrapExecutor
    from services.downloader.market_data_ledger import MarketDataJobLedger
    with TemporaryDirectory() as cache_executor_temp:
        executor_base = _SharedBaseClient()
        executor_shared = SharedFinMindRequestClient(executor_base)
        cached_request = BootstrapHttpRequest(
            "SyntheticExecutorCache", "synthetic", None, "2026-09-07", "2026-09-07"
        )
        executor_shared.seed_data(
            dataset=cached_request.dataset,
            start_date=cached_request.start_date,
            end_date=cached_request.end_date,
            frame=pd.DataFrame({"date": ["2026-09-07"], "value": [1]}),
        )
        cached_manifest = BootstrapRequestManifest(
            as_of_date="2026-09-07",
            full_range_start="2026-09-07",
            registry_fingerprint="1" * 64,
            manifest_fingerprint="2" * 64,
            historical_instrument_count=0,
            requests=(cached_request,),
        )
        cached_ledger = MarketDataJobLedger(Path(cache_executor_temp) / "ledger.sqlite3", workload_namespace="synthetic_cache")
        cached_executor = MarketDataBootstrapExecutor(
            ledger=cached_ledger,
            client=executor_shared,
            policy=get_market_data_execution_policy(),
            now_fn=lambda: datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc),
            sleep_fn=lambda _seconds: None,
            blocking_waits=False,
        )
        cached_summary = cached_executor.run(
            manifest=cached_manifest,
            sink=lambda _request, frame: MarketDataCommitReceipt(
                committed=True, row_count=len(frame), content_sha256="e" * 64
            ),
        )
    add_check(results, "market_data", case_id, "executor_cache_hit_consumes_zero_provider_data_calls", 0, executor_base.data_request_count)
    add_check(results, "market_data", case_id, "executor_cache_hit_does_not_increment_http_attempts", 0, cached_summary.http_attempts)
    add_check(results, "market_data", case_id, "executor_cache_hit_still_commits_logical_job", 1, cached_summary.done)

    class _BulkPriceBaseClient:
        def __init__(self, *, omit_current=()):
            self.data_request_count = 0
            self.usage_request_count = 0
            self.calls = []
            self.omit_current = set(omit_current)
        def get_usage(self):
            self.usage_request_count += 1
            return FinMindUsage(user_count=0, api_request_limit=6000)
        def get_data(self, **kwargs):
            self.data_request_count += 1
            self.calls.append(dict(kwargs))
            start = kwargs.get("start_date")
            end = kwargs.get("end_date")
            if kwargs.get("dataset") != "TaiwanStockPriceAdj" or kwargs.get("data_id") is not None:
                raise AssertionError(f"unexpected synthetic bulk request: {kwargs}")
            if start == "2026-07-01" and end == "2026-09-07":
                dates = ["2026-07-01", "2026-09-07"]
            elif start == "2026-01-01" and end == "2026-06-30":
                dates = ["2026-01-02", "2026-06-30"]
            else:
                raise AssertionError(f"unexpected synthetic date range: {start}~{end}")
            rows = []
            for d in dates:
                for sid, base in (("2330", 100.0), ("2317", 80.0)):
                    if d == "2026-09-07" and sid in self.omit_current:
                        continue
                    rows.append({
                        "date": d, "stock_id": sid, "open": base, "max": base + 2,
                        "min": base - 1, "close": base + 1, "Trading_Volume": 1000,
                    })
            return pd.DataFrame(rows)

    with TemporaryDirectory() as bulk_temp_dir:
        from services.downloader import runtime as downloader_runtime
        bulk_base = _BulkPriceBaseClient()
        bulk_shared = SharedFinMindRequestClient(bulk_base)
        with patch.object(downloader_runtime, "SAVE_DIR", str(Path(bulk_temp_dir) / "prices")), \
             patch.object(downloader_runtime, "OUTPUT_DIR", str(Path(bulk_temp_dir) / "outputs")), \
             patch.object(downloader_runtime, "PRICE_HISTORY_START_DATE", "2026-01-01"), \
             patch.object(downloader_runtime, "CANONICAL_PRICE_BULK_CHUNK_MONTHS", 6):
            probe = probe_latest_adjusted_price_market_date(
                client=bulk_shared,
                now=datetime(2026, 9, 7, 18, 0, tzinfo=ZoneInfo("Asia/Taipei")),
            )
            bulk_summary = refresh_trading_adjusted_price_dataset(
                ["2330", "2317"],
                "2026-09-07",
                client=bulk_shared,
                probe=probe,
                universe_tickers=["2330", "2317"],
                verbose=False,
            )
            csv_2330 = pd.read_csv(Path(downloader_runtime.SAVE_DIR) / "2330.csv", index_col=0)
            calls_before_exact = bulk_base.data_request_count
            exact_cached = bulk_shared.get_data(
                dataset="TaiwanStockPriceAdj",
                start_date="2026-09-07",
                end_date="2026-09-07",
            )
        add_check(results, "market_data", case_id, "bulk_price_probe_and_history_use_two_provider_data_calls", 2, bulk_base.data_request_count)
        add_check(results, "market_data", case_id, "bulk_price_refresh_uses_one_quota_capacity_probe", 1, bulk_base.usage_request_count)
        add_check(results, "market_data", case_id, "bulk_price_refresh_preserves_full_current_vintage_history", 4, len(csv_2330))
        add_check(results, "market_data", case_id, "bulk_price_refresh_reports_current_vintage_strategy", "full_market_range_current_vintage", bulk_summary.get("price_fetch_strategy"))
        add_check(results, "market_data", case_id, "bulk_price_target_exact_request_reuses_same_response", calls_before_exact, bulk_base.data_request_count)
        add_check(results, "market_data", case_id, "bulk_price_target_exact_cache_contains_full_market_rows", {"2317", "2330"}, set(exact_cached["stock_id"].astype(str)))
        add_check(results, "market_data", case_id, "bulk_price_historical_raw_chunk_is_not_retained_in_shared_cache", False, bulk_shared.has_cached_data(dataset="TaiwanStockPriceAdj", start_date="2026-01-01", end_date="2026-06-30"))
        add_check(results, "market_data", case_id, "bulk_price_reports_uncached_historical_fetch", 1, bulk_shared.snapshot().get("uncached_fetches"))

    with TemporaryDirectory() as suspended_temp_dir:
        from services.downloader import runtime as downloader_runtime
        suspended_base = _BulkPriceBaseClient(omit_current={"2317"})
        suspended_shared = SharedFinMindRequestClient(suspended_base)
        with patch.object(downloader_runtime, "SAVE_DIR", str(Path(suspended_temp_dir) / "prices")), \
             patch.object(downloader_runtime, "OUTPUT_DIR", str(Path(suspended_temp_dir) / "outputs")), \
             patch.object(downloader_runtime, "PRICE_HISTORY_START_DATE", "2026-01-01"), \
             patch.object(downloader_runtime, "CANONICAL_PRICE_BULK_CHUNK_MONTHS", 6):
            suspended_probe = probe_latest_adjusted_price_market_date(
                client=suspended_shared,
                now=datetime(2026, 9, 7, 18, 0, tzinfo=ZoneInfo("Asia/Taipei")),
            )
            suspended_summary = refresh_trading_adjusted_price_dataset(
                ["2330", "2317"],
                "2026-09-07",
                client=suspended_shared,
                probe=suspended_probe,
                universe_tickers=["2330", "2317"],
                verbose=False,
            )
            suspended_2317 = pd.read_csv(Path(downloader_runtime.SAVE_DIR) / "2317.csv", index_col=0)
        add_check(results, "market_data", case_id, "bulk_price_preserves_legacy_suspended_ticker_semantic", "2026-07-01", str(pd.Timestamp(suspended_2317.index[-1]).date()))
        add_check(results, "market_data", case_id, "bulk_price_reports_cached_universe_missing_target_row", 1, suspended_summary.get("current_universe_missing_price_row_count"))

    with TemporaryDirectory() as truncated_temp_dir:
        from services.downloader import runtime as downloader_runtime
        from services.downloader.trading_price_refresh import TradingBulkPriceUnsupported
        truncated_price_dir = Path(truncated_temp_dir) / "prices"
        truncated_price_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"Open": [90.0], "High": [91.0], "Low": [89.0], "Close": [90.5], "Volume": [100]},
            index=[pd.Timestamp("2026-03-01")],
        ).to_csv(truncated_price_dir / "2330.csv")
        truncated_original_bytes = (truncated_price_dir / "2330.csv").read_bytes()
        truncated_base = _BulkPriceBaseClient()
        truncated_shared = SharedFinMindRequestClient(truncated_base)
        with patch.object(downloader_runtime, "SAVE_DIR", str(truncated_price_dir)), \
             patch.object(downloader_runtime, "OUTPUT_DIR", str(Path(truncated_temp_dir) / "outputs")), \
             patch.object(downloader_runtime, "PRICE_HISTORY_START_DATE", "2026-01-01"), \
             patch.object(downloader_runtime, "CANONICAL_PRICE_BULK_CHUNK_MONTHS", 6):
            truncated_probe = probe_latest_adjusted_price_market_date(
                client=truncated_shared,
                now=datetime(2026, 9, 7, 18, 0, tzinfo=ZoneInfo("Asia/Taipei")),
            )
            try:
                refresh_trading_adjusted_price_dataset(
                    ["2330", "2317"],
                    "2026-09-07",
                    client=truncated_shared,
                    probe=truncated_probe,
                    universe_tickers=["2330", "2317"],
                    verbose=False,
                )
            except TradingBulkPriceUnsupported:
                truncated_rejected = True
            else:
                truncated_rejected = False
        add_check(results, "market_data", case_id, "bulk_price_rejects_history_that_drops_existing_legacy_dates", True, truncated_rejected)
        add_check(results, "market_data", case_id, "bulk_price_coverage_failure_publishes_no_partial_csv", truncated_original_bytes, (truncated_price_dir / "2330.csv").read_bytes())

    # Repair windows are minimum re-query horizons.  A long scheduler outage
    # must not leave a silent hole between the Provider Snapshot and the recent
    # seven-day repair window.
    long_gap_provider = {**provider, "as_of_date": "2026-07-31"}
    long_gap_manifest = build_trading_sync_request_manifest(
        specs=specs,
        provider_snapshot=long_gap_provider,
        target_date="2026-09-07",
        previous_sync_date=None,
        policy=policy,
        selected_datasets={"TaiwanStockPriceAdj"},
        previous_ready_dates_by_dataset={"TaiwanStockPriceAdj": "2026-08-01"},
    )
    long_gap_dates = tuple(request.start_date for request in long_gap_manifest.requests)
    add_check(results, "market_data", case_id, "recent_repair_long_gap_starts_after_last_ready_date", "2026-08-02", min(long_gap_dates))
    add_check(results, "market_data", case_id, "recent_repair_long_gap_remains_contiguous_through_target", "2026-09-07", max(long_gap_dates))
    add_check(results, "market_data", case_id, "recent_repair_long_gap_has_no_silent_calendar_hole", 37, len(long_gap_manifest.requests))

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        trading_root = resolve_trading_market_data_v2_root(root)
        state_path = resolve_trading_market_data_v2_state_path(root)
        add_check(results, "market_data", case_id, "trading_archive_is_under_trading_data_domain", True, str(trading_root).replace("\\", "/").endswith("data/trading/market_data_v2"))
        add_check(results, "market_data", case_id, "trading_state_is_under_trading_state_domain", True, str(state_path).replace("\\", "/").endswith("state/trading/market_data_v2/archive_state.json"))
        add_check(results, "market_data", case_id, "trading_overlay_does_not_share_neutral_bootstrap_root", False, "/data/market_data_v2/bootstrap/" in str(trading_root).replace("\\", "/"))

        class _NoCallClient:
            data_request_count = 0
            usage_request_count = 0

            def __getattr__(self, name):
                raise AssertionError(f"provider snapshot missing 時不應呼叫 FinMind client: {name}")

        no_provider = sync_market_data_v2_trading_archive(
            project_root=root,
            target_date="2026-09-07",
            token="",
            output_dir=root / "outputs",
            client=_NoCallClient(),
        )
        add_check(results, "market_data", case_id, "no_provider_snapshot_reports_not_bootstrapped", "NOT_BOOTSTRAPPED", no_provider["status"])
        add_check(results, "market_data", case_id, "no_provider_snapshot_consumes_zero_data_requests", 0, no_provider["request_count"])
        add_check(results, "market_data", case_id, "no_provider_snapshot_is_non_blocking", False, no_provider["execution_blocking"])

        provider_identity = {
            "schema_version": MARKET_DATA_PROVIDER_SNAPSHOT_SCHEMA_VERSION,
            "provider": MARKET_DATA_PROVIDER_NAME,
            "snapshot_role": MARKET_DATA_PROVIDER_SNAPSHOT_ROLE,
            "status": "READY",
            "as_of_date": "2026-09-04",
            "registry_fingerprint": registry_fp,
            "manifest_fingerprint": "b" * 64,
            "historical_instrument_count": 3210,
            "total_requests": 0,
            "total_rows": 0,
            "artifacts_fingerprint": "c" * 64,
            "datasets": [],
        }
        provider_payload = {
            **provider_identity,
            "snapshot_fingerprint": canonical_json_sha256(provider_identity),
            "finalized_at": "2026-09-07T00:00:00+00:00",
        }
        provider_path = resolve_market_data_provider_snapshot_path(root, "b" * 64)
        atomic_write_json(provider_path, provider_payload)

        class _QuotaFullClient:
            data_request_count = 0
            usage_request_count = 0

            def get_usage(self):
                self.usage_request_count += 1
                return FinMindUsage(user_count=1590, api_request_limit=1600)

            def get_data(self, **_kwargs):
                self.data_request_count += 1
                raise AssertionError("Trading V2 quota不足時不得等待後再發data request")

        class _NoCommitSink:
            def validate_activation_readiness(self):
                return None

            def recover_committed(self, _request):
                return None

            def __call__(self, _request, _frame):
                raise AssertionError("Trading V2 quota不足時不得進storage commit")

        quota_client = _QuotaFullClient()
        sleep_calls = []
        quota_deferred = sync_market_data_v2_trading_archive(
            project_root=root,
            target_date="2026-09-07",
            token="synthetic-token",
            output_dir=root / "outputs",
            client=quota_client,
            sink=_NoCommitSink(),
            sleep_fn=lambda seconds: sleep_calls.append(float(seconds)),
        )
        add_check(results, "market_data", case_id, "quota_wait_returns_stale_without_sleeping_workbench", "STALE", quota_deferred["status"])
        add_check(results, "market_data", case_id, "quota_wait_consumes_zero_data_requests", 0, quota_client.data_request_count)
        add_check(results, "market_data", case_id, "quota_wait_does_not_sleep_in_workbench_sidecar", [], sleep_calls)
        add_check(results, "market_data", case_id, "quota_wait_does_not_advance_done_jobs", 0, quota_deferred["done"])
        add_check(results, "market_data", case_id, "quota_wait_is_reported_as_incomplete_sidecar", True, "WAIT_QUOTA" in str(quota_deferred.get("error") or ""))
        add_check(results, "market_data", case_id, "trading_sync_exposes_last_observed_quota_used", 1590, quota_deferred.get("quota_user_count"))
        add_check(results, "market_data", case_id, "trading_sync_exposes_last_observed_quota_limit", 1600, quota_deferred.get("quota_limit"))
        from services.trading.market_data_v2_state import build_trading_market_data_v2_read_model
        quota_read_model = build_trading_market_data_v2_read_model(root)
        add_check(results, "market_data", case_id, "trading_v2_state_persists_last_observed_quota_used", 1590, quota_read_model.get("quota_user_count"))
        add_check(results, "market_data", case_id, "trading_v2_state_persists_last_observed_quota_limit", 1600, quota_read_model.get("quota_limit"))

    from core.market_data_freshness_contract import (
        CADENCE_CURRENT_VINTAGE,
        CADENCE_EVENT_DRIVEN,
        CADENCE_PERIODIC,
        COMPLETENESS_EVENT_NO_ROW_VALID,
        EXPECTED_DATE_NONE,
        EXPECTED_DATE_PERIOD_DUE,
        FRESHNESS_STATUSES,
        ROW_EXPECTATION_OPTIONAL,
        get_market_data_freshness_contracts,
        validate_market_data_freshness_contracts,
    )

    freshness_contracts = get_market_data_freshness_contracts(specs=specs)
    freshness_stats = validate_market_data_freshness_contracts(
        specs=specs,
        contracts=freshness_contracts,
    )
    by_dataset = {item.dataset: item for item in freshness_contracts}
    static_contracts = [item for item in freshness_contracts if item.cadence == CADENCE_CURRENT_VINTAGE]
    event_contracts = [item for item in freshness_contracts if item.cadence == CADENCE_EVENT_DRIVEN]
    periodic_contracts = [item for item in freshness_contracts if item.cadence == CADENCE_PERIODIC]
    add_check(results, "market_data", case_id, "freshness_contract_covers_all_included_datasets", len(specs), freshness_stats["contract_count"])
    add_check(results, "market_data", case_id, "freshness_contract_dataset_identity_matches_registry", {spec.dataset for spec in specs}, set(by_dataset))
    add_check(results, "market_data", case_id, "freshness_status_vocabulary_is_stable", {"READY", "DUE", "WAIT_PUBLISH", "WAIT_QUOTA", "STALE", "ERROR", "BLOCKED", "NOT_APPLICABLE"}, set(FRESHNESS_STATUSES))
    add_check(results, "market_data", case_id, "event_datasets_allow_legal_no_row_window", True, bool(event_contracts) and all(item.row_expectation == ROW_EXPECTATION_OPTIONAL and item.completeness_mode == COMPLETENESS_EVENT_NO_ROW_VALID and item.expected_date_mode == EXPECTED_DATE_NONE for item in event_contracts))
    add_check(results, "market_data", case_id, "static_datasets_use_current_vintage_semantics", True, bool(static_contracts) and all(item.expected_date_mode == EXPECTED_DATE_NONE for item in static_contracts))
    add_check(results, "market_data", case_id, "periodic_datasets_use_due_period_semantics", True, bool(periodic_contracts) and all(item.expected_date_mode == EXPECTED_DATE_PERIOD_DUE for item in periodic_contracts))
    add_check(results, "market_data", case_id, "publication_schedule_partition_is_complete", len(specs), freshness_stats["verified_schedule_count"] + freshness_stats["fallback_schedule_count"])
    add_check(results, "market_data", case_id, "provider_verified_publication_schedule_exists", True, freshness_stats["verified_schedule_count"] > 0)
    add_check(results, "market_data", case_id, "unknown_publication_times_use_conservative_fallback", True, freshness_stats["fallback_schedule_count"] > 0 and all(item.publication_schedule_source for item in freshness_contracts))
    add_check(results, "market_data", case_id, "documented_price_schedule_uses_grace", ("17:45", 0, True), (by_dataset["TaiwanStockPrice"].publication_first_check_time, by_dataset["TaiwanStockPrice"].publication_day_offset, by_dataset["TaiwanStockPrice"].publication_schedule_verified))
    add_check(results, "market_data", case_id, "documented_per_schedule_uses_grace", ("18:15", 0, True), (by_dataset["TaiwanStockPER"].publication_first_check_time, by_dataset["TaiwanStockPER"].publication_day_offset, by_dataset["TaiwanStockPER"].publication_schedule_verified))
    add_check(results, "market_data", case_id, "documented_day_trading_schedule_waits_for_close_values", ("21:45", 0, True), (by_dataset["TaiwanStockDayTrading"].publication_first_check_time, by_dataset["TaiwanStockDayTrading"].publication_day_offset, by_dataset["TaiwanStockDayTrading"].publication_schedule_verified))
    add_check(results, "market_data", case_id, "undocumented_schedule_uses_next_day_fallback", ("01:45", 1, False), (by_dataset["TaiwanStockHoldingSharesPer"].publication_first_check_time, by_dataset["TaiwanStockHoldingSharesPer"].publication_day_offset, by_dataset["TaiwanStockHoldingSharesPer"].publication_schedule_verified))

    from datetime import datetime
    from zoneinfo import ZoneInfo
    from core.market_data_due_planner import plan_market_data_due_datasets
    from core.market_data_trading_storage_contract import resolve_trading_market_data_v2_dataset_state_path
    from services.trading.market_data_dataset_state import (
        build_market_data_dataset_state_read_model,
        load_market_data_dataset_state,
        record_market_data_sync_success,
        refresh_market_data_due_state,
        schedule_market_data_auto_update_outcomes,
    )

    with TemporaryDirectory() as state_temp_dir:
        state_root = Path(state_temp_dir)
        state_path = resolve_trading_market_data_v2_dataset_state_path(state_root)
        state_now = datetime(2026, 9, 8, 2, 0, tzinfo=ZoneInfo("Asia/Taipei"))
        observations = {}
        for contract in freshness_contracts:
            if contract.cadence == CADENCE_EVENT_DRIVEN:
                observations[contract.dataset] = {
                    "row_count": 0,
                    "observed_min_date": None,
                    "observed_max_date": None,
                }
            elif contract.cadence == CADENCE_PERIODIC:
                observations[contract.dataset] = {
                    "row_count": 1,
                    "observed_min_date": "2026-06-30",
                    "observed_max_date": "2026-06-30",
                }
            elif contract.cadence == CADENCE_CURRENT_VINTAGE:
                observations[contract.dataset] = {
                    "row_count": 1,
                    "observed_min_date": None,
                    "observed_max_date": None,
                }
            else:
                observations[contract.dataset] = {
                    "row_count": 1,
                    "observed_min_date": "2026-09-07",
                    "observed_max_date": "2026-09-07",
                }
        dataset_state = record_market_data_sync_success(
            state_root,
            target_date="2026-09-07",
            finished_at=state_now,
            observations=observations,
        )
        add_check(results, "market_data", case_id, "dataset_state_is_persisted_under_trading_state_domain", True, str(state_path).replace("\\", "/").endswith("state/trading/market_data_v2/dataset_state.json") and state_path.is_file())
        add_check(results, "market_data", case_id, "dataset_state_covers_all_freshness_contracts", len(freshness_contracts), len(dataset_state["datasets"]))
        add_check(results, "market_data", case_id, "periodic_refresh_can_be_ready_with_older_real_data_date", "READY", dataset_state["datasets"]["TaiwanStockFinancialStatements"]["status"])
        add_check(results, "market_data", case_id, "periodic_state_preserves_actual_latest_data_date", "2026-06-30", dataset_state["datasets"]["TaiwanStockFinancialStatements"]["latest_data_date"])
        add_check(results, "market_data", case_id, "event_no_row_window_can_be_ready", "READY", dataset_state["datasets"]["TaiwanStockDelisting"]["status"])
        same_target_plan = plan_market_data_due_datasets(
            target_date="2026-09-07",
            now=datetime(2026, 9, 8, 2, 1, tzinfo=ZoneInfo("Asia/Taipei")),
            state=dataset_state,
            contracts=freshness_contracts,
        )
        add_check(results, "market_data", case_id, "fully_ready_same_target_has_zero_due_datasets", (), same_target_plan.due_datasets)
        add_check(results, "market_data", case_id, "fully_ready_same_target_requires_zero_provider_requests", False, same_target_plan.provider_requests_required)
        price_contract = (by_dataset["TaiwanStockPrice"],)
        before_price = plan_market_data_due_datasets(
            target_date="2026-09-08",
            now=datetime(2026, 9, 8, 17, 30, tzinfo=ZoneInfo("Asia/Taipei")),
            state=dataset_state,
            contracts=price_contract,
        )
        after_price = plan_market_data_due_datasets(
            target_date="2026-09-08",
            now=datetime(2026, 9, 8, 17, 50, tzinfo=ZoneInfo("Asia/Taipei")),
            state=dataset_state,
            contracts=price_contract,
        )
        add_check(results, "market_data", case_id, "due_planner_waits_until_dataset_publication_window", (), before_price.due_datasets)
        add_check(results, "market_data", case_id, "due_planner_releases_dataset_after_publication_window", ("TaiwanStockPrice",), after_price.due_datasets)
        read_model = build_market_data_dataset_state_read_model(state_root)
        add_check(results, "market_data", case_id, "dataset_state_read_model_joins_dynamic_state_with_contract", (True, len(freshness_contracts)), (read_model["state_ready"], read_model["dataset_count"]))
        price_read_row = next(row for row in read_model["datasets"] if row["dataset"] == "TaiwanStockPrice")
        add_check(results, "market_data", case_id, "dataset_read_model_exposes_publication_source", True, bool(price_read_row.get("publication_schedule_source")))
        add_check(results, "market_data", case_id, "dataset_read_model_exposes_completeness_semantics", by_dataset["TaiwanStockPrice"].completeness_mode, price_read_row.get("completeness_mode"))
        add_check(results, "market_data", case_id, "dataset_read_model_exposes_primary_key_hint", by_dataset["TaiwanStockPrice"].primary_key_hint, price_read_row.get("primary_key_hint"))
        from services.trading.market_data_ops import build_market_data_ops_read_model
        ops_model = build_market_data_ops_read_model(state_root, now=state_now)
        add_check(results, "market_data", case_id, "data_ops_local_refresh_uses_zero_provider_calls", 0, ops_model.get("provider_calls"))
        add_check(results, "market_data", case_id, "data_ops_local_read_model_exposes_scheduler_status", True, bool(ops_model.get("scheduler_registration_status")))
        add_check(results, "market_data", case_id, "data_ops_read_model_covers_all_51_datasets", len(freshness_contracts), ops_model.get("dataset_count"))
        loaded_state = load_market_data_dataset_state(state_root, required=True)
        add_check(results, "market_data", case_id, "dataset_state_round_trip_preserves_fingerprint", dataset_state["state_fingerprint"], loaded_state["state_fingerprint"])
        projected_plan, projected_state = refresh_market_data_due_state(
            state_root,
            target_date="2026-09-08",
            now=datetime(2026, 9, 8, 17, 30, tzinfo=ZoneInfo("Asia/Taipei")),
        )
        price_projection = projected_state["datasets"]["TaiwanStockPrice"]
        add_check(results, "market_data", case_id, "local_due_projection_persists_wait_publish_status", "WAIT_PUBLISH", price_projection["status"])
        add_check(results, "market_data", case_id, "local_due_projection_persists_next_check_at", "2026-09-08T17:45:00+08:00", price_projection["next_check_at"])
        add_check(results, "market_data", case_id, "local_due_projection_needs_no_provider_before_publication", False, "TaiwanStockPrice" in projected_plan.due_datasets)
        tampered_state = dict(projected_state)
        tampered_datasets = {key: dict(value) for key, value in projected_state["datasets"].items()}
        tampered_datasets["TaiwanStockPrice"]["status"] = "ERROR"
        tampered_state["datasets"] = tampered_datasets
        atomic_write_json(state_path, tampered_state)
        try:
            load_market_data_dataset_state(state_root, required=True)
        except ValueError:
            tamper_blocked = True
        else:
            tamper_blocked = False
        add_check(results, "market_data", case_id, "dataset_state_fingerprint_blocks_manual_tamper", True, tamper_blocked)

    with TemporaryDirectory() as partial_temp_dir:
        partial_state = record_market_data_sync_success(
            Path(partial_temp_dir),
            target_date="2026-09-07",
            finished_at=state_now,
            observations={
                "TaiwanStockPrice": {
                    "row_count": 1,
                    "observed_min_date": "2026-09-07",
                    "observed_max_date": "2026-09-07",
                }
            },
            attempted_datasets={"TaiwanStockPrice"},
        )
        add_check(results, "market_data", case_id, "partial_batch_only_advances_attempted_dataset", "READY", partial_state["datasets"]["TaiwanStockPrice"]["status"])
        add_check(results, "market_data", case_id, "partial_batch_does_not_advance_unattempted_dataset", "NOT_APPLICABLE", partial_state["datasets"]["TaiwanStockPER"]["status"])

    missing_obs = {dataset: dict(values) for dataset, values in observations.items()}
    missing_obs["TaiwanStockPrice"] = {"row_count": 1, "observed_min_date": "2026-09-06", "observed_max_date": "2026-09-06"}
    with TemporaryDirectory() as stale_temp_dir:
        stale_state = record_market_data_sync_success(
            Path(stale_temp_dir),
            target_date="2026-09-07",
            finished_at=state_now,
            observations=missing_obs,
        )
        add_check(results, "market_data", case_id, "required_daily_missing_target_evidence_waits_publish", "WAIT_PUBLISH", stale_state["datasets"]["TaiwanStockPrice"]["status"])
        add_check(results, "market_data", case_id, "required_daily_missing_target_does_not_advance_ready_target", None, stale_state["datasets"]["TaiwanStockPrice"]["last_ready_target_date"])
        from core.market_data_auto_update_policy import get_market_data_auto_update_policy
        auto_policy = get_market_data_auto_update_policy()
        retry_state = schedule_market_data_auto_update_outcomes(
            Path(stale_temp_dir),
            target_date="2026-09-07",
            now=state_now,
            publication_retry_minutes=auto_policy.publication_retry_minutes,
            max_publication_retries=auto_policy.max_publication_retries,
            quota_defer_minutes=auto_policy.quota_defer_minutes,
            error_defer_minutes=auto_policy.error_defer_minutes,
            wait_publish_datasets={"TaiwanStockPrice"},
        )
        add_check(results, "market_data", case_id, "publication_retry_uses_first_backoff_after_first_stale_attempt", "2026-09-08T02:15:00+08:00", retry_state["datasets"]["TaiwanStockPrice"]["next_check_at"])
        before_retry = plan_market_data_due_datasets(
            target_date="2026-09-07",
            now=datetime(2026, 9, 8, 2, 10, tzinfo=ZoneInfo("Asia/Taipei")),
            state=retry_state,
            contracts=(by_dataset["TaiwanStockPrice"],),
        )
        after_retry = plan_market_data_due_datasets(
            target_date="2026-09-07",
            now=datetime(2026, 9, 8, 2, 16, tzinfo=ZoneInfo("Asia/Taipei")),
            state=retry_state,
            contracts=(by_dataset["TaiwanStockPrice"],),
        )
        add_check(results, "market_data", case_id, "publication_retry_backoff_suppresses_early_provider_call", (), before_retry.due_datasets)
        add_check(results, "market_data", case_id, "publication_retry_becomes_due_after_next_check", ("TaiwanStockPrice",), after_retry.due_datasets)

    from services.trading.market_data_auto_update import run_trading_market_data_auto_update
    with TemporaryDirectory() as auto_no_due_dir:
        auto_root = Path(auto_no_due_dir)
        record_market_data_sync_success(
            auto_root,
            target_date="2026-09-07",
            finished_at=state_now,
            observations=observations,
        )
        class _AutoNoCallClient:
            data_request_count = 0
            usage_request_count = 0
            def __getattr__(self, name):
                raise AssertionError(f"NO_DUE 不得呼叫 provider: {name}")
        auto_no_due = run_trading_market_data_auto_update(
            project_root=auto_root,
            target_date="2026-09-07",
            client=_AutoNoCallClient(),
            now_fn=lambda: datetime(2026, 9, 8, 2, 5, tzinfo=ZoneInfo("Asia/Taipei")),
        )
        add_check(results, "market_data", case_id, "auto_updater_no_due_consumes_zero_data_requests", 0, auto_no_due["data_requests"])
        add_check(results, "market_data", case_id, "auto_updater_no_due_consumes_zero_usage_requests", 0, auto_no_due["usage_requests"])
        add_check(results, "market_data", case_id, "auto_updater_no_due_reports_provider_not_required", False, auto_no_due["provider_requests_required"])

    from unittest.mock import patch

    # Default scheduler mode must discover a newer execution target without
    # turning every 15-minute wake-up into a provider request.  Explicit
    # --target-date remains the deterministic test/recovery override above.
    from core.market_data_auto_update_policy import get_market_data_auto_update_policy
    from services.trading.market_data_market_date_discovery import (
        DISCOVERY_RESULT_NO_NEW_DATE,
        load_market_date_discovery_state,
        plan_market_date_discovery,
        record_market_date_probe_result,
    )
    discovery_policy = get_market_data_auto_update_policy()
    with TemporaryDirectory() as discovery_dir:
        discovery_root = Path(discovery_dir)
        before_window = plan_market_date_discovery(
            discovery_root,
            current_market_date="2026-09-07",
            now=datetime(2026, 9, 8, 4, 0, tzinfo=ZoneInfo("Asia/Taipei")),
            policy=discovery_policy,
        )
        add_check(results, "market_data", case_id, "market_date_discovery_before_price_publication_is_local_only_not_due", False, before_window["due"])
        add_check(results, "market_data", case_id, "market_date_discovery_first_probe_is_price_publication_window", "2026-09-08T17:45:00+08:00", before_window["next_probe_at"])
        due_window = plan_market_date_discovery(
            discovery_root,
            current_market_date="2026-09-07",
            now=datetime(2026, 9, 8, 17, 46, tzinfo=ZoneInfo("Asia/Taipei")),
            policy=discovery_policy,
        )
        add_check(results, "market_data", case_id, "market_date_discovery_becomes_due_after_price_publication_window", True, due_window["due"])
        deferred = record_market_date_probe_result(
            discovery_root,
            current_market_date="2026-09-07",
            observed_market_date="2026-09-07",
            now=datetime(2026, 9, 8, 17, 46, tzinfo=ZoneInfo("Asia/Taipei")),
            policy=discovery_policy,
            result=DISCOVERY_RESULT_NO_NEW_DATE,
        )
        add_check(results, "market_data", case_id, "market_date_discovery_no_new_date_uses_bounded_first_retry", "2026-09-08T18:01:00+08:00", deferred["next_probe_at"])
        add_check(results, "market_data", case_id, "market_date_discovery_state_is_persisted_and_fingerprinted", "NO_NEW_DATE", load_market_date_discovery_state(discovery_root, required=True)["last_probe_result"])

    with TemporaryDirectory() as scheduler_local_dir:
        scheduler_root = Path(scheduler_local_dir)
        record_market_data_sync_success(
            scheduler_root,
            target_date="2026-09-07",
            finished_at=state_now,
            observations=observations,
        )
        class _SchedulerNoCallClient:
            data_request_count = 0
            usage_request_count = 0
            def __getattr__(self, name):
                raise AssertionError(f"discovery 尚未 due 時不得呼叫 provider: {name}")
        with patch(
            "services.trading.market_data_auto_update.load_trading_market_data_snapshot",
            return_value={"market_date": "2026-09-07"},
        ):
            scheduler_local = run_trading_market_data_auto_update(
                project_root=scheduler_root,
                client=_SchedulerNoCallClient(),
                now_fn=lambda: datetime(2026, 9, 8, 4, 0, tzinfo=ZoneInfo("Asia/Taipei")),
            )
        add_check(results, "market_data", case_id, "default_auto_worker_pre_discovery_window_consumes_zero_data_requests", 0, scheduler_local["data_requests"])
        add_check(results, "market_data", case_id, "default_auto_worker_pre_discovery_window_consumes_zero_usage_requests", 0, scheduler_local["usage_requests"])
        add_check(results, "market_data", case_id, "default_auto_worker_exposes_next_market_date_probe", "2026-09-08T17:45:00+08:00", scheduler_local["market_date_discovery_next_check_at"])

    with TemporaryDirectory() as discovery_new_day_dir:
        discovery_root = Path(discovery_new_day_dir)
        # Pre-seed dataset state at the new target so this regression isolates
        # market-date discovery + canonical execution refresh from V2 due work.
        record_market_data_sync_success(
            discovery_root,
            target_date="2026-09-08",
            finished_at=state_now,
            observations={
                dataset: {
                    **dict(values),
                    "observed_min_date": (
                        "2026-09-08"
                        if by_dataset[dataset].expected_date_mode not in {"none", "period_due"}
                        else values.get("observed_min_date")
                    ),
                    "observed_max_date": (
                        "2026-09-08"
                        if by_dataset[dataset].expected_date_mode not in {"none", "period_due"}
                        else values.get("observed_max_date")
                    ),
                }
                for dataset, values in observations.items()
            },
        )
        class _DiscoveryBaseClient:
            def __init__(self):
                self.data_request_count = 0
                self.usage_request_count = 0
        discovery_base = _DiscoveryBaseClient()
        canonical_calls = []
        def _fake_probe(*, client, now):
            client.base_client.data_request_count += 1
            from services.downloader.trading_price_refresh import PriceRange, TradingPriceProbe
            return TradingPriceProbe(
                candidate_date="2026-09-08",
                market_date="2026-09-08",
                current_range=PriceRange("2026-07-01", "2026-09-08"),
                current_frame=pd.DataFrame(),
            )
        def _fake_canonical(**kwargs):
            canonical_calls.append(dict(kwargs))
            return {"market_date": "2026-09-08"}
        with (
            patch(
                "services.trading.market_data_auto_update.load_trading_market_data_snapshot",
                return_value={"market_date": "2026-09-07"},
            ),
            patch(
                "services.trading.market_data_auto_update.probe_latest_adjusted_price_market_date",
                side_effect=_fake_probe,
            ),
            patch(
                "services.trading.market_data_update.run_trading_market_data_update",
                side_effect=_fake_canonical,
            ),
        ):
            discovery_new_day = run_trading_market_data_auto_update(
                project_root=discovery_root,
                client=discovery_base,
                now_fn=lambda: datetime(2026, 9, 8, 17, 46, tzinfo=ZoneInfo("Asia/Taipei")),
            )
        add_check(results, "market_data", case_id, "default_auto_worker_discovers_new_completed_trading_day", "2026-09-08", discovery_new_day["target_date"])
        add_check(results, "market_data", case_id, "default_auto_worker_new_day_probe_costs_one_data_request_in_isolation", 1, discovery_new_day["data_requests"])
        add_check(results, "market_data", case_id, "default_auto_worker_new_day_runs_execution_update", True, discovery_new_day["execution_data_updated"])
        add_check(results, "market_data", case_id, "market_date_discovery_reuses_shared_client_for_canonical_execution_update", True, len(canonical_calls) == 1 and canonical_calls[0].get("provider_client") is not None)
        add_check(results, "market_data", case_id, "market_date_discovery_defers_full_v2_to_publication_due_planner", False, canonical_calls[0].get("sync_v2_archive"))

    with TemporaryDirectory() as auto_quota_dir:
        auto_root = Path(auto_quota_dir)
        provider_path = resolve_market_data_provider_snapshot_path(auto_root, "b" * 64)
        atomic_write_json(provider_path, provider_payload)
        quota_client = _QuotaFullClient()
        auto_sleep_calls = []
        auto_quota = run_trading_market_data_auto_update(
            project_root=auto_root,
            target_date="2026-09-07",
            client=quota_client,
            sink=_NoCommitSink(),
            now_fn=lambda: datetime(2026, 9, 8, 2, 0, tzinfo=ZoneInfo("Asia/Taipei")),
            sleep_fn=lambda seconds: auto_sleep_calls.append(float(seconds)),
        )
        quota_state = load_market_data_dataset_state(auto_root, required=True)
        add_check(results, "market_data", case_id, "auto_updater_quota_defer_uses_one_usage_probe", 1, quota_client.usage_request_count)
        add_check(results, "market_data", case_id, "auto_updater_quota_defer_consumes_zero_data_requests", 0, quota_client.data_request_count)
        add_check(results, "market_data", case_id, "auto_updater_quota_defer_never_sleeps_scheduler_worker", [], auto_sleep_calls)
        add_check(results, "market_data", case_id, "auto_updater_quota_defer_persists_wait_quota", "WAIT_QUOTA", quota_state["datasets"]["TaiwanStockPrice"]["status"])
        add_check(results, "market_data", case_id, "auto_updater_quota_defer_schedules_next_check", "2026-09-08T02:15:00+08:00", quota_state["datasets"]["TaiwanStockPrice"]["next_check_at"])

    with TemporaryDirectory() as auto_success_dir:
        auto_root = Path(auto_success_dir)
        provider_path = resolve_market_data_provider_snapshot_path(auto_root, "b" * 64)
        atomic_write_json(provider_path, provider_payload)
        ready_state = record_market_data_sync_success(
            auto_root,
            target_date="2026-09-07",
            finished_at=state_now,
            observations=observations,
        )
        from services.trading.market_data_dataset_state import publish_market_data_dataset_state
        reset_datasets = {key: dict(value) for key, value in ready_state["datasets"].items()}
        reset_row = reset_datasets["TaiwanStockTradingDate"]
        reset_row.update({
            "status": "NOT_APPLICABLE",
            "last_attempt_at": None,
            "last_attempt_target_date": None,
            "last_success_at": None,
            "last_success_target_date": None,
            "last_ready_at": None,
            "last_ready_target_date": None,
            "latest_data_date": None,
            "next_check_at": None,
        })
        publish_market_data_dataset_state(
            auto_root,
            {
                **{key: value for key, value in ready_state.items() if key not in {"state_fingerprint", "datasets", "updated_at"}},
                "updated_at": state_now.isoformat(),
                "datasets": reset_datasets,
            },
        )
        from services.downloader.market_data_executor import MarketDataCommitReceipt
        class _AutoSuccessClient:
            def __init__(self):
                self.data_request_count = 0
                self.usage_request_count = 0
            def get_usage(self):
                self.usage_request_count += 1
                return FinMindUsage(user_count=0, api_request_limit=1600)
            def get_data(self, **_kwargs):
                self.data_request_count += 1
                return pd.DataFrame({"date": ["2026-09-07"]})
        class _AutoSuccessSink:
            def __init__(self):
                self.calls = 0
            def validate_activation_readiness(self):
                return None
            def recover_committed(self, _request):
                return None
            def __call__(self, _request, frame):
                self.calls += 1
                return MarketDataCommitReceipt(committed=True, row_count=len(frame), content_sha256="d" * 64)
            def dataset_observations(self):
                return {
                    "TaiwanStockTradingDate": {
                        "request_count": self.calls,
                        "nonempty_request_count": self.calls,
                        "row_count": self.calls,
                        "observed_min_date": "2026-09-07",
                        "observed_max_date": "2026-09-07",
                    }
                }
        success_client = _AutoSuccessClient()
        success_result = run_trading_market_data_auto_update(
            project_root=auto_root,
            target_date="2026-09-07",
            client=success_client,
            sink=_AutoSuccessSink(),
            now_fn=lambda: datetime(2026, 9, 7, 18, 20, tzinfo=ZoneInfo("Asia/Taipei")),
        )
        success_state = load_market_data_dataset_state(auto_root, required=True)
        add_check(results, "market_data", case_id, "auto_updater_executes_only_single_due_dataset", ("TaiwanStockTradingDate",), success_result["due_datasets"])
        add_check(results, "market_data", case_id, "auto_updater_single_due_dataset_uses_one_data_request", 1, success_client.data_request_count)
        add_check(results, "market_data", case_id, "auto_updater_single_due_dataset_uses_one_usage_request", 1, success_client.usage_request_count)
        add_check(results, "market_data", case_id, "auto_updater_success_advances_dataset_ready", "READY", success_state["datasets"]["TaiwanStockTradingDate"]["status"])
        add_check(results, "market_data", case_id, "auto_updater_all_ready_rolls_archive_synced", "SYNCED", success_result["archive_status"])

    from services.trading.market_data_scheduler import (
        SCHEDULER_STATUS_DRIFTED_ENABLED,
        SCHEDULER_STATUS_INSTALLED_DISABLED,
        SCHEDULER_STATUS_INSTALLED_ENABLED,
        SCHEDULER_STATUS_NOT_INSTALLED,
        SCHEDULER_STATUS_UNSUPPORTED,
        build_market_data_scheduler_spec,
        get_market_data_scheduler_status,
        install_or_update_market_data_scheduler,
        remove_market_data_scheduler,
        set_market_data_scheduler_enabled,
    )
    scheduler_spec = build_market_data_scheduler_spec(Path.cwd())
    add_check(results, "market_data", case_id, "scheduler_spec_targets_canonical_one_shot_app", "apps/market_data_auto_update.py", scheduler_spec["app_path"])
    add_check(results, "market_data", case_id, "scheduler_spec_uses_hidden_powershell_launcher", "powershell.exe", scheduler_spec["execute"])
    unsupported_scheduler = get_market_data_scheduler_status(Path.cwd(), platform_name="posix")
    add_check(results, "market_data", case_id, "scheduler_non_windows_status_is_local_unsupported", SCHEDULER_STATUS_UNSUPPORTED, unsupported_scheduler["status"])

    fake_task = {"installed": False}
    def _fake_scheduler_runner(script, env):
        if "Register-ScheduledTask" in script:
            fake_task.clear()
            fake_task.update({
                "installed": True,
                "state": "Ready",
                "enabled": True,
                "execute": env["MQP_TASK_EXECUTE"],
                "arguments": env["MQP_TASK_ARGUMENTS"],
                "working_directory": env["MQP_WORKING_DIRECTORY"],
                "interval_minutes": int(env["MQP_WAKE_MINUTES"]),
                "logon_trigger": True,
                "next_run_at": "2026-09-08T08:15:00+08:00",
                "last_run_at": None,
                "last_task_result": 0,
                "missed_runs": 0,
            })
            return 0, "", ""
        if "Disable-ScheduledTask" in script:
            fake_task["enabled"] = False
            fake_task["state"] = "Disabled"
            return 0, "", ""
        if "Enable-ScheduledTask" in script:
            fake_task["enabled"] = True
            fake_task["state"] = "Ready"
            return 0, "", ""
        if "Unregister-ScheduledTask" in script:
            fake_task.clear()
            fake_task["installed"] = False
            return 0, "", ""
        if "ConvertTo-Json" in script:
            import json as _json
            return 0, _json.dumps(fake_task), ""
        raise AssertionError("unexpected scheduler PowerShell script")

    scheduler_missing = get_market_data_scheduler_status(
        Path.cwd(), runner=_fake_scheduler_runner, platform_name="nt"
    )
    add_check(results, "market_data", case_id, "scheduler_status_reports_not_installed_without_mutation", SCHEDULER_STATUS_NOT_INSTALLED, scheduler_missing["status"])
    scheduler_installed = install_or_update_market_data_scheduler(
        Path.cwd(), runner=_fake_scheduler_runner, platform_name="nt"
    )
    add_check(results, "market_data", case_id, "scheduler_install_registers_enabled_canonical_task", SCHEDULER_STATUS_INSTALLED_ENABLED, scheduler_installed["status"])
    add_check(results, "market_data", case_id, "scheduler_install_has_logon_trigger", True, scheduler_installed["logon_trigger"])
    add_check(results, "market_data", case_id, "scheduler_install_uses_configured_wake_interval", scheduler_spec["wake_minutes"], scheduler_installed["wake_minutes"])
    fake_task["interval_minutes"] = int(scheduler_spec["wake_minutes"]) + 1
    scheduler_drifted = get_market_data_scheduler_status(
        Path.cwd(), runner=_fake_scheduler_runner, platform_name="nt"
    )
    add_check(results, "market_data", case_id, "scheduler_status_detects_registration_drift", SCHEDULER_STATUS_DRIFTED_ENABLED, scheduler_drifted["status"])
    fake_task["interval_minutes"] = int(scheduler_spec["wake_minutes"])
    scheduler_disabled = set_market_data_scheduler_enabled(
        Path.cwd(), enabled=False, runner=_fake_scheduler_runner, platform_name="nt"
    )
    add_check(results, "market_data", case_id, "scheduler_can_be_disabled_without_deleting_task", SCHEDULER_STATUS_INSTALLED_DISABLED, scheduler_disabled["status"])
    scheduler_reenabled = set_market_data_scheduler_enabled(
        Path.cwd(), enabled=True, runner=_fake_scheduler_runner, platform_name="nt"
    )
    add_check(results, "market_data", case_id, "scheduler_can_be_reenabled", SCHEDULER_STATUS_INSTALLED_ENABLED, scheduler_reenabled["status"])
    scheduler_removed = remove_market_data_scheduler(
        Path.cwd(), runner=_fake_scheduler_runner, platform_name="nt"
    )
    add_check(results, "market_data", case_id, "scheduler_remove_only_removes_os_registration", SCHEDULER_STATUS_NOT_INSTALLED, scheduler_removed["status"])

    from core.trading_data_dependencies import (
        TradingDataDependencySpec,
        get_trading_data_dependency_spec,
        validate_trading_data_dependency_registry,
    )
    from services.trading import data_readiness as trading_data_readiness
    from services.trading.market_data_dataset_state import (
        VALIDATION_STATUS_NOT_EVALUATED,
        VALIDATION_STATUS_READY,
    )
    dependency_stats = validate_trading_data_dependency_registry()
    current_dependency = get_trading_data_dependency_spec("full_rule_based_no_dl")
    add_check(results, "market_data", case_id, "trading_dependency_registry_has_current_strategy", 1, dependency_stats["strategy_count"])
    add_check(results, "market_data", case_id, "current_rule_based_strategy_requires_execution_data", True, current_dependency.execution_market_data_required)
    add_check(results, "market_data", case_id, "current_rule_based_strategy_requires_no_v2_dataset", (), current_dependency.required_v2_datasets)
    current_ready = trading_data_readiness.build_trading_data_readiness_from_evidence(
        strategy_id="full_rule_based_no_dl",
        target_date="2026-09-07",
        execution_market_data_ready=True,
        dataset_state=None,
    )
    add_check(results, "market_data", case_id, "current_rule_based_strategy_is_not_blocked_by_optional_v2_state", True, current_ready["ready"])
    current_blocked = trading_data_readiness.build_trading_data_readiness_from_evidence(
        strategy_id="full_rule_based_no_dl",
        target_date="2026-09-07",
        execution_market_data_ready=False,
        execution_market_data_reason="execution missing",
        dataset_state=None,
    )
    add_check(results, "market_data", case_id, "current_rule_based_strategy_fails_closed_without_execution_data", False, current_blocked["ready"])

    future_dependency = TradingDataDependencySpec(
        strategy_id="synthetic_v2_strategy",
        execution_market_data_required=True,
        required_v2_datasets=("TaiwanStockPER",),
    )
    synthetic_v2_state = {
        "datasets": {
            "TaiwanStockPER": {
                "status": "READY",
                "last_ready_target_date": "2026-09-07",
                "last_attempt_target_date": "2026-09-07",
                "latest_data_date": "2026-09-07",
                "schema_status": VALIDATION_STATUS_READY,
                "coverage_status": VALIDATION_STATUS_NOT_EVALUATED,
            }
        }
    }
    from unittest.mock import patch as _patch
    with _patch.object(trading_data_readiness, "get_trading_data_dependency_spec", return_value=future_dependency):
        future_blocked = trading_data_readiness.build_trading_data_readiness_from_evidence(
            strategy_id="synthetic_v2_strategy",
            target_date="2026-09-07",
            execution_market_data_ready=True,
            dataset_state=synthetic_v2_state,
        )
        add_check(results, "market_data", case_id, "future_v2_dependency_fails_closed_when_completeness_not_evaluated", False, future_blocked["ready"])
        synthetic_v2_state["datasets"]["TaiwanStockPER"]["coverage_status"] = VALIDATION_STATUS_READY
        future_ready = trading_data_readiness.build_trading_data_readiness_from_evidence(
            strategy_id="synthetic_v2_strategy",
            target_date="2026-09-07",
            execution_market_data_ready=True,
            dataset_state=synthetic_v2_state,
        )
        add_check(results, "market_data", case_id, "future_v2_dependency_ready_only_after_fresh_schema_coverage_evidence", True, future_ready["ready"])
        synthetic_v2_state["datasets"]["TaiwanStockPER"]["status"] = "ERROR"
        future_error = trading_data_readiness.build_trading_data_readiness_from_evidence(
            strategy_id="synthetic_v2_strategy",
            target_date="2026-09-07",
            execution_market_data_ready=True,
            dataset_state=synthetic_v2_state,
        )
        add_check(results, "market_data", case_id, "future_v2_dependency_fails_closed_on_same_target_error", False, future_error["ready"])

    project_root = Path(__file__).resolve().parents[2]
    workflow_source = (project_root / "services" / "trading" / "daily_workflow.py").read_text(encoding="utf-8")
    update_source = (project_root / "services" / "trading" / "market_data_update.py").read_text(encoding="utf-8")
    downloader_source = (project_root / "services" / "downloader" / "main.py").read_text(encoding="utf-8")
    scanner_source = (project_root / "services" / "trading" / "scanner_state.py").read_text(encoding="utf-8")
    readiness_source = (project_root / "services" / "trading" / "data_readiness.py").read_text(encoding="utf-8")
    operations_source = (project_root / "services" / "trading" / "operations_status.py").read_text(encoding="utf-8")
    data_ops_source = (project_root / "services" / "trading" / "market_data_ops.py").read_text(encoding="utf-8")
    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    state_source = (project_root / "services" / "trading" / "market_data_v2_state.py").read_text(encoding="utf-8")
    from services.trading import daily_workflow as daily_workflow_module
    from services.trading import market_data_update as market_data_update_module

    legacy_pos = update_source.find("run_trading_dataset_update(")
    snapshot_pos = update_source.find("publish_trading_market_data_snapshot(")
    v2_pos = update_source.find("sync_market_data_v2_trading_archive(")
    add_check(results, "market_data", case_id, "canonical_update_owner_preserves_execution_then_snapshot_then_v2_order", True, 0 <= legacy_pos < snapshot_pos < v2_pos)
    executor_source = (project_root / "services" / "downloader" / "market_data_executor.py").read_text(encoding="utf-8")
    price_refresh_source = (project_root / "services" / "downloader" / "trading_price_refresh.py").read_text(encoding="utf-8")
    add_check(results, "market_data", case_id, "canonical_update_shares_one_provider_client_across_legacy_and_v2", True, "provider_client=shared_client" in update_source and "client=shared_client" in update_source and "SharedFinMindRequestClient" in update_source)
    add_check(results, "market_data", case_id, "executor_does_not_count_cache_hit_as_http_attempt", True, "will_issue_data_request" in executor_source and "if will_issue:" in executor_source)
    add_check(results, "market_data", case_id, "canonical_price_refresh_never_calculates_adjustment_locally", True, "TaiwanStockPriceAdj" in price_refresh_source and "adjusted-price calculator" in price_refresh_source and "full_market_range_current_vintage" in price_refresh_source)
    add_check(results, "market_data", case_id, "daily_workflow_reuses_canonical_market_data_update_owner", True, daily_workflow_module.run_trading_market_data_update is market_data_update_module.run_trading_market_data_update and "def run_trading_market_data_update" not in workflow_source)
    add_check(results, "market_data", case_id, "smart_downloader_reuses_canonical_market_data_update_owner", True, "from services.trading.market_data_update import run_trading_market_data_update" in downloader_source and "run_trading_market_data_update(project_root=PROJECT_ROOT)" in downloader_source)
    import importlib
    from unittest.mock import patch
    downloader_main = importlib.import_module("services.downloader.main")
    smart_calls = []
    with patch(
        "services.trading.market_data_update.run_trading_market_data_update",
        side_effect=lambda **kwargs: smart_calls.append(dict(kwargs)) or {"status": "READY"},
    ):
        smart_exit = downloader_main._run_trading_dataset_update()
    add_check(results, "market_data", case_id, "smart_downloader_runtime_calls_canonical_market_data_update_owner", 0, smart_exit)
    add_check(results, "market_data", case_id, "smart_downloader_runtime_passes_project_root_to_canonical_owner", [str(downloader_main.PROJECT_ROOT)], [str(item.get("project_root")) for item in smart_calls])
    add_check(results, "market_data", case_id, "sidecar_failure_is_persisted_without_advancing_execution_truth", True, "publish_trading_market_data_v2_failure(" in update_source and "last_attempt_target_date" in state_source)
    add_check(results, "market_data", case_id, "workbench_exposes_v2_archive_status", True, "V2 Archive" in panel_source)
    add_check(results, "market_data", case_id, "scanner_snapshot_exposes_sidecar_without_replacing_market_ready", True, "market_data_v2_archive_status" in scanner_source and '"market_data_ready": market_ready' in scanner_source)
    add_check(results, "market_data", case_id, "scanner_runtime_uses_canonical_trading_data_readiness_gate", True, "assert_trading_data_readiness" in scanner_source and "build_trading_data_readiness_for_execution_evidence" in scanner_source)
    add_check(results, "market_data", case_id, "operations_status_uses_strategy_data_readiness_not_aggregate_v2_synced", True, "trading_data_ready" in operations_source and "overall_v2_ready" not in operations_source)
    add_check(results, "market_data", case_id, "data_ops_reads_canonical_trading_data_readiness_without_provider_call", True, "build_trading_data_readiness" in data_ops_source and '"provider_calls": 0' in data_ops_source)
    add_check(results, "market_data", case_id, "readiness_gate_requires_schema_and_coverage_for_v2_dependencies", True, "schema_status" in readiness_source and "coverage_status" in readiness_source and "VALIDATION_STATUS_NOT_EVALUATED" not in readiness_source)

    summary.update({
        "checks": len(results),
        "trading_request_count": manifest_a.total_requests,
        "freshness_contract_count": freshness_stats["contract_count"],
        "verified_schedule_count": freshness_stats["verified_schedule_count"],
        "fallback_schedule_count": freshness_stats["fallback_schedule_count"],
    })
    return results, summary


def validate_market_data_v2_research_candidate_contract_case(_base_params):
    """Round-9 Research V2 stays pinned, PIT-audited and explicitly NOT_READY."""

    from datetime import date, datetime, timedelta, timezone
    from pathlib import Path
    import sqlite3
    from tempfile import TemporaryDirectory

    from config.market_data import ACTIVE_RESEARCH_DATA_GENERATION, RESEARCH_DATA_GENERATION_V1, RESEARCH_REQUIRED_CUTOFF
    from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
    from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest, build_registry_fingerprint
    from core.market_data_dataset_registry import (
        BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
        BOOTSTRAP_SINGLE_FULL_RANGE,
        BOOTSTRAP_SINGLE_NO_DATES,
        get_market_dataset_specs,
    )
    from core.market_data_provider_snapshot import ProviderArtifactEvidence, build_provider_snapshot_payload
    from core.market_data_research_storage_contract import (
        resolve_research_v2_candidate_manifest_path,
        resolve_research_v2_daily_universe_path,
    )
    from core.market_data_research_v2 import (
        RESEARCH_V2_CANDIDATE_STATUS_NOT_READY,
        RESEARCH_V2_DATASET_STATUS_CURRENT_VINTAGE_BLOCKED,
        build_daily_pit_universe,
        validate_research_v2_candidate_contract,
    )
    from core.market_data_storage_contract import (
        resolve_market_data_provider_snapshot_path,
        resolve_market_data_request_parquet_path,
    )
    from services.downloader.market_data_ledger import MarketDataJobLedger
    from services.research.market_data_v2 import (
        ResearchV2ProviderView,
        build_research_v2_candidate,
        load_research_v2_candidate,
    )
    from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive

    case_id = "MARKET_DATA_V2_RESEARCH_CANDIDATE"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}
    stats = validate_research_v2_candidate_contract()
    add_check(results, "market_data", case_id, "research_v2_registry_assesses_all_51_datasets", 51, stats["dataset_count"])
    add_check(results, "market_data", case_id, "research_v2_exact_candidate_count_is_registry_driven", 4, stats["exact_candidate_count"])
    add_check(results, "market_data", case_id, "research_v2_review_required_datasets_remain_unpromoted", 46, stats["review_required_count"])
    add_check(results, "market_data", case_id, "research_v2_current_vintage_adjusted_price_is_blocked", 1, stats["current_vintage_blocked_count"])

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        research_cutoff_date = date.fromisoformat(RESEARCH_REQUIRED_CUTOFF)
        before_cutoff = (research_cutoff_date - timedelta(days=1)).isoformat()
        research_cutoff = research_cutoff_date.isoformat()
        after_cutoff = (research_cutoff_date + timedelta(days=1)).isoformat()
        provider_as_of = (research_cutoff_date + timedelta(days=190)).isoformat()
        primitive_universe = build_daily_pit_universe(
            pd.DataFrame({
                "date": [before_cutoff, research_cutoff],
                "stock_id": ["2330", "2330"],
            }),
            historical_instruments=("2330",),
            transition_excluded_through={"2330": before_cutoff},
            trading_dates=(before_cutoff, research_cutoff),
            provider_as_of_date=research_cutoff,
        )
        add_check(
            results, "market_data", case_id,
            "canonical_daily_pit_universe_primitive_requires_date_specific_market_state",
            [(research_cutoff, "2330")],
            list(primitive_universe[["date", "stock_id"]].itertuples(index=False, name=None)),
        )
        manifest_fingerprint = "7" * 64
        registry_fingerprint = build_registry_fingerprint(get_market_dataset_specs(included_only=True))
        requests = (
            BootstrapHttpRequest("TaiwanStockTradingDate", BOOTSTRAP_SINGLE_NO_DATES, None, None, None),
            BootstrapHttpRequest("TaiwanStockInfo", BOOTSTRAP_SINGLE_NO_DATES, None, None, None),
            BootstrapHttpRequest("TaiwanStockDelisting", BOOTSTRAP_SINGLE_FULL_RANGE, None, "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPrice", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPrice", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "2330", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceAdj", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPER", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceLimit", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceLimit", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "2330", "1900-01-01", provider_as_of),
        )
        manifest = BootstrapRequestManifest(
            as_of_date=provider_as_of,
            full_range_start="1900-01-01",
            registry_fingerprint=registry_fingerprint,
            manifest_fingerprint=manifest_fingerprint,
            historical_instrument_count=2,
            requests=requests,
        )
        frame_by_request = {
            requests[0].request_id: pd.DataFrame({"date": [before_cutoff, research_cutoff, after_cutoff]}),
            requests[1].request_id: pd.DataFrame([
                {"date": provider_as_of, "stock_id": "0050", "type": "twse", "industry_category": "ETF"},
                {"date": before_cutoff, "stock_id": "2330", "type": "emerging", "industry_category": "半導體業"},
                {"date": provider_as_of, "stock_id": "2330", "type": "twse", "industry_category": "半導體業"},
            ]),
            requests[2].request_id: pd.DataFrame(columns=["date", "stock_id"]),
            requests[3].request_id: pd.DataFrame({"date": [before_cutoff, research_cutoff, after_cutoff], "stock_id": ["0050"] * 3, "Trading_Volume": [1000.0, 1100.0, 1200.0]}),
            requests[4].request_id: pd.DataFrame({"date": [before_cutoff, research_cutoff, after_cutoff], "stock_id": ["2330"] * 3, "Trading_Volume": [2000.0, 2100.0, 2200.0]}),
            requests[5].request_id: pd.DataFrame({"date": [before_cutoff], "stock_id": ["0050"], "open": [99.0], "max": [101.0], "min": [98.0], "close": [100.0]}),
            requests[6].request_id: pd.DataFrame({"date": [before_cutoff, research_cutoff, after_cutoff], "stock_id": ["0050"] * 3, "PER": [20.0, 21.0, 22.0]}),
            requests[7].request_id: pd.DataFrame({"date": [before_cutoff], "stock_id": ["0050"]}),
            requests[8].request_id: pd.DataFrame({"date": [before_cutoff, research_cutoff, after_cutoff], "stock_id": ["2330"] * 3}),
        }

        ledger_path = root / "data" / "market_data_v2" / "bootstrap" / manifest_fingerprint / "bootstrap_ledger.sqlite3"
        ledger = MarketDataJobLedger(ledger_path)
        workload_id = ledger.seed_manifest(manifest, now=datetime(2026, 9, 4, tzinfo=timezone.utc))
        evidence = []
        now = datetime(2026, 9, 4, tzinfo=timezone.utc)
        while True:
            job = ledger.claim_next_job(
                workload_id,
                owner_id="synthetic",
                now=now,
                lease_until=now + timedelta(minutes=5),
            )
            if job is None:
                break
            request = job.to_request()
            path = resolve_market_data_request_parquet_path(root, manifest_fingerprint, request)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((request.request_id + "\n").encode("utf-8"))
            digest = compute_file_sha256(path)
            frame = frame_by_request[request.request_id]
            ledger.mark_done(
                workload_id,
                request.request_id,
                row_count=len(frame),
                content_sha256=digest,
                now=now,
            )
            evidence.append(
                ProviderArtifactEvidence(
                    request_id=request.request_id,
                    dataset=request.dataset,
                    row_count=len(frame),
                    content_sha256=digest,
                )
            )
            now += timedelta(seconds=1)
        ledger.set_workload_status(workload_id, status="DONE", now=now)
        provider_payload = build_provider_snapshot_payload(
            manifest=manifest,
            artifacts=evidence,
            finalized_at="2026-09-04T00:00:00+00:00",
        )
        provider_path = resolve_market_data_provider_snapshot_path(root, manifest_fingerprint)
        atomic_write_json(provider_path, provider_payload)

        def frame_reader(path, columns):
            frame = frame_by_request[path.stem].copy()
            if columns:
                missing = [column for column in columns if column not in frame.columns]
                if missing:
                    raise ValueError(f"synthetic frame missing columns: {missing}")
                frame = frame.loc[:, list(columns)]
            return frame

        archive = load_ready_provider_snapshot_archive(root)
        add_check(results, "market_data", case_id, "research_provider_view_reuses_neutral_ready_snapshot", provider_payload["snapshot_fingerprint"], archive.snapshot_fingerprint)
        view = ResearchV2ProviderView(project_root=root, archive=archive, frame_reader=frame_reader)
        adjusted_blocked = False
        try:
            next(view.iter_dataset_frames("TaiwanStockPriceAdj", columns=("date", "stock_id")))
        except RuntimeError:
            adjusted_blocked = True
        add_check(results, "market_data", case_id, "current_vintage_adjusted_price_cannot_enter_exact_pit_audit", True, adjusted_blocked)
        adjusted_contract_audit_blocked = False
        try:
            next(view.iter_dataset_contract_audit_frames("TaiwanStockPriceAdj", columns=("date",)))
        except RuntimeError:
            adjusted_contract_audit_blocked = True
        add_check(results, "market_data", case_id, "current_vintage_adjusted_price_cannot_enter_review_contract_audit", True, adjusted_contract_audit_blocked)
        per_exact_blocked = False
        try:
            next(view.iter_dataset_frames("TaiwanStockPER", columns=("date",)))
        except RuntimeError:
            per_exact_blocked = True
        add_check(results, "market_data", case_id, "review_required_per_cannot_enter_exact_pit_reader", True, per_exact_blocked)
        per_audit_frame = next(view.iter_dataset_contract_audit_frames("TaiwanStockPER", columns=("date",)))
        add_check(results, "market_data", case_id, "review_required_per_can_enter_contract_audit_reader_only", 3, len(per_audit_frame))

        candidate = build_research_v2_candidate(
            root,
            frame_reader=frame_reader,
            now=datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc),
        )
        add_check(results, "market_data", case_id, "research_v2_candidate_never_promotes_itself", RESEARCH_V2_CANDIDATE_STATUS_NOT_READY, candidate["status"])
        add_check(results, "market_data", case_id, "research_v2_candidate_has_no_frozen_cutoff", None, candidate["frozen_cutoff"])
        add_check(results, "market_data", case_id, "research_v2_candidate_has_no_research_common_complete_cutoff", None, candidate["research_common_complete_cutoff"])
        add_check(results, "market_data", case_id, "research_v2_candidate_keeps_active_research_v1", RESEARCH_DATA_GENERATION_V1, candidate["active_research_generation"])
        add_check(results, "market_data", case_id, "research_v2_candidate_build_consumes_zero_provider_calls", 0, candidate["provider_calls"])
        add_check(results, "market_data", case_id, "research_v2_candidate_records_fixed_required_cutoff", research_cutoff, candidate["required_cutoff"])
        add_check(results, "market_data", case_id, "exact_candidate_ceiling_retreats_before_incomplete_required_cutoff", before_cutoff, candidate["exact_candidate_ceiling_date"])
        add_check(results, "market_data", case_id, "daily_universe_is_price_presence_based_market_state_guarded_and_cutoff_capped", 3, candidate["daily_universe_row_count"])
        add_check(results, "market_data", case_id, "research_v2_candidate_pins_market_state_guard_identity", 64, len(str(candidate.get("historical_market_state_guard_fingerprint") or "")))
        add_check(results, "market_data", case_id, "research_v2_transition_guard_detects_one_future_board_transition", 1, int(candidate.get("historical_market_state_transition_count") or 0))
        blockers = {str(item.get("code")) for item in candidate["blockers"]}
        add_check(results, "market_data", case_id, "research_v2_candidate_pins_authorized_required_dataset_scope", True, bool(candidate.get("research_scope_contract_fingerprint")) and bool(candidate.get("required_dataset_scope")))
        adjusted_scope = next(row for row in candidate["research_scope_contracts"] if row["dataset"] == "TaiwanStockPriceAdj")
        add_check(results, "market_data", case_id, "research_v2_candidate_keeps_raw_adjusted_price_levels_fail_closed_in_scope_contract", False, any(rule["direct_scientific_use_authorized"] for rule in adjusted_scope["field_authorizations"]))
        add_check(results, "market_data", case_id, "research_v2_candidate_keeps_scientific_common_complete_blocked", True, "RESEARCH_REQUIRED_SCOPE_COMMON_COMPLETE_NOT_READY" in blockers)
        add_check(results, "market_data", case_id, "research_v2_candidate_keeps_archive_wide_mechanical_audit_diagnostic_only", False, any(code.startswith("MECHANICAL_") for code in blockers))
        add_check(results, "market_data", case_id, "research_v2_candidate_blocks_incomplete_required_cutoff_exact_coverage", True, "RESEARCH_REQUIRED_CUTOFF_EXACT_COVERAGE_NOT_READY" in blockers)
        add_check(results, "market_data", case_id, "research_v2_candidate_preserves_configured_active_generation", ACTIVE_RESEARCH_DATA_GENERATION, candidate["active_research_generation"])

        universe_path = resolve_research_v2_daily_universe_path(root, provider_payload["snapshot_fingerprint"])
        manifest_path = resolve_research_v2_candidate_manifest_path(root, provider_payload["snapshot_fingerprint"])
        add_check(results, "market_data", case_id, "research_v2_daily_universe_is_research_domain_artifact", True, universe_path.is_file() and "data/research/market_data_v2" in universe_path.as_posix())
        add_check(results, "market_data", case_id, "research_v2_candidate_manifest_is_persisted", True, manifest_path.is_file())
        conn = sqlite3.connect(universe_path)
        try:
            universe_rows = int(conn.execute("SELECT COUNT(*) FROM daily_universe").fetchone()[0])
            max_universe_date = conn.execute("SELECT MAX(date) FROM daily_universe").fetchone()[0]
            pre_transition_2330_count = int(conn.execute(
                "SELECT COUNT(*) FROM daily_universe WHERE date = ? AND stock_id = ?",
                (before_cutoff, "2330"),
            ).fetchone()[0])
            post_transition_2330_count = int(conn.execute(
                "SELECT COUNT(*) FROM daily_universe WHERE date = ? AND stock_id = ?",
                (research_cutoff, "2330"),
            ).fetchone()[0])
            last_coverage = conn.execute("SELECT date, missing_price_limit_count, exact_complete FROM exact_coverage ORDER BY date DESC LIMIT 1").fetchone()
        finally:
            conn.close()
        add_check(results, "market_data", case_id, "research_v2_daily_universe_sqlite_preserves_only_date_eligible_membership_rows", 3, universe_rows)
        add_check(results, "market_data", case_id, "research_v2_daily_universe_excludes_provider_rows_after_research_cutoff", research_cutoff, max_universe_date)
        add_check(results, "market_data", case_id, "future_twse_transition_does_not_authorize_prior_emerging_row", 0, pre_transition_2330_count)
        add_check(results, "market_data", case_id, "twse_transition_allows_rows_strictly_after_final_emerging_date", 1, post_transition_2330_count)
        add_check(results, "market_data", case_id, "research_v2_exact_coverage_records_required_cutoff_missing_member", (research_cutoff, 1, 0), tuple(last_coverage))

        loaded = load_research_v2_candidate(root, required=True)
        add_check(results, "market_data", case_id, "research_v2_candidate_fingerprint_roundtrips", candidate["candidate_fingerprint"], loaded["candidate_fingerprint"])
        original_universe_bytes = universe_path.read_bytes()
        universe_path.write_bytes(original_universe_bytes + b"\n")
        tamper_blocked = False
        try:
            load_research_v2_candidate(root, required=True)
        except ValueError as exc:
            tamper_blocked = "SHA256 drift" in str(exc)
        add_check(results, "market_data", case_id, "research_v2_candidate_rejects_daily_universe_file_tamper", True, tamper_blocked)
        universe_path.write_bytes(original_universe_bytes)
        persisted = load_json_strict(manifest_path)
        add_check(results, "market_data", case_id, "research_v2_candidate_paths_are_project_relative", False, str(persisted["provider_snapshot_path"]).startswith(str(root)))

        service_source = (Path(__file__).resolve().parents[2] / "services" / "research" / "market_data_v2.py").read_text(encoding="utf-8")
        add_check(results, "market_data", case_id, "research_v2_builder_never_reads_trading_v2_overlay", False, "data/trading/market_data_v2" in service_source or "services.trading.market_data_v2_state" in service_source)

    summary.update(
        {
            "checks": len(results),
            "dataset_count": stats["dataset_count"],
            "exact_candidate_count": stats["exact_candidate_count"],
            "review_required_count": stats["review_required_count"],
        }
    )
    return results, summary


def validate_market_data_v2_research_pit_review_contract_case(_base_params):
    """Round-10 PIT review classification and mechanical common-tail contract."""

    from core.market_data_dataset_registry import (
        PIT_CURRENT_VINTAGE,
        PIT_EXACT_CANDIDATE,
        PIT_REVIEW_REQUIRED,
        get_market_dataset_specs,
    )
    from core.market_data_freshness_contract import (
        CADENCE_CALENDAR_DAILY,
        CADENCE_CURRENT_VINTAGE,
        CADENCE_EVENT_DRIVEN,
        CADENCE_PERIODIC,
        CADENCE_TRADING_DAILY,
        get_market_data_freshness_contracts,
    )
    from core.market_data_research_pit_contract import (
        AUDIT_MODE_CALENDAR_DAILY_DATE_PRESENCE,
        AUDIT_MODE_CURRENT_VINTAGE_BLOCKED,
        AUDIT_MODE_EVENT_INFORMATION_TIME,
        AUDIT_MODE_EXACT_CANDIDATE,
        AUDIT_MODE_PERIODIC_PUBLICATION,
        AUDIT_MODE_STATIC_CURRENT_VINTAGE,
        AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE,
        DATE_AUDIT_STATUS_NO_ARTIFACTS,
        DATE_AUDIT_STATUS_READY,
        PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED,
        ResearchV2DatasetDateAudit,
        build_research_v2_pit_review_contracts,
        research_v2_pit_review_contract_fingerprint,
        summarize_mechanical_common_complete_tail,
        validate_research_v2_pit_review_contracts,
    )

    case_id = "MARKET_DATA_V2_RESEARCH_PIT_REVIEW"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    specs = tuple(get_market_dataset_specs(included_only=True))
    freshness = {row.dataset: row for row in get_market_data_freshness_contracts()}
    contracts = build_research_v2_pit_review_contracts()
    contract_by_dataset = {row.dataset: row for row in contracts}
    stats = validate_research_v2_pit_review_contracts(contracts)

    add_check(results, "market_data", case_id, "pit_review_contract_covers_every_included_dataset", len(specs), len(contracts))
    add_check(results, "market_data", case_id, "pit_review_contract_dataset_identity_is_unique", len(contracts), len(contract_by_dataset))
    add_check(results, "market_data", case_id, "pit_review_never_auto_authorizes_model_input", 0, sum(row.scientific_input_authorized for row in contracts))
    add_check(results, "market_data", case_id, "pit_review_contract_fingerprint_is_deterministic", research_v2_pit_review_contract_fingerprint(contracts), research_v2_pit_review_contract_fingerprint())

    expected_modes = {}
    for spec in specs:
        cadence = freshness[spec.dataset].cadence
        if spec.pit_class == PIT_CURRENT_VINTAGE:
            expected = AUDIT_MODE_CURRENT_VINTAGE_BLOCKED
        elif spec.pit_class == PIT_EXACT_CANDIDATE:
            expected = AUDIT_MODE_EXACT_CANDIDATE
        elif spec.pit_class == PIT_REVIEW_REQUIRED and cadence == CADENCE_TRADING_DAILY:
            expected = AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE
        elif spec.pit_class == PIT_REVIEW_REQUIRED and cadence == CADENCE_CALENDAR_DAILY:
            expected = AUDIT_MODE_CALENDAR_DAILY_DATE_PRESENCE
        elif spec.pit_class == PIT_REVIEW_REQUIRED and cadence == CADENCE_EVENT_DRIVEN:
            expected = AUDIT_MODE_EVENT_INFORMATION_TIME
        elif spec.pit_class == PIT_REVIEW_REQUIRED and cadence == CADENCE_PERIODIC:
            expected = AUDIT_MODE_PERIODIC_PUBLICATION
        elif spec.pit_class == PIT_REVIEW_REQUIRED and cadence == CADENCE_CURRENT_VINTAGE:
            expected = AUDIT_MODE_STATIC_CURRENT_VINTAGE
        else:
            expected = "UNEXPECTED"
        expected_modes[spec.dataset] = expected
    actual_modes = {row.dataset: row.audit_mode for row in contracts}
    add_check(results, "market_data", case_id, "pit_review_classification_is_registry_and_cadence_driven", expected_modes, actual_modes)

    adjusted = contract_by_dataset["TaiwanStockPriceAdj"]
    add_check(results, "market_data", case_id, "adjusted_price_remains_current_vintage_hard_block", PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED, adjusted.review_status)
    review_daily_expected = sum(
        spec.pit_class == PIT_REVIEW_REQUIRED and freshness[spec.dataset].cadence in {CADENCE_TRADING_DAILY, CADENCE_CALENDAR_DAILY}
        for spec in specs
    )
    add_check(results, "market_data", case_id, "only_review_daily_classes_get_automatic_date_presence_audit", review_daily_expected, stats["automatic_date_audit_count"])

    audits = (
        ResearchV2DatasetDateAudit("A", AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE, DATE_AUDIT_STATUS_READY, 3, "2026-09-01", "2026-09-03", "synthetic"),
        ResearchV2DatasetDateAudit("B", AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE, DATE_AUDIT_STATUS_READY, 2, "2026-09-01", "2026-09-02", "synthetic"),
    )
    mechanical = summarize_mechanical_common_complete_tail(
        trading_dates=("2026-09-01", "2026-09-02", "2026-09-03"),
        exact_complete_dates=("2026-09-01", "2026-09-02", "2026-09-03"),
        dataset_audits=audits,
        observed_dates_by_dataset={
            "A": ("2026-09-01", "2026-09-02", "2026-09-03"),
            "B": ("2026-09-01", "2026-09-02"),
        },
        participating_datasets=("A", "B"),
    )
    add_check(results, "market_data", case_id, "mechanical_common_tail_retreats_to_latest_shared_date", "2026-09-02", mechanical.common_complete_ceiling_date)
    add_check(results, "market_data", case_id, "mechanical_common_tail_keeps_contiguous_start", "2026-09-01", mechanical.common_complete_tail_start)
    add_check(results, "market_data", case_id, "mechanical_common_tail_reports_two_shared_dates", 2, mechanical.common_complete_tail_date_count)
    add_check(results, "market_data", case_id, "mechanical_common_tail_is_complete_only_when_all_participants_audited", True, mechanical.audit_complete)

    incomplete = summarize_mechanical_common_complete_tail(
        trading_dates=("2026-09-01", "2026-09-02"),
        exact_complete_dates=("2026-09-01", "2026-09-02"),
        dataset_audits=(
            ResearchV2DatasetDateAudit("A", AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE, DATE_AUDIT_STATUS_READY, 2, "2026-09-01", "2026-09-02", "synthetic"),
            ResearchV2DatasetDateAudit("B", AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE, DATE_AUDIT_STATUS_NO_ARTIFACTS, 0, None, None, "synthetic"),
        ),
        observed_dates_by_dataset={"A": ("2026-09-01", "2026-09-02"), "B": ()},
        participating_datasets=("A", "B"),
    )
    add_check(results, "market_data", case_id, "missing_dataset_audit_clears_mechanical_ceiling", None, incomplete.common_complete_ceiling_date)
    add_check(results, "market_data", case_id, "missing_dataset_audit_marks_mechanical_audit_incomplete", False, incomplete.audit_complete)

    import sqlite3
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import pandas as pd
    from services.research.market_data_v2 import _build_review_date_audits, _init_universe_db

    class _SyntheticArchive:
        as_of_date = "2026-09-03"

    class _SyntheticReviewView:
        archive = _SyntheticArchive()

        @staticmethod
        def dataset_artifacts(_dataset):
            return (object(),)

        @staticmethod
        def iter_dataset_contract_audit_frames(_dataset, *, columns=None):
            del columns
            yield pd.DataFrame({"date": ["2026-09-01", "2026-09-02", "2026-09-03"]})

    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "review.sqlite3"
        conn = sqlite3.connect(db_path)
        try:
            _init_universe_db(conn)
            for date_value in ("2026-09-01", "2026-09-02", "2026-09-03"):
                conn.execute("INSERT INTO trading_dates(date) VALUES (?)", (date_value,))
                conn.execute(
                    "INSERT INTO exact_coverage(date, universe_count, price_limit_count, missing_price_limit_count, extra_price_limit_count, exact_complete) VALUES (?, ?, ?, ?, ?, ?)",
                    (date_value, 1, 1, 0, 0, 1),
                )
            service_audits, service_summary = _build_review_date_audits(
                conn,
                _SyntheticReviewView(),
                trading_dates={"2026-09-01", "2026-09-02", "2026-09-03"},
                research_cutoff="2026-09-03",
            )
            persisted_audits = int(conn.execute("SELECT COUNT(*) FROM dataset_date_audit").fetchone()[0])
            persisted_common_dates = int(conn.execute("SELECT COUNT(*) FROM mechanical_common_complete").fetchone()[0])
        finally:
            conn.close()
    add_check(results, "market_data", case_id, "service_persists_one_review_audit_row_per_dataset", len(contracts), persisted_audits)
    add_check(results, "market_data", case_id, "service_success_path_audits_all_contract_rows", len(contracts), len(service_audits))
    add_check(results, "market_data", case_id, "service_success_path_establishes_mechanical_ceiling", "2026-09-03", service_summary.common_complete_ceiling_date)
    add_check(results, "market_data", case_id, "service_success_path_persists_common_complete_dates", 3, persisted_common_dates)

    summary.update({
        "checks": len(results),
        "dataset_count": stats["dataset_count"],
        "automatic_date_audit_count": stats["automatic_date_audit_count"],
        "mode_counts": stats["mode_counts"],
    })
    return results, summary


def validate_market_data_v2_research_required_cutoff_isolation_contract_case(_base_params):
    """Round-11 pins Research V2 scientific evidence to the fixed Research horizon."""

    import sqlite3
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import pandas as pd

    from config.market_data import RESEARCH_DATA_GENERATION_V1, RESEARCH_DATA_GENERATION_V2, RESEARCH_REQUIRED_CUTOFF
    from core.market_data_contract import get_research_data_generation
    from core.market_data_research_v2 import (
        RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS,
        RESEARCH_V2_CANDIDATE_SCHEMA_VERSION,
        validate_research_v2_required_cutoff,
    )
    from services.research.market_data_v2 import _build_review_date_audits, _init_universe_db

    case_id = "MARKET_DATA_V2_RESEARCH_REQUIRED_CUTOFF_ISOLATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    v1 = get_research_data_generation(RESEARCH_DATA_GENERATION_V1)
    v2 = get_research_data_generation(RESEARCH_DATA_GENERATION_V2)
    add_check(results, "market_data", case_id, "research_v1_cutoff_uses_required_cutoff_ssot", RESEARCH_REQUIRED_CUTOFF, v1.cutoff)
    add_check(results, "market_data", case_id, "research_v1_required_cutoff_uses_ssot", RESEARCH_REQUIRED_CUTOFF, v1.required_cutoff)
    add_check(results, "market_data", case_id, "research_v2_required_cutoff_uses_same_ssot", RESEARCH_REQUIRED_CUTOFF, v2.required_cutoff)
    add_check(results, "market_data", case_id, "research_v2_remains_unfrozen_before_promotion", None, v2.cutoff)
    add_check(results, "market_data", case_id, "research_v2_candidate_schema_still_carries_cutoff_identity", True, RESEARCH_V2_CANDIDATE_SCHEMA_VERSION >= 3)
    add_check(results, "market_data", case_id, "required_cutoff_participates_in_candidate_identity", True, "required_cutoff" in RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS)
    add_check(
        results,
        "market_data",
        case_id,
        "provider_snapshot_may_extend_beyond_research_horizon",
        "2030-01-15",
        validate_research_v2_required_cutoff(provider_as_of_date="2030-02-01", required_cutoff="2030-01-15"),
    )
    provider_too_early_blocked = False
    try:
        validate_research_v2_required_cutoff(provider_as_of_date="2030-01-14", required_cutoff="2030-01-15")
    except ValueError:
        provider_too_early_blocked = True
    add_check(results, "market_data", case_id, "provider_snapshot_before_required_cutoff_fails_closed", True, provider_too_early_blocked)

    class _SyntheticArchive:
        as_of_date = "2030-02-01"

    class _SyntheticReviewView:
        archive = _SyntheticArchive()

        @staticmethod
        def dataset_artifacts(_dataset):
            return (object(),)

        @staticmethod
        def iter_dataset_contract_audit_frames(_dataset, *, columns=None):
            del columns
            yield pd.DataFrame({"date": ["2030-01-15", "2030-01-16"]})

    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "round11_cutoff.sqlite3"
        conn = sqlite3.connect(db_path)
        try:
            _init_universe_db(conn)
            conn.execute("INSERT INTO trading_dates(date) VALUES (?)", ("2030-01-15",))
            conn.execute(
                "INSERT INTO exact_coverage(date, universe_count, price_limit_count, missing_price_limit_count, extra_price_limit_count, exact_complete) VALUES (?, ?, ?, ?, ?, ?)",
                ("2030-01-15", 1, 1, 0, 0, 1),
            )
            audits, mechanical = _build_review_date_audits(
                conn,
                _SyntheticReviewView(),
                trading_dates={"2030-01-15"},
                research_cutoff="2030-01-15",
            )
            max_presence_date = conn.execute("SELECT MAX(date) FROM dataset_date_presence").fetchone()[0]
        finally:
            conn.close()
    add_check(results, "market_data", case_id, "review_date_presence_excludes_provider_rows_after_research_cutoff", "2030-01-15", max_presence_date)
    add_check(results, "market_data", case_id, "mechanical_common_complete_cannot_extend_past_research_cutoff", "2030-01-15", mechanical.common_complete_ceiling_date)
    add_check(results, "market_data", case_id, "round11_review_audit_still_covers_full_registry", 51, len(audits))

    summary.update({"checks": len(results), "required_cutoff": v2.required_cutoff})
    return results, summary



def validate_market_data_v2_research_non_daily_pit_legality_contract_case(_base_params):
    """Round-12 closes non-daily information-time/revision legality without authorizing model inputs."""

    from core.market_data_freshness_contract import (
        CADENCE_CURRENT_VINTAGE,
        CADENCE_EVENT_DRIVEN,
        CADENCE_PERIODIC,
    )
    from core.market_data_research_pit_contract import (
        AUDIT_MODE_CURRENT_VINTAGE_BLOCKED,
        PIT_LEGALITY_STATUS_CURRENT_VINTAGE_BLOCKED,
        PIT_LEGALITY_STATUS_EVENT_ANCHOR_READY,
        PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED,
        RESEARCH_PIT_REVIEW_SCHEMA_VERSION,
        build_research_v2_pit_review_contracts,
        research_v2_pit_review_contract_fingerprint,
        validate_research_v2_pit_review_contracts,
    )
    from core.market_data_research_v2 import (
        RESEARCH_V2_CANDIDATE_SCHEMA_VERSION,
        RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS,
    )
    from core.market_data_research_scope import (
        SCOPE_STATUS_OPTIONAL_NOT_SELECTED,
        build_research_v2_dataset_scope_contracts,
    )

    case_id = "MARKET_DATA_V2_RESEARCH_NON_DAILY_PIT_LEGALITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    contracts = build_research_v2_pit_review_contracts()
    by_dataset = {row.dataset: row for row in contracts}
    stats = validate_research_v2_pit_review_contracts(contracts)
    non_daily = [
        row for row in contracts
        if row.audit_mode in {
            "event_information_time_review",
            "periodic_publication_revision_review",
            "static_current_vintage_review",
        }
    ]
    events = [row for row in non_daily if row.cadence == CADENCE_EVENT_DRIVEN]
    blocked_vintage = [
        row for row in non_daily if row.cadence in {CADENCE_PERIODIC, CADENCE_CURRENT_VINTAGE}
    ]

    add_check(results, "market_data", case_id, "round12_pit_contract_schema_is_versioned", True, RESEARCH_PIT_REVIEW_SCHEMA_VERSION >= 2)
    add_check(results, "market_data", case_id, "round12_non_daily_policy_covers_registry_derived_review_datasets", len(non_daily), stats["non_daily_policy_count"])
    add_check(results, "market_data", case_id, "round12_event_information_time_anchor_count_is_registry_driven", len(events), stats["event_information_time_anchor_ready_count"])
    add_check(results, "market_data", case_id, "round12_periodic_static_historical_vintage_block_count_is_registry_driven", len(blocked_vintage), stats["historical_publication_vintage_blocked_count"])
    add_check(results, "market_data", case_id, "event_contracts_are_anchor_ready_not_model_authorized", True, bool(events) and all(row.pit_legality_status == PIT_LEGALITY_STATUS_EVENT_ANCHOR_READY and row.information_time_anchor_ready and row.field_scope_required and not row.scientific_input_authorized for row in events))
    add_check(results, "market_data", case_id, "periodic_static_contracts_fail_closed_without_historical_vintage", True, bool(blocked_vintage) and all(row.pit_legality_status == PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED and not row.information_time_anchor_ready and row.field_scope_required and not row.scientific_input_authorized for row in blocked_vintage))

    dividend = by_dataset["TaiwanStockDividend"]
    add_check(results, "market_data", case_id, "dividend_uses_intrinsic_announcement_timestamp_anchor", ("AnnouncementDate", "AnnouncementTime"), dividend.information_time_columns)
    add_check(results, "market_data", case_id, "dividend_disallows_same_day_assumption", "next_taiwan_trading_session_after_announcement_timestamp", dividend.information_time_rule)
    disposition = by_dataset["TaiwanStockDispositionSecuritiesPeriod"]
    add_check(results, "market_data", case_id, "disposition_uses_documented_announcement_date_anchor", ("date",), disposition.information_time_columns)
    add_check(results, "market_data", case_id, "disposition_missing_intraday_time_uses_next_session", "next_taiwan_trading_session_after_announcement_date", disposition.information_time_rule)

    month_revenue = by_dataset["TaiwanStockMonthRevenue"]
    add_check(results, "market_data", case_id, "month_revenue_historical_create_time_gap_stays_blocked", PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED, month_revenue.pit_legality_status)
    add_check(results, "market_data", case_id, "month_revenue_evidence_records_provider_schema_transition", True, "2026-04-21" in month_revenue.evidence_source)
    static_rows = [row for row in non_daily if row.cadence == CADENCE_CURRENT_VINTAGE]
    add_check(results, "market_data", case_id, "all_registry_static_current_vintage_review_datasets_fail_closed", True, bool(static_rows) and all(row.pit_legality_status == PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED for row in static_rows))

    adjusted = by_dataset["TaiwanStockPriceAdj"]
    add_check(results, "market_data", case_id, "adjusted_price_retains_separate_current_vintage_hard_block", PIT_LEGALITY_STATUS_CURRENT_VINTAGE_BLOCKED, adjusted.pit_legality_status)
    add_check(results, "market_data", case_id, "round12_never_auto_authorizes_any_model_input", 0, sum(row.scientific_input_authorized for row in contracts))
    add_check(results, "market_data", case_id, "round14_demotes_archive_wide_pit_matrix_from_foundation_identity", True, RESEARCH_V2_CANDIDATE_SCHEMA_VERSION >= 6 and "pit_review_contract_fingerprint" not in RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS and "research_scope_contract_fingerprint" in RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS)
    add_check(results, "market_data", case_id, "pit_contract_fingerprint_is_deterministic_after_legality_extension", research_v2_pit_review_contract_fingerprint(contracts), research_v2_pit_review_contract_fingerprint())

    scope_rows = build_research_v2_dataset_scope_contracts()
    scope_by_dataset = {row.dataset: row for row in scope_rows}
    optional_events = [row for row in events if not scope_by_dataset[row.dataset].required_for_generation]
    optional_blocked = [row for row in blocked_vintage if not scope_by_dataset[row.dataset].required_for_generation]
    add_check(results, "market_data", case_id, "round12_event_anchor_evidence_remains_available_after_scope_selection", len(events), stats["event_information_time_anchor_ready_count"])
    add_check(results, "market_data", case_id, "round12_historical_vintage_blocks_remain_preserved_after_scope_selection", len(blocked_vintage), stats["historical_publication_vintage_blocked_count"])
    add_check(results, "market_data", case_id, "round14_optional_non_daily_review_datasets_are_not_auto_selected", True, all(scope_by_dataset[row.dataset].scope_status == SCOPE_STATUS_OPTIONAL_NOT_SELECTED for row in optional_events + optional_blocked))

    summary.update({
        "checks": len(results),
        "non_daily_policy_count": stats["non_daily_policy_count"],
        "event_anchor_ready_count": stats["event_information_time_anchor_ready_count"],
        "historical_vintage_blocked_count": stats["historical_publication_vintage_blocked_count"],
    })
    return results, summary



def validate_market_data_v2_research_scope_field_authorization_contract_case(_base_params):
    """Round-14 freezes the narrow foundation scope and fail-closed field authorization."""

    from core.market_data_adjusted_price_invariance import (
        PRICE_REPRESENTATION_STATUS_BLOCKED,
        PRICE_REPRESENTATION_STATUS_DEFERRED,
        PRICE_REPRESENTATION_STATUS_INVARIANT,
        PROVIDER_PRICE_FIELDS,
        PROVIDER_VOLUME_FIELD,
        get_adjusted_price_representation_contract,
    )
    from core.market_data_dataset_registry import get_market_dataset_specs
    from core.market_data_research_pit_contract import validate_research_v2_pit_review_contracts
    from core.market_data_research_scope import (
        FIELD_USE_DIRECT_INPUT,
        FIELD_USE_TRANSFORM_SOURCE_ONLY,
        RESEARCH_V2_ADJUSTED_PRICE_DATASET,
        RESEARCH_V2_COMMON_COMPLETE_DATASETS,
        RESEARCH_V2_RAW_VOLUME_DATASET,
        RESEARCH_V2_REQUIRED_DATASETS,
        RESEARCH_V2_SCOPE_SCHEMA_VERSION,
        SCOPE_STATUS_OPTIONAL_NOT_SELECTED,
        build_research_v2_dataset_scope_contracts,
        research_v2_dataset_scope_contract_fingerprint,
        research_v2_dataset_scope_contract_payloads,
        validate_research_v2_dataset_scope_contracts,
    )
    from core.market_data_research_v2 import (
        RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS,
        RESEARCH_V2_CANDIDATE_SCHEMA_VERSION,
    )
    from services.research.market_data_v2 import _build_candidate_blockers

    case_id = "MARKET_DATA_V2_RESEARCH_SCOPE_FIELD_AUTHORIZATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    rows = build_research_v2_dataset_scope_contracts()
    stats = validate_research_v2_dataset_scope_contracts(rows)
    by_dataset = {row.dataset: row for row in rows}
    registry_count = len(tuple(get_market_dataset_specs(included_only=True)))
    add_check(results, "market_data", case_id, "round14_scope_contract_is_versioned", True, RESEARCH_V2_SCOPE_SCHEMA_VERSION >= 1)
    add_check(results, "market_data", case_id, "scope_covers_current_included_registry", registry_count, stats["dataset_count"])
    add_check(results, "market_data", case_id, "required_dataset_scope_is_explicit", set(RESEARCH_V2_REQUIRED_DATASETS), {row.dataset for row in rows if row.required_for_generation})
    add_check(results, "market_data", case_id, "optional_count_is_registry_driven", registry_count - len(RESEARCH_V2_REQUIRED_DATASETS), stats["optional_not_selected_count"])
    add_check(results, "market_data", case_id, "common_complete_scope_is_explicit", set(RESEARCH_V2_COMMON_COMPLETE_DATASETS), {row.dataset for row in rows if row.contributes_to_common_complete})
    add_check(results, "market_data", case_id, "delisting_event_is_not_daily_common_complete_denominator", False, by_dataset["TaiwanStockDelisting"].contributes_to_common_complete)
    add_check(results, "market_data", case_id, "required_scope_authorization_is_complete", True, stats["required_scope_authorization_complete"])
    add_check(results, "market_data", case_id, "scope_fingerprint_is_deterministic", research_v2_dataset_scope_contract_fingerprint(rows), research_v2_dataset_scope_contract_fingerprint())
    from dataclasses import replace
    optional_index = next(index for index, row in enumerate(rows) if not row.required_for_generation)
    optional_mutated = list(rows)
    optional_mutated[optional_index] = replace(optional_mutated[optional_index], pit_contract_status="SYNTHETIC_OPTIONAL_DIAGNOSTIC_CHANGE")
    add_check(results, "market_data", case_id, "optional_pit_diagnostic_does_not_pollute_foundation_fingerprint", research_v2_dataset_scope_contract_fingerprint(rows), research_v2_dataset_scope_contract_fingerprint(optional_mutated))
    add_check(results, "market_data", case_id, "scope_payload_normalizes_tuple_fields_to_json_lists", True, isinstance(research_v2_dataset_scope_contract_payloads(rows)[0]["evidence_fields"], list))

    raw = by_dataset[RESEARCH_V2_RAW_VOLUME_DATASET]
    raw_direct = [rule for rule in raw.field_authorizations if rule.direct_scientific_use_authorized]
    raw_direct_fields = {field for rule in raw_direct for field in rule.fields}
    add_check(results, "market_data", case_id, "raw_daily_price_authorizes_only_share_volume_directly", {PROVIDER_VOLUME_FIELD}, raw_direct_fields)
    add_check(results, "market_data", case_id, "raw_daily_volume_has_next_session_availability", "next_taiwan_trading_session_after_market_date", raw_direct[0].availability_rule)
    add_check(results, "market_data", case_id, "raw_ohlc_remains_prohibited_as_model_price_input", set(), set(PROVIDER_PRICE_FIELDS).intersection(raw_direct_fields))
    add_check(results, "market_data", case_id, "raw_volume_rule_is_direct_input", FIELD_USE_DIRECT_INPUT, raw_direct[0].use_mode)

    adjusted = by_dataset[RESEARCH_V2_ADJUSTED_PRICE_DATASET]
    adjusted_rules = list(adjusted.field_authorizations)
    adjusted_direct_fields = {field for rule in adjusted_rules if rule.direct_scientific_use_authorized for field in rule.fields}
    adjusted_transform_fields = {field for rule in adjusted_rules if rule.use_mode == FIELD_USE_TRANSFORM_SOURCE_ONLY for field in rule.fields}
    representation = get_adjusted_price_representation_contract()
    invariant_ids = {rule.representation_id for rule in representation.rules if rule.status == PRICE_REPRESENTATION_STATUS_INVARIANT}
    blocked_ids = {rule.representation_id for rule in representation.rules if rule.status == PRICE_REPRESENTATION_STATUS_BLOCKED}
    deferred_ids = {rule.representation_id for rule in representation.rules if rule.status == PRICE_REPRESENTATION_STATUS_DEFERRED}
    add_check(results, "market_data", case_id, "adjusted_price_ohlc_are_transform_source_only", set(PROVIDER_PRICE_FIELDS), adjusted_transform_fields)
    add_check(results, "market_data", case_id, "adjusted_price_has_no_direct_raw_fields", set(), adjusted_direct_fields)
    add_check(results, "market_data", case_id, "adjusted_price_volume_is_not_authorized", False, any(PROVIDER_VOLUME_FIELD in rule.fields for rule in adjusted_rules))
    add_check(results, "market_data", case_id, "adjusted_price_authorizes_exact_round13_invariant_representation_set", invariant_ids, set(adjusted.authorized_representation_ids))
    add_check(results, "market_data", case_id, "blocked_absolute_representation_is_not_authorized", set(), blocked_ids.intersection(adjusted.authorized_representation_ids))
    add_check(results, "market_data", case_id, "deferred_adjusted_volume_representation_is_not_authorized", set(), deferred_ids.intersection(adjusted.authorized_representation_ids))

    optional_dividend = by_dataset["TaiwanStockDividend"]
    optional_revenue = by_dataset["TaiwanStockMonthRevenue"]
    add_check(results, "market_data", case_id, "event_archive_dataset_is_optional_until_experiment_selects_it", SCOPE_STATUS_OPTIONAL_NOT_SELECTED, optional_dividend.scope_status)
    add_check(results, "market_data", case_id, "periodic_archive_dataset_is_optional_until_experiment_selects_it", SCOPE_STATUS_OPTIONAL_NOT_SELECTED, optional_revenue.scope_status)
    add_check(results, "market_data", case_id, "optional_datasets_receive_no_implicit_fields", True, not optional_dividend.evidence_fields and not optional_dividend.field_authorizations and not optional_revenue.evidence_fields and not optional_revenue.field_authorizations)

    add_check(results, "market_data", case_id, "candidate_schema_pins_scope_contract", True, RESEARCH_V2_CANDIDATE_SCHEMA_VERSION >= 6 and "research_scope_contract_fingerprint" in RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS)
    add_check(results, "market_data", case_id, "candidate_schema_pins_required_dataset_list", True, "required_dataset_scope" in RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS)
    pit_stats = validate_research_v2_pit_review_contracts()
    blockers = _build_candidate_blockers(
        {"latest_exact_complete_date": "2030-01-15"},
        required_cutoff="2030-01-15",
        scope_stats=stats,
    )
    blocker_codes = {str(row.get("code")) for row in blockers}
    add_check(results, "market_data", case_id, "archive_wide_pit_review_no_longer_blocks_unselected_datasets", False, "DATASET_SPECIFIC_PIT_REVIEW_REQUIRED" in blocker_codes)
    add_check(results, "market_data", case_id, "archive_wide_mechanical_diagnostic_no_longer_blocks_foundation", False, any(code.startswith("MECHANICAL_") for code in blocker_codes))
    add_check(results, "market_data", case_id, "scope_authorization_blocker_is_cleared", False, "REQUIRED_SCOPE_AUTHORIZATION_INCOMPLETE" in blocker_codes)
    add_check(results, "market_data", case_id, "round15_common_complete_gate_remains", True, "RESEARCH_COMMON_COMPLETE_NOT_AUTHORIZED" in blocker_codes)

    summary.update({
        "checks": len(results),
        "required_dataset_count": stats["required_dataset_count"],
        "optional_not_selected_count": stats["optional_not_selected_count"],
        "contract_fingerprint": stats["contract_fingerprint"],
    })
    return results, summary


def validate_market_data_v2_research_required_common_complete_freeze_contract_case(_base_params):
    """Round-15 proves field-aware required-scope completeness and immutable freeze-candidate semantics."""

    from datetime import date, datetime, timedelta, timezone
    from pathlib import Path
    import sqlite3
    from tempfile import TemporaryDirectory

    from config.market_data import (
        ACTIVE_RESEARCH_DATA_GENERATION,
        RESEARCH_DATA_GENERATION_V1,
        RESEARCH_DATA_GENERATION_V2,
        RESEARCH_REQUIRED_CUTOFF,
    )
    from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
    from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest, build_registry_fingerprint
    from core.market_data_contract import RESEARCH_STATUS_AUTHORIZED_NOT_READY, get_research_data_generation
    from core.market_data_dataset_registry import (
        BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
        BOOTSTRAP_SINGLE_FULL_RANGE,
        BOOTSTRAP_SINGLE_NO_DATES,
        get_market_dataset_specs,
    )
    from core.market_data_provider_snapshot import ProviderArtifactEvidence, build_provider_snapshot_payload
    from core.market_data_research_freeze import (
        RESEARCH_V2_FREEZE_CANDIDATE_STATUS_READY,
        RESEARCH_V2_REQUIRED_COMMON_COMPLETE_SCHEMA_VERSION,
        build_research_v2_required_common_complete,
    )
    from core.market_data_research_scope import (
        RESEARCH_V2_COMMON_COMPLETE_DATASETS,
        RESEARCH_V2_REQUIRED_DATASETS,
        validate_research_v2_dataset_scope_contracts,
    )
    from core.market_data_research_storage_contract import (
        resolve_research_v2_candidate_manifest_path,
        resolve_research_v2_daily_universe_path,
        resolve_research_v2_freeze_candidate_manifest_path,
        resolve_research_v2_frozen_daily_universe_path,
        resolve_research_v2_frozen_source_candidate_manifest_path,
    )
    from core.market_data_storage_contract import (
        resolve_market_data_provider_snapshot_path,
        resolve_market_data_request_parquet_path,
    )
    from services.downloader.market_data_ledger import MarketDataJobLedger
    from services.research.market_data_v2 import (
        ResearchV2ProviderView,
        build_research_v2_candidate,
        build_research_v2_freeze_candidate,
        load_research_v2_candidate,
        load_research_v2_freeze_candidate,
    )
    from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive

    case_id = "MARKET_DATA_V2_RESEARCH_REQUIRED_COMMON_COMPLETE_FREEZE"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    scope_stats = validate_research_v2_dataset_scope_contracts()
    synthetic_dates = ("2030-01-13", "2030-01-14", "2030-01-15")
    complete_map = {dataset: synthetic_dates for dataset in RESEARCH_V2_COMMON_COMPLETE_DATASETS}
    common_dates, complete_summary = build_research_v2_required_common_complete(
        trading_dates=synthetic_dates,
        complete_dates_by_dataset=complete_map,
        required_cutoff="2030-01-15",
        research_scope_contract_fingerprint=str(scope_stats["contract_fingerprint"]),
    )
    add_check(results, "market_data", case_id, "round15_required_common_complete_contract_is_versioned", True, RESEARCH_V2_REQUIRED_COMMON_COMPLETE_SCHEMA_VERSION >= 1)
    add_check(results, "market_data", case_id, "required_common_complete_uses_exact_round14_participant_scope", set(RESEARCH_V2_COMMON_COMPLETE_DATASETS), set(complete_summary.participating_datasets))
    add_check(results, "market_data", case_id, "required_common_complete_reaches_fixed_cutoff_only_when_all_participants_complete", "2030-01-15", complete_summary.common_complete_cutoff)
    add_check(results, "market_data", case_id, "required_common_complete_preserves_contiguous_tail_start", "2030-01-13", complete_summary.common_complete_tail_start)
    add_check(results, "market_data", case_id, "required_common_complete_date_set_is_deterministic", synthetic_dates, common_dates)
    incomplete_map = dict(complete_map)
    incomplete_map["TaiwanStockPriceAdj"] = synthetic_dates[:-1]
    _incomplete_dates, incomplete_summary = build_research_v2_required_common_complete(
        trading_dates=synthetic_dates,
        complete_dates_by_dataset=incomplete_map,
        required_cutoff="2030-01-15",
        research_scope_contract_fingerprint=str(scope_stats["contract_fingerprint"]),
    )
    add_check(results, "market_data", case_id, "missing_adjusted_price_operand_at_cutoff_fail_closes_freeze_readiness", None, incomplete_summary.common_complete_cutoff)
    extra_optional_map = {**complete_map, "TaiwanStockDividend": synthetic_dates}
    _optional_dates, optional_summary = build_research_v2_required_common_complete(
        trading_dates=synthetic_dates,
        complete_dates_by_dataset=extra_optional_map,
        required_cutoff="2030-01-15",
        research_scope_contract_fingerprint=str(scope_stats["contract_fingerprint"]),
    )
    add_check(results, "market_data", case_id, "optional_archive_dates_do_not_pollute_required_common_complete_identity", complete_summary.coverage_fingerprint, optional_summary.coverage_fingerprint)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cutoff_date = date.fromisoformat(RESEARCH_REQUIRED_CUTOFF)
        d1 = (cutoff_date - timedelta(days=4)).isoformat()
        d2 = (cutoff_date - timedelta(days=3)).isoformat()
        cutoff = cutoff_date.isoformat()
        provider_as_of = (cutoff_date + timedelta(days=190)).isoformat()
        dates = [d1, d2, cutoff]
        manifest_fingerprint = "8" * 64
        registry_fingerprint = build_registry_fingerprint(get_market_dataset_specs(included_only=True))
        requests = (
            BootstrapHttpRequest("TaiwanStockTradingDate", BOOTSTRAP_SINGLE_NO_DATES, None, None, None),
            BootstrapHttpRequest("TaiwanStockDelisting", BOOTSTRAP_SINGLE_FULL_RANGE, None, "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPrice", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPrice", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "2330", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceLimit", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceLimit", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "2330", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceAdj", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceAdj", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "2330", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockInfo", BOOTSTRAP_SINGLE_NO_DATES, None, None, None),
        )
        manifest = BootstrapRequestManifest(
            as_of_date=provider_as_of,
            full_range_start="1900-01-01",
            registry_fingerprint=registry_fingerprint,
            manifest_fingerprint=manifest_fingerprint,
            historical_instrument_count=2,
            requests=requests,
        )
        frame_by_request = {
            requests[0].request_id: pd.DataFrame({"date": dates}),
            requests[1].request_id: pd.DataFrame(columns=["date", "stock_id"]),
            requests[2].request_id: pd.DataFrame({"date": dates, "stock_id": ["0050"] * 3, "Trading_Volume": [1000.0, 1100.0, 1200.0]}),
            requests[3].request_id: pd.DataFrame({"date": dates, "stock_id": ["2330"] * 3, "Trading_Volume": [2000.0, 2100.0, 2200.0]}),
            requests[4].request_id: pd.DataFrame({"date": dates, "stock_id": ["0050"] * 3}),
            requests[5].request_id: pd.DataFrame({"date": dates, "stock_id": ["2330"] * 3}),
            requests[6].request_id: pd.DataFrame({
                "date": dates,
                "stock_id": ["0050"] * 3,
                "open": [100.0, 101.0, 102.0],
                "max": [102.0, 103.0, 104.0],
                "min": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
            }),
            requests[7].request_id: pd.DataFrame({
                "date": dates,
                "stock_id": ["2330"] * 3,
                "open": [500.0, 501.0, 502.0],
                "max": [502.0, 503.0, 504.0],
                "min": [499.0, 500.0, 501.0],
                "close": [501.0, 502.0, 503.0],
            }),
            requests[8].request_id: pd.DataFrame([
                {"date": provider_as_of, "stock_id": "0050", "type": "twse", "industry_category": "ETF"},
                {"date": provider_as_of, "stock_id": "2330", "type": "twse", "industry_category": "半導體業"},
            ]),
        }

        legacy_dir = root / "data" / "tw_stock_data_vip"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for stock_id, adj_index, volume_index in (("0050", 6, 2), ("2330", 7, 3)):
            adjusted = frame_by_request[requests[adj_index].request_id]
            volume = frame_by_request[requests[volume_index].request_id]["Trading_Volume"].tolist()
            pd.DataFrame({
                "Date": adjusted["date"],
                "Open": adjusted["open"],
                "High": adjusted["max"],
                "Low": adjusted["min"],
                "Close": adjusted["close"],
                "Volume": volume,
            }).to_csv(legacy_dir / f"{stock_id}.csv", index=False)

        ledger_path = root / "data" / "market_data_v2" / "bootstrap" / manifest_fingerprint / "bootstrap_ledger.sqlite3"
        ledger = MarketDataJobLedger(ledger_path)
        workload_id = ledger.seed_manifest(manifest, now=datetime(2026, 9, 8, tzinfo=timezone.utc))
        evidence = []
        current = datetime(2026, 9, 8, tzinfo=timezone.utc)
        while True:
            job = ledger.claim_next_job(
                workload_id,
                owner_id="synthetic-round15",
                now=current,
                lease_until=current + timedelta(minutes=5),
            )
            if job is None:
                break
            request = job.to_request()
            path = resolve_market_data_request_parquet_path(root, manifest_fingerprint, request)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((request.request_id + "\n").encode("utf-8"))
            digest = compute_file_sha256(path)
            frame = frame_by_request[request.request_id]
            ledger.mark_done(
                workload_id,
                request.request_id,
                row_count=len(frame),
                content_sha256=digest,
                now=current,
            )
            evidence.append(
                ProviderArtifactEvidence(
                    request_id=request.request_id,
                    dataset=request.dataset,
                    row_count=len(frame),
                    content_sha256=digest,
                )
            )
            current += timedelta(seconds=1)
        ledger.set_workload_status(workload_id, status="DONE", now=current)
        provider_payload = build_provider_snapshot_payload(
            manifest=manifest,
            artifacts=evidence,
            finalized_at="2026-09-08T00:00:00+00:00",
        )
        provider_path = resolve_market_data_provider_snapshot_path(root, manifest_fingerprint)
        atomic_write_json(provider_path, provider_payload)

        def frame_reader(path, columns):
            frame = frame_by_request[path.stem].copy()
            if columns:
                missing = [column for column in columns if column not in frame.columns]
                if missing:
                    raise ValueError(f"synthetic round15 frame missing columns: {missing}")
                frame = frame.loc[:, list(columns)]
            return frame

        archive = load_ready_provider_snapshot_archive(root)
        view = ResearchV2ProviderView(project_root=root, archive=archive, frame_reader=frame_reader)
        adjusted_scope_frame = next(view.iter_dataset_scope_frames("TaiwanStockPriceAdj", columns=("date", "stock_id", "open", "max", "min", "close")))
        add_check(results, "market_data", case_id, "scope_reader_allows_only_authorized_adjusted_price_operands", 3, len(adjusted_scope_frame))
        adjusted_volume_blocked = False
        try:
            next(view.iter_dataset_scope_frames("TaiwanStockPriceAdj", columns=("date", "stock_id", "Trading_Volume")))
        except RuntimeError:
            adjusted_volume_blocked = True
        add_check(results, "market_data", case_id, "scope_reader_rejects_unlisted_adjusted_volume_field", True, adjusted_volume_blocked)

        candidate = build_research_v2_candidate(
            root,
            frame_reader=frame_reader,
            now=datetime(2026, 9, 8, 8, 0, tzinfo=timezone.utc),
        )
        add_check(results, "market_data", case_id, "candidate_computes_true_required_scope_common_complete_cutoff", cutoff, candidate["research_common_complete_cutoff"])
        add_check(results, "market_data", case_id, "candidate_common_complete_tail_starts_at_first_fully_usable_synthetic_date", d1, candidate["required_common_complete_start_date"])
        add_check(results, "market_data", case_id, "candidate_required_common_complete_tail_count_matches_calendar", 3, candidate["required_common_complete_tail_date_count"])
        add_check(results, "market_data", case_id, "candidate_has_no_foundation_blockers_when_required_scope_is_complete", [], candidate["blockers"])
        add_check(results, "market_data", case_id, "candidate_remains_unfrozen_even_after_common_complete_proof", None, candidate["frozen_cutoff"])
        add_check(results, "market_data", case_id, "candidate_never_authorizes_promotion", False, candidate["promotion_authorized"])
        add_check(results, "market_data", case_id, "candidate_keeps_active_research_v1", RESEARCH_DATA_GENERATION_V1, candidate["active_research_generation"])
        add_check(results, "market_data", case_id, "candidate_required_scope_coverage_is_persisted_and_fingerprinted", True, int(candidate["required_scope_coverage_row_count"]) == 3 and len(str(candidate["required_scope_coverage_fingerprint"])) == 64)
        add_check(results, "market_data", case_id, "candidate_required_common_complete_fingerprint_is_scientific_identity", str(candidate["required_common_complete"]["coverage_fingerprint"]), candidate["required_common_complete_fingerprint"])
        add_check(results, "market_data", case_id, "candidate_required_source_projection_is_self_fingerprinted", candidate["required_source_projection_fingerprint"], canonical_json_sha256(candidate["required_source_projection"]))
        add_check(results, "market_data", case_id, "candidate_pins_ready_adjusted_price_revision_proof", "REVISION_EQUIVALENCE_PROVEN", candidate["adjusted_price_revision_proof"]["status"])
        add_check(results, "market_data", case_id, "candidate_adjusted_price_revision_proof_identity_roundtrips", candidate["adjusted_price_revision_proof_fingerprint"], candidate["adjusted_price_revision_proof"]["proof_fingerprint"])

        universe_path = resolve_research_v2_daily_universe_path(root, provider_payload["snapshot_fingerprint"])
        conn = sqlite3.connect(universe_path)
        try:
            last_required = conn.execute(
                "SELECT date, missing_raw_volume_count, missing_price_limit_count, missing_adjusted_price_operand_count, required_complete "
                "FROM required_scope_coverage ORDER BY date DESC LIMIT 1"
            ).fetchone()
            common_cutoff = conn.execute("SELECT MAX(date) FROM required_common_complete").fetchone()[0]
            transient_tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                if str(row[0]) in {"raw_volume_field_evidence", "price_limit_evidence", "adjusted_price_operand_evidence", "delisting_event_evidence"}
            }
        finally:
            conn.close()
        add_check(results, "market_data", case_id, "required_scope_coverage_cutoff_row_is_field_complete", (cutoff, 0, 0, 0, 1), tuple(last_required))
        add_check(results, "market_data", case_id, "required_common_complete_sqlite_ceiling_equals_fixed_cutoff", cutoff, common_cutoff)
        add_check(results, "market_data", case_id, "row_level_transient_completeness_evidence_is_not_duplicated_in_persistent_sqlite", set(), transient_tables)

        loaded_candidate = load_research_v2_candidate(root, required=True)
        add_check(results, "market_data", case_id, "round15_candidate_identity_roundtrips", candidate["candidate_fingerprint"], loaded_candidate["candidate_fingerprint"])
        candidate_manifest_path = resolve_research_v2_candidate_manifest_path(root, provider_payload["snapshot_fingerprint"])
        add_check(results, "market_data", case_id, "candidate_manifest_remains_candidate_namespace_before_promotion", True, candidate_manifest_path.is_file() and "/candidates/" in candidate_manifest_path.as_posix())

        freeze = build_research_v2_freeze_candidate(
            root,
            frame_reader=frame_reader,
            now=datetime(2026, 9, 8, 8, 5, tzinfo=timezone.utc),
        )
        add_check(results, "market_data", case_id, "freeze_candidate_is_ready_artifact_not_active_generation", RESEARCH_V2_FREEZE_CANDIDATE_STATUS_READY, freeze["status"])
        add_check(results, "market_data", case_id, "freeze_candidate_pins_fixed_frozen_cutoff", cutoff, freeze["frozen_cutoff"])
        add_check(results, "market_data", case_id, "freeze_candidate_keeps_promotion_unauthorized", False, freeze["promotion_authorized"])
        add_check(results, "market_data", case_id, "freeze_candidate_does_not_claim_active_generation_changed", False, freeze["active_research_generation_changed"])
        add_check(results, "market_data", case_id, "freeze_candidate_pins_source_candidate_identity", candidate["candidate_fingerprint"], freeze["candidate_fingerprint"])
        add_check(results, "market_data", case_id, "freeze_candidate_pins_required_common_complete_identity", candidate["required_common_complete_fingerprint"], freeze["required_common_complete_fingerprint"])
        add_check(results, "market_data", case_id, "freeze_candidate_pins_required_source_projection_identity", candidate["required_source_projection_fingerprint"], freeze["required_source_projection_fingerprint"])
        add_check(results, "market_data", case_id, "freeze_candidate_pins_adjusted_price_revision_proof_identity", candidate["adjusted_price_revision_proof_fingerprint"], freeze["adjusted_price_revision_proof_fingerprint"])
        add_check(results, "market_data", case_id, "freeze_candidate_requires_exact_round14_required_scope", set(RESEARCH_V2_REQUIRED_DATASETS), set(freeze["required_dataset_scope"]))
        add_check(results, "market_data", case_id, "freeze_candidate_requires_exact_round14_common_complete_scope", set(RESEARCH_V2_COMMON_COMPLETE_DATASETS), set(freeze["common_complete_dataset_scope"]))
        freeze_manifest = resolve_research_v2_freeze_candidate_manifest_path(root, freeze["freeze_candidate_fingerprint"])
        add_check(results, "market_data", case_id, "freeze_candidate_is_stored_under_fingerprint_namespace", True, freeze_manifest.is_file() and f"/freeze_candidates/{freeze['freeze_candidate_fingerprint']}/" in freeze_manifest.as_posix())
        add_check(results, "market_data", case_id, "freeze_candidate_build_consumes_zero_provider_calls", 0, freeze["provider_calls"])

        reused = build_research_v2_freeze_candidate(
            root,
            frame_reader=frame_reader,
            now=datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc),
        )
        add_check(results, "market_data", case_id, "freeze_candidate_rebuild_reuses_same_immutable_identity", freeze["freeze_candidate_fingerprint"], reused["freeze_candidate_fingerprint"])
        add_check(results, "market_data", case_id, "freeze_candidate_rebuild_reports_reuse", True, reused["reused"])
        loaded_freeze = load_research_v2_freeze_candidate(
            root,
            freeze_candidate_fingerprint=freeze["freeze_candidate_fingerprint"],
            required=True,
        )
        add_check(results, "market_data", case_id, "freeze_candidate_loader_roundtrips_identity", freeze["freeze_candidate_fingerprint"], loaded_freeze["freeze_candidate_fingerprint"])
        frozen_universe = resolve_research_v2_frozen_daily_universe_path(root, freeze["freeze_candidate_fingerprint"])
        frozen_candidate_manifest = resolve_research_v2_frozen_source_candidate_manifest_path(root, freeze["freeze_candidate_fingerprint"])
        add_check(results, "market_data", case_id, "freeze_candidate_owns_fingerprint_addressed_daily_universe_copy", True, frozen_universe.is_file() and compute_file_sha256(frozen_universe) == freeze["daily_universe_file_sha256"])
        add_check(results, "market_data", case_id, "freeze_candidate_owns_fingerprint_addressed_source_manifest_copy", True, frozen_candidate_manifest.is_file() and compute_file_sha256(frozen_candidate_manifest) == freeze["candidate_manifest_sha256"])
        original_universe_bytes = universe_path.read_bytes()
        original_candidate_manifest_bytes = candidate_manifest_path.read_bytes()
        universe_path.write_bytes(original_universe_bytes + b"candidate-slot-drift")
        candidate_manifest_path.write_bytes(original_candidate_manifest_bytes + b"\n")
        frozen_after_source_drift = load_research_v2_freeze_candidate(
            root,
            freeze_candidate_fingerprint=freeze["freeze_candidate_fingerprint"],
            required=True,
        )
        add_check(results, "market_data", case_id, "freeze_loader_is_independent_of_mutable_candidate_physical_slot", freeze["freeze_candidate_fingerprint"], frozen_after_source_drift["freeze_candidate_fingerprint"])
        universe_path.write_bytes(original_universe_bytes)
        candidate_manifest_path.write_bytes(original_candidate_manifest_bytes)

        v2_generation = get_research_data_generation(RESEARCH_DATA_GENERATION_V2)
        add_check(results, "market_data", case_id, "round15_does_not_change_configured_v2_status", RESEARCH_STATUS_AUTHORIZED_NOT_READY, v2_generation.status)
        add_check(results, "market_data", case_id, "round15_does_not_change_configured_v2_cutoff", None, v2_generation.cutoff)
        add_check(results, "market_data", case_id, "round15_does_not_change_active_research_generation", RESEARCH_DATA_GENERATION_V1, ACTIVE_RESEARCH_DATA_GENERATION)

        persisted_freeze = load_json_strict(freeze_manifest)
        add_check(results, "market_data", case_id, "freeze_candidate_manifest_paths_are_project_relative", False, str(persisted_freeze["candidate_manifest_path"]).startswith(str(root)))

    summary.update(
        {
            "checks": len(results),
            "required_dataset_count": len(RESEARCH_V2_REQUIRED_DATASETS),
            "common_complete_dataset_count": len(RESEARCH_V2_COMMON_COMPLETE_DATASETS),
            "scope_contract_fingerprint": scope_stats["contract_fingerprint"],
        }
    )
    return results, summary


def validate_market_data_v2_research_promotion_consumer_integration_contract_case(_base_params):
    """Round-16 proves explicit promotion, immutable materialization and active consumer routing."""

    from datetime import date, datetime, timedelta, timezone
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from config.market_data import ACTIVE_RESEARCH_DATA_GENERATION, RESEARCH_DATA_GENERATION_V1, RESEARCH_DATA_GENERATION_V2, RESEARCH_REQUIRED_CUTOFF
    from core.dataset_profiles import get_dataset_dir
    from core.file_integrity import atomic_write_json, compute_file_sha256
    from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest, build_registry_fingerprint
    from core.market_data_dataset_registry import BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, BOOTSTRAP_SINGLE_FULL_RANGE, BOOTSTRAP_SINGLE_NO_DATES, get_market_dataset_specs
    from core.market_data_research_promotion import (
        build_effective_market_data_contract_snapshot,
        get_effective_research_data_generation,
    )
    from core.market_data_provider_snapshot import ProviderArtifactEvidence, build_provider_snapshot_payload
    from core.market_data_research_storage_contract import resolve_active_research_generation_path, resolve_research_v2_promotion_manifest_path
    from core.market_data_storage_contract import resolve_market_data_provider_snapshot_path, resolve_market_data_request_parquet_path
    from core.runtime_domains import (
        assert_runtime_write_path_is_not_research_dataset,
        build_runtime_domain_contract_snapshot,
    )
    from services.downloader.market_data_ledger import MarketDataJobLedger
    from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
    from services.research.market_data_generation import (
        load_active_research_v2_read_view,
        promote_research_v2,
        validate_research_v2_compatibility_materialization,
    )
    from services.research.market_data_v2 import build_research_v2_candidate, build_research_v2_freeze_candidate

    case_id = "MARKET_DATA_V2_RESEARCH_PROMOTION_CONSUMER_INTEGRATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cutoff_date = date.fromisoformat(RESEARCH_REQUIRED_CUTOFF)
        dates = [
            (cutoff_date - timedelta(days=4)).isoformat(),
            (cutoff_date - timedelta(days=3)).isoformat(),
            cutoff_date.isoformat(),
        ]
        cutoff = dates[-1]
        after_cutoff = (cutoff_date + timedelta(days=1)).isoformat()
        provider_dates = [*dates, after_cutoff]
        provider_as_of = (cutoff_date + timedelta(days=190)).isoformat()
        manifest_fingerprint = "9" * 64
        registry_fingerprint = build_registry_fingerprint(get_market_dataset_specs(included_only=True))
        requests = (
            BootstrapHttpRequest("TaiwanStockTradingDate", BOOTSTRAP_SINGLE_NO_DATES, None, None, None),
            BootstrapHttpRequest("TaiwanStockDelisting", BOOTSTRAP_SINGLE_FULL_RANGE, None, "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPrice", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceLimit", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockPriceAdj", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, "0050", "1900-01-01", provider_as_of),
            BootstrapHttpRequest("TaiwanStockInfo", BOOTSTRAP_SINGLE_NO_DATES, None, None, None),
        )
        manifest = BootstrapRequestManifest(
            as_of_date=provider_as_of,
            full_range_start="1900-01-01",
            registry_fingerprint=registry_fingerprint,
            manifest_fingerprint=manifest_fingerprint,
            historical_instrument_count=1,
            requests=requests,
        )
        frame_by_request = {
            requests[0].request_id: pd.DataFrame({"date": dates}),
            requests[1].request_id: pd.DataFrame(columns=["date", "stock_id"]),
            requests[2].request_id: pd.DataFrame({
                "date": provider_dates,
                "stock_id": ["0050"] * 4,
                "Trading_Volume": [1000.0, 1100.0, 1200.0, 9999.0],
            }),
            requests[3].request_id: pd.DataFrame({"date": dates, "stock_id": ["0050"] * 3}),
            requests[4].request_id: pd.DataFrame({
                "date": provider_dates,
                "stock_id": ["0050"] * 4,
                "open": [100.0, 101.0, 102.0, 999.0],
                "max": [102.0, 103.0, 104.0, 1000.0],
                "min": [99.0, 100.0, 101.0, 998.0],
                "close": [101.0, 102.0, 103.0, 999.5],
            }),
            requests[5].request_id: pd.DataFrame([
                {"date": provider_as_of, "stock_id": "0050", "type": "twse", "industry_category": "ETF"},
            ]),
        }
        legacy_dir = root / "data" / "tw_stock_data_vip"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        adjusted = frame_by_request[requests[4].request_id]
        pd.DataFrame({
            "Date": adjusted.iloc[:3]["date"],
            "Open": adjusted.iloc[:3]["open"],
            "High": adjusted.iloc[:3]["max"],
            "Low": adjusted.iloc[:3]["min"],
            "Close": adjusted.iloc[:3]["close"],
            "Volume": frame_by_request[requests[2].request_id].iloc[:3]["Trading_Volume"].tolist(),
        }).to_csv(legacy_dir / "0050.csv", index=False)

        ledger_path = root / "data" / "market_data_v2" / "bootstrap" / manifest_fingerprint / "bootstrap_ledger.sqlite3"
        ledger = MarketDataJobLedger(ledger_path)
        workload_id = ledger.seed_manifest(manifest, now=datetime(2026, 9, 8, tzinfo=timezone.utc))
        evidence = []
        current = datetime(2026, 9, 8, tzinfo=timezone.utc)
        while True:
            job = ledger.claim_next_job(
                workload_id,
                owner_id="synthetic-round16",
                now=current,
                lease_until=current + timedelta(minutes=5),
            )
            if job is None:
                break
            request = job.to_request()
            path = resolve_market_data_request_parquet_path(root, manifest_fingerprint, request)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((request.request_id + "\n").encode("utf-8"))
            digest = compute_file_sha256(path)
            frame = frame_by_request[request.request_id]
            ledger.mark_done(workload_id, request.request_id, row_count=len(frame), content_sha256=digest, now=current)
            evidence.append(ProviderArtifactEvidence(request_id=request.request_id, dataset=request.dataset, row_count=len(frame), content_sha256=digest))
            current += timedelta(seconds=1)
        ledger.set_workload_status(workload_id, status="DONE", now=current)
        provider_payload = build_provider_snapshot_payload(
            manifest=manifest,
            artifacts=evidence,
            finalized_at="2026-09-08T00:00:00+00:00",
        )
        atomic_write_json(resolve_market_data_provider_snapshot_path(root, manifest_fingerprint), provider_payload)

        def frame_reader(path, columns):
            frame = frame_by_request[path.stem].copy()
            if columns:
                missing = [column for column in columns if column not in frame.columns]
                if missing:
                    raise ValueError(f"synthetic round16 frame missing columns: {missing}")
                frame = frame.loc[:, list(columns)]
            return frame

        load_ready_provider_snapshot_archive(root)
        before = get_effective_research_data_generation(root)
        add_check(results, "market_data", case_id, "promotion_starts_from_safe_v1_fallback", RESEARCH_DATA_GENERATION_V1, before.generation_id)
        add_check(results, "market_data", case_id, "static_config_remains_safe_v1_fallback", RESEARCH_DATA_GENERATION_V1, ACTIVE_RESEARCH_DATA_GENERATION)
        add_check(results, "market_data", case_id, "full_dataset_before_promotion_uses_legacy_v1_path", str((root / "data" / "tw_stock_data_vip").resolve()), str(Path(get_dataset_dir(root, "full")).resolve()))

        candidate = build_research_v2_candidate(root, frame_reader=frame_reader, now=datetime(2026, 9, 8, 8, 0, tzinfo=timezone.utc))
        freeze = build_research_v2_freeze_candidate(root, frame_reader=frame_reader, now=datetime(2026, 9, 8, 8, 5, tzinfo=timezone.utc))
        add_check(results, "market_data", case_id, "promotion_fixture_candidate_reaches_fixed_common_complete_cutoff", cutoff, candidate["research_common_complete_cutoff"])
        add_check(results, "market_data", case_id, "promotion_consumes_ready_freeze_candidate", cutoff, freeze["frozen_cutoff"])

        permission_blocked = False
        try:
            promote_research_v2(
                root,
                freeze_candidate_fingerprint=freeze["freeze_candidate_fingerprint"],
                explicit_authorization=False,
                frame_reader=frame_reader,
            )
        except PermissionError:
            permission_blocked = True
        add_check(results, "market_data", case_id, "promotion_requires_explicit_authorization", True, permission_blocked)
        add_check(results, "market_data", case_id, "rejected_promotion_does_not_publish_active_pointer", False, resolve_active_research_generation_path(root).exists())

        promoted = promote_research_v2(
            root,
            freeze_candidate_fingerprint=freeze["freeze_candidate_fingerprint"],
            explicit_authorization=True,
            frame_reader=frame_reader,
            now=datetime(2026, 9, 8, 8, 10, tzinfo=timezone.utc),
        )
        materialization = dict(promoted["materialization"])
        add_check(results, "market_data", case_id, "explicit_promotion_publishes_v2", RESEARCH_DATA_GENERATION_V2, promoted["generation_id"])
        add_check(results, "market_data", case_id, "promotion_pins_fixed_cutoff", cutoff, promoted["frozen_cutoff"])
        add_check(results, "market_data", case_id, "promotion_and_materialization_make_zero_provider_calls", 0, int(promoted["provider_calls"]) + int(materialization["provider_calls"]))
        add_check(results, "market_data", case_id, "compatibility_materialization_has_one_ticker_file", 1, materialization["csv_file_count"])
        add_check(results, "market_data", case_id, "compatibility_materialization_has_three_eligible_rows", 3, materialization["row_count"])

        promoted_dir = Path(get_dataset_dir(root, "full")).resolve()
        csv_path = promoted_dir / "0050.csv"
        csv_frame = pd.read_csv(csv_path)
        add_check(results, "market_data", case_id, "legacy_compatibility_schema_is_exact_six_column_ohlcv", ["Date", "Open", "High", "Low", "Close", "Volume"], list(csv_frame.columns))
        add_check(results, "market_data", case_id, "compatibility_ohlc_comes_from_canonical_price_adj", [100.0, 101.0, 102.0], csv_frame["Open"].tolist())
        add_check(results, "market_data", case_id, "compatibility_volume_comes_from_raw_price", [1000, 1100, 1200], csv_frame["Volume"].tolist())
        add_check(results, "market_data", case_id, "full_consumer_routes_to_promoted_immutable_materialization", True, "/materializations/" in promoted_dir.as_posix())
        add_check(results, "market_data", case_id, "reduced_fixture_is_not_redirected_by_research_promotion", str((root / "data" / "tw_stock_data_vip_reduced").resolve()), str(Path(get_dataset_dir(root, "reduced")).resolve()))

        active = get_effective_research_data_generation(root)
        add_check(results, "market_data", case_id, "effective_active_generation_switches_only_after_pointer", RESEARCH_DATA_GENERATION_V2, active.generation_id)
        add_check(results, "market_data", case_id, "effective_active_contract_is_frozen", cutoff, active.cutoff)
        contract_snapshot = build_effective_market_data_contract_snapshot(root)
        add_check(results, "market_data", case_id, "market_data_contract_snapshot_reports_effective_v2", RESEARCH_DATA_GENERATION_V2, contract_snapshot["active_research_generation"]["generation_id"])
        runtime = build_runtime_domain_contract_snapshot(root)
        add_check(results, "market_data", case_id, "runtime_domain_reports_research_v2", RESEARCH_DATA_GENERATION_V2, runtime["research"]["market_data_generation"])
        add_check(results, "market_data", case_id, "runtime_domain_routes_research_data_dir_to_materialization", promoted_dir, Path(runtime["research"]["data_dir"]).resolve())
        add_check(results, "market_data", case_id, "runtime_domain_keeps_trading_root_separate", True, Path(runtime["trading"]["data_dir"]).resolve() != promoted_dir)
        legacy_write_blocked = False
        promoted_write_blocked = False
        try:
            assert_runtime_write_path_is_not_research_dataset(root, legacy_dir / "0050.csv")
        except RuntimeError:
            legacy_write_blocked = True
        try:
            assert_runtime_write_path_is_not_research_dataset(root, promoted_dir / "0050.csv")
        except RuntimeError:
            promoted_write_blocked = True
        add_check(results, "market_data", case_id, "promotion_keeps_legacy_v1_research_root_write_protected", True, legacy_write_blocked)
        add_check(results, "market_data", case_id, "promotion_keeps_v2_materialization_root_write_protected", True, promoted_write_blocked)

        from filters.breakout_quality.source_inventory import build_source_data_inventory
        source_inventory = build_source_data_inventory(root, "full")
        source_lineage = dict(source_inventory.get("research_market_data_lineage") or {})
        add_check(results, "market_data", case_id, "full_research_source_inventory_pins_v2_generation", RESEARCH_DATA_GENERATION_V2, source_lineage.get("generation_id"))
        add_check(results, "market_data", case_id, "full_research_source_inventory_pins_promotion_identity", promoted["promotion_fingerprint"], source_lineage.get("promotion_fingerprint"))
        add_check(results, "market_data", case_id, "full_research_source_inventory_pins_materialization_identity", materialization["materialization_fingerprint"], source_lineage.get("materialization_fingerprint"))

        read_view = load_active_research_v2_read_view(root, frame_reader=frame_reader)
        add_check(results, "market_data", case_id, "active_v2_read_view_pins_frozen_cutoff", cutoff, read_view.frozen_cutoff)
        add_check(results, "market_data", case_id, "active_v2_read_view_uses_exact_daily_universe", ("0050",), read_view.eligible_stock_ids(cutoff))
        post_cutoff_blocked = False
        try:
            read_view.eligible_stock_ids((cutoff_date + timedelta(days=1)).isoformat())
        except ValueError:
            post_cutoff_blocked = True
        add_check(results, "market_data", case_id, "active_v2_read_view_rejects_post_cutoff_queries", True, post_cutoff_blocked)
        scoped_dates = []
        for frame in read_view.iter_dataset_scope_frames(
            "TaiwanStockPrice", columns=("date", "stock_id", "Trading_Volume")
        ):
            scoped_dates.extend(frame["date"].astype(str).tolist())
        add_check(results, "market_data", case_id, "active_scope_reader_filters_provider_rows_after_frozen_cutoff", dates, scoped_dates)
        scoped_without_date_rows = sum(
            len(frame)
            for frame in read_view.iter_dataset_scope_frames(
                "TaiwanStockPrice", columns=("stock_id", "Trading_Volume")
            )
        )
        add_check(results, "market_data", case_id, "active_scope_reader_enforces_cutoff_even_when_consumer_omits_date", 3, scoped_without_date_rows)

        promotion_manifest = resolve_research_v2_promotion_manifest_path(root, promoted["promotion_fingerprint"])
        add_check(results, "market_data", case_id, "promotion_manifest_is_immutable_fingerprint_addressed", True, promotion_manifest.is_file() and f"/promotions/{promoted['promotion_fingerprint']}/" in promotion_manifest.as_posix())
        add_check(results, "market_data", case_id, "active_pointer_is_small_separate_state", True, resolve_active_research_generation_path(root).is_file() and resolve_active_research_generation_path(root) != promotion_manifest)

        reused = promote_research_v2(
            root,
            freeze_candidate_fingerprint=freeze["freeze_candidate_fingerprint"],
            explicit_authorization=True,
            frame_reader=frame_reader,
        )
        add_check(results, "market_data", case_id, "same_promotion_identity_reuses_existing_activation", True, reused["reused"])
        add_check(results, "market_data", case_id, "same_promotion_identity_keeps_same_fingerprint", promoted["promotion_fingerprint"], reused["promotion_fingerprint"])

        resolve_active_research_generation_path(root).unlink()
        pointer_loss_fail_closed = False
        try:
            get_effective_research_data_generation(root)
        except RuntimeError:
            pointer_loss_fail_closed = True
        add_check(results, "market_data", case_id, "published_promotion_with_missing_pointer_fails_closed_instead_of_silent_v1_rollback", True, pointer_loss_fail_closed)
        recovered = promote_research_v2(
            root,
            freeze_candidate_fingerprint=freeze["freeze_candidate_fingerprint"],
            explicit_authorization=True,
            frame_reader=frame_reader,
            now=datetime(2026, 9, 8, 8, 20, tzinfo=timezone.utc),
        )
        add_check(results, "market_data", case_id, "explicit_same_freeze_promotion_recovers_missing_active_pointer", True, recovered.get("recovered_pointer"))
        add_check(results, "market_data", case_id, "pointer_recovery_restores_v2_without_new_scientific_identity", promoted["promotion_fingerprint"], recovered["promotion_fingerprint"])

        candidate_after_promotion_blocked = False
        try:
            build_research_v2_candidate(root, frame_reader=frame_reader)
        except RuntimeError:
            candidate_after_promotion_blocked = True
        add_check(results, "market_data", case_id, "active_v2_blocks_new_candidate_build_on_same_research_generation", True, candidate_after_promotion_blocked)

        deep = validate_research_v2_compatibility_materialization(
            root,
            materialization_fingerprint=materialization["materialization_fingerprint"],
            deep=True,
        )
        add_check(results, "market_data", case_id, "deep_materialization_validation_roundtrips_inventory", materialization["dataset_inventory_sha256"], deep["dataset_inventory_sha256"])
        original = csv_path.read_bytes()
        csv_path.write_bytes(original + b"\n")
        active_routing_tamper_blocked = False
        try:
            get_dataset_dir(root, "full")
        except ValueError:
            active_routing_tamper_blocked = True
        add_check(results, "market_data", case_id, "normal_active_routing_detects_materialized_csv_tamper", True, active_routing_tamper_blocked)
        tamper_blocked = False
        try:
            validate_research_v2_compatibility_materialization(
                root,
                materialization_fingerprint=materialization["materialization_fingerprint"],
                deep=True,
            )
        except ValueError:
            tamper_blocked = True
        add_check(results, "market_data", case_id, "deep_validation_detects_materialized_csv_tamper", True, tamper_blocked)
        csv_path.write_bytes(original)
        add_check(results, "market_data", case_id, "active_routing_recovers_after_exact_materialized_bytes_are_restored", promoted_dir, Path(get_dataset_dir(root, "full")).resolve())

    summary.update({"checks": len(results), "promotion_is_explicit": True, "active_generation": RESEARCH_DATA_GENERATION_V2})
    return results, summary


def validate_market_data_rounds_1_16_repair2_research_pit_projection_contract_case(_base_params):
    """Repair-2 proves post-cutoff PriceAdj equivalence and required-source identity isolation."""

    from pathlib import Path
    import sqlite3
    from tempfile import TemporaryDirectory

    from core.file_integrity import canonical_json_sha256
    from core.market_data_adjusted_price_revision_proof import (
        ADJUSTED_PRICE_PROVIDER_CORRECTION_DATE,
        ADJUSTED_PRICE_REVISION_STATUS_BLOCKED,
        ADJUSTED_PRICE_REVISION_STATUS_READY,
    )
    from core.market_data_provider_snapshot import provider_snapshot_identity_from_payload
    from core.market_data_research_freeze import RESEARCH_V2_FREEZE_CANDIDATE_IDENTITY_FIELDS
    from core.market_data_research_materialization import RESEARCH_V2_COMPAT_MATERIALIZATION_IDENTITY_FIELDS
    from core.market_data_research_promotion import RESEARCH_V2_PROMOTION_IDENTITY_FIELDS
    from core.market_data_research_v2 import RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS
    from services.market_data.provider_snapshot_repository import validate_provider_snapshot_payload
    from services.research.adjusted_price_revision_proof import build_adjusted_price_revision_proof

    case_id = "MARKET_DATA_ROUNDS_1_16_REPAIR2"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    identity_sets = (
        set(RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS),
        set(RESEARCH_V2_FREEZE_CANDIDATE_IDENTITY_FIELDS),
        set(RESEARCH_V2_COMPAT_MATERIALIZATION_IDENTITY_FIELDS),
        set(RESEARCH_V2_PROMOTION_IDENTITY_FIELDS),
    )
    add_check(results, "market_data", case_id, "required_source_projection_propagates_across_candidate_freeze_materialization_promotion", True, all("required_source_projection_fingerprint" in fields for fields in identity_sets))
    add_check(results, "market_data", case_id, "adjusted_revision_proof_propagates_across_candidate_freeze_materialization_promotion", True, all("adjusted_price_revision_proof_fingerprint" in fields for fields in identity_sets))
    add_check(results, "market_data", case_id, "full_provider_snapshot_is_provenance_not_foundation_identity", True, all("provider_snapshot_fingerprint" not in fields for fields in identity_sets))
    add_check(results, "market_data", case_id, "post_cutoff_provider_rebuild_date_is_explicit", "2026-09-01", ADJUSTED_PRICE_PROVIDER_CORRECTION_DATE)

    synthetic_identity = {
        "schema_version": 1,
        "provider": "FinMind",
        "snapshot_role": "neutral_provider_bootstrap",
        "status": "READY",
        "as_of_date": "2030-01-15",
        "registry_fingerprint": "1" * 64,
        "manifest_fingerprint": "2" * 64,
        "historical_instrument_count": 0,
        "total_requests": 0,
        "total_rows": 0,
        "artifacts_fingerprint": canonical_json_sha256([]),
        "datasets": [],
    }
    synthetic_snapshot = {
        **synthetic_identity,
        "snapshot_fingerprint": canonical_json_sha256(synthetic_identity),
        "finalized_at": "2030-01-16T00:00:00+00:00",
    }
    validated_snapshot = validate_provider_snapshot_payload(synthetic_snapshot)
    add_check(results, "market_data", case_id, "historical_provider_snapshot_validates_from_persisted_self_identity", synthetic_snapshot["snapshot_fingerprint"], validated_snapshot["snapshot_fingerprint"])
    add_check(results, "market_data", case_id, "provider_snapshot_self_identity_preserves_registry_provenance_without_current_registry_dependency", "1" * 64, provider_snapshot_identity_from_payload(validated_snapshot)["registry_fingerprint"])

    class _ProviderView:
        def __init__(self, frame):
            self.frame = frame

        def iter_dataset_scope_frames(self, _dataset, *, columns=None, data_id=None):
            frame = self.frame.copy()
            if data_id is not None:
                frame = frame.loc[frame["stock_id"].astype(str) == str(data_id)]
            if columns:
                frame = frame.loc[:, list(columns)]
            if not frame.empty:
                yield frame.reset_index(drop=True)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        legacy_dir = root / "data" / "tw_stock_data_vip"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy = pd.DataFrame({
            "Date": ["2030-01-14", "2030-01-15"],
            "Open": [100.0, 110.0], "High": [102.0, 112.0], "Low": [99.0, 109.0], "Close": [101.0, 111.0],
            "Volume": [1000.0, 1100.0],
        })
        legacy.to_csv(legacy_dir / "0050.csv", index=False)
        universe_path = root / "daily_universe.sqlite3"
        conn = sqlite3.connect(universe_path)
        try:
            conn.execute("CREATE TABLE daily_universe(date TEXT NOT NULL, stock_id TEXT NOT NULL)")
            conn.executemany("INSERT INTO daily_universe(date, stock_id) VALUES (?, ?)", [("2030-01-14", "0050"), ("2030-01-15", "0050")])
            conn.commit()
        finally:
            conn.close()

        scalar_current = pd.DataFrame({
            "date": ["2030-01-14", "2030-01-15"], "stock_id": ["0050", "0050"],
            "open": [50.0, 55.0], "max": [51.0, 56.0], "min": [49.5, 54.5], "close": [50.5, 55.5],
        })
        ready = build_adjusted_price_revision_proof(root, provider_view=_ProviderView(scalar_current), daily_universe_path=universe_path, required_cutoff="2030-01-15")
        add_check(results, "market_data", case_id, "uniform_per_instrument_scalar_revision_is_proven_equivalent", ADJUSTED_PRICE_REVISION_STATUS_READY, ready["status"])
        add_check(results, "market_data", case_id, "ready_revision_proof_is_content_addressed", 64, len(str(ready["proof_fingerprint"])))

        non_scalar = scalar_current.copy()
        non_scalar.loc[1, "close"] = 55.0
        blocked = build_adjusted_price_revision_proof(root, provider_view=_ProviderView(non_scalar), daily_universe_path=universe_path, required_cutoff="2030-01-15")
        add_check(results, "market_data", case_id, "non_uniform_historical_rebuild_fails_closed", ADJUSTED_PRICE_REVISION_STATUS_BLOCKED, blocked["status"])
        add_check(results, "market_data", case_id, "non_uniform_revision_has_distinct_proof_identity", True, blocked["proof_fingerprint"] != ready["proof_fingerprint"])

    summary.update({"checks": len(results), "repair_contract": "research_pit_required_source_projection_v1"})
    return results, summary


def validate_market_data_rounds_1_16_repair1_contract_case(_base_params):
    """Audit Repair 1 keeps date identity and historical eligibility fail closed."""

    from datetime import timedelta
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.market_data_contract import (
        ADJUSTED_PRICE_NO_PRICE_DAY_POLICY,
        RESEARCH_DAILY_BAR_CLOCK,
        get_market_price_source_contract,
    )
    from core.market_data_instrument_universe import (
        build_historical_market_state_guard,
        historical_market_state_guard_fingerprint,
        is_historical_market_state_eligible,
    )
    from core.trading_market_clock import trading_daily_bar_complete_time
    from services.downloader import runtime as downloader_runtime
    from services.downloader.universe import _load_reusable_universe_cache, _publish_universe_cache

    case_id = "MARKET_DATA_ROUNDS_1_16_REPAIR1"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    with TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "universe_cache_v3.json"
        _publish_universe_cache(cache_path, qualified_tickers=["2330"], market_date="2026-09-01")
        now = downloader_runtime.get_taipei_now()
        stale = _load_reusable_universe_cache(cache_path, now=now, market_date="2026-09-08")
        same_date = _load_reusable_universe_cache(cache_path, now=now, market_date="2026-09-01")
        add_check(results, "market_data", case_id, "cache_reuse_is_bound_to_requested_market_date", None, stale)
        add_check(results, "market_data", case_id, "same_market_date_cache_remains_reusable", ["2330"], same_date)

    from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET
    from core.market_data_dataset_registry import get_market_dataset_spec
    from core.market_data_freshness_contract import build_market_data_freshness_contract

    publish_time = trading_daily_bar_complete_time()
    freshness = build_market_data_freshness_contract(get_market_dataset_spec(FINMIND_ADJUSTED_PRICE_DATASET))
    expected_publish_time = tuple(
        int(part) for part in str(freshness.publication_first_check_time).split(":", 1)
    )
    add_check(results, "market_data", case_id, "daily_completion_uses_provider_verified_publication_policy", expected_publish_time, publish_time)

    stock_info = pd.DataFrame(
        [
            {"stock_id": "9999", "type": "emerging", "industry_category": "其他", "date": "2025-01-02"},
            {"stock_id": "9999", "type": "twse", "industry_category": "其他", "date": "2026-09-08"},
            {"stock_id": "2330", "type": "twse", "industry_category": "半導體業", "date": "2026-09-08"},
            {"stock_id": "8888", "type": "emerging", "industry_category": "其他", "date": "2024-06-30"},
        ]
    )
    guard = build_historical_market_state_guard(
        stock_info, historical_instruments=("9999", "2330", "8888")
    )
    guard_fp = historical_market_state_guard_fingerprint(guard)
    add_check(
        results, "market_data", case_id, "transition_guard_covers_listed_and_delisting_only_archive_members",
        {"8888": "2024-06-30", "9999": "2025-01-02"}, guard,
    )
    add_check(results, "market_data", case_id, "transition_guard_identity_is_content_addressed", 64, len(guard_fp))
    add_check(
        results,
        "market_data",
        case_id,
        "future_board_transition_cannot_authorize_earlier_emerging_sample",
        False,
        is_historical_market_state_eligible(stock_id="9999", date_value="2024-12-31", transition_excluded_through=guard),
    )
    add_check(
        results,
        "market_data",
        case_id,
        "post_transition_date_can_be_eligible",
        True,
        is_historical_market_state_eligible(stock_id="9999", date_value="2025-01-03", transition_excluded_through=guard),
    )

    price_contract = get_market_price_source_contract()
    add_check(results, "market_data", case_id, "research_bar_clock_semantics_are_explicit", RESEARCH_DAILY_BAR_CLOCK, price_contract.research_daily_bar_clock)
    add_check(
        results,
        "market_data",
        case_id,
        "no_price_day_policy_preserves_existing_provider_calendar_row_semantics",
        ADJUSTED_PRICE_NO_PRICE_DAY_POLICY,
        price_contract.adjusted_price_no_price_day_policy,
    )

    summary.update({"checks": len(results), "repair_contract": "market_date_and_historical_market_state_v1"})
    return results, summary

def validate_market_data_rounds_1_16_repair3_lifecycle_integrity_contract_case(_base_params):
    """Repair-3 generic lifecycle invariants: durable activation, immutable roots and content integrity."""

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from config.market_data import RESEARCH_DATA_GENERATION_V1
    from core.file_integrity import canonical_json_sha256, compute_file_sha256
    from core.market_data_research_materialization import validate_research_v2_materialization_file_integrity
    from core.market_data_research_promotion import (
        discover_published_research_v2_promotion_fingerprints,
        get_effective_research_data_generation,
    )
    from core.market_data_research_storage_contract import (
        RESEARCH_MARKET_DATA_V2_PROMOTIONS_DIRNAME,
        RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT,
        resolve_research_v2_compatibility_dataset_dir,
    )
    from core.runtime_domains import assert_runtime_write_path_is_not_research_dataset

    case_id = "MARKET_DATA_ROUNDS_1_16_REPAIR3"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        legacy_full = root / "data" / "tw_stock_data_vip"
        legacy_reduced = root / "data" / "tw_stock_data_vip_reduced"
        v2_root = root / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT
        trading_root = root / "data" / "trading" / "tw_stock_data_vip"
        for path in (legacy_full, legacy_reduced, v2_root, trading_root):
            path.mkdir(parents=True, exist_ok=True)

        blocked = []
        for path in (legacy_full / "x.csv", legacy_reduced / "x.csv", v2_root / "x.bin"):
            try:
                assert_runtime_write_path_is_not_research_dataset(root, path)
            except RuntimeError:
                blocked.append(path)
        add_check(results, "market_data", case_id, "write_guard_protects_all_legacy_and_v2_research_roots_independent_of_active_generation", 3, len(blocked))
        trading_allowed = True
        try:
            assert_runtime_write_path_is_not_research_dataset(root, trading_root / "x.csv")
        except RuntimeError:
            trading_allowed = False
        add_check(results, "market_data", case_id, "write_guard_still_allows_trading_namespace", True, trading_allowed)

        promotions_root = root / RESEARCH_MARKET_DATA_V2_RELATIVE_ROOT / RESEARCH_MARKET_DATA_V2_PROMOTIONS_DIRNAME
        promotions_root.mkdir(parents=True, exist_ok=True)
        (promotions_root / ".staged-crash.tmp").mkdir()
        add_check(results, "market_data", case_id, "hidden_stage_directory_is_not_published_promotion_truth", (), discover_published_research_v2_promotion_fingerprints(root))
        add_check(results, "market_data", case_id, "stage_only_state_preserves_never_promoted_v1_fallback", RESEARCH_DATA_GENERATION_V1, get_effective_research_data_generation(root).generation_id)

        published_fp = "a" * 64
        (promotions_root / published_fp).mkdir()
        add_check(results, "market_data", case_id, "final_fingerprint_directory_is_durable_promotion_state_evidence", (published_fp,), discover_published_research_v2_promotion_fingerprints(root))
        missing_pointer_blocked = False
        try:
            get_effective_research_data_generation(root)
        except RuntimeError:
            missing_pointer_blocked = True
        add_check(results, "market_data", case_id, "published_promotion_without_pointer_fails_closed", True, missing_pointer_blocked)
        (promotions_root / published_fp).rmdir()

        materialization_fp = "b" * 64
        dataset_dir = resolve_research_v2_compatibility_dataset_dir(root, materialization_fp)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dataset_dir / "0050.csv"
        original = b"Date,Close\n2026-03-02,100\n"
        csv_path.write_bytes(original)
        files = [{
            "ticker": "0050",
            "relative_path": "0050.csv",
            "row_count": 1,
            "size_bytes": len(original),
            "content_sha256": compute_file_sha256(csv_path),
        }]
        payload = {"files": files, "dataset_inventory_sha256": canonical_json_sha256(files)}
        validated_dir = validate_research_v2_materialization_file_integrity(
            root, materialization_fingerprint=materialization_fp, payload=payload
        )
        add_check(results, "market_data", case_id, "content_addressed_materialization_integrity_accepts_exact_bytes", dataset_dir, validated_dir)
        tampered = original.replace(b",100\n", b",999\n")
        add_check(results, "market_data", case_id, "same_length_tamper_fixture_preserves_size", len(original), len(tampered))
        csv_path.write_bytes(tampered)
        content_tamper_blocked = False
        try:
            validate_research_v2_materialization_file_integrity(
                root, materialization_fingerprint=materialization_fp, payload=payload
            )
        except ValueError:
            content_tamper_blocked = True
        add_check(results, "market_data", case_id, "content_addressed_integrity_rejects_same_size_byte_tamper", True, content_tamper_blocked)

    summary.update({"checks": len(results), "repair_round": 3})
    return results, summary



def validate_market_data_rounds_1_16_repair4_downstream_generation_identity_contract_case(_base_params):
    """Repair-4 generic downstream identity invariants across Optimizer, Compare and BQ."""

    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    from config.market_data import RESEARCH_DATA_GENERATION_V2, RESEARCH_REQUIRED_CUTOFF
    from core.dataset_profiles import (
        build_dataset_generation_identity,
        get_dataset_generation_namespace,
    )
    from core.market_data_contract import ResearchDataGenerationContract
    from core.walk_forward_policy import (
        build_optimizer_effective_policy_fingerprint,
        build_optimizer_runtime_policy,
        load_walk_forward_policy,
    )
    from filters.breakout_quality.paths import (
        resolve_existing_filter_artifact_paths,
        resolve_filter_model_dir,
        resolve_filter_output_dir,
    )
    from services.optimizer.application import _build_optimizer_study_db_file_path
    from services.research.strategy_compare_reuse import _pair_cache_fingerprint_from_payload

    case_id = "MARKET_DATA_ROUNDS_1_16_REPAIR4"
    results = []
    summary = {"ticker": case_id, "synthetic": True, "training_performed": False}

    active = ResearchDataGenerationContract(
        generation_id=RESEARCH_DATA_GENERATION_V2,
        status="active_frozen",
        lifecycle="immutable",
        cutoff_mode="fixed",
        cutoff=RESEARCH_REQUIRED_CUTOFF,
        required_cutoff=RESEARCH_REQUIRED_CUTOFF,
        universe_mode="daily_pit_eligibility",
    )

    def promoted(materialization_char: str):
        materialization_fp = materialization_char * 64
        promotion_fp = ("a" if materialization_char != "a" else "b") * 64
        payload = {
            "required_source_projection_fingerprint": "c" * 64,
            "adjusted_price_revision_proof_fingerprint": "d" * 64,
        }
        return SimpleNamespace(
            promotion_fingerprint=promotion_fp,
            materialization_fingerprint=materialization_fp,
            frozen_cutoff=RESEARCH_REQUIRED_CUTOFF,
            payload=payload,
        )

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        with patch(
            "core.market_data_research_promotion.get_effective_research_data_generation",
            return_value=active,
        ), patch(
            "core.market_data_research_promotion.load_active_research_v2_promotion",
            return_value=promoted("1"),
        ):
            identity_a = build_dataset_generation_identity(root, "full")
            namespace_a = get_dataset_generation_namespace(root, "full")

        with patch(
            "core.market_data_research_promotion.get_effective_research_data_generation",
            return_value=active,
        ), patch(
            "core.market_data_research_promotion.load_active_research_v2_promotion",
            return_value=promoted("2"),
        ):
            identity_b = build_dataset_generation_identity(root, "full")
            namespace_b = get_dataset_generation_namespace(root, "full")

        add_check(results, "market_data", case_id, "canonical_v2_generation_identity_changes_with_materialized_truth", False, identity_a["identity_fingerprint"] == identity_b["identity_fingerprint"])
        add_check(results, "market_data", case_id, "canonical_v2_generation_namespace_changes_with_materialized_truth", False, namespace_a == namespace_b)
        add_check(results, "market_data", case_id, "canonical_generation_namespace_uses_complete_sha_identity", 64, len(str(namespace_a).split("research_v2_", 1)[1]))

        base_policy = load_walk_forward_policy(Path(__file__).resolve().parents[2])
        runtime_policy = build_optimizer_runtime_policy(
            base_policy, "study", latest_data_date=RESEARCH_REQUIRED_CUTOFF, study_scope="full"
        )
        legacy_fp = build_optimizer_effective_policy_fingerprint(runtime_policy)["fingerprint_sha256"]
        runtime_a = dict(runtime_policy)
        runtime_a["research_dataset_generation_identity"] = identity_a
        runtime_b = dict(runtime_policy)
        runtime_b["research_dataset_generation_identity"] = identity_b
        fp_a = build_optimizer_effective_policy_fingerprint(runtime_a)["fingerprint_sha256"]
        fp_b = build_optimizer_effective_policy_fingerprint(runtime_b)["fingerprint_sha256"]
        add_check(results, "market_data", case_id, "optimizer_v2_policy_identity_differs_from_legacy_v1", False, fp_a == legacy_fp)
        add_check(results, "market_data", case_id, "optimizer_different_v2_generations_cannot_share_policy_fingerprint", False, fp_a == fp_b)

        legacy_db = _build_optimizer_study_db_file_path(
            output_dir=str(root / "outputs" / "ml_optimizer"), dataset_profile_key="full", study_scope="full"
        )
        v2_db = _build_optimizer_study_db_file_path(
            output_dir=str(root / "outputs" / "ml_optimizer"),
            dataset_profile_key="full",
            study_scope="full",
            dataset_generation_namespace=namespace_a,
        )
        add_check(results, "market_data", case_id, "optimizer_v1_study_db_path_remains_legacy_compatible", "optimizer_study_full_full.db", Path(legacy_db).name)
        add_check(results, "market_data", case_id, "optimizer_v2_study_db_is_generation_scoped", True, namespace_a in Path(v2_db).name)

        settings_payload = {
            "dataset": "full",
            "param_policy": "base-finalist-best",
            "max_positions": 10,
            "rotation": "off",
            "parameter_sources": {
                "p": {
                    "path_template": "x",
                    "identity_manifest_path": None,
                    "trained_with_dl_id": None,
                }
            },
            "dl_sources": {},
        }
        off_arm = {
            "arm_id": "off", "param_source": "p", "param_policy": "base-finalist-best",
            "rule_policy": "all_off", "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None,
        }
        on_arm = dict(off_arm, arm_id="on")
        common = dict(
            settings_payload=settings_payload,
            artifact_identities={},
            comparison_period={"start": "2021-01-01", "end": RESEARCH_REQUIRED_CUTOFF},
            off_arm_payload=off_arm,
            on_arm_payload=on_arm,
            engine_schema_version=1,
            parameter_evaluation_sha256="e" * 64,
        )
        compare_legacy = _pair_cache_fingerprint_from_payload(**common)
        compare_a = _pair_cache_fingerprint_from_payload(**common, dataset_generation_identity=identity_a)
        compare_b = _pair_cache_fingerprint_from_payload(**common, dataset_generation_identity=identity_b)
        add_check(results, "market_data", case_id, "strategy_compare_v2_pair_cache_cannot_reuse_legacy_v1_identity", False, compare_a == compare_legacy)
        add_check(results, "market_data", case_id, "strategy_compare_different_v2_generations_cannot_share_pair_cache", False, compare_a == compare_b)

        legacy_model = resolve_filter_model_dir(root, "breakout_quality_v1", "inception_time_v1", "unique_group_sampling")
        legacy_output = resolve_filter_output_dir(root, "breakout_quality_v1")
        add_check(results, "market_data", case_id, "breakout_quality_v1_model_path_remains_legacy_namespace", False, "research_generations" in legacy_model.parts)
        add_check(results, "market_data", case_id, "breakout_quality_v1_output_path_remains_legacy_namespace", False, "research_generations" in legacy_output.parts)

        with patch("filters.breakout_quality.paths._research_generation_namespace", return_value=namespace_a):
            v2_model_a = resolve_filter_model_dir(root, "breakout_quality_v1", "inception_time_v1", "unique_group_sampling")
            v2_output_a = resolve_filter_output_dir(root, "breakout_quality_v1")
            legacy_manifest = legacy_model / "manifest.json"
            legacy_manifest.parent.mkdir(parents=True, exist_ok=True)
            legacy_manifest.write_text("{}", encoding="utf-8")
            existing_v2 = resolve_existing_filter_artifact_paths(
                root, "breakout_quality_v1", "inception_time_v1", "unique_group_sampling"
            )
        with patch("filters.breakout_quality.paths._research_generation_namespace", return_value=namespace_b):
            v2_model_b = resolve_filter_model_dir(root, "breakout_quality_v1", "inception_time_v1", "unique_group_sampling")

        add_check(results, "market_data", case_id, "breakout_quality_v2_model_artifact_is_generation_scoped", True, namespace_a in v2_model_a.parts)
        add_check(results, "market_data", case_id, "breakout_quality_v2_dataset_output_is_generation_scoped", True, namespace_a in v2_output_a.parts)
        add_check(results, "market_data", case_id, "breakout_quality_different_v2_generations_have_distinct_paths", False, v2_model_a == v2_model_b)
        add_check(results, "market_data", case_id, "breakout_quality_v2_never_falls_back_to_v1_artifact_path", v2_model_a, existing_v2.model_dir)

    summary.update({"checks": len(results), "repair_round": 4})
    return results, summary

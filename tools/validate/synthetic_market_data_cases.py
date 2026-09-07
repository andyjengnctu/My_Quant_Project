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


__all__ = [
    "validate_market_data_v2_preflight_planner_contract_case",
    "validate_market_data_v2_resumable_executor_contract_case",
    "validate_market_data_v2_parquet_storage_contract_case",
    "validate_market_data_v2_bootstrap_activation_contract_case",
    "validate_market_data_v2_provider_snapshot_completion_contract_case",
    "validate_market_data_v2_trading_workbench_sidecar_contract_case",
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

    project_root = Path(__file__).resolve().parents[2]
    workflow_source = (project_root / "services" / "trading" / "daily_workflow.py").read_text(encoding="utf-8")
    update_source = (project_root / "services" / "trading" / "market_data_update.py").read_text(encoding="utf-8")
    downloader_source = (project_root / "services" / "downloader" / "main.py").read_text(encoding="utf-8")
    scanner_source = (project_root / "services" / "trading" / "scanner_state.py").read_text(encoding="utf-8")
    panel_source = (project_root / "services" / "workbench_ui" / "trading_account_panel.py").read_text(encoding="utf-8")
    state_source = (project_root / "services" / "trading" / "market_data_v2_state.py").read_text(encoding="utf-8")
    from services.trading import daily_workflow as daily_workflow_module
    from services.trading import market_data_update as market_data_update_module

    legacy_pos = update_source.find("run_trading_dataset_update(")
    snapshot_pos = update_source.find("publish_trading_market_data_snapshot(")
    v2_pos = update_source.find("sync_market_data_v2_trading_archive(")
    add_check(results, "market_data", case_id, "canonical_update_owner_preserves_execution_then_snapshot_then_v2_order", True, 0 <= legacy_pos < snapshot_pos < v2_pos)
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

    summary.update({
        "checks": len(results),
        "trading_request_count": manifest_a.total_requests,
        "freshness_contract_count": freshness_stats["contract_count"],
        "verified_schedule_count": freshness_stats["verified_schedule_count"],
        "fallback_schedule_count": freshness_stats["fallback_schedule_count"],
    })
    return results, summary

"""Canonical FinMind Market Data dataset registry for the V2 bootstrap.

The registry is intentionally provider-facing: it declares what the project plans
 to archive and how a preflight/bootstrap planner may query it.  It does not make
 any dataset PIT-safe for Research; PIT eligibility remains a later-layer concern.
"""
from __future__ import annotations

from dataclasses import dataclass

ARCHIVE_INCLUDE = "include"
ARCHIVE_EXCLUDE = "exclude"

BOOTSTRAP_SINGLE_NO_DATES = "single_no_dates"
BOOTSTRAP_SINGLE_FULL_RANGE = "single_full_range"
BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE = "per_instrument_full_range"
BOOTSTRAP_BULK_REFERENCE_DATES = "bulk_reference_dates"
BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE = "fixed_data_id_full_range"

DAILY_STATIC_REFRESH = "static_refresh"
DAILY_INCREMENTAL = "incremental"
DAILY_RECENT_REPAIR = "recent_repair"
DAILY_PERIODIC_REPAIR = "periodic_repair"
DAILY_EVENT_REPAIR = "event_repair"

PIT_EXACT_CANDIDATE = "exact_candidate"
PIT_REVIEW_REQUIRED = "review_required"
PIT_ARCHIVE_ONLY = "archive_only"
PIT_CURRENT_VINTAGE = "current_vintage"


@dataclass(frozen=True)
class MarketDatasetSpec:
    dataset: str
    category: str
    archive_policy: str
    bootstrap_mode: str
    daily_mode: str
    pit_class: str
    primary_key_hint: tuple[str, ...] = ()
    fixed_data_ids: tuple[str, ...] = ()
    probe_data_id: str | None = None
    full_market_exact_date_expected: bool = False
    rationale: str = ""

    @property
    def included(self) -> bool:
        return self.archive_policy == ARCHIVE_INCLUDE


def _include(
    dataset: str,
    category: str,
    bootstrap_mode: str,
    daily_mode: str,
    pit_class: str,
    *,
    primary_key_hint: tuple[str, ...] = (),
    fixed_data_ids: tuple[str, ...] = (),
    probe_data_id: str | None = None,
    full_market_exact_date_expected: bool = False,
    rationale: str = "",
) -> MarketDatasetSpec:
    return MarketDatasetSpec(
        dataset=dataset,
        category=category,
        archive_policy=ARCHIVE_INCLUDE,
        bootstrap_mode=bootstrap_mode,
        daily_mode=daily_mode,
        pit_class=pit_class,
        primary_key_hint=primary_key_hint,
        fixed_data_ids=fixed_data_ids,
        probe_data_id=probe_data_id,
        full_market_exact_date_expected=full_market_exact_date_expected,
        rationale=rationale,
    )


def _exclude(dataset: str, category: str, rationale: str) -> MarketDatasetSpec:
    return MarketDatasetSpec(
        dataset=dataset,
        category=category,
        archive_policy=ARCHIVE_EXCLUDE,
        bootstrap_mode=BOOTSTRAP_SINGLE_NO_DATES,
        daily_mode=DAILY_STATIC_REFRESH,
        pit_class=PIT_ARCHIVE_ONLY,
        rationale=rationale,
    )


# A probe stock is execution-only diagnostic input. It is not a scientific or
# dataset identity and may fall back to another TWSE/TPEx stock at runtime.
DEFAULT_EQUITY_PROBE_DATA_ID = "2330"

MARKET_DATASET_SPECS: tuple[MarketDatasetSpec, ...] = (
    # Market/security master and price truth.
    _include("TaiwanStockInfo", "security_master", BOOTSTRAP_SINGLE_NO_DATES, DAILY_STATIC_REFRESH, PIT_REVIEW_REQUIRED, primary_key_hint=("stock_id", "type", "date"), rationale="Historical/current market identity source; transition rows are preserved."),
    _include("TaiwanStockTradingDate", "calendar", BOOTSTRAP_SINGLE_NO_DATES, DAILY_INCREMENTAL, PIT_EXACT_CANDIDATE, primary_key_hint=("date",)),
    _include("TaiwanStockDelisting", "security_master", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_EXACT_CANDIDATE, primary_key_hint=("date", "stock_id")),
    _include("TaiwanStockIndustryChain", "security_master", BOOTSTRAP_SINGLE_NO_DATES, DAILY_STATIC_REFRESH, PIT_REVIEW_REQUIRED, primary_key_hint=("stock_id", "industry", "sub_industry", "date"), rationale="Current-vintage industry classification; archive now, PIT legality is reviewed later."),
    _include("TaiwanStockActiveETFInfo", "security_master", BOOTSTRAP_SINGLE_NO_DATES, DAILY_STATIC_REFRESH, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), rationale="Small security-master supplement for active ETFs; holdings remain Sponsor-only and excluded."),
    _include("TaiwanStockPrice", "price", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_EXACT_CANDIDATE, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Raw archive/evidence only; current model/strategy direct consumption remains prohibited."),
    _include("TaiwanStockPriceAdj", "price", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_CURRENT_VINTAGE, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Canonical vendor adjusted-price source; Research legality still requires representation invariance."),
    _include("TaiwanStockPER", "valuation", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockDayTrading", "trading_activity", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockPriceLimit", "trading_constraint", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_EXACT_CANDIDATE, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Bootstrap is per-instrument because daily all-market reference dates would exceed the historical instrument count; daily refresh may still use the verified all-market exact-date capability."),
    _include("TaiwanStockSuspended", "trading_constraint", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("stock_id", "date")),
    _include("TaiwanStockDayTradingSuspension", "trading_constraint", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("stock_id", "date")),
    _include("TaiwanStockTotalReturnIndex", "market_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), fixed_data_ids=("TAIEX", "TPEx")),

    # Chip / positioning.
    _include("TaiwanStockMarginPurchaseShortSale", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockTotalMarginPurchaseShortSale", "chip_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "name")),
    _include("TaiwanStockInstitutionalInvestorsBuySellWide", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Canonical flattened view; the long-format table is excluded as duplicate source data."),
    _include("TaiwanStockTotalInstitutionalInvestors", "chip_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "name")),
    _include("TaiwanStockShareholding", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockHoldingSharesPer", "ownership", BOOTSTRAP_BULK_REFERENCE_DATES, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "HoldingSharesLevel"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockSecuritiesLending", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockMarginShortSaleSuspension", "trading_constraint", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID),
    _include("TaiwanDailyShortSaleBalances", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID),
    _include("TaiwanTotalExchangeMarginMaintenance", "chip_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED),
    _include("TaiwanStockDispositionSecuritiesPeriod", "trading_constraint", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID),
    _include("TaiwanStockDayTradingBorrowingFeeRate", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID),

    # Fundamental / corporate actions.
    _include("TaiwanStockFinancialStatements", "fundamental", BOOTSTRAP_BULK_REFERENCE_DATES, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type", "origin_name"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockBalanceSheet", "fundamental", BOOTSTRAP_BULK_REFERENCE_DATES, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type", "origin_name"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockCashFlowsStatement", "fundamental", BOOTSTRAP_BULK_REFERENCE_DATES, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type", "origin_name"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockDividend", "corporate_action", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockDividendResult", "corporate_action", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockMonthRevenue", "fundamental", BOOTSTRAP_BULK_REFERENCE_DATES, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "revenue_year", "revenue_month"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockCapitalReductionReferencePrice", "corporate_action", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id")),
    _include("TaiwanStockMarketValue", "market_cap", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockMarketValueWeight", "market_context", BOOTSTRAP_BULK_REFERENCE_DATES, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockSplitPrice", "corporate_action", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id")),
    _include("TaiwanStockParValueChange", "corporate_action", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id")),

    # Low-volume regime/context datasets useful for future input experiments.
    _include("TaiwanBusinessIndicator", "macro_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED),
    _include("CnnFearGreedIndex", "macro_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, primary_key_hint=("date",)),
    _include("TaiwanExchangeRate", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, fixed_data_ids=("AUD", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "IDR", "JPY", "KRW", "MYR", "NZD", "PHP", "SEK", "SGD", "THB", "USD", "VND", "ZAR")),
    _include("InterestRate", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("BOE", "RBA", "FED", "PBOC", "BOC", "ECB", "RBNZ", "RBI", "CBR", "BCB", "BOJ", "SNB")),
    _include("GovernmentBondsYield", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, fixed_data_ids=("United States 1-Month", "United States 2-Month", "United States 3-Month", "United States 6-Month", "United States 1-Year", "United States 2-Year", "United States 3-Year", "United States 5-Year", "United States 7-Year", "United States 10-Year", "United States 20-Year", "United States 30-Year")),
    _include("GoldPrice", "macro_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, primary_key_hint=("date",)),
    _include("CrudeOilPrices", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, fixed_data_ids=("Brent", "WTI")),
    _include("TaiwanFutOptDailyInfo", "derivative_master", BOOTSTRAP_SINGLE_NO_DATES, DAILY_STATIC_REFRESH, PIT_REVIEW_REQUIRED, primary_key_hint=("code", "type", "name"), rationale="Canonical derivative product/code inventory used to interpret futures/options datasets; no scientific product selection is implied."),
    _include("TaiwanFuturesDaily", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TX",)),
    _include("TaiwanFuturesInstitutionalInvestors", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TX",)),
    _include("TaiwanFuturesOpenInterestLargeTraders", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TXF",), rationale="This dataset uses futures product code TXF; it is not the TX code used by FuturesDaily/InstitutionalInvestors."),
    _include("TaiwanOptionDaily", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TXO",)),
    _include("TaiwanOptionInstitutionalInvestors", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TXO",)),
    _include("TaiwanOptionOpenInterestLargeTraders", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TXO",)),
    _include("TaiwanOptionVix", "derivative_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, primary_key_hint=("date",)),

    # Explicit exclusions prevent later downloader expansion from silently adding
    # redundant, oversized, unsupported, or currently out-of-scope datasets.
    _exclude("TaiwanStockInstitutionalInvestorsBuySell", "chip", "Same source as the included Wide table; avoid duplicate physical truth."),
    _exclude("TaiwanStockWeekPrice", "derived_price", "Deterministically derivable from canonical daily price."),
    _exclude("TaiwanStockMonthPrice", "derived_price", "Deterministically derivable from canonical daily price."),
    _exclude("TaiwanStock10Year", "derived_price", "Deterministically derivable rolling statistic; do not archive a second derived truth."),
    _exclude("TaiwanStockPriceTick", "intraday", "Tick volume/storage cost is disproportionate to the current daily-input research scope."),
    _exclude("TaiwanStockKBar", "intraday", "Minute data is outside the current daily-input research scope."),
    _exclude("TaiwanStockNews", "nlp", "News/NLP is a separate research family and is not part of this tabular/time-series bootstrap."),
    _exclude("TaiwanStockMarginMaintenance", "chip", "Sponsor-only according to the current FinMind dataset documentation; must not become a required Backer dependency."),
    _exclude("TaiwanStockActiveETFHolding", "etf", "Sponsor-only according to the current FinMind dataset documentation; must not become a required Backer dependency."),
    _exclude("TaiwanStockActiveETFHoldingChange", "etf", "Sponsor-only according to the current FinMind dataset documentation; must not become a required Backer dependency."),
    _exclude("TaiwanStockIndustryChainMoneyFlow", "market_context", "Sponsor-only according to the current FinMind dataset documentation; current Backer bootstrap archives the underlying industry classification instead."),
)


def get_market_dataset_specs(*, included_only: bool = False) -> tuple[MarketDatasetSpec, ...]:
    if not included_only:
        return MARKET_DATASET_SPECS
    return tuple(spec for spec in MARKET_DATASET_SPECS if spec.included)


def get_market_dataset_spec(dataset: str) -> MarketDatasetSpec:
    key = str(dataset or "").strip()
    matches = [spec for spec in MARKET_DATASET_SPECS if spec.dataset == key]
    if len(matches) != 1:
        raise ValueError(f"Market Data dataset registry identity 無法唯一解析: {key!r}")
    return matches[0]


def validate_market_dataset_registry() -> dict[str, int]:
    datasets = [spec.dataset for spec in MARKET_DATASET_SPECS]
    duplicates = sorted({dataset for dataset in datasets if datasets.count(dataset) > 1})
    if duplicates:
        raise ValueError(f"Market Data dataset identity 重複: {duplicates}")

    supported_bootstrap_modes = {
        BOOTSTRAP_SINGLE_NO_DATES,
        BOOTSTRAP_SINGLE_FULL_RANGE,
        BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
        BOOTSTRAP_BULK_REFERENCE_DATES,
        BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE,
    }
    invalid_modes = sorted({spec.bootstrap_mode for spec in MARKET_DATASET_SPECS if spec.bootstrap_mode not in supported_bootstrap_modes})
    if invalid_modes:
        raise ValueError(f"Market Data bootstrap_mode 未支援: {invalid_modes}")

    for spec in MARKET_DATASET_SPECS:
        if spec.bootstrap_mode == BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE and not spec.fixed_data_ids:
            raise ValueError(f"{spec.dataset} fixed_data_id mode 缺少 fixed_data_ids")
        if spec.bootstrap_mode in {BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, BOOTSTRAP_BULK_REFERENCE_DATES} and not spec.probe_data_id:
            raise ValueError(f"{spec.dataset} bootstrap probe 缺少 probe_data_id")
        if spec.archive_policy not in {ARCHIVE_INCLUDE, ARCHIVE_EXCLUDE}:
            raise ValueError(f"{spec.dataset} archive_policy 未支援: {spec.archive_policy}")

    return {
        "total": len(MARKET_DATASET_SPECS),
        "included": sum(spec.included for spec in MARKET_DATASET_SPECS),
        "excluded": sum(not spec.included for spec in MARKET_DATASET_SPECS),
        "per_instrument": sum(spec.included and spec.bootstrap_mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE for spec in MARKET_DATASET_SPECS),
    }


__all__ = [
    "ARCHIVE_INCLUDE",
    "ARCHIVE_EXCLUDE",
    "BOOTSTRAP_SINGLE_NO_DATES",
    "BOOTSTRAP_SINGLE_FULL_RANGE",
    "BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE",
    "BOOTSTRAP_BULK_REFERENCE_DATES",
    "BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE",
    "DAILY_STATIC_REFRESH",
    "DAILY_INCREMENTAL",
    "DAILY_RECENT_REPAIR",
    "DAILY_PERIODIC_REPAIR",
    "DAILY_EVENT_REPAIR",
    "PIT_EXACT_CANDIDATE",
    "PIT_REVIEW_REQUIRED",
    "PIT_ARCHIVE_ONLY",
    "PIT_CURRENT_VINTAGE",
    "DEFAULT_EQUITY_PROBE_DATA_ID",
    "MarketDatasetSpec",
    "MARKET_DATASET_SPECS",
    "get_market_dataset_specs",
    "get_market_dataset_spec",
    "validate_market_dataset_registry",
]

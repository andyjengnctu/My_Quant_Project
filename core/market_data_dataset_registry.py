"""Canonical FinMind Market Data dataset registry for the V2 bootstrap.

The registry is intentionally provider-facing: it declares what the project plans
 to archive and how a preflight/bootstrap planner may query it.  It does not make
 any dataset PIT-safe for Research; PIT eligibility remains a later-layer concern.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from core.market_data_contract import (
    FINMIND_ADJUSTED_PRICE_DATASET,
    FINMIND_RAW_PRICE_ARCHIVE_DATASET,
)

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

TRADING_QUERY_AUTO = "auto"
TRADING_QUERY_RECENT_DATES = "recent_dates"
TRADING_QUERY_MONTH_STARTS = "month_starts"
TRADING_QUERY_QUARTER_ENDS = "quarter_ends"
TRADING_QUERY_RANGE = "range"

PIT_EXACT_CANDIDATE = "exact_candidate"
PIT_REVIEW_REQUIRED = "review_required"
PIT_ARCHIVE_ONLY = "archive_only"
PIT_CURRENT_VINTAGE = "current_vintage"


# Provider-facing Traditional Chinese dataset labels used by operator UIs.
# This mapping lives beside the canonical dataset registry so console/Workbench
# renderers do not maintain their own identity-to-label tables.
MARKET_DATASET_DISPLAY_NAMES_ZH: dict[str, str] = {
    "TaiwanStockInfo": "台股總覽",
    "TaiwanStockTradingDate": "台股交易日",
    "TaiwanStockDelisting": "台灣股票下市櫃表",
    "TaiwanStockIndustryChain": "個體公司所屬產業鏈",
    "TaiwanStockActiveETFInfo": "主動式ETF清單",
    "TaiwanStockPrice": "股價日成交資訊",
    "TaiwanStockPriceAdj": "台灣還原股價資料表",
    "TaiwanStockPER": "個股 PER、PBR 資料表",
    "TaiwanStockDayTrading": "當日沖銷交易標的及成交量值",
    "TaiwanStockPriceLimit": "每日漲跌停價",
    "TaiwanStockSuspended": "台股暫停交易公告",
    "TaiwanStockDayTradingSuspension": "暫停先賣後買當沖預告表",
    "TaiwanStockTotalReturnIndex": "加權、櫃買報酬指數",
    "TaiwanStockMarginPurchaseShortSale": "個股融資融劵表",
    "TaiwanStockTotalMarginPurchaseShortSale": "整體市場融資融劵表",
    "TaiwanStockInstitutionalInvestorsBuySellWide": "個股三大法人買賣表（寬表）",
    "TaiwanStockTotalInstitutionalInvestors": "整體三大市場法人買賣表",
    "TaiwanStockShareholding": "外資持股表",
    "TaiwanStockHoldingSharesPer": "股權持股分級表",
    "TaiwanStockSecuritiesLending": "借券成交明細",
    "TaiwanStockMarginShortSaleSuspension": "暫停融券賣出表",
    "TaiwanDailyShortSaleBalances": "信用額度總量管制餘額表",
    "TaiwanTotalExchangeMarginMaintenance": "台灣大盤融資維持率",
    "TaiwanStockDispositionSecuritiesPeriod": "公布處置有價證券表",
    "TaiwanStockDayTradingBorrowingFeeRate": "現股當日沖銷券差借券費率",
    "TaiwanStockFinancialStatements": "綜合損益表",
    "TaiwanStockBalanceSheet": "資產負債表",
    "TaiwanStockCashFlowsStatement": "現金流量表",
    "TaiwanStockDividend": "股利政策表",
    "TaiwanStockDividendResult": "除權除息結果表",
    "TaiwanStockMonthRevenue": "月營收表",
    "TaiwanStockCapitalReductionReferencePrice": "減資恢復買賣參考價格",
    "TaiwanStockMarketValue": "台灣股價市值表",
    "TaiwanStockMarketValueWeight": "台股市值比重表",
    "TaiwanStockSplitPrice": "台股分割後參考價",
    "TaiwanStockParValueChange": "台灣股票變更面額恢復買賣參考價格",
    "TaiwanBusinessIndicator": "台灣每月景氣對策信號表",
    "CnnFearGreedIndex": "CNN 恐懼與貪婪指數",
    "TaiwanExchangeRate": "外幣對台幣匯率",
    "InterestRate": "央行利率",
    "GovernmentBondsYield": "美國國債殖利率",
    "GoldPrice": "黃金價格",
    "CrudeOilPrices": "原油價格",
    "TaiwanFutOptDailyInfo": "期貨、選擇權日成交資訊總覽",
    "TaiwanFuturesDaily": "期貨日成交資訊",
    "TaiwanFuturesInstitutionalInvestors": "期貨三大法人買賣",
    "TaiwanFuturesOpenInterestLargeTraders": "期貨大額交易人未沖銷部位",
    "TaiwanOptionDaily": "選擇權日成交資訊",
    "TaiwanOptionInstitutionalInvestors": "選擇權三大法人買賣",
    "TaiwanOptionOpenInterestLargeTraders": "選擇權大額交易人未沖銷部位",
    "TaiwanOptionVix": "臺指選擇權波動率指數",
}


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
    trading_query_mode: str = TRADING_QUERY_AUTO
    trading_lookback_periods: int = 0
    trading_fixed_data_ids: tuple[str, ...] = ()
    bootstrap_start_date: str | None = None
    bootstrap_chunk_years: int = 0
    bootstrap_chunk_months: int = 0
    preflight_probe_calendar_days: int = 0

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
    trading_query_mode: str = TRADING_QUERY_AUTO,
    trading_lookback_periods: int = 0,
    trading_fixed_data_ids: tuple[str, ...] = (),
    bootstrap_start_date: str | None = None,
    bootstrap_chunk_years: int = 0,
    bootstrap_chunk_months: int = 0,
    preflight_probe_calendar_days: int = 0,
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
        trading_query_mode=trading_query_mode,
        trading_lookback_periods=int(trading_lookback_periods),
        trading_fixed_data_ids=tuple(trading_fixed_data_ids),
        bootstrap_start_date=bootstrap_start_date,
        bootstrap_chunk_years=int(bootstrap_chunk_years),
        bootstrap_chunk_months=int(bootstrap_chunk_months),
        preflight_probe_calendar_days=int(preflight_probe_calendar_days),
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
    _include(FINMIND_RAW_PRICE_ARCHIVE_DATASET, "price", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_EXACT_CANDIDATE, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Raw archive/evidence only; current model/strategy direct consumption remains prohibited."),
    _include(FINMIND_ADJUSTED_PRICE_DATASET, "price", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_CURRENT_VINTAGE, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Canonical vendor adjusted-price source; Research legality still requires representation invariance."),
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
    _include("TaiwanStockHoldingSharesPer", "ownership", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "HoldingSharesLevel"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Correctness-first bootstrap: do not infer the market-wide historical date inventory from one probe stock.", trading_query_mode=TRADING_QUERY_RECENT_DATES, trading_lookback_periods=21),
    _include("TaiwanStockSecuritiesLending", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockMarginShortSaleSuspension", "trading_constraint", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanDailyShortSaleBalances", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanTotalExchangeMarginMaintenance", "chip_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED),
    _include("TaiwanStockDispositionSecuritiesPeriod", "trading_constraint", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockDayTradingBorrowingFeeRate", "chip", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),

    # Fundamental / corporate actions.
    _include("TaiwanStockFinancialStatements", "fundamental", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type", "origin_name"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Correctness-first bootstrap: do not infer the market-wide historical date inventory from one probe stock.", trading_query_mode=TRADING_QUERY_QUARTER_ENDS, trading_lookback_periods=8),
    _include("TaiwanStockBalanceSheet", "fundamental", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type", "origin_name"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Correctness-first bootstrap: do not infer the market-wide historical date inventory from one probe stock.", trading_query_mode=TRADING_QUERY_QUARTER_ENDS, trading_lookback_periods=8),
    _include("TaiwanStockCashFlowsStatement", "fundamental", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type", "origin_name"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Correctness-first bootstrap: do not infer the market-wide historical date inventory from one probe stock.", trading_query_mode=TRADING_QUERY_QUARTER_ENDS, trading_lookback_periods=8),
    _include("TaiwanStockDividend", "corporate_action", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockDividendResult", "corporate_action", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockMonthRevenue", "fundamental", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "revenue_year", "revenue_month"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Correctness-first bootstrap: do not infer the market-wide historical date inventory from one probe stock.", trading_query_mode=TRADING_QUERY_MONTH_STARTS, trading_lookback_periods=15),
    _include("TaiwanStockCapitalReductionReferencePrice", "corporate_action", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id")),
    _include("TaiwanStockMarketValue", "market_cap", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True),
    _include("TaiwanStockMarketValueWeight", "market_context", BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id", "type"), probe_data_id=DEFAULT_EQUITY_PROBE_DATA_ID, full_market_exact_date_expected=True, rationale="Correctness-first bootstrap: do not infer the market-wide historical date inventory from one probe stock.", trading_query_mode=TRADING_QUERY_RECENT_DATES, trading_lookback_periods=7),
    _include("TaiwanStockSplitPrice", "corporate_action", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id")),
    _include("TaiwanStockParValueChange", "corporate_action", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_EVENT_REPAIR, PIT_REVIEW_REQUIRED, primary_key_hint=("date", "stock_id")),

    # Low-volume regime/context datasets useful for future input experiments.
    _include("TaiwanBusinessIndicator", "macro_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, trading_query_mode=TRADING_QUERY_RANGE, trading_lookback_periods=450),
    _include("CnnFearGreedIndex", "macro_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, primary_key_hint=("date",)),
    _include("TaiwanExchangeRate", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, fixed_data_ids=("AUD", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "IDR", "JPY", "KRW", "MYR", "NZD", "PHP", "SEK", "SGD", "THB", "USD", "VND", "ZAR")),
    _include("InterestRate", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_PERIODIC_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("BOE", "RBA", "FED", "PBOC", "BOC", "ECB", "RBNZ", "RBI", "CBR", "BCB", "BOJ", "SNB"), trading_query_mode=TRADING_QUERY_RANGE, trading_lookback_periods=450),
    _include("GovernmentBondsYield", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, fixed_data_ids=("United States 1-Month", "United States 2-Month", "United States 3-Month", "United States 6-Month", "United States 1-Year", "United States 2-Year", "United States 3-Year", "United States 5-Year", "United States 7-Year", "United States 10-Year", "United States 20-Year", "United States 30-Year")),
    _include("GoldPrice", "macro_context", BOOTSTRAP_SINGLE_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, primary_key_hint=("date",)),
    _include("CrudeOilPrices", "macro_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_INCREMENTAL, PIT_REVIEW_REQUIRED, fixed_data_ids=("Brent", "WTI")),
    _include("TaiwanFutOptDailyInfo", "derivative_master", BOOTSTRAP_SINGLE_NO_DATES, DAILY_STATIC_REFRESH, PIT_REVIEW_REQUIRED, primary_key_hint=("code", "type", "name"), rationale="Canonical derivative product/code inventory used to interpret futures/options datasets; no scientific product selection is implied."),
    _include("TaiwanFuturesDaily", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TX",)),
    _include("TaiwanFuturesInstitutionalInvestors", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TX",)),
# FinMind package/bootstrap alias remains TXF; raw Data API daily rows expose the TAIEX futures product as TX.
    _include("TaiwanFuturesOpenInterestLargeTraders", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TXF",), trading_fixed_data_ids=("TX",), rationale="This dataset uses futures product code TXF; it is not the TX code used by FuturesDaily/InstitutionalInvestors."),
    _include("TaiwanOptionDaily", "derivative_context", BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE, DAILY_RECENT_REPAIR, PIT_REVIEW_REQUIRED, fixed_data_ids=("TXO",), bootstrap_start_date="2001-12-01", bootstrap_chunk_months=1, preflight_probe_calendar_days=2, rationale="High-density strike/expiry option history is chunked by calendar month; capability preflight uses only the latest two completed-market calendar days to avoid oversized provider responses."),
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


def get_market_dataset_display_name_zh(dataset: str) -> str:
    key = str(dataset or "").strip()
    try:
        return MARKET_DATASET_DISPLAY_NAMES_ZH[key]
    except KeyError as exc:
        raise ValueError(f"Market Data dataset 缺少中文顯示名稱: {key!r}") from exc


def validate_market_dataset_registry() -> dict[str, int]:
    datasets = [spec.dataset for spec in MARKET_DATASET_SPECS]
    duplicates = sorted({dataset for dataset in datasets if datasets.count(dataset) > 1})
    if duplicates:
        raise ValueError(f"Market Data dataset identity 重複: {duplicates}")

    included_datasets = {spec.dataset for spec in MARKET_DATASET_SPECS if spec.included}
    display_name_datasets = set(MARKET_DATASET_DISPLAY_NAMES_ZH)
    missing_display_names = sorted(included_datasets - display_name_datasets)
    extra_display_names = sorted(display_name_datasets - included_datasets)
    if missing_display_names:
        raise ValueError(f"Market Data included datasets 缺少中文顯示名稱: {missing_display_names}")
    if extra_display_names:
        raise ValueError(f"Market Data 中文顯示名稱包含非 included dataset: {extra_display_names}")
    if any(not str(value or "").strip() for value in MARKET_DATASET_DISPLAY_NAMES_ZH.values()):
        raise ValueError("Market Data 中文顯示名稱不得空白")

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

    supported_trading_query_modes = {
        TRADING_QUERY_AUTO,
        TRADING_QUERY_RECENT_DATES,
        TRADING_QUERY_MONTH_STARTS,
        TRADING_QUERY_QUARTER_ENDS,
        TRADING_QUERY_RANGE,
    }
    exact_date_query_modes = {
        TRADING_QUERY_RECENT_DATES,
        TRADING_QUERY_MONTH_STARTS,
        TRADING_QUERY_QUARTER_ENDS,
    }
    for spec in MARKET_DATASET_SPECS:
        if spec.bootstrap_mode == BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE and not spec.fixed_data_ids:
            raise ValueError(f"{spec.dataset} fixed_data_id mode 缺少 fixed_data_ids")
        if spec.bootstrap_mode in {BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE, BOOTSTRAP_BULK_REFERENCE_DATES} and not spec.probe_data_id:
            raise ValueError(f"{spec.dataset} bootstrap probe 缺少 probe_data_id")
        if spec.archive_policy not in {ARCHIVE_INCLUDE, ARCHIVE_EXCLUDE}:
            raise ValueError(f"{spec.dataset} archive_policy 未支援: {spec.archive_policy}")
        if spec.trading_query_mode not in supported_trading_query_modes:
            raise ValueError(f"{spec.dataset} trading_query_mode 未支援: {spec.trading_query_mode}")
        if spec.trading_query_mode != TRADING_QUERY_AUTO and spec.trading_lookback_periods < 1:
            raise ValueError(f"{spec.dataset} trading_query_mode 已指定但 trading_lookback_periods < 1")
        if spec.trading_query_mode == TRADING_QUERY_AUTO and spec.trading_lookback_periods != 0:
            raise ValueError(f"{spec.dataset} trading_query_mode=auto 時不得設定 trading_lookback_periods")
        if len(set(spec.trading_fixed_data_ids)) != len(spec.trading_fixed_data_ids):
            raise ValueError(f"{spec.dataset} trading_fixed_data_ids 不得重複")
        if any(not str(value or "").strip() for value in spec.trading_fixed_data_ids):
            raise ValueError(f"{spec.dataset} trading_fixed_data_ids 不得包含空白 identity")
        if spec.trading_fixed_data_ids and not spec.fixed_data_ids:
            raise ValueError(f"{spec.dataset} trading_fixed_data_ids 只能覆寫 fixed-data-id dataset")
        if spec.daily_mode == DAILY_PERIODIC_REPAIR and spec.trading_query_mode == TRADING_QUERY_AUTO:
            raise ValueError(f"{spec.dataset} periodic_repair 必須明確登記 Trading query policy")
        if spec.trading_query_mode in exact_date_query_modes and not spec.full_market_exact_date_expected:
            raise ValueError(f"{spec.dataset} exact-date Trading query 缺少 full-market capability 宣告")
        if spec.bootstrap_chunk_years < 0:
            raise ValueError(f"{spec.dataset} bootstrap_chunk_years 不得 < 0")
        if spec.bootstrap_chunk_months < 0:
            raise ValueError(f"{spec.dataset} bootstrap_chunk_months 不得 < 0")
        if spec.preflight_probe_calendar_days < 0:
            raise ValueError(f"{spec.dataset} preflight_probe_calendar_days 不得 < 0")
        if spec.bootstrap_chunk_years and spec.bootstrap_chunk_months:
            raise ValueError(f"{spec.dataset} bootstrap chunk 只能擇一使用 years 或 months")
        if (spec.bootstrap_chunk_years or spec.bootstrap_chunk_months) and spec.bootstrap_mode != BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE:
            raise ValueError(f"{spec.dataset} bootstrap chunk 目前只允許 fixed_data_id_full_range")
        if spec.preflight_probe_calendar_days and spec.bootstrap_mode != BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE:
            raise ValueError(f"{spec.dataset} preflight_probe_calendar_days 目前只允許 fixed_data_id_full_range")
        if spec.bootstrap_start_date is not None:
            try:
                date.fromisoformat(spec.bootstrap_start_date)
            except ValueError as exc:
                raise ValueError(f"{spec.dataset} bootstrap_start_date 不是 YYYY-MM-DD: {spec.bootstrap_start_date}") from exc

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
    "TRADING_QUERY_AUTO",
    "TRADING_QUERY_RECENT_DATES",
    "TRADING_QUERY_MONTH_STARTS",
    "TRADING_QUERY_QUARTER_ENDS",
    "TRADING_QUERY_RANGE",
    "PIT_EXACT_CANDIDATE",
    "PIT_REVIEW_REQUIRED",
    "PIT_ARCHIVE_ONLY",
    "PIT_CURRENT_VINTAGE",
    "DEFAULT_EQUITY_PROBE_DATA_ID",
    "MarketDatasetSpec",
    "MARKET_DATASET_SPECS",
    "MARKET_DATASET_DISPLAY_NAMES_ZH",
    "get_market_dataset_specs",
    "get_market_dataset_spec",
    "get_market_dataset_display_name_zh",
    "validate_market_dataset_registry",
]

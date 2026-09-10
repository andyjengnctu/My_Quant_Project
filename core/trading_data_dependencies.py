"""Canonical Trading strategy/model market-data dependency contracts.

The registry declares *what* data an execution profile requires.  It does not
inspect runtime state and does not perform provider calls.  Runtime readiness is
resolved by ``services.trading.data_readiness`` from this single declarative
owner plus canonical Trading/V2 state.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.file_integrity import canonical_json_sha256
from core.market_data_dataset_registry import get_market_dataset_specs

TRADING_DATA_DEPENDENCY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TradingDataDependencySpec:
    strategy_id: str
    execution_market_data_required: bool
    required_v2_datasets: tuple[str, ...] = ()
    rationale: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": TRADING_DATA_DEPENDENCY_SCHEMA_VERSION,
            "strategy_id": self.strategy_id,
            "execution_market_data_required": bool(self.execution_market_data_required),
            "required_v2_datasets": list(self.required_v2_datasets),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_json_sha256(self.identity_payload())


TRADING_DATA_DEPENDENCY_SPECS: tuple[TradingDataDependencySpec, ...] = (
    TradingDataDependencySpec(
        strategy_id="full_rule_based_no_dl",
        execution_market_data_required=False,
        required_v2_datasets=(
            "TaiwanStockPriceAdj",
            "TaiwanStockPrice",
            "TaiwanStockMarketValue",
            "TaiwanStockTradingDate",
            "TaiwanStockInfo",
            "TaiwanStockDelisting",
        ),
        rationale=(
            "Round-7 execution cutover: production rule-based Trading reads the verified Market Data V2 "
            "historical/latest view directly. PriceAdj+raw Price provide OHLCV, MarketValue+TradingDate+" 
            "StockInfo+Delisting provide the date-local execution pool and membership evidence. The "
            "transitional Legacy CSV/snapshot is no longer execution-required."
        ),
    ),
)


def get_trading_data_dependency_spec(strategy_id: str) -> TradingDataDependencySpec:
    key = str(strategy_id or "").strip()
    matches = [item for item in TRADING_DATA_DEPENDENCY_SPECS if item.strategy_id == key]
    if len(matches) != 1:
        raise ValueError(f"Trading data dependency identity 無法唯一解析: {key!r}")
    return matches[0]


def validate_trading_data_dependency_registry() -> dict[str, int]:
    strategy_ids = [item.strategy_id for item in TRADING_DATA_DEPENDENCY_SPECS]
    duplicates = sorted({item for item in strategy_ids if strategy_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Trading data dependency strategy_id 重複: {duplicates}")

    included = {spec.dataset for spec in get_market_dataset_specs(included_only=True)}
    for item in TRADING_DATA_DEPENDENCY_SPECS:
        if not item.strategy_id:
            raise ValueError("Trading data dependency strategy_id 不得為空")
        if len(set(item.required_v2_datasets)) != len(item.required_v2_datasets):
            raise ValueError(f"{item.strategy_id} required_v2_datasets 重複")
        unknown = sorted(set(item.required_v2_datasets) - included)
        if unknown:
            raise ValueError(f"{item.strategy_id} required_v2_datasets 不在 canonical included registry: {unknown}")

    return {
        "strategy_count": len(TRADING_DATA_DEPENDENCY_SPECS),
        "v2_dependency_count": sum(len(item.required_v2_datasets) for item in TRADING_DATA_DEPENDENCY_SPECS),
    }


__all__ = [
    "TRADING_DATA_DEPENDENCY_SCHEMA_VERSION",
    "TradingDataDependencySpec",
    "TRADING_DATA_DEPENDENCY_SPECS",
    "get_trading_data_dependency_spec",
    "validate_trading_data_dependency_registry",
]

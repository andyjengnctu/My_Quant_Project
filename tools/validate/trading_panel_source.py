"""Read the actual Trading Account Panel implementation across its owned modules.

The panel's implementation was split without moving any business logic into
tests. Source-based legacy guards inspect the real source of every component,
not an obsolete standalone-file layout or a replicated implementation.
"""
from __future__ import annotations

from pathlib import Path

_TRADING_ACCOUNT_COMPONENTS = (
    "trading_account_panel_data.py",
    "trading_account_panel_status.py",
    "trading_account_panel.py",
    "trading_account_panel_runtime.py",
    "trading_account_panel_layout.py",
    "trading_account_panel_protection.py",
    "trading_account_panel_dashboard.py",
    "trading_account_panel_promotion.py",
    "trading_account_panel_scanner.py",
    "trading_account_panel_pending.py",
    "trading_account_panel_orders.py",
    "trading_account_panel_account.py",
)


def read_trading_account_panel_source(project_root: Path) -> str:
    """Combine source files in widget/handler declaration order for legacy guards."""
    directory = Path(project_root) / "services" / "workbench_ui"
    return "\n\n".join(
        (directory / name).read_text(encoding="utf-8")
        for name in _TRADING_ACCOUNT_COMPONENTS
    )

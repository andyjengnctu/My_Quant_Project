from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one scheduler-safe Trading market-data automatic update iteration, including sparse new-market-date discovery."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--target-date",
        default=None,
        help="Optional YYYY-MM-DD override for deterministic recovery/testing. Default can sparsely discover a newer completed Trading day, then runs the canonical due planner.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON result.")
    args = parser.parse_args(None if argv is None else list(argv)[1:])

    from services.trading.market_data_auto_update import run_trading_market_data_auto_update

    result = run_trading_market_data_auto_update(
        project_root=Path(args.project_root).resolve(),
        target_date=args.target_date,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    else:
        print(
            "Market Data Auto Update | "
            f"status={result.get('status')} | target={result.get('target_date') or '-'} | "
            f"due={result.get('due_dataset_count', 0)} | data_req={result.get('data_requests', 0)} | "
            f"usage_req={result.get('usage_requests', 0)} | next={result.get('next_check_at') or '-'}"
        )
    return 2 if str(result.get("status") or "") == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

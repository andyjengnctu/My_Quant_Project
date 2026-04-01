from pathlib import Path
import re
from unittest.mock import patch

from .checks import add_check


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md"


def _extract_table_rows(text: str, heading: str):
    pattern = rf"^### {re.escape(heading)}\n\n((?:\|.*\n)+)"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"找不到表格段落: {heading}")
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip().startswith("|")]
    data_lines = []
    for line in lines[2:]:
        cols = [part.strip() for part in line.strip("|").split("|")]
        data_lines.append(cols)
    return data_lines





def _load_main_table_statuses():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    statuses = {}
    headings = [
        ("B1. 專案設定對應清單（不含暫時特例）", 3),
        ("B2. 未明列於專案設定，但正式 test suite 應納入", 4),
    ]
    for heading, status_idx in headings:
        rows = _extract_table_rows(text, heading)
        for cols in rows:
            if len(cols) > status_idx:
                statuses[cols[0]] = cols[status_idx]
    return statuses

def _load_done_d_rows():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    rows = _extract_table_rows(text, "F2. 目前所有 `DONE` 的建議測試項目摘要")
    parsed = []
    for cols in rows:
        if len(cols) < 4:
            continue
        parsed.append(
            {
                "id": cols[0],
                "name": cols[1].strip("`").strip(),
                "b_id": cols[2],
                "done_date": cols[3],
            }
        )
    return parsed


def _load_done_b_rows():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    rows = _extract_table_rows(text, "F1. 目前所有 `DONE` 的主表項目摘要")
    parsed = []
    for cols in rows:
        if len(cols) < 5:
            continue
        parsed.append(
            {
                "kind": cols[0],
                "b_id": cols[1],
                "item": cols[2],
                "entry": cols[3],
                "done_date": cols[4],
            }
        )
    return parsed


def validate_registry_checklist_entry_consistency_case(_base_params):
    from tools.validate.synthetic_cases import get_synthetic_validators

    case_id = "META_REGISTRY_CHECKLIST_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validators = get_synthetic_validators()
    validator_names = [validator.__name__ for validator in validators]
    validator_name_set = set(validator_names)
    done_d_rows = _load_done_d_rows()
    done_b_rows = _load_done_b_rows()
    main_statuses = _load_main_table_statuses()

    add_check(results, "meta_registry", case_id, "validator_registry_not_empty", True, len(validators) > 0)
    add_check(results, "meta_registry", case_id, "validator_registry_names_unique", len(validator_names), len(validator_name_set))

    done_d_names = [row["name"] for row in done_d_rows]
    done_d_name_set = set(done_d_names)
    add_check(results, "meta_registry", case_id, "done_d_names_unique", len(done_d_names), len(done_d_name_set))

    for row in done_d_rows:
        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['id']}_registered_in_main_entry",
            True,
            row["name"] in validator_name_set,
        )

    done_b_ids = [row["b_id"] for row in done_b_rows]
    mapped_b_ids = {row["b_id"] for row in done_d_rows}
    for row in done_b_rows:
        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['b_id']}_done_summary_status_matches_main_table",
            "DONE",
            main_statuses.get(row["b_id"]),
        )
        if row["entry"] == "既有 synthetic case":
            continue
        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['b_id']}_done_summary_has_done_d_mapping",
            True,
            row["b_id"] in mapped_b_ids,
        )
        entry_path = row["entry"].split(",")[0].strip().strip("`")
        add_check(
            results,
            "meta_registry",
            case_id,
            f"{row['b_id']}_declared_entry_file_exists",
            True,
            (PROJECT_ROOT / entry_path).exists(),
        )

    summary["done_d_count"] = len(done_d_rows)
    summary["done_b_count"] = len(done_b_ids)
    summary["validator_count"] = len(validators)
    return results, summary



def _count_failures(results):
    return sum(1 for row in results if row.get("status") == "FAIL")


def validate_known_bad_fault_injection_case(base_params):
    case_id = "META_KNOWN_BAD_FAULT_INJECTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.validate.synthetic_flow_cases import validate_synthetic_same_day_buy_sell_forbidden_case
    from tools.validate.synthetic_history_cases import validate_synthetic_portfolio_history_filter_only_case
    from tools.validate.synthetic_take_profit_cases import (
        validate_synthetic_fee_tax_net_equity_case,
        validate_synthetic_same_bar_stop_priority_case,
    )

    def broken_same_day_sell_stats(*args, **kwargs):
        stats = {
            "trade_count": 1,
            "portfolio_missed_buy_rows": 0,
            "total_missed": 0,
            "df_trades": __import__("pandas").DataFrame(
                [
                    {"Date": "2024-02-26", "Type": "買進(一般)"},
                    {"Date": "2024-02-26", "Type": "全倉結算(停損)"},
                ]
            ),
        }
        return stats

    with patch('tools.validate.synthetic_flow_cases.run_portfolio_core_check_for_dir', side_effect=broken_same_day_sell_stats), \
         patch('tools.validate.synthetic_flow_cases.run_portfolio_sim_tool_check_for_dir', side_effect=broken_same_day_sell_stats), \
         patch('tools.validate.synthetic_flow_cases.add_portfolio_stats_equality_checks', lambda *args, **kwargs: None):
        fault_results, _summary = validate_synthetic_same_day_buy_sell_forbidden_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'same_day_sell_fault_detected', True, _count_failures(fault_results) > 0)

    def broken_execute_bar_step(*args, **kwargs):
        position = dict(args[0])
        position['qty'] = 5
        position['sold_half'] = True
        position['realized_pnl'] = 25.0
        return position, 500.0, 25.0, ['TP_HALF']

    with patch('tools.validate.synthetic_take_profit_cases.execute_bar_step', side_effect=broken_execute_bar_step):
        fault_results, _summary = validate_synthetic_same_bar_stop_priority_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'same_bar_stop_priority_fault_detected', True, _count_failures(fault_results) > 0)

    def broken_run_portfolio_timeline(*args, **kwargs):
        module = __import__('tools.validate.synthetic_take_profit_cases', fromlist=['run_portfolio_timeline'])
        original = broken_run_portfolio_timeline._original
        timeline = list(original(*args, **kwargs))
        df_equity = timeline[0].copy()
        df_trades = timeline[1].copy()
        if not df_equity.empty:
            df_equity.loc[df_equity.index[-1], 'Equity'] = float(df_equity.iloc[-1]['Equity']) + 50.0
        if not df_trades.empty:
            mask = df_trades['Type'].fillna('').isin(['全倉結算(停損)', '全倉結算(指標)'])
            if mask.any():
                idx = df_trades[mask].index[0]
                df_trades.loc[idx, '該筆總損益'] = float(df_trades.loc[idx, '該筆總損益']) + 50.0
        timeline[0] = df_equity
        timeline[1] = df_trades
        timeline[2] = float(timeline[2]) + 0.5
        timeline[8] = float(timeline[8]) + 50.0
        return tuple(timeline)

    import tools.validate.synthetic_take_profit_cases as _tp_module
    broken_run_portfolio_timeline._original = _tp_module.run_portfolio_timeline
    with patch('tools.validate.synthetic_take_profit_cases.run_portfolio_timeline', side_effect=broken_run_portfolio_timeline):
        fault_results, _summary = validate_synthetic_fee_tax_net_equity_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'fee_tax_fault_detected', True, _count_failures(fault_results) > 0)

    def broken_run_v16_backtest(*args, **kwargs):
        return {
            'trade_count': 0,
            'is_candidate': False,
            'current_position': 0,
            'asset_growth': 0.0,
        }

    with patch('tools.validate.synthetic_history_cases.run_v16_backtest', side_effect=broken_run_v16_backtest):
        fault_results, _summary = validate_synthetic_portfolio_history_filter_only_case(base_params)
        add_check(results, 'meta_fault_injection', case_id, 'history_filter_misuse_fault_detected', True, _count_failures(fault_results) > 0)

    summary['fault_injections_checked'] = 4
    return results, summary

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from tools.validate.meta_contracts import extract_markdown_table_rows


STATUS_VALUES = frozenset({"DONE", "PARTIAL", "TODO", "N/A"})
MAIN_TABLE_SPECS = (
    ("B1", "B1. 長期固定核心規則（不含暫時特例）", 3),
    ("B2", "B2. 長期固定補充契約", 4),
    ("B3", "B3. 可隨策略升級調整的測試", 4),
)
T_TABLE_HEADING = "T. 目前所有 `DONE` 的建議測試項目摘要"
G_TABLE_HEADING = "G. 逐項收斂紀錄"


def load_checklist_tables(checklist_path: Path) -> Dict[str, List[List[str]]]:
    """Load only persisted checklist truth tables.

    B is the current-state truth, T is the minimal completed-test index, and G is
    the append-only transition log.  PARTIAL/TODO summaries are derived from B/G
    and intentionally are not persisted as duplicate Markdown tables.
    """

    text = Path(checklist_path).read_text(encoding="utf-8")
    tables = {
        key: extract_markdown_table_rows(text, heading)
        for key, heading, _status_idx in MAIN_TABLE_SPECS
    }
    tables["T"] = extract_markdown_table_rows(text, T_TABLE_HEADING)
    tables["G"] = extract_markdown_table_rows(text, G_TABLE_HEADING)
    return tables


def load_main_statuses(tables: Dict[str, List[List[str]]]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for key, _heading, status_idx in MAIN_TABLE_SPECS:
        for cols in tables.get(key, []):
            if len(cols) <= status_idx:
                continue
            statuses[cols[0].strip()] = cols[status_idx].strip()
    return statuses


def load_main_catalog(checklist_path: Path) -> Dict[str, Dict[str, str]]:
    tables = load_checklist_tables(checklist_path)
    catalog: Dict[str, Dict[str, str]] = {}

    for cols in tables["B1"]:
        if len(cols) > 5:
            catalog[cols[0].strip()] = {
                "kind": "規則",
                "item": cols[2].strip(),
                "entry": cols[5].strip(),
                "status": cols[3].strip(),
            }
    for key in ("B2", "B3"):
        for cols in tables[key]:
            if len(cols) > 6:
                catalog[cols[0].strip()] = {
                    "kind": cols[2].strip(),
                    "item": cols[3].strip(),
                    "entry": cols[6].strip(),
                    "status": cols[4].strip(),
                }
    return catalog


def load_done_test_rows(checklist_path: Path) -> List[Dict[str, str]]:
    tables = load_checklist_tables(checklist_path)
    parsed: List[Dict[str, str]] = []
    for cols in tables["T"]:
        if len(cols) < 3:
            continue
        parsed.append(
            {
                "id": cols[0].strip(),
                "name": cols[1].replace("`", "").strip(),
                "b_id": cols[2].strip(),
            }
        )
    return parsed


def load_done_b_rows(checklist_path: Path) -> List[Dict[str, str]]:
    catalog = load_main_catalog(checklist_path)
    return [
        {
            "kind": row["kind"],
            "b_id": b_id,
            "item": row["item"],
            "entry": row["entry"],
        }
        for b_id, row in sorted(catalog.items())
        if row.get("status") == "DONE"
    ]



def ids_from_table(rows: List[List[str]], idx: int = 1) -> List[str]:
    return [cols[idx].strip() for cols in rows if len(cols) > idx and cols[idx].strip()]


def sorted_unique(values: List[str]) -> List[str]:
    return sorted(dict.fromkeys(values))

def latest_statuses_from_convergence_rows(rows: List[List[str]]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for cols in rows:
        if len(cols) < 4:
            continue
        item_id = cols[1].strip()
        transition = cols[3].strip()
        if not item_id or "->" not in transition:
            continue
        statuses[item_id] = transition.split("->", 1)[-1].strip()
    return statuses


def load_convergence_latest_statuses(checklist_path: Path) -> Dict[str, str]:
    tables = load_checklist_tables(checklist_path)
    return latest_statuses_from_convergence_rows(tables["G"])


def derive_checklist_state(checklist_path: Path) -> Dict[str, object]:
    """Derive current summaries from the canonical B/T/G persisted truth."""

    tables = load_checklist_tables(checklist_path)
    main_statuses = load_main_statuses(tables)
    convergence_statuses = latest_statuses_from_convergence_rows(tables["G"])

    partial_ids = sorted(item_id for item_id, status in main_statuses.items() if status == "PARTIAL")
    todo_ids = sorted(item_id for item_id, status in main_statuses.items() if status == "TODO")
    done_ids = sorted(item_id for item_id, status in main_statuses.items() if status == "DONE")
    g_done_test_ids = sorted(
        item_id
        for item_id, status in convergence_statuses.items()
        if item_id.startswith("T") and status == "DONE"
    )
    g_unfinished_test_ids = sorted(
        item_id
        for item_id, status in convergence_statuses.items()
        if item_id.startswith("T") and status in {"PARTIAL", "TODO"}
    )

    return {
        "tables": tables,
        "main_statuses": main_statuses,
        "convergence_statuses": convergence_statuses,
        "partial_ids": partial_ids,
        "todo_ids": todo_ids,
        "done_ids": done_ids,
        "g_done_test_ids": g_done_test_ids,
        "g_unfinished_test_ids": g_unfinished_test_ids,
    }


__all__ = [
    "STATUS_VALUES",
    "MAIN_TABLE_SPECS",
    "T_TABLE_HEADING",
    "G_TABLE_HEADING",
    "load_checklist_tables",
    "load_main_statuses",
    "load_main_catalog",
    "load_done_test_rows",
    "load_done_b_rows",
    "ids_from_table",
    "sorted_unique",
    "latest_statuses_from_convergence_rows",
    "load_convergence_latest_statuses",
    "derive_checklist_state",
]

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tools.validate.meta_contracts import extract_markdown_table_rows


# v2 is a CURRENT contract.  Completed repairs are not status events and do
# not belong in the formal test catalog.
STATUS_VALUES = frozenset({"DONE", "PARTIAL", "TODO"})
MAIN_TABLE_SPECS = (
    ("B1", "B1. 長期固定核心規則（不含暫時特例）", 3),
    ("B2", "B2. 長期固定補充契約", 4),
    ("B3", "B3. 可隨策略升級調整的測試", 4),
)
T_TABLE_HEADING = "T. 目前所有 `DONE` 的建議測試項目摘要"
CONTRACT_SCHEMA_VERSION = 2
CONTRACT_FILENAME = "TEST_SUITE_CHECKLIST_CONTRACT.json"
CONTRACT_ID = "test_suite_checklist_current_v2"


def _tracking_id_sort_key(item_id: str) -> tuple[str, int, str]:
    normalized = str(item_id or "").strip()
    match = re.fullmatch(r"([A-Za-z]+)(\d+)(.*)", normalized)
    if not match:
        return (normalized, -1, "")
    return (match.group(1), int(match.group(2)), match.group(3))


def _contract_path_for(checklist_path: Path) -> Path:
    return Path(checklist_path).with_name(CONTRACT_FILENAME)


def load_checklist_tables(checklist_path: Path) -> Dict[str, List[List[str]]]:
    """Parse only the current B/T generated views (never historical events)."""
    text = Path(checklist_path).read_text(encoding="utf-8")
    tables = {
        key: extract_markdown_table_rows(text, heading)
        for key, heading, _status_idx in MAIN_TABLE_SPECS
    }
    tables["T"] = extract_markdown_table_rows(text, T_TABLE_HEADING)
    return tables


def load_main_statuses(tables: Mapping[str, List[List[str]]]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for key, _heading, status_idx in MAIN_TABLE_SPECS:
        for cols in tables.get(key, []):
            if len(cols) > status_idx:
                statuses[cols[0].strip()] = cols[status_idx].strip()
    return statuses


def ids_from_table(rows: List[List[str]], idx: int = 1) -> List[str]:
    return [cols[idx].strip() for cols in rows if len(cols) > idx and cols[idx].strip()]


def sorted_unique(values: List[str]) -> List[str]:
    return sorted(dict.fromkeys(values), key=_tracking_id_sort_key)


def validate_checklist_contract(contract: Mapping[str, Any]) -> Dict[str, str]:
    """Validate the one current contract, including active B-to-T mappings.

    v1 status history, N/A retirement markers, and initial-status migration
    flags are rejected rather than silently kept in the v2 live contract.
    """
    if set(contract) != {"schema_version", "contract_id", "main_items", "tests"}:
        raise ValueError("checklist v2 requires schema_version/contract_id/main_items/tests only")
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION or contract["contract_id"] != CONTRACT_ID:
        raise ValueError(f"unsupported checklist current contract identity: {contract.get('schema_version')}/{contract.get('contract_id')}")
    if not isinstance(contract["main_items"], list) or not isinstance(contract["tests"], list):
        raise ValueError("checklist main_items and tests must be lists")

    main = contract["main_items"]
    tests = contract["tests"]
    definitions = [*main, *tests]
    if any(not isinstance(row, dict) for row in definitions):
        raise ValueError("checklist definitions must be objects")
    ids = [str(row.get("id", "")).strip() for row in definitions]
    duplicates = [item_id for item_id, count in Counter(ids).items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate checklist IDs: {duplicates}")
    resolved: Dict[str, str] = {}
    main_ids: set[str] = set()
    main_orders: set[int] = set()
    test_orders: set[int] = set()
    for row in main:
        item_id = str(row.get("id", "")).strip()
        section = str(row.get("section", ""))
        required = {"id", "section", "priority", "item", "gap", "entry", "status", "order"}
        allowed = required | ({"category"} if section in {"B2", "B3"} else set())
        if set(row) != allowed:
            raise ValueError(f"{item_id}: invalid v2 B fields: {set(row) ^ allowed}")
        if not re.fullmatch(r"B\d+", item_id) or section not in {spec[0] for spec in MAIN_TABLE_SPECS}:
            raise ValueError(f"invalid checklist main ID/section: {item_id}/{section}")
        if not str(row["item"]).strip() or not str(row["entry"]).strip():
            raise ValueError(f"{item_id}: current rule and implementation required")
        if row["status"] not in STATUS_VALUES:
            raise ValueError(f"{item_id}: retired/invalid main status: {row['status']}")
        if type(row["order"]) is not int or row["order"] < 0 or row["order"] in main_orders:
            raise ValueError(f"{item_id}: invalid/duplicate main order")
        main_orders.add(row["order"])
        main_ids.add(item_id)
        resolved[item_id] = row["status"]
    for row in tests:
        item_id = str(row.get("id", "")).strip()
        if set(row) != {"id", "description", "b_id", "status", "order"}:
            raise ValueError(f"{item_id}: invalid v2 T fields")
        if not re.fullmatch(r"T\d+", item_id):
            raise ValueError(f"invalid checklist test ID: {item_id}")
        if row["status"] not in STATUS_VALUES:
            raise ValueError(f"{item_id}: retired/invalid test status")
        if str(row["b_id"]).strip() not in main_ids:
            raise ValueError(f"{item_id}: active test references missing/retired B={row['b_id']!r}")
        if not str(row["description"]).strip():
            raise ValueError(f"{item_id}: current test requires description")
        if type(row["order"]) is not int or row["order"] < 0 or row["order"] in test_orders:
            raise ValueError(f"{item_id}: invalid/duplicate test order")
        test_orders.add(row["order"])
        resolved[item_id] = row["status"]
    return resolved


def load_checklist_contract(checklist_path: Path) -> Dict[str, Any]:
    path = _contract_path_for(Path(checklist_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checklist v2 contract root must be an object")
    validate_checklist_contract(payload)
    return payload


def render_contract_tables(contract: Mapping[str, Any]) -> Dict[str, List[List[str]]]:
    resolved = validate_checklist_contract(contract)
    tables: Dict[str, List[List[str]]] = {key: [] for key, _heading, _idx in MAIN_TABLE_SPECS}
    for row in sorted(contract["main_items"], key=lambda item: item["order"]):
        item_id = row["id"]
        base = [item_id, row["priority"]]
        if row["section"] == "B1":
            base += [row["item"], resolved[item_id], row["gap"], row["entry"]]
        else:
            base += [row["category"], row["item"], resolved[item_id], row["gap"], row["entry"]]
        tables[row["section"]].append([str(v) for v in base])
    tables["T"] = [
        [row["id"], row["description"], row["b_id"]]
        for row in sorted(contract["tests"], key=lambda x: _tracking_id_sort_key(x["id"]))
        if resolved[row["id"]] == "DONE"
    ]
    return tables


def compare_persisted_views(checklist_path: Path) -> Dict[str, Any]:
    checklist_path = Path(checklist_path)
    if not _contract_path_for(checklist_path).exists():
        # Mutation fixtures can intentionally contain only a Markdown view.
        return {"ok": True, "mismatches": {}, "mode": "markdown_only_fixture"}
    persisted = load_checklist_tables(checklist_path)
    canonical = render_contract_tables(load_checklist_contract(checklist_path))
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key in (*[spec[0] for spec in MAIN_TABLE_SPECS], "T"):
        if persisted.get(key, []) == canonical.get(key, []):
            continue
        left_rows, right_rows = persisted.get(key, []), canonical.get(key, [])
        first_diff = None
        for index in range(max(len(left_rows), len(right_rows))):
            left = left_rows[index] if index < len(left_rows) else None
            right = right_rows[index] if index < len(right_rows) else None
            if left != right:
                first_diff = {"index": index, "persisted": left, "canonical": right}
                break
        mismatches[key] = {"persisted_count": len(left_rows), "canonical_count": len(right_rows), "first_diff": first_diff}
    if re.search(r"^## G\. 逐項收斂紀錄\s*$", checklist_path.read_text(encoding="utf-8"), re.M):
        mismatches["legacy_G"] = {"reason": "retired status-history table must not be present"}
    return {"ok": not mismatches, "mismatches": mismatches}


def _find_table_blocks(lines: List[str], heading: str) -> List[tuple[int, int]]:
    heading_candidates = (f"### {heading}", f"## {heading}")
    heading_index = next((i for i, line in enumerate(lines) if line.strip() in heading_candidates), None)
    if heading_index is None:
        raise ValueError(f"找不到 checklist heading={heading}")
    section_end = len(lines)
    for index in range(heading_index + 1, len(lines)):
        if lines[index].strip().startswith(("## ", "### ")):
            section_end = index
            break
    blocks: List[tuple[int, int]] = []
    cursor = heading_index + 1
    while cursor < section_end:
        if not lines[cursor].strip().startswith("|"):
            cursor += 1
            continue
        start = cursor
        while cursor < section_end and lines[cursor].strip().startswith("|"):
            cursor += 1
        blocks.append((start, cursor))
    if not blocks:
        raise ValueError(f"找不到 checklist table={heading}")
    return blocks


def _render_markdown_row(cols: Sequence[str]) -> str:
    return "| " + " | ".join(str(value) for value in cols) + " |"


def render_checklist_markdown(checklist_path: Path, contract: Mapping[str, Any]) -> str:
    """Update only B/T tables; preserve current documentation around them."""
    text = Path(checklist_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    tables = render_contract_tables(contract)
    headings = {key: heading for key, heading, _idx in MAIN_TABLE_SPECS}
    headings["T"] = T_TABLE_HEADING
    replacements = []
    for key, heading in headings.items():
        blocks = _find_table_blocks(lines, heading)
        start, end = blocks[0]
        existing = lines[start:end]
        if len(existing) < 2:
            raise ValueError(f"checklist table {key} lacks header/separator")
        replacements.append((start, end, [existing[0], existing[1], *[_render_markdown_row(row) for row in tables[key]]]))
        replacements.extend((start, end, []) for start, end in blocks[1:])
    for start, end, rows in sorted(replacements, reverse=True):
        lines[start:end] = rows
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_checklist_contract(checklist_path: Path, contract: Mapping[str, Any], *, sync_markdown: bool = True) -> None:
    checklist_path = Path(checklist_path)
    payload = dict(contract)
    validate_checklist_contract(payload)
    contract_path = _contract_path_for(checklist_path)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_checklist_markdown(checklist_path, payload) if sync_markdown else None
    old_json = contract_path.read_text(encoding="utf-8") if contract_path.exists() else None
    try:
        _write_text_atomic(contract_path, json_text)
        if markdown_text is not None:
            _write_text_atomic(checklist_path, markdown_text)
    except Exception:
        if old_json is None:
            contract_path.unlink(missing_ok=True)
        else:
            _write_text_atomic(contract_path, old_json)
        raise


def apply_checklist_transaction(
    checklist_path: Path,
    *,
    main_definitions: Iterable[Mapping[str, Any]] = (),
    test_definitions: Iterable[Mapping[str, Any]] = (),
    main_updates: Iterable[Mapping[str, Any]] = (),
    test_updates: Iterable[Mapping[str, Any]] = (),
    retire_main_ids: Iterable[str] = (),
    retire_test_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """One canonical current-state transaction; retire obsolete rows, not events."""
    contract = load_checklist_contract(checklist_path)
    existing_ids = {row["id"] for row in [*contract["main_items"], *contract["tests"]]}
    next_main_order = max((row["order"] for row in contract["main_items"]), default=-1) + 1
    next_test_order = max((row["order"] for row in contract["tests"]), default=-1) + 1
    for raw in main_definitions:
        row = dict(raw)
        if row.get("id") in existing_ids:
            raise ValueError(f"checklist definition already exists: {row['id']}")
        row.setdefault("status", "TODO")
        row.setdefault("order", next_main_order)
        next_main_order += 1
        contract["main_items"].append(row)
        existing_ids.add(row.get("id"))
    for raw in test_definitions:
        row = dict(raw)
        if row.get("id") in existing_ids:
            raise ValueError(f"checklist test already exists: {row['id']}")
        row.setdefault("status", "TODO")
        row.setdefault("order", next_test_order)
        next_test_order += 1
        contract["tests"].append(row)
        existing_ids.add(row.get("id"))
    for collection, updates in (("main_items", main_updates), ("tests", test_updates)):
        by_id = {row["id"]: row for row in contract[collection]}
        for raw in updates:
            update = dict(raw)
            item_id = update.pop("id", "")
            if item_id not in by_id:
                raise ValueError(f"checklist update references unknown definition: {item_id}")
            if {"section", "order", "id"} & set(update):
                raise ValueError(f"{item_id}: immutable checklist identity/order")
            by_id[item_id].update(update)
    retired_b = set(retire_main_ids)
    retired_t = set(retire_test_ids)
    if retired_b - {row["id"] for row in contract["main_items"]} or retired_t - {row["id"] for row in contract["tests"]}:
        raise ValueError("checklist retirement references unknown IDs")
    contract["main_items"] = [row for row in contract["main_items"] if row["id"] not in retired_b]
    contract["tests"] = [row for row in contract["tests"] if row["id"] not in retired_t]
    # The v2 validator forbids orphaned active tests in the same transaction.
    write_checklist_contract(checklist_path, contract)
    return contract


def derive_checklist_state(checklist_path: Path) -> Dict[str, object]:
    checklist_path = Path(checklist_path)
    contract = load_checklist_contract(checklist_path) if _contract_path_for(checklist_path).exists() else None
    tables = render_contract_tables(contract) if contract is not None else load_checklist_tables(checklist_path)
    main_statuses = load_main_statuses(tables)
    test_statuses = {row["id"]: row["status"] for row in contract["tests"]} if contract is not None else {row[0]: "DONE" for row in tables["T"]}
    return {
        "contract": contract,
        "tables": tables,
        "main_statuses": main_statuses,
        "test_statuses": test_statuses,
        "partial_ids": sorted((k for k, v in main_statuses.items() if v == "PARTIAL"), key=_tracking_id_sort_key),
        "todo_ids": sorted((k for k, v in main_statuses.items() if v == "TODO"), key=_tracking_id_sort_key),
        "done_ids": sorted((k for k, v in main_statuses.items() if v == "DONE"), key=_tracking_id_sort_key),
        "unfinished_test_ids": sorted((k for k, v in test_statuses.items() if v != "DONE"), key=_tracking_id_sort_key),
    }


def load_main_catalog(checklist_path: Path) -> Dict[str, Dict[str, str]]:
    tables = derive_checklist_state(checklist_path)["tables"]
    catalog: Dict[str, Dict[str, str]] = {}
    for cols in tables["B1"]:
        if len(cols) > 5:
            catalog[cols[0]] = {"kind": "規則", "item": cols[2], "entry": cols[5], "status": cols[3]}
    for key in ("B2", "B3"):
        for cols in tables[key]:
            if len(cols) > 6:
                catalog[cols[0]] = {"kind": cols[2], "item": cols[3], "entry": cols[6], "status": cols[4]}
    return catalog


def load_done_test_rows(checklist_path: Path) -> List[Dict[str, str]]:
    return [
        {"id": cols[0], "name": cols[1].replace("`", "").strip(), "b_id": cols[2]}
        for cols in derive_checklist_state(checklist_path)["tables"]["T"] if len(cols) >= 3
    ]


def load_done_b_rows(checklist_path: Path) -> List[Dict[str, str]]:
    return [
        {"kind": row["kind"], "b_id": b_id, "item": row["item"], "entry": row["entry"]}
        for b_id, row in sorted(load_main_catalog(checklist_path).items(), key=lambda pair: _tracking_id_sort_key(pair[0]))
        if row["status"] == "DONE"
    ]


__all__ = [
    "STATUS_VALUES", "MAIN_TABLE_SPECS", "T_TABLE_HEADING", "CONTRACT_SCHEMA_VERSION", "CONTRACT_FILENAME",
    "load_checklist_tables", "load_main_statuses", "load_main_catalog", "load_done_test_rows", "load_done_b_rows",
    "ids_from_table", "sorted_unique", "load_checklist_contract", "render_contract_tables", "compare_persisted_views",
    "render_checklist_markdown", "write_checklist_contract", "apply_checklist_transaction",
    "validate_checklist_contract", "derive_checklist_state",
]

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tools.validate.meta_contracts import extract_markdown_table_rows


STATUS_VALUES = frozenset({"DONE", "PARTIAL", "TODO", "N/A"})
INITIAL_STATUS_VALUES = STATUS_VALUES | {"NEW"}
MAIN_TABLE_SPECS = (
    ("B1", "B1. 長期固定核心規則（不含暫時特例）", 3),
    ("B2", "B2. 長期固定補充契約", 4),
    ("B3", "B3. 可隨策略升級調整的測試", 4),
)
T_TABLE_HEADING = "T. 目前所有 `DONE` 的建議測試項目摘要"
G_TABLE_HEADING = "G. 逐項收斂紀錄"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_FILENAME = "TEST_SUITE_CHECKLIST_CONTRACT.json"


def _tracking_id_sort_key(item_id: str) -> tuple[str, int, str]:
    normalized = str(item_id or "").strip()
    match = re.fullmatch(r"([A-Za-z]+)(\d+)(.*)", normalized)
    if not match:
        return (normalized, -1, "")
    return (match.group(1), int(match.group(2)), match.group(3))


def _contract_path_for(checklist_path: Path) -> Path:
    return Path(checklist_path).with_name(CONTRACT_FILENAME)


def load_checklist_tables(checklist_path: Path) -> Dict[str, List[List[str]]]:
    """Load the persisted Markdown B/T/G views.

    These tables are generated views once a sibling machine-readable contract
    exists.  Keeping this parser separate is intentional: meta-quality can
    prove that the persisted human-readable view still matches the canonical
    contract instead of silently trusting either side.
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


def ids_from_table(rows: List[List[str]], idx: int = 1) -> List[str]:
    return [cols[idx].strip() for cols in rows if len(cols) > idx and cols[idx].strip()]


def sorted_unique(values: List[str]) -> List[str]:
    return sorted(dict.fromkeys(values), key=_tracking_id_sort_key)


def _split_transition(raw_transition: str) -> tuple[str, str]:
    normalized = str(raw_transition or "").strip()
    if "->" not in normalized:
        raise ValueError(f"invalid checklist transition: {normalized!r}")
    from_status, to_status = [part.strip() for part in normalized.split("->", 1)]
    if from_status not in INITIAL_STATUS_VALUES:
        raise ValueError(f"invalid checklist from_status={from_status!r}")
    if to_status not in STATUS_VALUES:
        raise ValueError(f"invalid checklist to_status={to_status!r}")
    if from_status == to_status:
        raise ValueError(f"checklist transition must change state: {normalized!r}")
    return from_status, to_status


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


def _events_by_id(events: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event.get("id", "")).strip(), []).append(event)
    return grouped


def _definition_index(contract: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    index: Dict[str, Mapping[str, Any]] = {}
    for row in contract.get("main_items", []):
        index[str(row.get("id", "")).strip()] = row
    for row in contract.get("tests", []):
        index[str(row.get("id", "")).strip()] = row
    return index


def _resolved_statuses(contract: Mapping[str, Any]) -> Dict[str, str]:
    definitions = _definition_index(contract)
    grouped_events = _events_by_id(contract.get("transitions", []))
    resolved: Dict[str, str] = {}
    for item_id, definition in definitions.items():
        status = str(definition.get("initial_status", "")).strip()
        if status not in INITIAL_STATUS_VALUES:
            raise ValueError(f"{item_id}: invalid initial_status={status!r}")
        for event in grouped_events.get(item_id, []):
            from_status = str(event.get("from_status", "")).strip()
            to_status = str(event.get("to_status", "")).strip()
            if from_status != status:
                raise ValueError(
                    f"{item_id}: transition chain mismatch; expected from={status!r}, got {from_status!r}"
                )
            if to_status not in STATUS_VALUES:
                raise ValueError(f"{item_id}: invalid to_status={to_status!r}")
            if from_status == to_status:
                raise ValueError(f"{item_id}: no-op transition {from_status!r} -> {to_status!r}")
            status = to_status
        if status == "NEW":
            raise ValueError(f"{item_id}: unresolved NEW status is not a persisted current state")
        resolved[item_id] = status
    return resolved


def validate_checklist_contract(contract: Mapping[str, Any]) -> Dict[str, str]:
    schema_version = int(contract.get("schema_version", -1))
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checklist contract schema_version={schema_version}; expected={CONTRACT_SCHEMA_VERSION}"
        )

    main_items = list(contract.get("main_items", []))
    tests = list(contract.get("tests", []))
    transitions = list(contract.get("transitions", []))
    definitions = [*main_items, *tests]
    ids = [str(row.get("id", "")).strip() for row in definitions]
    if any(not item_id for item_id in ids):
        raise ValueError("checklist definitions require non-empty IDs")
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1}, key=_tracking_id_sort_key)
    if duplicates:
        raise ValueError(f"duplicate checklist definition IDs: {duplicates}")

    main_ids = {str(row.get("id", "")).strip() for row in main_items}
    test_ids = {str(row.get("id", "")).strip() for row in tests}
    for row in definitions:
        item_id = str(row.get("id", "")).strip()
        initial_status = str(row.get("initial_status", "")).strip()
        if initial_status != "NEW" and not bool(row.get("legacy_initial_status", False)):
            raise ValueError(
                f"{item_id}: non-NEW initial_status is migration-only and requires legacy_initial_status=true"
            )
    for row in main_items:
        item_id = str(row.get("id", "")).strip()
        if not item_id.startswith("B"):
            raise ValueError(f"main checklist definition must use B ID: {item_id}")
        if str(row.get("section", "")) not in {spec[0] for spec in MAIN_TABLE_SPECS}:
            raise ValueError(f"{item_id}: invalid section={row.get('section')!r}")
    for row in tests:
        item_id = str(row.get("id", "")).strip()
        if not item_id.startswith("T"):
            raise ValueError(f"test checklist definition must use T ID: {item_id}")

    previous_date = ""
    seen_event_ids: set[str] = set()
    for event in transitions:
        item_id = str(event.get("id", "")).strip()
        if item_id not in main_ids | test_ids:
            raise ValueError(f"transition references unknown checklist ID={item_id}")
        date = str(event.get("date", "")).strip()
        if previous_date and date < previous_date:
            raise ValueError(f"checklist transition dates must be non-decreasing: {previous_date} -> {date}")
        previous_date = date
        from_status = str(event.get("from_status", "")).strip()
        to_status = str(event.get("to_status", "")).strip()
        if from_status not in INITIAL_STATUS_VALUES or to_status not in STATUS_VALUES:
            raise ValueError(f"{item_id}: invalid transition={from_status!r}->{to_status!r}")
        if from_status == to_status:
            raise ValueError(f"{item_id}: no-op transition={from_status!r}->{to_status!r}")
        if from_status == "NEW" and item_id in seen_event_ids:
            raise ValueError(f"{item_id}: NEW transition may only appear on first occurrence")
        seen_event_ids.add(item_id)

    resolved = _resolved_statuses(contract)
    for row in tests:
        test_id = str(row.get("id", "")).strip()
        if resolved.get(test_id) != "DONE":
            continue
        b_id = str(row.get("b_id", "")).strip()
        description = str(row.get("description", "")).strip()
        if not b_id or b_id not in main_ids:
            raise ValueError(f"{test_id}: DONE test requires valid b_id, got={b_id!r}")
        if not description:
            raise ValueError(f"{test_id}: DONE test requires non-empty description")

    return resolved


def _initial_status_from_rows(item_id: str, current_status: str, g_rows_by_id: Mapping[str, Sequence[List[str]]]) -> str:
    rows = list(g_rows_by_id.get(item_id, []))
    if not rows:
        return current_status
    if len(rows[0]) < 4:
        return current_status
    from_status, _to_status = _split_transition(rows[0][3])
    return from_status


def build_contract_from_markdown(checklist_path: Path) -> Dict[str, Any]:
    """One-time/compatibility migration from the historical Markdown B/T/G tables."""

    tables = load_checklist_tables(checklist_path)
    g_rows_by_id: Dict[str, List[List[str]]] = {}
    for row in tables["G"]:
        if len(row) > 1:
            g_rows_by_id.setdefault(row[1].strip(), []).append(row)

    main_items: List[Dict[str, Any]] = []
    order = 0
    for section, _heading, status_idx in MAIN_TABLE_SPECS:
        for cols in tables[section]:
            if len(cols) <= status_idx:
                continue
            item_id = cols[0].strip()
            current_status = cols[status_idx].strip()
            item: Dict[str, Any] = {
                "id": item_id,
                "section": section,
                "priority": cols[1].strip(),
                "initial_status": _initial_status_from_rows(item_id, current_status, g_rows_by_id),
                "order": order,
            }
            item["legacy_initial_status"] = item["initial_status"] != "NEW"
            if section == "B1":
                if len(cols) < 6:
                    raise ValueError(f"{item_id}: malformed B1 row={cols}")
                item.update({"item": cols[2].strip(), "gap": cols[4].strip(), "entry": cols[5].strip()})
            else:
                if len(cols) < 7:
                    raise ValueError(f"{item_id}: malformed {section} row={cols}")
                item.update(
                    {
                        "category": cols[2].strip(),
                        "item": cols[3].strip(),
                        "gap": cols[5].strip(),
                        "entry": cols[6].strip(),
                    }
                )
            main_items.append(item)
            order += 1

    current_t_rows = {row[0].strip(): row for row in tables["T"] if len(row) >= 3}
    all_t_ids = sorted(
        {row[1].strip() for row in tables["G"] if len(row) > 1 and row[1].strip().startswith("T")},
        key=_tracking_id_sort_key,
    )
    tests: List[Dict[str, Any]] = []
    for order, test_id in enumerate(all_t_ids):
        current_row = current_t_rows.get(test_id)
        history = g_rows_by_id.get(test_id, [])
        if current_row is not None:
            description = current_row[1].strip()
            b_id = current_row[2].strip()
        else:
            description = history[0][2].strip() if history and len(history[0]) > 2 else ""
            b_id = ""
        current_status = ""
        if history and len(history[-1]) > 3 and "->" in history[-1][3]:
            current_status = history[-1][3].split("->", 1)[-1].strip()
        initial_status = _initial_status_from_rows(test_id, current_status, g_rows_by_id)
        tests.append(
            {
                "id": test_id,
                "description": description,
                "b_id": b_id,
                "initial_status": initial_status,
                "legacy_initial_status": initial_status != "NEW",
                "order": order,
            }
        )

    transitions: List[Dict[str, str]] = []
    for row in tables["G"]:
        if len(row) < 4:
            continue
        from_status, to_status = _split_transition(row[3])
        transitions.append(
            {
                "date": row[0].strip(),
                "id": row[1].strip(),
                "description": row[2].strip(),
                "from_status": from_status,
                "to_status": to_status,
                "note": row[4].strip() if len(row) > 4 else "",
            }
        )

    contract: Dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": "test_suite_checklist_ssot_v1",
        "main_items": main_items,
        "tests": tests,
        "transitions": transitions,
    }
    validate_checklist_contract(contract)
    return contract


def load_checklist_contract(checklist_path: Path) -> Dict[str, Any]:
    checklist_path = Path(checklist_path)
    contract_path = _contract_path_for(checklist_path)
    if contract_path.exists():
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"checklist contract root must be object: {contract_path}")
        validate_checklist_contract(payload)
        return payload
    # Mutation tests and old snapshots may intentionally provide only Markdown.
    return build_contract_from_markdown(checklist_path)


def render_contract_tables(contract: Mapping[str, Any]) -> Dict[str, List[List[str]]]:
    resolved = validate_checklist_contract(contract)
    tables: Dict[str, List[List[str]]] = {key: [] for key, _heading, _status_idx in MAIN_TABLE_SPECS}
    for row in sorted(contract.get("main_items", []), key=lambda item: int(item.get("order", 0))):
        section = str(row.get("section", ""))
        item_id = str(row.get("id", "")).strip()
        status = resolved[item_id]
        if section == "B1":
            tables[section].append(
                [
                    item_id,
                    str(row.get("priority", "")),
                    str(row.get("item", "")),
                    status,
                    str(row.get("gap", "")),
                    str(row.get("entry", "")),
                ]
            )
        else:
            tables[section].append(
                [
                    item_id,
                    str(row.get("priority", "")),
                    str(row.get("category", "")),
                    str(row.get("item", "")),
                    status,
                    str(row.get("gap", "")),
                    str(row.get("entry", "")),
                ]
            )

    tables["T"] = []
    for row in sorted(contract.get("tests", []), key=lambda item: _tracking_id_sort_key(str(item.get("id", "")))):
        test_id = str(row.get("id", "")).strip()
        if resolved.get(test_id) != "DONE":
            continue
        tables["T"].append([test_id, str(row.get("description", "")), str(row.get("b_id", ""))])

    tables["G"] = [
        [
            str(event.get("date", "")),
            str(event.get("id", "")),
            str(event.get("description", "")),
            f"{event.get('from_status', '')} -> {event.get('to_status', '')}",
            str(event.get("note", "")),
        ]
        for event in contract.get("transitions", [])
    ]
    return tables


def compare_persisted_views(checklist_path: Path) -> Dict[str, Any]:
    checklist_path = Path(checklist_path)
    if not _contract_path_for(checklist_path).exists():
        # Historical/mutation fixtures intentionally contain only Markdown.
        # They remain governed by the legacy parser so old fault-injection tests
        # can mutate B/T/G independently without pretending to be canonical data.
        return {"ok": True, "mismatches": {}, "mode": "legacy_markdown_fixture"}
    persisted = load_checklist_tables(checklist_path)
    canonical = render_contract_tables(load_checklist_contract(checklist_path))
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key in (*[spec[0] for spec in MAIN_TABLE_SPECS], "T", "G"):
        if persisted.get(key, []) == canonical.get(key, []):
            continue
        persisted_rows = persisted.get(key, [])
        canonical_rows = canonical.get(key, [])
        first_diff = None
        for index in range(max(len(persisted_rows), len(canonical_rows))):
            left = persisted_rows[index] if index < len(persisted_rows) else None
            right = canonical_rows[index] if index < len(canonical_rows) else None
            if left != right:
                first_diff = {"index": index, "persisted": left, "canonical": right}
                break
        mismatches[key] = {
            "persisted_count": len(persisted_rows),
            "canonical_count": len(canonical_rows),
            "first_diff": first_diff,
        }
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
    """Render canonical B/T/G tables while preserving the hand-written narrative."""

    text = Path(checklist_path).read_text(encoding="utf-8")
    trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    tables = render_contract_tables(contract)
    heading_by_key = {key: heading for key, heading, _status_idx in MAIN_TABLE_SPECS}
    heading_by_key.update({"T": T_TABLE_HEADING, "G": G_TABLE_HEADING})

    # Replace from bottom to top so earlier line offsets remain stable.
    replacements = []
    for key, heading in heading_by_key.items():
        blocks = _find_table_blocks(lines, heading)
        first_start, first_end = blocks[0]
        existing = lines[first_start:first_end]
        if len(existing) < 2:
            raise ValueError(f"checklist table={key} missing header/separator")
        replacement = [existing[0], existing[1], *[_render_markdown_row(row) for row in tables[key]]]
        replacements.append((first_start, first_end, replacement))
        for start, end in blocks[1:]:
            replacements.append((start, end, []))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = replacement
    rendered = "\n".join(lines)
    if trailing_newline:
        rendered += "\n"
    return rendered


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def write_checklist_contract(checklist_path: Path, contract: Mapping[str, Any], *, sync_markdown: bool = True) -> None:
    checklist_path = Path(checklist_path)
    payload = deepcopy(dict(contract))
    validate_checklist_contract(payload)
    contract_path = _contract_path_for(checklist_path)
    contract_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_checklist_markdown(checklist_path, payload) if sync_markdown else None

    old_contract = contract_path.read_text(encoding="utf-8") if contract_path.exists() else None
    try:
        _write_text_atomic(contract_path, contract_text)
        if markdown_text is not None:
            _write_text_atomic(checklist_path, markdown_text)
    except Exception:
        if old_contract is None:
            contract_path.unlink(missing_ok=True)
        else:
            _write_text_atomic(contract_path, old_contract)
        raise


def apply_checklist_transaction(
    checklist_path: Path,
    *,
    main_definitions: Iterable[Mapping[str, Any]] = (),
    test_definitions: Iterable[Mapping[str, Any]] = (),
    transitions: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Atomically update canonical definitions/events and regenerate all B/T/G views.

    This is the write seam for future checklist maintenance: callers register a
    B definition, optional T bindings, and append transitions once.  They must
    never hand-edit the three Markdown tables independently.
    """

    contract = deepcopy(load_checklist_contract(checklist_path))
    existing_ids = set(_definition_index(contract))
    next_main_order = max((int(row.get("order", -1)) for row in contract.get("main_items", [])), default=-1) + 1
    next_test_order = max((int(row.get("order", -1)) for row in contract.get("tests", [])), default=-1) + 1

    for raw in main_definitions:
        row = dict(raw)
        item_id = str(row.get("id", "")).strip()
        if item_id in existing_ids:
            raise ValueError(f"checklist definition already exists: {item_id}")
        row.setdefault("initial_status", "NEW")
        if str(row.get("initial_status", "")).strip() != "NEW":
            raise ValueError("new checklist definitions must start at NEW and transition explicitly")
        row.pop("legacy_initial_status", None)
        row.setdefault("order", next_main_order)
        next_main_order += 1
        contract.setdefault("main_items", []).append(row)
        existing_ids.add(item_id)

    for raw in test_definitions:
        row = dict(raw)
        item_id = str(row.get("id", "")).strip()
        if item_id in existing_ids:
            raise ValueError(f"checklist definition already exists: {item_id}")
        row.setdefault("initial_status", "NEW")
        if str(row.get("initial_status", "")).strip() != "NEW":
            raise ValueError("new checklist test definitions must start at NEW and transition explicitly")
        row.pop("legacy_initial_status", None)
        row.setdefault("order", next_test_order)
        next_test_order += 1
        contract.setdefault("tests", []).append(row)
        existing_ids.add(item_id)

    previous_date = str(contract.get("transitions", [])[-1].get("date", "")) if contract.get("transitions") else ""
    for raw in transitions:
        event = dict(raw)
        item_id = str(event.get("id", "")).strip()
        if item_id not in existing_ids:
            raise ValueError(f"transition references unknown checklist ID={item_id}")
        date = str(event.get("date", "")).strip()
        if previous_date and date < previous_date:
            raise ValueError(f"new transition date {date} precedes append-only tail {previous_date}")
        previous_date = date
        contract.setdefault("transitions", []).append(event)

    validate_checklist_contract(contract)
    write_checklist_contract(checklist_path, contract, sync_markdown=True)
    return contract


def load_main_catalog(checklist_path: Path) -> Dict[str, Dict[str, str]]:
    tables = derive_checklist_state(checklist_path)["tables"]
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
    tables = derive_checklist_state(checklist_path)["tables"]
    parsed: List[Dict[str, str]] = []
    for cols in tables["T"]:
        if len(cols) < 3:
            continue
        parsed.append({"id": cols[0].strip(), "name": cols[1].replace("`", "").strip(), "b_id": cols[2].strip()})
    return parsed


def load_done_b_rows(checklist_path: Path) -> List[Dict[str, str]]:
    catalog = load_main_catalog(checklist_path)
    return [
        {"kind": row["kind"], "b_id": b_id, "item": row["item"], "entry": row["entry"]}
        for b_id, row in sorted(catalog.items(), key=lambda item: _tracking_id_sort_key(item[0]))
        if row.get("status") == "DONE"
    ]


def load_convergence_latest_statuses(checklist_path: Path) -> Dict[str, str]:
    return dict(derive_checklist_state(checklist_path)["convergence_statuses"])


def derive_checklist_state(checklist_path: Path) -> Dict[str, object]:
    """Resolve current B/T summaries from the canonical contract when present.

    The real project checklist always has a sibling JSON owner.  Historical and
    mutation fixtures that intentionally contain Markdown only keep the legacy
    parser semantics so fault-injection tests can exercise malformed views.
    """

    checklist_path = Path(checklist_path)
    contract_path = _contract_path_for(checklist_path)
    if contract_path.exists():
        contract = load_checklist_contract(checklist_path)
        tables = render_contract_tables(contract)
    else:
        contract = None
        tables = load_checklist_tables(checklist_path)
    main_statuses = load_main_statuses(tables)
    convergence_statuses = latest_statuses_from_convergence_rows(tables["G"])

    partial_ids = sorted((item_id for item_id, status in main_statuses.items() if status == "PARTIAL"), key=_tracking_id_sort_key)
    todo_ids = sorted((item_id for item_id, status in main_statuses.items() if status == "TODO"), key=_tracking_id_sort_key)
    done_ids = sorted((item_id for item_id, status in main_statuses.items() if status == "DONE"), key=_tracking_id_sort_key)
    g_done_test_ids = sorted(
        (item_id for item_id, status in convergence_statuses.items() if item_id.startswith("T") and status == "DONE"),
        key=_tracking_id_sort_key,
    )
    g_unfinished_test_ids = sorted(
        (
            item_id
            for item_id, status in convergence_statuses.items()
            if item_id.startswith("T") and status in {"PARTIAL", "TODO"}
        ),
        key=_tracking_id_sort_key,
    )

    return {
        "contract": contract,
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
    "INITIAL_STATUS_VALUES",
    "MAIN_TABLE_SPECS",
    "T_TABLE_HEADING",
    "G_TABLE_HEADING",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_FILENAME",
    "load_checklist_tables",
    "load_main_statuses",
    "load_main_catalog",
    "load_done_test_rows",
    "load_done_b_rows",
    "ids_from_table",
    "sorted_unique",
    "latest_statuses_from_convergence_rows",
    "load_convergence_latest_statuses",
    "build_contract_from_markdown",
    "load_checklist_contract",
    "render_contract_tables",
    "compare_persisted_views",
    "render_checklist_markdown",
    "write_checklist_contract",
    "apply_checklist_transaction",
    "validate_checklist_contract",
    "derive_checklist_state",
]

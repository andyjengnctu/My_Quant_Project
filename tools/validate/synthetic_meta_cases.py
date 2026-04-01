from pathlib import Path
import re

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

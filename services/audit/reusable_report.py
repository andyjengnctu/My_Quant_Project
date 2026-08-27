"""Shared presentation/persistence helpers for reusable read-only Audit reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config.audit import AUDIT_OUTPUT_ROOT
from core.research_report_contract import ReportTableContract, format_contract_value
from core.console_report import (
    console_color_enabled,
    paint,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from core.report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_NEUTRAL,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    markdown_tone,
    signal_for_delta,
    styled_signal,
)


def audit_title(text: str, *, target: str) -> str:
    if target == "console":
        return render_title(paint(text, "cyan", enabled=console_color_enabled(), bold=True))
    if target == "markdown":
        return f"# {markdown_tone(text, 'blue', bold=True)}"
    return str(text)


def audit_section(text: str, number: int, *, target: str) -> str:
    label = f"{int(number)}. {text}"
    if target == "console":
        colored = paint(label, "cyan", enabled=console_color_enabled(), bold=True)
        return f"\n{colored}\n{'-' * len(label)}"
    if target == "markdown":
        return f"## {markdown_tone(label, 'blue', bold=True)}"
    return label


def status_text(text: str, signal: str, *, target: str, bold: bool = False) -> str:
    return styled_signal(text, signal, target=target, bold=bold)


def evidence_signal(status: str) -> str:
    value = str(status).strip().upper()
    if value in {"PRESENT", "IMPROVED", "STABLE", "AVAILABLE", "READY", "RECONCILED", "SAME"}:
        return SIGNAL_POSITIVE
    if value in {"COLLAPSED", "WORSE", "BLOCKED", "FAILED", "DIVERGED", "MISSING"}:
        return SIGNAL_NEGATIVE
    if value in {"MIXED", "WARNING", "PARTIAL"}:
        return SIGNAL_WARNING
    return SIGNAL_NEUTRAL


def evidence_status_for_delta(value: Any, *, preference: str) -> str:
    signal = signal_for_delta(value, preference=str(preference))
    if signal == SIGNAL_POSITIVE:
        return "IMPROVED"
    if signal == SIGNAL_NEGATIVE:
        return "WORSE"
    return "MIXED"


def render_evidence_rows(rows: Iterable[tuple[str, str, str]], *, target: str) -> str:
    styled = [
        (label, status_text(status, evidence_signal(status), target=target, bold=True), detail)
        for label, status, detail in rows
    ]
    if target == "console":
        return render_table(("Evidence", "Status", "Detail"), styled)
    header = "| Evidence | Status | Detail |\n|---|---|---|"
    body = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in styled)
    return header + ("\n" + body if body else "")


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{int(digits)}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)




def format_contract_row(table: ReportTableContract, values: Mapping[str, Any]) -> list[str]:
    row = dict(values or {})
    return [format_contract_value(column, row.get(column.key)) for column in table.columns]

def truth_cell(cell: Mapping[str, Any]) -> str:
    row = dict(cell or {})
    n = int(row.get("n", 0) or 0)
    pct = row.get("population_pct")
    enrich = row.get("independence_enrichment")
    return f"{n:,} / {fmt(pct, 2, '%')} / {fmt(enrich, 2, '×')}"


def render_truth_5x5(geometry: Mapping[str, Any], *, target: str) -> str:
    rows: list[list[str]] = []
    for s_idx, row in enumerate(list(geometry.get("cells") or []), start=1):
        rows.append([f"S{s_idx}", *(truth_cell(dict(cell or {})) for cell in row)])
    headers = ["Actual Safety \\ Pure-MFE", "M1", "M2", "M3", "M4", "M5"]
    if target == "console":
        return render_table(headers, rows)
    header = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return header + ("\n" + body if body else "")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(str(value) for value in headers) + " |"
    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def persist_reusable_report(
    *,
    definition,
    project_root: Path,
    payload: Mapping[str, Any],
    console_text: str,
    markdown_text: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = root / AUDIT_OUTPUT_ROOT / Path(definition.output_subdir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    latest_dir = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    result = dict(payload)
    result.setdefault("schema_version", 1)
    result.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    result["report_id"] = definition.report_id
    result["report_type"] = definition.report_type
    result["read_only"] = True
    result["artifacts"] = {
        "markdown": project_relative_display_path(run_dir / "audit.md", project_root=root),
        "json": project_relative_display_path(run_dir / "audit.json", project_root=root),
    }
    json_text = json.dumps(result, ensure_ascii=False, indent=2)
    (run_dir / "audit.md").write_text(markdown_text.rstrip() + "\n", encoding="utf-8")
    (run_dir / "audit.json").write_text(json_text + "\n", encoding="utf-8")
    (run_dir / "console.txt").write_text(console_text.rstrip() + "\n", encoding="utf-8")
    (latest_dir / "audit.md").write_text(markdown_text.rstrip() + "\n", encoding="utf-8")
    (latest_dir / "audit.json").write_text(json_text + "\n", encoding="utf-8")
    return result


def source_summary(definition, *, target: str) -> str:
    source = dict(definition.source)
    rows = (
        ("Report", definition.report_id),
        ("Comparison", f"{source.get('treatment_arm_id', '-')} vs {source.get('control_arm_id', '-')}") ,
        ("Evaluation", " + ".join(str(x) for x in source.get("evaluation_profile_ids", ()))),
        ("Breakdown", str(definition.dimensions.get("breakdown", "overall"))),
        ("Contract", "READ ONLY / canonical artifacts only"),
    )
    if target == "console":
        return render_key_values(rows)
    return "\n".join(f"- **{label}**: `{value}`" for label, value in rows)


__all__ = [
    "audit_section",
    "audit_title",
    "evidence_signal",
    "evidence_status_for_delta",
    "fmt",
    "format_contract_row",
    "markdown_table",
    "persist_reusable_report",
    "render_evidence_rows",
    "render_truth_5x5",
    "source_summary",
    "status_text",
    "truth_cell",
]

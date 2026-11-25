"""Compact execution-status views of already resolved Research plans."""
from collections import Counter

from core.console_report import DEFAULT_REPORT_WIDTH, _wrap_display_cell, render_section, render_table
from core.report_style import styled_workflow_status


def single_line(value):
    return " ".join(str(value).split())


def render_seed_status_matrix(label, *, seeds, rows):
    """Use Target's S1/S2 legend; callers supply every actual per-seed decision."""
    seeds = tuple(seeds)
    rows = tuple((str(identity), tuple(actions)) for identity, actions in rows)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seed status matrix requires distinct seeds")
    if len({identity for identity, _ in rows}) != len(rows):
        raise ValueError("Seed status matrix has duplicate identities")
    if any(len(actions) != len(seeds) or any(not str(action).strip() for action in actions)
           for _, actions in rows):
        raise ValueError("Seed status matrix must include each seed decision")
    legend = "Seed：" + "  ".join(f"S{index}={seed}" for index, seed in enumerate(seeds, 1))
    counts = Counter(str(action) for _, actions in rows for action in actions)
    return "\n".join((
        legend,
        render_table((label, *(f"S{index}" for index in range(1, len(seeds) + 1))),
                     [(identity, *(styled_workflow_status(action) for action in actions))
                      for identity, actions in rows]),
        " / ".join(f"{styled_workflow_status(action)} {count}" for action, count in counts.items()),
    ))


def render_action_details(rows, *, title="需處理項目"):
    """Keep actionable reasons, wrapping complete text instead of clipping it."""
    grouped = {}
    for action, identity, reason in rows:
        grouped.setdefault((str(action), single_line(reason)), []).append(single_line(identity))
    if not grouped:
        return ""
    return render_section(title) + "\n" + render_table(
        ("項目", "動作", "原因"),
        [(" / ".join(dict.fromkeys(identities)), styled_workflow_status(action), reason)
         for (action, reason), identities in grouped.items()],
        max_column_widths=(28, 16, DEFAULT_REPORT_WIDTH - 48),
    )


def render_dependency_summary(rows):
    """Completed dependencies need a count; pending dependencies retain reasons."""
    rows = tuple(rows)
    if not rows:
        return ""
    counts = Counter(str(action) for action, _, _ in rows)
    summary = "前置／其他工作：" + " / ".join(
        f"{styled_workflow_status(action)} {count}" for action, count in counts.items()
    )
    pending = [(action, identity, reason) for action, identity, reason in rows if action != "REUSE"]
    details = render_action_details(pending)
    return summary + ("\n" + details if details else "")


def render_plan_metadata(rows):
    """Do not put internal hashes or profile paths ahead of the work matrix."""
    lines = []
    for label, value in rows:
        if label in {"設定檔", "Config fingerprint", "Seed generator", "Benchmark ID", "Benchmark seeds"}:
            continue
        if label == "整體狀態":
            value = styled_workflow_status(value)
        line = f"{label}：{value}"
        lines.extend(_wrap_display_cell(line, DEFAULT_REPORT_WIDTH))
    return "\n".join(lines)

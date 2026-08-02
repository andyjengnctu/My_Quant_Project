"""Prepare the configured continuous target artifacts before ranker training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config.breakout_quality import get_breakout_quality_workflow_settings
from filters.breakout_quality.continuous_target import (
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    TARGET_AUDIT_MARKDOWN_FILENAME,
    load_validated_continuous_target_arrays,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import DEFAULT_FILTER_ID, DEFAULT_LABEL_POLICY
from tools.filters.breakout_quality.audit_continuous_target import main as build_base_target
from tools.filters.breakout_quality.audit_no_time_continuous_target import (
    main as build_no_time_target,
)
from tools.filters.breakout_quality.common import PROJECT_ROOT, load_validated_dataset_bundle
from filters.breakout_quality.console_report import (
    compact_console_enabled,
    console_color_enabled,
    paint,
    print_artifact_paths,
)

SUPPORTED_TARGET_IDS = (
    STRATEGY_ALIGNED_TARGET_ID,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
)


def _readable_report_path(*, filter_id: str, target_id: str) -> Path:
    return (
        resolve_continuous_target_dir(
            PROJECT_ROOT,
            filter_id,
            target_id=target_id,
        )
        / TARGET_AUDIT_MARKDOWN_FILENAME
    )


def parse_args(argv=None) -> argparse.Namespace:
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "檢查目前continuous target與Dataset identity；缺少或過期時自動重建。"
            "本命令不訓練模型"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--target-id",
        choices=SUPPORTED_TARGET_IDS,
        default=settings.continuous_target_id,
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="只供歷史重現；預設要求Dataset來源inventory為目前版本",
    )
    args = parser.parse_args(argv)
    if not str(args.target_id or "").strip():
        parser.error("目前workflow沒有設定continuous target")
    return args


def _dataset_identity(filter_id: str, *, allow_stale_source: bool) -> tuple[dict[str, Any], int]:
    summary, indexed_features, _context, _labels, _events = load_validated_dataset_bundle(
        filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(allow_stale_source),
    )
    return summary, int(len(indexed_features.feature_bank))


def _target_is_current(
    *,
    filter_id: str,
    target_id: str,
    summary: dict[str, Any],
    group_count: int,
) -> tuple[bool, str]:
    try:
        load_validated_continuous_target_arrays(
            PROJECT_ROOT,
            filter_id,
            target_id=target_id,
            expected_group_count=group_count,
            expected_dataset_policy=summary.get("policy"),
            expected_dataset_artifacts=summary.get("dataset_artifacts"),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "current"


def _build_target(
    *,
    filter_id: str,
    target_id: str,
    allow_stale_source: bool,
) -> int:
    common_args = ["--filter-id", str(filter_id)]
    if allow_stale_source:
        common_args.append("--allow-stale-source")

    if target_id == STRATEGY_ALIGNED_TARGET_ID:
        return int(build_base_target(common_args) or 0)
    if target_id != STRATEGY_ALIGNED_NO_TIME_TARGET_ID:
        raise ValueError(f"不支援的continuous target: {target_id}")

    summary, group_count = _dataset_identity(
        filter_id,
        allow_stale_source=allow_stale_source,
    )
    source_current, source_reason = _target_is_current(
        filter_id=filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
        summary=summary,
        group_count=group_count,
    )
    if not source_current:
        if compact_console_enabled():
            print("[Continuous Target] 先建立必要的基礎 target")
        else:
            print("[Continuous Target] 基礎component target缺少或過期，先重建：")
            print(f"- {source_reason}")
        code = int(build_base_target(common_args) or 0)
        if code != 0:
            return code

    no_time_args = [*common_args, "--approved-workflow-rebuild"]
    return int(build_no_time_target(no_time_args) or 0)


def main(argv=None) -> int:
    args = parse_args(argv)
    color_enabled = console_color_enabled()
    summary, group_count = _dataset_identity(
        args.filter_id,
        allow_stale_source=bool(args.allow_stale_source),
    )
    target_id = str(args.target_id)
    current, reason = _target_is_current(
        filter_id=args.filter_id,
        target_id=target_id,
        summary=summary,
        group_count=group_count,
    )
    compact_console = compact_console_enabled()
    if current:
        if not compact_console:
            print(
                paint(
                    f"[略過] Continuous Target 已符合目前 Dataset：{target_id}",
                    "green",
                    enabled=color_enabled,
                    bold=True,
                )
            )
            report_path = _readable_report_path(
                filter_id=args.filter_id,
                target_id=target_id,
            )
            if report_path.is_file():
                print_artifact_paths((("Continuous Target Markdown", report_path),), project_root=PROJECT_ROOT)
        return 0

    print(
        paint(
            f"[Continuous Target] 建立／更新：{target_id}",
            "yellow",
            enabled=color_enabled,
            bold=True,
        )
    )
    if not compact_console:
        print(f"- 原因：{reason}")
    code = _build_target(
        filter_id=args.filter_id,
        target_id=target_id,
        allow_stale_source=bool(args.allow_stale_source),
    )
    if code != 0:
        return code

    refreshed_summary, refreshed_group_count = _dataset_identity(
        args.filter_id,
        allow_stale_source=bool(args.allow_stale_source),
    )
    current, reason = _target_is_current(
        filter_id=args.filter_id,
        target_id=target_id,
        summary=refreshed_summary,
        group_count=refreshed_group_count,
    )
    if not current:
        raise ValueError(f"Continuous Target重建後仍未通過identity驗證: {reason}")
    if not compact_console:
        print(
            paint(
                f"[Continuous Target] identity 驗證完成：{target_id}",
                "green",
                enabled=color_enabled,
                bold=True,
            )
        )
        report_path = _readable_report_path(
            filter_id=args.filter_id,
            target_id=target_id,
        )
        if report_path.is_file():
            print_artifact_paths((("Continuous Target Markdown", report_path),), project_root=PROJECT_ROOT)
    return 0


__all__ = ["SUPPORTED_TARGET_IDS", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical dataset date inspection helpers shared by Optimizer and Trading runtime."""
from __future__ import annotations

import os


def resolve_latest_dataset_date(data_dir: str | os.PathLike[str]) -> str:
    import pandas as pd
    from core.data_utils import discover_unique_csv_inputs

    latest = None
    csv_inputs, _duplicate_issue_lines = discover_unique_csv_inputs(os.fspath(data_dir))
    for csv_entry in csv_inputs:
        if isinstance(csv_entry, (tuple, list)) and len(csv_entry) >= 2:
            _ticker, csv_path = csv_entry[0], csv_entry[1]
        else:
            _ticker, csv_path = "", csv_entry
        try:
            df = pd.read_csv(csv_path, usecols=lambda col: str(col).strip().lower() in {"date", "time"})
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        columns = list(df.columns)
        if not columns:
            continue
        series = pd.to_datetime(df[columns[0]], errors="coerce").dropna()
        if series.empty:
            continue
        value = series.max().normalize()
        if latest is None or value > latest:
            latest = value
    if latest is None:
        raise ValueError("無法從資料集 CSV 解析最新交易日，請確認資料包含 Date/Time 欄位。")
    return latest.strftime("%Y-%m-%d")


__all__ = ["resolve_latest_dataset_date"]

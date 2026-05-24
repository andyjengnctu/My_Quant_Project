"""Export dl_pass score table from a trained breakout quality filter."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import shutil

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import DEFAULT_FILTER_ID, DEFAULT_MODEL_FILENAME, DEFAULT_SCORE_FILENAME, LABEL_PASS
from filters.breakout_quality.model import build_model, require_torch
from tools.filters.breakout_quality.common import dataset_npz_path, events_csv_path, model_dir, read_breakout_quality_csv, scores_csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="匯出 breakout quality score table")
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--copy-to-model-dir", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch, _nn = require_torch()
    data = np.load(dataset_npz_path(args.filter_id))
    X = data["features"].astype(np.float32)
    C = data["context"].astype(np.float32)
    events = read_breakout_quality_csv(events_csv_path(args.filter_id))
    checkpoint_path = model_dir(args.filter_id) / DEFAULT_MODEL_FILENAME
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(int(checkpoint["feature_count"]), int(checkpoint["context_count"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X), torch.from_numpy(C))
        prob = torch.softmax(logits, dim=1).cpu().numpy()
    events = events.copy()
    events["dl_quality_score"] = prob[:, LABEL_PASS]
    events["dl_pass"] = np.argmax(prob, axis=1) == LABEL_PASS
    out_cols = ["ticker", "date", "high_len", "dl_pass", "dl_quality_score", "label", "label_reason"]
    scores_path = scores_csv_path(args.filter_id)
    events[out_cols].to_csv(scores_path, index=False, encoding="utf-8-sig")
    print(f"已輸出: {scores_path}")
    if args.copy_to_model_dir:
        target = model_dir(args.filter_id) / DEFAULT_SCORE_FILENAME
        shutil.copy2(scores_path, target)
        print(f"已複製: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

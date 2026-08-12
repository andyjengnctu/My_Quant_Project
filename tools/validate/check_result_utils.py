from core.runtime_utils import is_insufficient_data_error
import copy
import math

import pandas as pd


FLOAT_TOL = 1e-6


def normalize_ticker_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text[:-2].isdigit() and text.endswith(".0"):
        text = text[:-2]

    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)

    return text


def make_consistency_params(base_params):
    params = copy.deepcopy(base_params)
    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0
    return params


def add_check(results, module_name, ticker, metric, expected, actual, tol=FLOAT_TOL, note=""):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        expected_float = float(expected)
        actual_float = float(actual)
        if math.isinf(expected_float) or math.isinf(actual_float):
            diff = 0.0 if expected_float == actual_float else math.inf
            passed = expected_float == actual_float
        else:
            diff = abs(expected_float - actual_float)
            passed = diff <= tol
    else:
        diff = None
        passed = expected == actual

    status = "PASS" if passed else "FAIL"

    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "abs_diff": diff,
        "passed": passed,
        "status": status,
        "note": note
    })


def add_skip_result(results, module_name, ticker, metric, note):
    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": "SKIP",
        "actual": "SKIP",
        "abs_diff": None,
        "passed": True,
        "status": "SKIP",
        "note": note
    })


def add_fail_result(results, module_name, ticker, metric, expected, actual, note):
    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "abs_diff": None,
        "passed": False,
        "status": "FAIL",
        "note": note
    })

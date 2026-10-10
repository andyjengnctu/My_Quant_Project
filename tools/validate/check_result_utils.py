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


def bind_checks(results, module_name, ticker):
    """Bind stable result/category/case context for dense contract validators."""

    def check(metric, expected, actual, tol=FLOAT_TOL, note=""):
        add_check(results, module_name, ticker, metric, expected, actual, tol=tol, note=note)

    def check_true(metric, actual, tol=FLOAT_TOL, note=""):
        add_check(results, module_name, ticker, metric, True, actual, tol=tol, note=note)

    return check, check_true


def bind_synthetic_case(case_id, module_name, *, training_performed=None):
    """Create standard synthetic result state and bind its stable check context."""

    results = []
    summary = {"ticker": case_id, "synthetic": True}
    if training_performed is not None:
        summary["training_performed"] = bool(training_performed)
    check, check_true = bind_checks(results, module_name, case_id)
    return results, summary, check, check_true


def raises_expected(error_type, action):
    """Return whether a zero-argument validation action raises the expected error type."""

    try:
        action()
    except error_type:
        return True
    return False


def run_bound_checks(bound_check, cases):
    """Run a declarative sequence through one already-bound synthetic check function."""
    for args in cases:
        bound_check(*args)


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


def run_unittest_capability_cases(case_id, module_name, test_classes):
    """Report selected isolated unittest fixtures as canonical synthetic cases.

    Test classes are reusable capability invariants, not current model identities.
    GUI-only tests remain independent of the headless synthetic runner.
    """
    import unittest

    results, summary, _check, check_true = bind_synthetic_case(case_id, module_name)
    completed = 0
    failures = {}
    for test_class in test_classes:
        for method_name in unittest.defaultTestLoader.getTestCaseNames(test_class):
            result = unittest.TestResult()
            test_class(method_name).run(result)
            metric = f"{test_class.__name__}.{method_name}"
            problems = [*result.failures, *result.errors, *result.unexpectedSuccesses]
            if result.skipped:
                add_skip_result(results, module_name, case_id, metric, str(result.skipped[0][1]))
            else:
                check_true(metric, not problems)
            if problems:
                first = problems[0]
                failures[metric] = (str(first[1]) if isinstance(first, tuple) else str(first))[-1000:]
            completed += result.testsRun
    summary["test_count"] = completed
    summary["failures"] = failures
    return results, summary

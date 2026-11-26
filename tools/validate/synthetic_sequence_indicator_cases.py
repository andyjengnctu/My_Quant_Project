"""Full-history indicator invariants exercised by the reusable model case."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from filters.breakout_quality.daily_ranker_data import LazyDailyFeatureBank
from filters.breakout_quality.models.architectures import SUPPORTED_MODEL_ARCHITECTURES
from filters.breakout_quality.models.spec import get_model_spec, model_spec_from_manifest
from filters.breakout_quality.sequence_indicators import full_history_stock_emas


def check_sequence_indicators(check, check_true):
    from .synthetic_calendar_bar_cases import check_calendar_bar_inputs

    check_calendar_bar_inputs(check, check_true)
    spans = (2, 4)
    closes = np.asarray([10.0, 20.0, 14.0, 26.0])
    expected = np.repeat(closes[:1, None], len(spans), axis=1)
    rows = [expected[0].copy()]
    for close in closes[1:]:
        previous = rows[-1]
        rows.append(np.asarray([
            2.0 / (span + 1) * close + (1 - 2.0 / (span + 1)) * previous[i]
            for i, span in enumerate(spans)
        ]))
    check_true("full_history_ema_matches_hand_recurrence", np.allclose(
        full_history_stock_emas(closes, spans), np.asarray(rows), rtol=0, atol=1e-14,
    ))
    missing = full_history_stock_emas(np.asarray([np.nan, 10.0, np.nan, 20.0]), spans)
    check_true("ema_initializes_at_first_observation_and_skips_missing", (
        np.isnan(missing[0]).all() and np.array_equal(missing[1], missing[2])
        and np.allclose(missing[-1], rows[1])
    ))

    close = 100.0 + np.arange(360) * 0.1
    close[:35] += 500.0
    frame = pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.02, "Low": close * 0.98,
        "Close": close, "Volume": 1000.0 + np.arange(len(close)) * 3,
    }, index=pd.date_range("2010-01-01", periods=len(close), freq="B"))
    positions = np.asarray([340, 299, 350])

    def bank(source, ema_spans):
        return LazyDailyFeatureBank(
            frames=(source,), benchmark=frame,
            ticker_ids=np.zeros(len(positions), dtype=np.int32),
            source_positions=positions, benchmark_positions=positions,
            policy=SimpleNamespace(feature_window_bars=300),
            stock_close_ema_spans=ema_spans,
        )

    baseline = bank(frame, ())[:]
    for architecture in SUPPORTED_MODEL_ARCHITECTURES:
        spec = get_model_spec(architecture)
        if not spec.stock_close_ema_spans:
            continue
        spans = spec.stock_close_ema_spans
        indicator_bank = bank(frame, spans)
        features = indicator_bank[:]
        prefix = f"{architecture}_"
        check(prefix + "input_shape", (3, 300, 10 + len(spans)), features.shape)
        check_true(prefix + "preserves_every_original_input", np.array_equal(features[:, :, :10], baseline))
        full = full_history_stock_emas(close, spans)
        reference = np.asarray([
            full[pos - 299:pos + 1] / close[pos] - 1.0 for pos in positions
        ], dtype=np.float32)
        check_true(prefix + "complete_history_then_window_slice", np.array_equal(features[:, :, 10:], reference))
        reset = full_history_stock_emas(close[41:341], spans) / close[340] - 1.0
        check_true(prefix + "retains_history_before_300_day_window", not np.allclose(
            features[0, :, 10:], reset, rtol=0, atol=1e-5,
        ))
        check_true(prefix + "batch_scalar_and_repeated_reordered_reads_match", np.array_equal(
            indicator_bank[[2, 0, 2]], np.stack([indicator_bank[2], indicator_bank[0], indicator_bank[2]]),
        ))
        future = frame.copy()
        future.iloc[341:, :4] *= 5
        check_true(prefix + "future_prices_cannot_change_past_inputs", np.array_equal(
            indicator_bank[0], bank(future, spans)[0],
        ))
        historical = frame.copy()
        historical.iloc[:41, :4] *= 2
        changed = bank(historical, spans)[0]
        check_true(prefix + "only_ema_changes_when_pre_window_history_changes", (
            np.array_equal(changed[:, :10], features[0, :, :10])
            and not np.array_equal(changed[:, 10:], features[0, :, 10:])
        ))
        scaled = frame.copy()
        scaled.loc[:, ["Open", "High", "Low", "Close"]] *= 8
        check_true(prefix + "uniform_adjustment_scale_invariance", np.array_equal(
            features, bank(scaled, spans)[:],
        ))
        check_true(prefix + "manifest_uses_actual_feature_order", tuple(
            indicator_bank.manifest_feature_columns
        ) == indicator_bank.feature_columns)
        payload = spec.as_manifest_payload()
        check(prefix + "model_manifest_roundtrip", spec, model_spec_from_manifest(payload))
        other_spans = (*spans, spans[-1] + 1)
        check_true(prefix + "changed_spans_change_fit_spec", (
            payload != replace(spec, stock_close_ema_spans=other_spans).as_manifest_payload()
        ))
        from filters.breakout_quality.models.active import build_active_model
        import torch

        model = build_active_model(10 + len(spans), 0, architecture=architecture)
        model.eval()
        with torch.no_grad():
            result = model(torch.from_numpy(features[:2]), torch.empty((2, 0)))
        check(prefix + "formal_model_accepts_appended_inputs", (2, 2), tuple(result.shape))

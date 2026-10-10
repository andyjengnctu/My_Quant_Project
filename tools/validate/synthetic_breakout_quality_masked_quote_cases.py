"""Quote-aware raw model and shared market observation capability tests.

Run independently: PYTHONPATH=. python -m unittest
  tools.validate.synthetic_breakout_quality_masked_quote_cases -v
This is not the project's formal integration suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE, BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
)
from core.breakout_quality_registry import (
    get_breakout_quality_experiment_profile, get_continuous_ranker_research_spec,
)
from core.file_integrity import canonical_json_sha256, compute_file_sha256
from core.market_data_raw_quote import raw_research_observed_ohlc_mask
from filters.breakout_quality.daily_ranker_data import (
    LazyDailyFeatureBank, load_aligned_observed_raw_quote_frame,
)
from filters.breakout_quality.model_source_provenance import recorded_input_price_source_issues
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.models.spec import get_model_spec
from services.breakout_quality.price_input_source import resolve_feature_price_source
from services.research.raw_price_strategy_dataset import (
    RAW_PRICE_QUOTE_POLICY_STRICT, RAW_PRICE_QUOTE_POLICY_OBSERVED,
    _build_files, inspect_raw_price_strategy_dataset,
)


class MaskedRawQuoteTests(unittest.TestCase):
    @staticmethod
    def frame(n=308, start="2020-01-01", slope=0.15):
        dates = pd.date_range(start, periods=n, freq="B")
        close = 100 + np.arange(n, dtype=np.float64) * slope
        return pd.DataFrame({
            "Open": close * .99, "High": close * 1.01,
            "Low": close * .98, "Close": close,
            "Volume": np.full(n, 100.0),
        }, index=dates)

    def test_new_scientific_pair_without_bw_reuse(self):
        model_id, profile_id = BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
        self.assertEqual(model_id, "MR-13BX")
        self.assertIn((model_id, profile_id), BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        self.assertIn(("MR-13AO", "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"),
                      BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        spec = get_continuous_ranker_research_spec(profile_id)
        self.assertEqual(spec.model_research_id, model_id)
        self.assertFalse(spec.selection_pit_authorized)
        ao_id = "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
        ao = get_breakout_quality_experiment_profile(ao_id).as_manifest_payload()
        bx = get_breakout_quality_experiment_profile(profile_id).as_manifest_payload()
        self.assertEqual({k: v for k,v in ao.items() if k not in {"name", "model_architecture"}},
                         {k: v for k,v in bx.items() if k not in {"name", "model_architecture"}})
        ao_arch = get_model_spec(get_breakout_quality_experiment_profile(ao_id).model_architecture).as_manifest_payload()
        bx_arch = get_model_spec(get_breakout_quality_experiment_profile(profile_id).model_architecture).as_manifest_payload()
        self.assertEqual({k:v for k,v in ao_arch.items() if k != "architecture"},
                         {k:v for k,v in bx_arch.items() if k != "architecture"})
        self.assertNotEqual(get_breakout_quality_experiment_profile(profile_id).model_architecture,
                            get_breakout_quality_experiment_profile(
                                "daily_universal_raw_input_adjusted_target_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
                            ).model_architecture)

    def test_quote_mask_semantics_and_emerging_open(self):
        raw = self.frame(6)
        raw.iloc[1, :4] = 0
        raw.iloc[2, raw.columns.get_loc("Open")] = raw.iloc[2]["High"] + 2.0
        raw.iloc[3, raw.columns.get_loc("Close")] = raw.iloc[3]["High"] + 1.0
        raw.iloc[4, raw.columns.get_loc("Low")] = 0
        mask = raw_research_observed_ohlc_mask(raw)
        self.assertEqual(mask.tolist(), [True, False, True, False, False, True])
        self.assertTrue((raw.iloc[1, :4] == 0.0).all())  # source unchanged

    def test_mask_loader_preserves_date_volume_and_zero_evidence(self):
        with TemporaryDirectory() as tmp:
            adjusted = self.frame()
            raw = self.frame(slope=0.41)
            raw.iloc[21, :4] = 0
            raw.iloc[36, raw.columns.get_loc("Open")] = raw.iloc[36]["High"] * 1.03
            raw.reset_index(names="Date").to_csv(Path(tmp) / "0052.csv", index=False)
            output = load_aligned_observed_raw_quote_frame(Path(tmp), "0052", adjusted)
            self.assertTrue(output.index.equals(adjusted.index))
            self.assertTrue(np.array_equal(output["Volume"], adjusted["Volume"]))
            self.assertEqual(output.iloc[21]["Close"], 0.0)
            self.assertTrue(raw_research_observed_ohlc_mask(output).iloc[36])
            raw.loc[adjusted.index[20], "Volume"] = 12
            raw.reset_index(names="Date").to_csv(Path(tmp) / "0052.csv", index=False)
            with self.assertRaisesRegex(ValueError, "volume mismatch"):
                load_aligned_observed_raw_quote_frame(Path(tmp), "0052", adjusted)
            raw = raw.drop(raw.index[20])
            raw.reset_index(names="Date").to_csv(Path(tmp) / "0052.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing required adjusted dates"):
                load_aligned_observed_raw_quote_frame(Path(tmp), "0052", adjusted)

    def test_two_masks_and_no_synthetic_prices_or_future_leak(self):
        adjusted = self.frame()
        raw = self.frame(slope=0.39)
        benchmark_adj = self.frame(slope=.04)
        benchmark_raw = self.frame(slope=.14)
        raw.iloc[42, :4] = 0  # no published market quote
        raw.iloc[299, :4] = 0  # score day has no raw quote
        benchmark_raw.iloc[48, :4] = 0
        bank = LazyDailyFeatureBank(
            frames=(adjusted,), benchmark=benchmark_adj,
            feature_frames=(raw,), feature_benchmark=benchmark_raw,
            feature_quote_mask=True, ticker_ids=np.array([0, 0]),
            source_positions=np.array([299, 300]),
            benchmark_positions=np.array([299, 300]),
            policy=SimpleNamespace(feature_window_bars=300),
        )
        self.assertEqual(bank.shape, (2, 300, 12))
        data = bank[:]
        self.assertTrue(np.isfinite(data).all())
        self.assertEqual(data[0, 42, 10], 0)
        self.assertTrue(np.all(data[0, 42, :4] == 0))
        self.assertEqual(data[0, 299, 10], 0)
        self.assertEqual(data[0, 48, 11], 0)
        self.assertEqual(data[0, 36, 11], 1)
        self.assertTrue(np.array_equal(bank._frame_arrays[0][:, 3], adjusted["Close"]))
        self.assertTrue(np.array_equal(bank._feature_frame_arrays[0][42, :4], np.zeros(4)))
        self.assertEqual(bank.feature_columns[-2:], ("Stock_Raw_Quote_Available", "Benchmark_Raw_Quote_Available"))
        # Changes after a historical scoring date must not affect earlier features.
        future = raw.copy()
        future.loc[future.index[300:], "Close"] = 999999.0
        future.loc[future.index[300:], "High"] = 999999.0
        future.loc[future.index[300:], "Open"] = 999999.0
        future.loc[future.index[300:], "Low"] = 999999.0
        replay = LazyDailyFeatureBank(
            frames=(adjusted,), benchmark=benchmark_adj,
            feature_frames=(future,), feature_benchmark=benchmark_raw,
            feature_quote_mask=True, ticker_ids=np.array([0]),
            source_positions=np.array([299]), benchmark_positions=np.array([299]),
            policy=SimpleNamespace(feature_window_bars=300),
        )
        np.testing.assert_array_equal(replay[0], data[0])

    def test_all_unquoted_window_is_neutral_not_price(self):
        adj = self.frame()
        raw = self.frame(slope=.4)
        raw.iloc[:, :4] = 0
        bank = LazyDailyFeatureBank(
            frames=(adj,), benchmark=adj, feature_frames=(raw,), feature_benchmark=raw,
            feature_quote_mask=True, ticker_ids=np.array([0]),
            source_positions=np.array([299]), benchmark_positions=np.array([299]),
            policy=SimpleNamespace(feature_window_bars=300),
        )
        data = bank[0]
        self.assertEqual(data.shape, (300, 12))
        np.testing.assert_array_equal(data[:, :4], np.zeros((300, 4)))
        np.testing.assert_array_equal(data[:, 5:9], np.zeros((300, 4)))
        np.testing.assert_array_equal(data[:, 10:], np.zeros((300, 2)))
        self.assertTrue(np.isfinite(data).all())

    def test_scientific_model_can_consume_12_inputs(self):
        import torch
        from config.breakout_quality import BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
        arch = get_breakout_quality_experiment_profile(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[1]).model_architecture
        model = build_model(12, 0, architecture=arch)
        with torch.no_grad():
            outputs = model(torch.zeros((2, 300, 12)), torch.zeros((2, 0)))
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_separate_source_policy_and_deep_reuse(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control"
            control.mkdir()
            adjusted = self.frame(10).reset_index(names="Date")
            adjusted.to_csv(control / "0052.csv", index=False)
            provider = adjusted.rename(columns={
                "Date": "date", "Open": "open", "High": "max", "Low": "min", "Close": "close",
            })
            provider["date"] = provider["date"].dt.strftime("%Y-%m-%d")
            provider["stock_id"] = "0052"
            provider.loc[provider.index[2], ["open", "max", "min", "close"]] = 0.0
            view = SimpleNamespace(iter_raw_price_strategy_frames=lambda **kw: iter([provider]))
            source = {
                "frozen_cutoff": "2026-03-02", "provider_snapshot_fingerprint": "a"*64,
                "control_dataset_inventory_fingerprint": "b"*64,
                "contract_id": "research_raw_observed_quote_mask_v1",
                "quote_policy": RAW_PRICE_QUOTE_POLICY_OBSERVED,
            }
            stage = root / "stage"
            files, count = _build_files(
                root=root, view=view, control_dir=control, stage_dir=stage,
                source_identity=source,
            )
            self.assertEqual(count, 10)
            self.assertEqual(len(files), 1)
            raw_csv = stage / "tw_stock_data_vip" / "0052.csv"
            self.assertEqual(float(pd.read_csv(raw_csv).iloc[2]["Close"]), 0.0)
            strict_source = dict(source, quote_policy=RAW_PRICE_QUOTE_POLICY_STRICT)
            with self.assertRaisesRegex(ValueError, "RAW_SOURCE_QUALITY_BLOCKED"):
                _build_files(root=root, view=view, control_dir=control, stage_dir=root/"strict",
                             source_identity=strict_source)
            source_no_fp = dict(source)
            source_fp = canonical_json_sha256(source_no_fp)
            source["source_identity_fingerprint"] = source_fp
            location = root / "outputs" / "research" / "raw_price_strategy_dataset" / source_fp
            location.mkdir(parents=True)
            import shutil
            shutil.move(str(stage / "tw_stock_data_vip"), location / "tw_stock_data_vip")
            inv = canonical_json_sha256(files)
            mat = canonical_json_sha256({"source_identity_fingerprint": source_fp,
                                         "dataset_inventory_sha256": inv})
            (location / "manifest.json").write_text(json.dumps({
                "source_identity": source, "source_identity_fingerprint": source_fp,
                "dataset_inventory_sha256": inv, "files": files,
                "data_identity": {"materialization_fingerprint": mat},
            }), encoding="utf-8")
            with patch("services.research.raw_price_strategy_dataset._source_identity",
                       return_value=(source, view, control)):
                state = inspect_raw_price_strategy_dataset(
                    root, deep=True, quote_policy=RAW_PRICE_QUOTE_POLICY_OBSERVED,
                )
                self.assertTrue(state["ready"])
                with self.assertRaisesRegex(ValueError, "RAW_SOURCE_QUALITY_BLOCKED"):
                    inspect_raw_price_strategy_dataset(root, deep=True)
            record = {
                "price_basis": "raw_unadjusted_ohlc_canonical_volume",
                "target_price_basis": "canonical_adjusted",
                "source_contract_id": source["contract_id"],
                "manifest_relative_path": (location / "manifest.json").relative_to(root).as_posix(),
                "source_identity_fingerprint": source_fp,
                "dataset_inventory_sha256": inv, "materialization_fingerprint": mat,
                "input_representation": "raw_observed_quote_mask_300x12_v1",
                "quote_policy": RAW_PRICE_QUOTE_POLICY_OBSERVED,
            }
            self.assertEqual(recorded_input_price_source_issues(published=record,project_root=root), ())
            forged = dict(record)
            forged.pop("quote_policy")
            self.assertIn("quote representation mismatch",
                          ";".join(recorded_input_price_source_issues(published=forged,project_root=root)))
            raw_csv = location / "tw_stock_data_vip" / "0052.csv"
            raw_csv.write_text(raw_csv.read_text().replace("0.0,0.0", "1.0,1.0"))
            self.assertIn("CSV SHA256 mismatch",
                          ";".join(recorded_input_price_source_issues(published=record,project_root=root)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

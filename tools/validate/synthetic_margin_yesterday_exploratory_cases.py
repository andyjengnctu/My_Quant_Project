"""MR-13BY isolated current-vintage feature/date/identity regression tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.breakout_quality import margin_yesterday_exploratory as source
from services.breakout_quality import margin_yesterday_research as runner

CAL = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06"])
DATE = "2024-02-01"


def frame(*rows):
    return pd.DataFrame(rows, columns=["date", *source.SOURCE_COLUMNS[2:]])


class FakeBank:
    feature_columns = tuple(f"AO_{i}" for i in range(10))

    def __init__(self):
        self._ticker_ids = np.array([0, 0, 1], dtype=np.int32)
        self._source_positions = np.array([3, 4, 5], dtype=np.int32)
        self._frame_dates = (CAL.to_numpy(dtype="datetime64[D]"), CAL.to_numpy(dtype="datetime64[D]"))
        self.shape = (3, 3, 10)
        self.example = np.arange(3 * 3 * 10, dtype=np.float32).reshape(3, 3, 10) / 1000

    def __getitem__(self, idx):
        return self.example[idx]

    def __len__(self):
        return len(self.example)


class ExploratoryInputCases(unittest.TestCase):
    def test_friday_uses_thursday_report_even_after_weekend(self):
        values, mask = source.align_confirmed_balances(
            pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-05"]), CAL,
            frame(["2024-02-01", 7, 0], ["2024-02-02", 11, 3], ["2024-02-05", 9999, 9999]),
        )
        np.testing.assert_array_equal(mask, [[False, False], [True, True], [True, True]])
        np.testing.assert_array_equal(values, [[0, 0], [7, 0], [11, 3]])
        self.assertEqual(values[1, 0], 7)
        self.assertEqual(values[2, 0], 11)

    def test_source_same_day_not_used_ever(self):
        controls = pd.to_datetime(["2024-02-02"])
        yesterday = frame([DATE, 47, 0])
        earlier, earlier_mask = source.align_confirmed_balances(controls, CAL, yesterday)
        tomorrow, tomorrow_mask = source.align_confirmed_balances(
            controls, CAL, frame([DATE, 47, 0], ["2024-02-02", 999999, 999999]))
        np.testing.assert_array_equal(earlier, tomorrow)
        np.testing.assert_array_equal(earlier_mask, tomorrow_mask)

    def test_no_forward_fill_and_zero_balance_is_real(self):
        values, masks = source.align_confirmed_balances(pd.to_datetime(["2024-02-02", "2024-02-05"]), CAL,
                                                        frame([DATE, 0, None]))
        np.testing.assert_array_equal(values, [[0, 0], [0, 0]])
        np.testing.assert_array_equal(masks, [[True, False], [False, False]])
        encoded = source.encode_margin_window(values, masks)
        self.assertEqual(encoded[0, 2], 1)
        self.assertEqual(encoded[0, 3], 0)
        self.assertEqual(encoded[1, 2], 0)

    def test_duplicate_source_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, "重複"):
            source.align_confirmed_balances(pd.to_datetime(["2024-02-02"]), CAL,
                                            frame([DATE, 1, 1], [DATE, 2, 2]))

    def test_bad_values_rejected(self):
        for bad in (-1, -0.2, 1.4, "invalid", float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                source.align_confirmed_balances(pd.to_datetime(["2024-02-02"]), CAL,
                                                frame([DATE, bad, 2]))

    def test_unverified_day_and_unsorted_dates_rejected(self):
        with self.assertRaises(ValueError):
            source.align_confirmed_balances(pd.to_datetime(["2024-02-03"]), CAL, frame())
        with self.assertRaises(ValueError):
            source.align_confirmed_balances(pd.to_datetime(["2024-02-02", "2024-02-01"]), CAL, frame())
        with self.assertRaises(ValueError):
            source.align_confirmed_balances(pd.to_datetime(["2024-02-02"]), CAL,
                                            frame(["2024-02-03", 1, 2]))
        with self.assertRaisesRegex(ValueError, "日曆尾端"):
            source.align_confirmed_balances(pd.to_datetime(["2024-02-07"]), CAL, frame())

    def test_calendar_start_precedes_coverage_as_missing_not_future(self):
        controls = pd.to_datetime(["1995-01-05", "2024-01-30", "2024-01-31"])
        vals, masks = source.align_confirmed_balances(controls, CAL, frame())
        self.assertEqual(vals.shape, (3, 2))
        self.assertFalse(masks.any())

    def test_window_normalization_causal_no_leakage(self):
        values = np.array([[0, 100], [1, 0], [0, 0]], dtype=np.float64)
        mask = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
        initial = source.encode_margin_window(values, mask)
        extended = np.concatenate([values, np.array([[90000., 80000.]])])
        extended_mask = np.concatenate([mask, np.array([[1, 1]], dtype=bool)])
        # Forward scores for past dates cannot be recomputed using later windows;
        # the bank takes only the past slice when materializing each sample.
        early = source.encode_margin_window(extended[:2], extended_mask[:2])
        late = source.encode_margin_window(extended[:3], extended_mask[:3])
        self.assertEqual(early.shape, (2, 4))
        self.assertEqual(late.shape, (3, 4))
        np.testing.assert_array_equal(source.encode_margin_window(values, mask), initial)
        self.assertEqual(initial[2, 2], 0.)
        self.assertEqual(initial[0, 2], 1.)
        self.assertEqual(initial[1, 3], 0.)

    def test_bank_batch_matches_simple_reference(self):
        bank = FakeBank()
        arrays = (
            (np.array([[10, 0], [20, 1], [30, 2], [40, 3], [50, 4], [60, 5.]], dtype=float),
             np.ones((6, 2), dtype=bool)),
            (np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6.]], dtype=float),
             np.ones((6, 2), dtype=bool)),
        )
        for index in (3, 5):
            arrays[0][1][index, 1] = False
            arrays[0][0][index, 1] = 0
        features = source.MarginAugmentedDailyFeatureBank(bank, arrays)
        self.assertEqual(features.shape, (3, 3, 14))
        self.assertEqual(features.feature_columns[-4:], source.FEATURE_COLUMNS)
        for j in range(3):
            combined = features[j]
            np.testing.assert_array_equal(combined[:, :10], bank[j])
            ticker, pos = int(bank._ticker_ids[j]), int(bank._source_positions[j])
            expected = source.encode_margin_window(arrays[ticker][0][pos-2:pos+1],
                                                   arrays[ticker][1][pos-2:pos+1])
            np.testing.assert_allclose(combined[:, 10:], expected, atol=1e-6)
        np.testing.assert_array_equal(features[[2, 0]], np.stack([features[2], features[0]]))
        np.testing.assert_array_equal(features[:2], np.stack([features[0], features[1]]))
        np.testing.assert_array_equal(features[np.array([True, False, True])], np.stack([features[0], features[2]]))
        self.assertEqual(features[[]].shape, (0, 3, 14))
        with self.assertRaises(IndexError):
            _ = features[[5]]

    def test_bank_feature_window_does_not_read_later_rows(self):
        bank = FakeBank()
        values = np.arange(6*2, dtype=np.float64).reshape(6,2)
        mask = np.ones_like(values, dtype=bool)
        features = source.MarginAugmentedDailyFeatureBank(bank, ((values.copy(), mask.copy()), (values.copy(), mask.copy())))
        early_before = features[0].copy()
        values[4:] = 1e12
        revised = source.MarginAugmentedDailyFeatureBank(bank, ((values.copy(), mask.copy()), (values.copy(), mask.copy())))
        np.testing.assert_allclose(early_before, revised[0])

    def test_ready_provider_snapshot_source_is_fingerprint_and_pit_bound(self):
        def artifact(name):
            return SimpleNamespace(request_id=name+":a", content_sha256="a"*64)

        class FakeView:
            def __init__(self, *args, **kwargs):
                pass
            def dataset_artifacts(self, dataset):
                return [artifact(dataset)]
            def iter_verified_frames(self, dataset, columns=None):
                yield pd.DataFrame({"date": CAL.strftime("%Y-%m-%d")})
            def iter_verified_frames_many_data_ids(self, dataset, columns=None, data_ids=()):
                self_ids = tuple(data_ids)
                assert dataset == source.DATASET
                assert set(self_ids) == {"0050", "2330"}
                yield pd.DataFrame([{"date": DATE, "stock_id": "0050",
                                     "MarginPurchaseYesterdayBalance": 20,
                                     "ShortSaleYesterdayBalance": 0},
                                    {"date": DATE, "stock_id": "2330",
                                     "MarginPurchaseYesterdayBalance": 5,
                                     "ShortSaleYesterdayBalance": 1}])

        bank = FakeBank()
        bundle = SimpleNamespace(feature_bank=bank,
                                 group_table=pd.DataFrame({"ticker": ["0050", "0050", "2330"]}))
        archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a" * 64,
                                  manifest_fingerprint="b" * 64)
        contract = SimpleNamespace(dataset=source.DATASET, review_status="REVIEW_REQUIRED")
        with (patch.object(source, 'ProviderSnapshotView', FakeView),
              patch.object(source, 'load_ready_provider_snapshot_archive', return_value=archive),
              patch.object(source, 'build_research_v2_pit_review_contracts', return_value=(contract,))):
            obj = source.load_margin_source_for_ao_bundle(ROOT, bundle)
        self.assertEqual(obj.source_identity['research_pit'], 'REVIEW_REQUIRED')
        self.assertFalse(obj.source_identity['model_input_authorized_in_formal_workflow'])
        self.assertEqual(obj.source_identity['snapshot_fingerprint'], 'a' * 64)
        np.testing.assert_array_equal(obj.arrays[0][1][3], [True, True])
        np.testing.assert_array_equal(obj.arrays[1][1][3], [True, True])
        self.assertEqual(obj.coverage['by_prediction_price_year']['2024']['stock_days'], 12)

    def test_source_mapping_preserves_unscored_categorical_price_frames(self):
        # AO may retain source-price frames without a single score-eligible group.
        # The frame identity must come from category codes, not observed IDs.
        class SparseBank:
            _ticker_ids = np.array([0, 2], dtype=np.int32)
            _source_positions = np.array([3, 3], dtype=np.int32)
            _frame_dates = tuple(CAL.to_numpy(dtype="datetime64[D]") for _ in range(3))

        class CategoricalView:
            def __init__(self, *args, **kwargs):
                pass
            def dataset_artifacts(self, dataset):
                return [SimpleNamespace(request_id=dataset+":f", content_sha256="a"*64)]
            def iter_verified_frames(self, dataset, columns=None):
                yield pd.DataFrame({"date": CAL.strftime("%Y-%m-%d")})
            def iter_verified_frames_many_data_ids(self, dataset, columns=None, data_ids=()):
                assert tuple(data_ids) == ("0050", "1200", "2330")
                yield pd.DataFrame([{"date": DATE, "stock_id": "0050",
                                     "MarginPurchaseYesterdayBalance": 33,
                                     "ShortSaleYesterdayBalance": 0},
                                    {"date": DATE, "stock_id": "2330",
                                     "MarginPurchaseYesterdayBalance": 9,
                                     "ShortSaleYesterdayBalance": 2}])

        table = pd.DataFrame({"ticker": pd.Categorical.from_codes(
            [0, 2], categories=["0050", "1200", "2330"]
        )})
        original = SimpleNamespace(feature_bank=SparseBank(), group_table=table)
        archive = SimpleNamespace(as_of_date="2026-03-02", snapshot_fingerprint="a"*64,
                                  manifest_fingerprint="b"*64)
        contract = SimpleNamespace(dataset=source.DATASET, review_status="REVIEW_REQUIRED")
        with (patch.object(source, "ProviderSnapshotView", CategoricalView),
              patch.object(source, "load_ready_provider_snapshot_archive", return_value=archive),
              patch.object(source, "build_research_v2_pit_review_contracts", return_value=(contract,))):
            result = source.load_margin_source_for_ao_bundle(ROOT, original)
        self.assertEqual(len(result.arrays), 3)
        self.assertFalse(result.arrays[1][1].any())
        np.testing.assert_array_equal(result.arrays[0][0][3], [33, 0])
        np.testing.assert_array_equal(result.arrays[2][0][3], [9, 2])

    def test_canonical_ao_backbone_accepts_14_features_dual_heads(self):
        import torch
        from filters.breakout_quality.models.factory import build_model
        model = build_model(14, 2, architecture=runner.AO_ARCHITECTURE)
        model.eval()
        with torch.no_grad():
            logits = model.forward_output_head(torch.ones(2, 300, 14),
                                               torch.ones(2, 2), 'both')
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.isfinite(logits).all())

    def test_menu_forward_producer_reuses_primary_score_and_isolates_outputs(self):
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        import json
        import torch
        bank = FakeBank()
        arrays = tuple((np.full((6, 2), 12., dtype=float),
                        np.ones((6, 2), dtype=bool)) for _ in range(2))
        evidence = source.MarginSourceEvidence(
            arrays=arrays,
            source_identity={"source_contract": source.FEATURE_CONTRACT,
                             "snapshot_fingerprint": "a"*64,
                             "historical_vintage_proven": False},
            coverage={"by_prediction_price_year": {"2024": {"stock_days": 12}}},
        )
        from filters.breakout_quality.continuous_ranker_data import ContinuousRankerDataBundle
        original = ContinuousRankerDataBundle(
            events=pd.DataFrame(), labels=np.zeros(3, dtype=np.int32),
            event_group_index=np.arange(3), target_valid=np.ones(3, dtype=bool),
            outer_policy={}, feature_bank=bank,
            summary={"dataset": "control", "training_universe_start_date": "2001-01-01"},
            target_manifest={},
            profile=SimpleNamespace(training_objective="daily_shared_safety_hs_conditional_mfe_pairwise_ranking",
                                    as_manifest_payload=lambda: {"profile": "ao"}),
            group_table=pd.DataFrame({"ticker": ["0050", "0050", "2330"],
                                      "date": ["2024-02-02", "2024-02-05", "2024-02-06"]}),
            raw_target=np.array([1., 2., 3.], dtype=np.float32),
            group_context=np.ones((3, 2), dtype=np.float32),
            model_spec=SimpleNamespace(as_manifest_payload=lambda: {"architecture": runner.AO_ARCHITECTURE}),
        )
        split = SimpleNamespace(inner_train_ids=np.array([0]), validation_ids=np.array([1]),
                                selection_ids=np.array([0, 1]), oos_ids=np.array([2]),
                                report={"selection_start_date": "2001-01-01", "counts": {"selection":2}})
        quality = {"raw_safety": {"mean_daily_spearman": 0.11},
                   "conditional_mfe_true_hs": {"mean_daily_spearman": 0.22},
                   "lexicographic_model_gate": {"predicted_hs_hmhs_pct": 20.,
                                                "selected_hmhs_pct": 22.,
                                                "predicted_hs_true_ls_pct": 31.}}
        with TemporaryDirectory() as tmp:
            with (patch.object(runner, "ROOT", Path(tmp)),
                  patch.object(runner, "load_daily_universal_ranker_data_with_price_source", return_value=original),
                  patch.object(runner, "load_margin_source_for_ao_bundle", return_value=evidence),
                  patch.object(runner, "build_daily_ranker_split", return_value=split),
                  patch.object(runner.ranker_api, "training_semantics", return_value={"ao":True}),
                  patch.object(runner.ranker_api, "build_daily_percentile_targets", return_value=np.zeros(3, dtype=np.float32)),
                  patch.object(runner, "resolve_ranker_execution_plan", return_value=(torch, object())),
                  patch.object(runner, "scientific_torch_execution", return_value={"device": "cpu", "dtype": "float32"}),
                  patch.object(runner.ranker_api, "select_epoch", return_value={"best_epoch":1}),
                  patch.object(runner.ranker_api, "fit_final", return_value=(torch.nn.Linear(1, 1), [])),
                  patch.object(runner, "predict_score_output_payload",
                               return_value=(np.array([.94], dtype=np.float32),
                                             {"raw_safety_score": np.array([.17], dtype=np.float32)})),
                  patch.object(runner.ranker_api, "hs_conditional_mfe_metrics", return_value=quality),
                  patch("filters.breakout_quality.hs_conditional_mfe.build_hs_conditional_mfe_targets", return_value=object())):
                report_file, human_file = runner.run()
            self.assertTrue(human_file.is_file())
            report = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertFalse(report["formal_model_input_authorized"])
            self.assertFalse(report["historical_contemporaneous_vintage_proven"])
            self.assertEqual(report["scientific_id"], "MR-13BY")
            self.assertEqual(report["oos_groups"], 1)
            self.assertEqual(tuple(report["fitting_identity"]["feature_columns"][-4:]), source.FEATURE_COLUMNS)
            scores = pd.read_csv(report_file.parent / "experimental_forward_oos_scores.csv.gz")
            self.assertAlmostEqual(scores.loc[0, "conditional_mfe_score"], .94, places=5)
            self.assertAlmostEqual(scores.loc[0, "raw_safety_score"], .17, places=5)
            self.assertTrue(str(report_file).startswith(str(Path(tmp) / runner.EXPERIMENT_RELATIVE_OUTPUT)))
            with self.assertRaises(FileExistsError):
                # Experiment outputs are single identity and must not overwrite.
                with (patch.object(runner, "ROOT", Path(tmp)),
                      patch.object(runner, "load_daily_universal_ranker_data_with_price_source", return_value=original),
                      patch.object(runner, "load_margin_source_for_ao_bundle", return_value=evidence),
                      patch.object(runner, "build_daily_ranker_split", return_value=split),
                      patch.object(runner.ranker_api, "training_semantics", return_value={"ao":True}),
                      patch.object(runner, "scientific_torch_execution", return_value={"device": "cpu", "dtype": "float32"}),
                      patch.object(runner, "resolve_ranker_execution_plan", return_value=(torch, object()))):
                    runner.run()

    def test_official_ao_selection_unmodified_by_exploration(self):
        from config import breakout_quality as cfg
        from core.breakout_quality_registry import get_continuous_ranker_research_spec
        self.assertEqual(get_continuous_ranker_research_spec(cfg.BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE).model_research_id,
                         'MR-13AO')
        self.assertIn('experiments/mr13by_', runner.EXPERIMENT_RELATIVE_OUTPUT.as_posix())
        self.assertIn('experiments/mr13by_', runner.EXPERIMENT_RELATIVE_MODEL.as_posix())
        self.assertFalse(runner._STUDY.formal_input_authorized)
        self.assertFalse(runner._STUDY.trading_promotion_authorized)


class ExploratoryMenuIntegrationCases(unittest.TestCase):
    def _settings(self):
        return SimpleNamespace(
            is_binary_classification=False, is_continuous_ranker=True,
            rolling_authorized=True,
            experiment_profile=runner._STUDY.reference_profile,
        )

    def test_menu_forward_isolated_from_formal_ao(self):
        from services.research import breakout_quality_application as app
        with (patch.object(app, "get_breakout_quality_model_research_settings", return_value=self._settings()),
              patch.object(app, "_configured_exploratory_research_study", return_value=runner._STUDY),
              patch.object(app, "_interactive_continuous_forward_model_gate") as formal,
              patch.object(app, "_invoke_exploratory_research_action", return_value=0) as explorer,
              patch("builtins.input", side_effect=["1"])):
            result = app._interactive_model_research("apps/research.py model")
        self.assertEqual(result, 0)
        explorer.assert_called_once_with(runner._STUDY, "forward")
        formal.assert_not_called()

    def test_menu_only_advertises_registered_callable_actions(self):
        import contextlib
        import io
        from services.research import breakout_quality_application as app
        output = io.StringIO()
        with (patch.object(app, "get_breakout_quality_model_research_settings", return_value=self._settings()),
              patch.object(app, "_configured_exploratory_research_study", return_value=runner._STUDY),
              patch.object(app, "_run_configured_model_comparison") as formal,
              patch.object(app, "_invoke_exploratory_research_action", return_value=0) as explorer,
              patch("builtins.input", side_effect=["2", "4", "5", "6", "7", "3", "0"]),
              contextlib.redirect_stdout(output)):
            result = app._interactive_model_research("apps/research.py model")
        self.assertEqual(result, 0)
        self.assertIn("Forward OOS 模型比較", output.getvalue())
        self.assertNotIn("Rolling OOS 模型訓練", output.getvalue())
        self.assertNotIn("Robustness 模型測試", output.getvalue())
        self.assertNotIn("[BLOCKED]", output.getvalue())
        explorer.assert_called_once_with(runner._STUDY, "compare")
        formal.assert_not_called()

    def test_new_registered_action_is_visible_and_dispatchable_without_menu_edit(self):
        from dataclasses import replace
        from services.research import breakout_quality_application as app
        study = replace(runner._STUDY, workflow_handlers=(
            ("forward", "run_forward_menu"),
            ("rolling", "run_synthetic_rolling"),
            ("compare", "run_forward_comparison"),
        ))
        with (patch.object(runner, "run_synthetic_rolling", return_value=17, create=True) as producer,
              patch.object(app, "get_breakout_quality_model_research_settings", return_value=self._settings()),
              patch.object(app, "_configured_exploratory_research_study", return_value=study),
              patch("builtins.input", side_effect=["2", "0"])):
            menu_items = app._exploratory_model_menu_work_items(study)
            self.assertEqual(tuple(int(item[0]) for item in menu_items), (1, 2, 3))
            self.assertEqual(app._interactive_model_research("apps/research.py model"), 0)
        producer.assert_called_once_with()

    def test_missing_registered_producer_fails_before_menu_is_shown(self):
        from dataclasses import replace
        from services.research import breakout_quality_application as app
        invalid = replace(runner._STUDY, workflow_handlers=(("forward", "missing_producer"),))
        with self.assertRaisesRegex(RuntimeError, "缺少可執行 producer"):
            app._exploratory_model_menu_work_items(invalid)
        unregistered = replace(runner._STUDY, workflow_handlers=(("not_a_work_type", "run_forward_menu"),))
        with self.assertRaisesRegex(ValueError, "未對應正式選單類型"):
            app._exploratory_model_menu_work_items(unregistered)

    def test_compare_missing_forward_delegates_canonical_producer(self):
        from tempfile import TemporaryDirectory
        import json
        from core.file_integrity import compute_file_sha256
        from services.breakout_quality import margin_yesterday_research as mod
        with TemporaryDirectory() as temp:
            root = Path(temp)
            score = root / "ao_scores.csv.gz"
            score.write_bytes(b"valid-reference")
            output = root / "comparison"
            output.mkdir()
            report = output / "experimental_report.json"
            comparison = {
                "fitting_fingerprint": "s" * 64,
                "same_oos_group_count": 3,
                "canonical_reference_score_sha256": compute_file_sha256(score),
                "rows": [],
            }
            with (patch.object(mod, "ROOT", root),
                  patch.object(mod, "_read_experimental_report", side_effect=[(None, None), (report, {"ok": True})]) as reader,
                  patch.object(mod, "run_forward_menu", return_value=0) as producer,
                  patch.object(mod, "_validate_current_exploratory_source"),
                  patch.object(mod, "_comparison_payload", return_value=comparison),
                  patch("filters.breakout_quality.ranking_score_store.resolve_continuous_ranker_oos_score_path", return_value=score)):
                self.assertEqual(mod.run_forward_comparison(), 0)
            self.assertEqual(reader.call_count, 2)
            producer.assert_called_once_with()
            self.assertTrue((output / "experimental_comparison.md").is_file())
            self.assertEqual(json.loads((output / "experimental_comparison.json").read_text(encoding="utf-8")), comparison)

    def test_compare_cancellation_stops_without_claiming_result(self):
        from services.breakout_quality import margin_yesterday_research as mod
        with (patch.object(mod, "_read_experimental_report", return_value=(None, None)),
              patch.object(mod, "run_forward_menu", return_value=0) as producer,
              patch.object(mod, "_comparison_payload") as consumer):
            self.assertEqual(mod.run_forward_comparison(), 0)
        producer.assert_called_once_with()
        consumer.assert_not_called()

    def test_menu_switch_restores_unmodified_formal_ao(self):
        from services.research import breakout_quality_application as app
        with (patch.object(app, "get_breakout_quality_model_research_settings", return_value=self._settings()),
              patch.object(app, "_configured_exploratory_research_study", return_value=runner._STUDY),
              patch.object(app, "_interactive_continuous_forward_model_gate", return_value=0) as formal,
              patch.object(app, "_invoke_exploratory_research_action") as explorer,
              patch("builtins.input", side_effect=["8", "1"])):
            result = app._interactive_model_research("apps/research.py model")
        self.assertEqual(result, 0)
        formal.assert_called_once()
        explorer.assert_not_called()

    def test_exploratory_reference_is_exact_group_ticker_date_join(self):
        from tempfile import TemporaryDirectory
        import numpy as np
        from services.breakout_quality import margin_yesterday_research as mod
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'canonical_ao_scores.csv.gz'
            pd.DataFrame({
                "group_index": [0, 1, 2], "ticker": ["0050", "2330", "2409"],
                "date": ["2024-02-02", "2024-02-02", "2024-02-05"],
                "raw_safety_score": [.1, .4, .8], "conditional_mfe_score": [.2, .5, .9],
            }).to_csv(path, index=False, compression='gzip')
            bundle = SimpleNamespace(group_table=pd.DataFrame({
                "ticker": ["0050", "2330", "2409"],
                "date": ["2024-02-02", "2024-02-02", "2024-02-05"],
            }))
            quality = {"raw_safety": {"mean_daily_spearman": .6},
                       "conditional_mfe_true_hs": {"mean_daily_spearman": .7},
                       "lexicographic_model_gate": {"predicted_hs_true_ls_pct": 30.}}
            with (patch.object(mod, 'load_continuous_ranker_oos_contract',
                               return_value=SimpleNamespace(score_path=path)),
                  patch.object(mod.ranker_api, 'hs_conditional_mfe_metrics', return_value=quality) as metrics):
                found = mod._ao_reference_quality(bundle, np.array([0,1]), object(), filter_id='breakout_quality_v1')
                self.assertEqual(found['reference_groups'], 2)
                self.assertEqual(found['metrics']['raw_safety']['mean_daily_spearman'], .6)
                self.assertEqual(metrics.call_count, 1)
                bundle.group_table.loc[0, "ticker"] = "006208"
                with self.assertRaisesRegex(ValueError, 'ticker錯配'):
                    mod._ao_reference_quality(bundle, np.array([0,1]), object(), filter_id='breakout_quality_v1')

    def test_exploratory_same_sample_comparison_reuse_and_ao_sha_guard(self):
        from tempfile import TemporaryDirectory
        import json
        from services.breakout_quality import margin_yesterday_research as mod
        from core.file_integrity import compute_file_sha256
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fingerprint = 'a' * 64
            output = root / mod.EXPERIMENT_RELATIVE_OUTPUT / fingerprint
            model = root / mod.EXPERIMENT_RELATIVE_MODEL / fingerprint
            output.mkdir(parents=True)
            model.mkdir(parents=True)
            (model / 'model.pt').write_bytes(b'frozen')
            (output / 'experimental_forward_oos_scores.csv.gz').write_bytes(b'controlled')
            ao = root / 'ao_oos.csv.gz'
            ao.write_bytes(b'ao_original')
            metrics = {"raw_safety": {"mean_daily_spearman": .4},
                       "conditional_mfe_true_hs": {"mean_daily_spearman": .5},
                       "lexicographic_model_gate": {"predicted_hs_true_ls_pct": 30.,
                                                    "selected_hmhs_pct": 22.,
                                                    "selected_high_safety_pct": 33.}}
            result = {"scientific_id": mod.SCIENTIFIC_ID,
                      "fitting_fingerprint": fingerprint,
                      "formal_model_input_authorized": False,
                      "historical_contemporaneous_vintage_proven": False,
                      "oos_groups": 5,
                      "metrics": metrics,
                      "reference_ao_same_oos": {
                          "reference_profile": mod._STUDY.reference_profile,
                          "reference_groups": 5,
                          "canonical_score_sha256": compute_file_sha256(ao),
                          "metrics": {"raw_safety": {"mean_daily_spearman": .3},
                                      "conditional_mfe_true_hs": {"mean_daily_spearman": .6},
                                      "lexicographic_model_gate": {"predicted_hs_true_ls_pct": 32.,
                                                                   "selected_hmhs_pct": 24.,
                                                                   "selected_high_safety_pct": 34.}},
                      }}
            (output / 'experimental_report.json').write_text(json.dumps(result), encoding='utf-8')
            with (patch.object(mod, 'ROOT', root),
                  patch.object(mod, '_validate_current_exploratory_source'),
                  patch('filters.breakout_quality.ranking_score_store.resolve_continuous_ranker_oos_score_path', return_value=ao)):
                self.assertEqual(mod.run_forward_comparison(), 0)
                comparison = json.loads((output / 'experimental_comparison.json').read_text(encoding='utf-8'))
                self.assertFalse(comparison['formal_pit_input_authorized'])
                self.assertEqual(comparison['same_oos_group_count'], 5)
                self.assertEqual(mod.run_forward_comparison(), 0)
                ao.write_bytes(b'tampered')
                with self.assertRaisesRegex(RuntimeError, '已變動'):
                    mod.run_forward_comparison()



if __name__ == '__main__':
    unittest.main()

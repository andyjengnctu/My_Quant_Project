"""Dedicated scientific invariants for alternate OHLC inputs with canonical labels.

Run with: python tools/validate/synthetic_breakout_quality_raw_input_cases.py
This is a focused synthetic suite, not the project's formal integration suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,
    BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
)
from core.breakout_quality_registry import get_breakout_quality_experiment_profile, get_continuous_ranker_research_spec
from core.file_integrity import canonical_json_sha256, compute_file_sha256
from filters.breakout_quality.daily_ranker_data import (
    LazyDailyFeatureBank, load_daily_universal_ranker_data, load_aligned_alternate_ohlcv_frame,
)
from filters.breakout_quality.model_source_provenance import (
    recorded_input_price_source_issues, _published_source_provenance_issues,
)
from filters.breakout_quality.models.architectures import get_architecture_descriptor
from filters.breakout_quality.models.spec import get_model_spec
from services.breakout_quality.fitted_model_artifacts import source_training_identity
from services.breakout_quality.price_input_source import (
    load_daily_universal_ranker_data_with_price_source,
    resolve_feature_price_source,
)
from services.research.raw_price_strategy_dataset import (
    _build_files, _raw_price_quality_failures, _raw_quote_diagnostics,
    _fail_on_raw_price_quality, inspect_raw_price_strategy_dataset,
)


class AlternatePriceInputTests(unittest.TestCase):
    def setUp(self):
        self.raw_profile = BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[1]
        self.ao_profile = "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
        self.raw_arch = get_breakout_quality_experiment_profile(self.raw_profile).model_architecture
        self.ao_arch = get_breakout_quality_experiment_profile(self.ao_profile).model_architecture

    def test_current_pair_and_reference(self):
        self.assertEqual(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[0], 'MR-13BW')
        self.assertEqual(get_continuous_ranker_research_spec(self.raw_profile).model_research_id,'MR-13BW')
        self.assertIn(('MR-13AO',self.ao_profile),BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        self.assertIn(('MR-13BW',self.raw_profile),BREAKOUT_QUALITY_MODEL_TEST_PROFILES)

    def test_architecture_preserves_ao_topology(self):
        self.assertTrue(get_architecture_descriptor(self.raw_arch).has_capability('raw_unadjusted_price_input'))
        self.assertFalse(get_architecture_descriptor(self.ao_arch).has_capability('raw_unadjusted_price_input'))
        ao = get_model_spec(self.ao_arch).as_manifest_payload()
        raw = get_model_spec(self.raw_arch).as_manifest_payload()
        self.assertEqual({k:v for k,v in ao.items() if k!='architecture'},
                         {k:v for k,v in raw.items() if k!='architecture'})

    def test_profile_keeps_label_loss_and_selection(self):
        ao=get_breakout_quality_experiment_profile(self.ao_profile).as_manifest_payload()
        raw=get_breakout_quality_experiment_profile(self.raw_profile).as_manifest_payload()
        self.assertEqual({k:v for k,v in ao.items() if k not in ('name','model_architecture')},
                         {k:v for k,v in raw.items() if k not in ('name','model_architecture')})
        self.assertEqual(get_continuous_ranker_research_spec(self.raw_profile).model_gate_reference_profile_name,
                         self.ao_profile)

    @staticmethod
    def _frame(close_slope):
        dates=pd.date_range('2021-01-01',periods=308,freq='B')
        close=np.asarray([100+i*close_slope for i in range(len(dates))],dtype=float)
        return pd.DataFrame(dict(Open=close*0.99,High=close*1.01,Low=close*0.98,
                                 Close=close,Volume=np.full(len(close),10000.0)),index=dates)

    def _bank(self, alternate:bool):
        adjusted=self._frame(0.12)
        raw=self._frame(0.31)
        benchmark_adjusted=self._frame(0.04)
        benchmark_raw=self._frame(0.21)
        bank=LazyDailyFeatureBank(frames=(adjusted,),benchmark=benchmark_adjusted,
          feature_frames=(raw,) if alternate else None,
          feature_benchmark=benchmark_raw if alternate else None,
          ticker_ids=np.asarray([0,0]),source_positions=np.asarray([299,300]),
          benchmark_positions=np.asarray([299,300]),
          policy=SimpleNamespace(feature_window_bars=300))
        return bank

    def test_raw_changes_only_features(self):
        ao=self._bank(False)
        raw=self._bank(True)
        self.assertEqual(ao.shape,raw.shape)
        self.assertEqual(ao.shape,(2,300,10))
        self.assertFalse(np.allclose(ao[:][:,:,:5],raw[:][:,:,:5]))
        self.assertFalse(np.allclose(ao[:][:,:,5:],raw[:][:,:,5:]))
        self.assertTrue(np.array_equal(ao._frame_arrays[0],raw._frame_arrays[0]))
        self.assertTrue(np.array_equal(ao._benchmark_array,raw._benchmark_array))

    def test_future_path_uses_adjusted_even_with_raw_feature(self):
        ao=self._bank(False)
        raw=self._bank(True)
        kwargs=dict(cutoff_date=pd.Timestamp('2022-12-30'),horizon_bars=3,
                    target_names=('max_upside_3d','max_downside_3d'))
        first=ao.future_path_statistics(np.asarray([0,1]),**kwargs)
        second=raw.future_path_statistics(np.asarray([0,1]),**kwargs)
        self.assertTrue(np.array_equal(first[0],second[0]))
        self.assertEqual(first[1],second[1])

    def test_date_alignment_fails_closed(self):
        adjusted=self._frame(0.1)
        shifted=self._frame(0.3).iloc[1:]
        with self.assertRaisesRegex(ValueError,'date grid'):
            LazyDailyFeatureBank(frames=(adjusted,),benchmark=adjusted,
                feature_frames=(shifted,),ticker_ids=np.asarray([0]),
                source_positions=np.asarray([299]),benchmark_positions=np.asarray([299]),
                policy=SimpleNamespace(feature_window_bars=300))

    def test_raw_csv_loader_requires_equal_dates_and_volume(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            baseline=self._frame(0.12)
            raw=self._frame(0.22)
            raw.reset_index(names='Date').to_csv(root/'2330.csv',index=False)
            aligned=load_aligned_alternate_ohlcv_frame(root,'2330',baseline)
            self.assertTrue(aligned.index.equals(baseline.index))
            self.assertFalse(np.array_equal(aligned['Close'].to_numpy(),baseline['Close'].to_numpy()))
            raw.loc[raw.index[0],'Volume']=555.0
            raw.reset_index(names='Date').to_csv(root/'2330.csv',index=False)
            with self.assertRaisesRegex(ValueError,'volume mismatch'):
                load_aligned_alternate_ohlcv_frame(root,'2330',baseline)
            raw=raw.iloc[1:]
            raw.reset_index(names='Date').to_csv(root/'2330.csv',index=False)
            with self.assertRaisesRegex(ValueError,'trading-date membership mismatch'):
                load_aligned_alternate_ohlcv_frame(root,'2330',baseline)

    def test_raw_extra_dates_are_projected_to_adjusted_sample_grid(self):
        # The raw materializer retains the original full control date grid.
        # Adjusted OHLCV sanitation can remove dates that must not become new
        # training samples or additional 300-bar history in this experiment.
        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            raw = self._frame(0.31)
            unclean_adjusted = self._frame(0.12)
            unclean_adjusted.iloc[18, unclean_adjusted.columns.get_loc("Open")] = 0.0
            unclean_adjusted.iloc[45, unclean_adjusted.columns.get_loc("High")] = 0.0
            from core.data_utils import sanitize_ohlcv_dataframe
            control, stats = sanitize_ohlcv_dataframe(
                unclean_adjusted.reset_index(names="Date"), "0052", min_rows=1,
            )
            self.assertEqual(stats["invalid_row_count"], 2)
            raw.reset_index(names="Date").to_csv(raw_dir / "0052.csv", index=False)
            aligned = load_aligned_alternate_ohlcv_frame(raw_dir, "0052", control)
            self.assertTrue(aligned.index.equals(control.index))
            self.assertEqual(len(aligned), len(control))
            self.assertTrue(np.array_equal(aligned["Volume"], control["Volume"]))
            self.assertTrue(np.allclose(aligned["Close"], raw.loc[control.index, "Close"], rtol=0, atol=1e-12))
            bank = LazyDailyFeatureBank(
                frames=(control,), benchmark=control,
                feature_frames=(aligned,), feature_benchmark=aligned,
                ticker_ids=np.asarray([0]), source_positions=np.asarray([300]),
                benchmark_positions=np.asarray([300]),
                policy=SimpleNamespace(feature_window_bars=300),
            )
            self.assertEqual(bank.shape, (1, 300, 10))
            self.assertTrue(np.isfinite(bank[:]).all())
            self.assertTrue(np.array_equal(bank._frame_arrays[0][:, 3], control["Close"]))

    def test_raw_missing_adjusted_date_fails_with_diagnostics(self):
        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            control = self._frame(0.12)
            raw = self._frame(0.31)
            missing_date = raw.index[19].date().isoformat()
            raw.iloc[19, raw.columns.get_loc("High")] = 0.0
            # Canonical sanitizer drops invalid raw bar; no price substitution or
            # forward filling is allowed for a date needed by adjusted targets.
            raw.reset_index(names="Date").to_csv(raw_dir / "0052.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing_raw=1") as raised:
                load_aligned_alternate_ohlcv_frame(raw_dir, "0052", control)
            self.assertIn(missing_date, str(raised.exception))

    def test_raw_sanitizer_rejection_reports_original_ohlc_without_filling(self):
        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            control = self._frame(0.12)
            raw = self._frame(0.31)
            first_date = raw.index[19].date().isoformat()
            second_date = raw.index[23].date().isoformat()
            raw.loc[raw.index[19], "Open"] = 0.0
            raw.loc[raw.index[23], "High"] = raw.loc[raw.index[23], "Close"] - 1.0
            raw.reset_index(names="Date").to_csv(raw_dir / "0052.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing_raw=2") as caught:
                load_aligned_alternate_ohlcv_frame(raw_dir, "0052", control)
            message = str(caught.exception)
            self.assertIn("raw_rows_removed_by_sanitizer=2", message)
            self.assertIn("raw_rows_absent=0", message)
            self.assertIn("present_but_rejected_by_canonical_sanitizer", message)
            self.assertIn("'Open': '0.0'", message)
            self.assertIn(first_date, message)
            self.assertIn(second_date, message)
            self.assertIn("data_source_blocked", message)

    def test_canonical_raw_source_quality_detects_all_six_zero_bars(self):
        control = self._frame(0.2).iloc[:30].reset_index(names="Date")
        zero_dates = list(control.loc[[2, 4, 8, 11, 15, 22], "Date"])
        control.loc[control["Date"].isin(zero_dates), ["Open", "High", "Low", "Close"]] = 0.0
        count, examples = _raw_price_quality_failures(control, ticker="0052")
        self.assertEqual(count, 6)
        self.assertEqual(len(examples), 6)
        self.assertTrue(all(item["ticker"] == "0052" for item in examples))
        self.assertTrue(all(item["ohlc"]["Open"] == "0.0" for item in examples))
        control.loc[control["Date"].isin(zero_dates), ["Open", "High", "Low", "Close"]] = 10.0
        count, examples = _raw_price_quality_failures(control, ticker="0052")
        self.assertEqual((count, examples), (0, []))

    def test_raw_no_published_quote_is_not_misclassified_as_corrupt_price(self):
        # FinMind can report an all-zero OHLC when no official price was published.
        # Controlled adjusted Volume can be zero OR positive on these dates.
        frame = self._frame(0.1).iloc[:6].reset_index(names="Date")
        frame.loc[frame.index[1], ["Open", "High", "Low", "Close"]] = 0.0
        frame.loc[frame.index[2], ["Open", "High", "Low", "Close"]] = 0.0
        frame.loc[frame.index[1], "Volume"] = 0.0
        frame.loc[frame.index[3], "High"] = 0.0  # other invalid, not all-zero
        removed, examples = _raw_price_quality_failures(frame, ticker="0052")
        counts = _raw_quote_diagnostics(frame)
        self.assertEqual(removed, 3)
        self.assertEqual(counts["all_zero_ohlc"], 2)
        self.assertEqual(counts["all_zero_control_volume_zero"], 1)
        self.assertEqual(counts["all_zero_control_volume_positive"], 1)
        self.assertEqual(examples[0]["quote_class"], "ALL_ZERO_NO_PUBLISHED_QUOTE_POSSIBLE")
        self.assertEqual(examples[2]["quote_class"], "OTHER_INVALID_OHLC")
        with self.assertRaisesRegex(ValueError, "all_zero_ohlc=2") as caught:
            _fail_on_raw_price_quality([{
                "ticker": "0052", "invalid_rows": removed,
                "samples": examples, "quote_diagnostics": counts,
            }], provider="a" * 64)
        message = str(caught.exception)
        self.assertIn("other_invalid=1", message)
        self.assertIn("no published market quote", message)
        self.assertIn("scientific authorization", message)

    def test_raw_materializer_rejects_invalid_provider_rows_before_ready(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctrl = root / "control"
            ctrl.mkdir()
            rows = self._frame(0.1).iloc[:8].reset_index(names="Date")
            rows.to_csv(ctrl / "0052.csv", index=False)
            provider = rows.rename(columns={"Date": "date", "Open": "open", "High": "max", "Low": "min", "Close": "close"})
            provider["date"] = provider["date"].dt.strftime("%Y-%m-%d")
            provider["stock_id"] = "0052"
            provider.loc[provider.index[2], ["open", "max", "min", "close"]] = 0.0
            view = SimpleNamespace(iter_raw_price_strategy_frames=lambda **_kw: iter([provider]))
            stage = root / "stage"
            with self.assertRaisesRegex(ValueError, "RAW_SOURCE_QUALITY_BLOCKED") as caught:
                _build_files(
                    root=root, view=view, control_dir=ctrl, stage_dir=stage,
                    source_identity={"frozen_cutoff": "2026-03-02", "provider_snapshot_fingerprint": "a" * 64},
                )
            self.assertIn("0052", str(caught.exception))
            self.assertIn("0.0", str(caught.exception))
            self.assertIn("provider_snapshot=", str(caught.exception))
            self.assertFalse((stage / "manifest.json").exists())

    def test_raw_materializer_keeps_healthy_prices_and_control_volume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control"
            control.mkdir()
            adjusted = self._frame(0.1).iloc[:8].reset_index(names="Date")
            adjusted.to_csv(control / "0052.csv", index=False)
            provider = adjusted.rename(columns={
                "Date": "date", "Open": "open", "High": "max", "Low": "min", "Close": "close",
            })
            provider["date"] = provider["date"].dt.strftime("%Y-%m-%d")
            provider["stock_id"] = "0052"
            provider["close"] += 0.05
            provider["max"] += 0.05
            view = SimpleNamespace(iter_raw_price_strategy_frames=lambda **_kw: iter([provider]))
            stage = root / "stage"
            files, total_rows = _build_files(
                root=root, view=view, control_dir=control, stage_dir=stage,
                source_identity={"frozen_cutoff": "2026-03-02", "provider_snapshot_fingerprint": "a" * 64},
            )
            self.assertEqual(total_rows, len(adjusted))
            self.assertEqual(len(files), 1)
            output = pd.read_csv(stage / "tw_stock_data_vip" / "0052.csv")
            self.assertTrue(np.array_equal(output["Volume"].to_numpy(), adjusted["Volume"].to_numpy()))
            self.assertTrue(np.allclose(output["Close"], provider["close"], atol=1e-10, rtol=0))
            self.assertEqual(files[0]["content_sha256"], compute_file_sha256(stage / "tw_stock_data_vip" / "0052.csv"))

    def test_existing_sha_valid_raw_materialization_rejects_zero_bars(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = {"provider_snapshot_fingerprint": "b" * 64}
            source_fp = "a" * 64
            source["source_identity_fingerprint"] = source_fp
            base = root / "outputs" / "research" / "raw_price_strategy_dataset" / source_fp
            data_dir = base / "tw_stock_data_vip"
            data_dir.mkdir(parents=True)
            frame = self._frame(0.15).iloc[:10].reset_index(names="Date")
            frame.loc[frame.index[3], ["Open", "High", "Low", "Close"]] = 0.0
            path = data_dir / "0052.csv"
            frame.to_csv(path, index=False)
            files = [{"ticker": "0052", "relative_path": "0052.csv", "size_bytes": path.stat().st_size,
                      "content_sha256": compute_file_sha256(path)}]
            inventory = canonical_json_sha256(files)
            materialization = canonical_json_sha256({"source_identity_fingerprint": source_fp,
                                                       "dataset_inventory_sha256": inventory})
            manifest = {"source_identity": source, "files": files,
                        "dataset_inventory_sha256": inventory,
                        "data_identity": {"materialization_fingerprint": materialization}}
            (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("services.research.raw_price_strategy_dataset._source_identity",
                       return_value=(source, None, None)):
                with self.assertRaisesRegex(ValueError, "RAW_SOURCE_QUALITY_BLOCKED") as raised:
                    inspect_raw_price_strategy_dataset(root, deep=True)
                self.assertIn("0052", str(raised.exception))
                self.assertIn("0.0", str(raised.exception))
            self.assertTrue(path.exists())  # immutable historical evidence untouched

    def test_existing_healthy_raw_materialization_remains_reusable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = {"provider_snapshot_fingerprint": "b" * 64}
            source_fp = "a" * 64
            source["source_identity_fingerprint"] = source_fp
            base = root / "outputs" / "research" / "raw_price_strategy_dataset" / source_fp
            data_dir = base / "tw_stock_data_vip"
            data_dir.mkdir(parents=True)
            frame = self._frame(0.15).iloc[:10].reset_index(names="Date")
            path = data_dir / "0052.csv"
            frame.to_csv(path, index=False)
            files = [{"ticker": "0052", "relative_path": "0052.csv", "size_bytes": path.stat().st_size,
                      "content_sha256": compute_file_sha256(path)}]
            inventory = canonical_json_sha256(files)
            materialization = canonical_json_sha256({"source_identity_fingerprint": source_fp,
                                                       "dataset_inventory_sha256": inventory})
            manifest = {"source_identity": source, "files": files,
                        "dataset_inventory_sha256": inventory,
                        "data_identity": {"materialization_fingerprint": materialization}}
            (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("services.research.raw_price_strategy_dataset._source_identity",
                       return_value=(source, None, None)):
                state = inspect_raw_price_strategy_dataset(root, deep=True)
            self.assertTrue(state["ready"])
            self.assertEqual(state["data_identity"]["materialization_fingerprint"], materialization)

    def test_raw_source_absence_reports_absent_without_backfill(self):
        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            control = self._frame(0.12)
            raw = self._frame(0.31).drop(self._frame(0.31).index[19])
            raw.reset_index(names="Date").to_csv(raw_dir / "0052.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing_raw=1") as caught:
                load_aligned_alternate_ohlcv_frame(raw_dir, "0052", control)
            message = str(caught.exception)
            self.assertIn("raw_rows_removed_by_sanitizer=0", message)
            self.assertIn("raw_rows_absent=1", message)
            self.assertIn("source_row_absent", message)

    def test_raw_only_dates_do_not_hide_shared_date_volume_mismatch(self):
        with TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            control = self._frame(0.12).iloc[1:]
            raw = self._frame(0.31)
            raw.loc[control.index[5], "Volume"] += 1.0
            raw.reset_index(names="Date").to_csv(raw_dir / "0052.csv", index=False)
            with self.assertRaisesRegex(ValueError, "volume mismatch"):
                load_aligned_alternate_ohlcv_frame(raw_dir, "0052", control)

    def test_service_source_is_pinned_to_verified_canonical_record(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=root/'manifest.json'
            payload={
                'ready':True, 'status':'READY', 'dataset_dir':root,
                'manifest_path':manifest,
                'source_identity':{'contract_id':'test'},
                'source_identity_fingerprint':'a'*64,
                'data_identity':{'materialization_fingerprint':'b'*64,'dataset_inventory_sha256':'c'*64},
            }
            with patch('services.breakout_quality.price_input_source.ensure_raw_price_strategy_dataset',return_value=payload):
                source=resolve_feature_price_source(model_architecture=self.raw_arch,project_root=root,create=True)
            self.assertEqual(source['dataset_dir'],root)
            self.assertEqual(source['identity']['materialization_fingerprint'],'b'*64)
            self.assertEqual(source['identity']['target_price_basis'],'canonical_adjusted')

    def test_missing_raw_source_fails_before_data_load(self):
        with self.assertRaisesRegex(ValueError, 'raw input capability'):
            load_daily_universal_ranker_data(
                filter_id='breakout_quality_v1',model_architecture=self.raw_arch,
                experiment_profile=self.raw_profile,preload_feature_bank=False,
                allow_stale_source=False,
            )

    def test_old_ao_identity_is_unchanged(self):
        summary=dict(dataset='full',policy={'a':1},source_data_inventory={'x':1},
                     source_dataset_artifacts={'a':'sha'},training_universe_start_date='2012-01-01')
        bundle=SimpleNamespace(summary=summary,target_manifest={'target_id':'test'})
        ao=source_training_identity(bundle)
        self.assertNotIn('input_price_source',ao)
        summary['input_price_source']={'materialization_fingerprint':'test'}
        raw=source_training_identity(bundle)
        self.assertEqual(raw['input_price_source'],summary['input_price_source'])
        self.assertEqual({k:v for k,v in raw.items() if k!='input_price_source'},ao)

    def test_source_loader_non_raw_has_no_materialization_side_effect(self):
        with patch('services.breakout_quality.price_input_source.ensure_raw_price_strategy_dataset',
                   side_effect=AssertionError('must not run')):
            self.assertIsNone(resolve_feature_price_source(model_architecture=self.ao_arch))

    def _fake_materialization(self,root:Path):
        fp='a'*64
        manifest_dir=root/'outputs'/'research'/'raw_price_strategy_dataset'/fp
        data_dir=manifest_dir/'tw_stock_data_vip'
        data_dir.mkdir(parents=True)
        file_path=data_dir/'0050.csv'
        file_path.write_text('Date,Open,High,Low,Close,Volume\n2020-01-02,10,11,9,10,10\n',encoding='utf-8')
        rows=[dict(relative_path='0050.csv',ticker='0050',row_count=1,
                   size_bytes=file_path.stat().st_size,content_sha256=compute_file_sha256(file_path))]
        source=dict(contract_id='research_raw_unadjusted_controlled_ohlcv_v2')
        source_fp=canonical_json_sha256(source)
        source['source_identity_fingerprint']=source_fp
        inv=canonical_json_sha256(rows)
        mat=canonical_json_sha256(dict(source_identity_fingerprint=source_fp,dataset_inventory_sha256=inv))
        record=dict(price_basis='raw_unadjusted_ohlc_canonical_volume',target_price_basis='canonical_adjusted',
                    source_contract_id=source['contract_id'],manifest_relative_path=(manifest_dir/'manifest.json').relative_to(root).as_posix(),
                    source_identity_fingerprint=source_fp,materialization_fingerprint=mat,
                    dataset_inventory_sha256=inv)
        payload=dict(source_identity=source,source_identity_fingerprint=source_fp,
                     data_identity={'materialization_fingerprint':mat},dataset_inventory_sha256=inv,
                     files=rows)
        (manifest_dir/'manifest.json').write_text(json.dumps(payload),encoding='utf-8')
        return record,file_path

    def test_pinned_source_verifies_content_and_rejects_tamper(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            record,path=self._fake_materialization(root)
            self.assertEqual(recorded_input_price_source_issues(published=record,project_root=root),())
            path.write_text(path.read_text('utf-8').replace(',10,10\n',',11,10\n'),'utf-8')
            self.assertIn('CSV SHA256 mismatch',';'.join(recorded_input_price_source_issues(published=record,project_root=root)))

    def test_source_path_escape_fails_closed(self):
        with TemporaryDirectory() as tmp:
            record,path=self._fake_materialization(Path(tmp))
            record['manifest_relative_path']='../../etc/passwd'
            self.assertIn('escapes research source namespace',';'.join(recorded_input_price_source_issues(published=record,project_root=tmp)))

    def test_published_source_requires_raw_record_and_rejects_mismatch(self):
        current=dict(dataset='full',policy={'x':1},source_data_inventory={'x':'sha'},
                     dataset_artifacts={'a':1},requires_input_price_source=True)
        published=dict(dataset='full',policy={'x':1},source_data_inventory={'x':'sha'},source_dataset_artifacts={'a':1})
        issues=_published_source_provenance_issues(label='test',source_dataset=published,
                   source_target=None,dataset_summary=current)
        self.assertIn('required raw input source missing',';'.join(issues))
        current['input_price_source']={'id':'new'}
        published['input_price_source']={'id':'old'}
        issues=_published_source_provenance_issues(label='test',source_dataset=published,
                   source_target=None,dataset_summary=current)
        self.assertIn('input_price_source mismatch',';'.join(issues))

if __name__=='__main__':
    unittest.main(verbosity=2)

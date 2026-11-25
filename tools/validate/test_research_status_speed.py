"""Performance paths preserve byte checks, date values and current source pins."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd

from core.file_integrity import compute_file_sha256, file_change_signature
from filters.breakout_quality import source_inventory
from filters.breakout_quality import strategy_score_projection as projection
from filters.breakout_quality.ranking_score_store import _normalized_score_dates
from services.research.oracle_numerical_sources import _Syntax
from services.research import strategy_compare_reuse as reports


class ResearchStatusSpeedTests(unittest.TestCase):
    def test_report_scope_reads_each_source_once_and_releases_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'report.json'
            path.write_text('{"status":"COMPLETED"}',encoding='utf-8')
            with patch.object(reports,'_load_report_json',wraps=reports._load_report_json) as read:
                with reports.comparison_report_read_scope():
                    for _ in range(5):
                        self.assertEqual(reports._read_json(path),{'status':'COMPLETED'})
                self.assertEqual(read.call_count,1)
                self.assertIsNone(reports._REPORT_READ_SCOPE.get())
                reports._read_json(path)
                self.assertEqual(read.call_count,2)

    def test_report_scope_rejects_changed_deleted_or_newly_published_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'report.json'
            for mutation in ('rewrite','delete','publish'):
                with self.subTest(mutation=mutation):
                    if mutation=='publish':
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text('{"value":1}',encoding='utf-8')
                    with self.assertRaisesRegex(RuntimeError,'report changed during'):
                        with reports.comparison_report_read_scope():
                            reports._read_json(path)
                            if mutation=='delete':
                                path.unlink()
                            elif mutation=='publish':
                                path.write_text('{"value":2}',encoding='utf-8')
                            else:
                                stamp=path.stat();time.sleep(0.01)
                                path.write_text('{"value":2}',encoding='utf-8')
                                os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
                    self.assertIsNone(reports._REPORT_READ_SCOPE.get())

    def test_report_scope_rejects_a_change_during_read_and_clears_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'report.json'
            def publish(_path):
                path.write_text('{}',encoding='utf-8');return None
            with self.assertRaisesRegex(RuntimeError,'report changed while reading'):
                with reports.comparison_report_read_scope(),patch.object(reports,'_load_report_json',side_effect=publish):
                    reports._read_json(path)
            self.assertIsNone(reports._REPORT_READ_SCOPE.get())

    def test_date_formatting_preserves_parser_values_order_missing_and_index(self):
        for values in (
            ['2020-01-01','2020-01-02','2020-01-01',None],
            [pd.Timestamp('2020-01-01'),pd.NaT,pd.Timestamp('2020-01-01')],
            ['2020-01-01T12:30:00+08:00','2020-01-01T13:30:00+08:00'],
            [],
        ):
            series=pd.Series(values,index=list(reversed(range(len(values)))))
            expected=pd.to_datetime(series,errors='raise').dt.strftime('%Y-%m-%d')
            pd.testing.assert_series_equal(_normalized_score_dates(series),expected)
        with self.assertRaises(ValueError):
            _normalized_score_dates(pd.Series(['2020-01-01','not a date']))

    def test_change_signature_keeps_equivalent_windows_paths_and_same_size_rewrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'sub').mkdir(); path=root/'file.txt'
            path.write_bytes(b'original')
            before=file_change_signature(path)
            self.assertEqual(file_change_signature(root/'sub'/'..'/'file.txt'),before)
            time.sleep(0.01)  # Put the rewrite in a distinct filesystem clock tick.
            stamp=path.stat(); path.write_bytes(b'modified')
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            self.assertNotEqual(file_change_signature(path),before)

    def test_syntax_projection_preserves_shared_ast_and_exact_normalization(self):
        source='"""module"""\nclass A:\n    """class"""\n    async def f(self):\n        """function"""\n        def inner():\n            """inner"""\n            return "kept"\n        return inner()\n'
        tree=ast.parse(source)
        original=ast.dump(tree,include_attributes=True)
        class Legacy(ast.NodeTransformer):
            def owner(self,node):
                self.generic_visit(node)
                if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                    node.body=node.body[1:]
                return node
            visit_Module=owner
            visit_ClassDef=owner
            visit_FunctionDef=owner
            visit_AsyncFunctionDef=owner
        expected=ast.dump(Legacy().visit(deepcopy(tree)),include_attributes=False)
        self.assertEqual(ast.dump(_Syntax().visit(tree),include_attributes=False),expected)
        self.assertEqual(ast.dump(tree,include_attributes=True),original)

    def test_published_price_scope_rechecks_both_boundaries_and_never_survives_exit(self):
        for changed in (False,True):
            calls=[]
            def read(*args):
                calls.append(args)
                return 'changed' if changed and len(calls)>1 else 'original'
            with patch.object(source_inventory,'_read_published_price_inventory',side_effect=read):
                def observe():
                    with source_inventory.source_data_inventory_read_scope():
                        for _ in range(4):
                            self.assertEqual(source_inventory.read_published_price_inventory('.', 'full'),'original')
                if changed:
                    with self.assertRaisesRegex(ValueError,'行情內容證據'):
                        observe()
                else:
                    observe()
                self.assertEqual(len(calls),2)
                self.assertIsNone(source_inventory._PRICE_READ_SCOPE.get())

    def test_persistent_projection_reuses_only_matching_bytes_and_recomputes_bad_receipts(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(projection,'_projection_implementation_sha256',return_value='a'*64):
            root=Path(temporary); path=root/'scores.csv'; cache=root/'cache'
            path.write_text('ticker,date,group_index,breakout_quality_score,fold_id,model_information_cutoff\n0050,2020-01-02,0,0.4,f,2019-12-31\n2330,2020-01-02,1,0.7,f,2019-12-31\n',encoding='utf-8')
            kwargs=dict(score_path=path,score_source=projection.SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                        columns=('breakout_quality_score','missing_optional'),projection_cache_dir=cache)
            original_hash=compute_file_sha256(path)
            expected=projection.compute_score_projection_sha256s(**kwargs,score_file_sha256=original_hash)
            self.assertEqual(set(expected),{'breakout_quality_score'})
            projection.clear_selection_point_in_time_projection_cache()
            with patch.object(projection,'_selection_pit_projection_cached',side_effect=AssertionError('CSV must not be parsed again')):
                self.assertEqual(projection.compute_score_projection_sha256s(**kwargs,score_file_sha256=original_hash),expected)
            receipt=next(cache.glob('*.json'))
            damaged=json.loads(receipt.read_text(encoding='utf-8'))
            damaged['projections']['breakout_quality_score']='0'*64
            receipt.write_text(json.dumps(damaged),encoding='utf-8')
            with patch.object(projection,'_selection_pit_projection_cached',wraps=projection._selection_pit_projection_cached) as derive:
                self.assertEqual(projection.compute_score_projection_sha256s(**kwargs,score_file_sha256=original_hash),expected)
                derive.assert_called_once()
            stamp=path.stat()
            path.write_bytes(path.read_bytes().replace(b'0.4',b'0.6'))
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'SHA256'):
                projection.compute_score_projection_sha256s(**kwargs,score_file_sha256=original_hash)
            changed=projection.compute_score_projection_sha256s(**kwargs)
            self.assertNotEqual(changed,expected)
            with patch.object(projection,'_projection_implementation_sha256',return_value='b'*64):
                self.assertEqual(projection.compute_score_projection_sha256s(**kwargs),changed)
            self.assertEqual(len(list(cache.glob('*.json'))),3)

    def test_projection_read_scope_rejects_implementation_change(self):
        with patch.object(projection,'_projection_implementation_sha256',return_value='changed'):
            with self.assertRaisesRegex(RuntimeError,'implementation changed'):
                with projection.score_projection_read_scope():
                    projection._READ_SCOPE.get()['implementation']='original'
        self.assertIsNone(projection._READ_SCOPE.get())


if __name__=='__main__':
    unittest.main()

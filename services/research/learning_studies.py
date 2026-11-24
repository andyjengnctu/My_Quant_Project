"""Controlled study orchestration; all FIT, parameter and replay work is delegated.

Stages are resumable. A negative screen closes the hypothesis; incomplete
evidence and execution failures remain distinct. Nothing promotes Trading.
"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd

from config.learning_studies import LEARNING_STUDIES, STUDY_OUTPUT_ROOT
from core.file_integrity import canonical_json_sha256, compute_file_sha256
from services.research.learning_study_metrics import paired_rank_metrics, screen_gate, strategy_gate

ROOT = Path(__file__).resolve().parents[2]


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')
    temp.replace(path)


def _publish(directory, report):
    report['updated_at'] = datetime.now(timezone.utc).isoformat()
    _write(directory / 'summary.json', report)
    lines = ['# CO 真實事件歷史擴充', '', f"狀態：{report['status']}",
             '', '2021 年起屬迭代研究 OOS；四 seed 是同一市場歷史的配對重複。', '']
    if report.get('reason'):
        lines += [str(report['reason']), '']
    lines += ['| 年度 | A Top-10 lift | B Top-10 lift | A Pair | B Pair |', '|---|---:|---:|---:|---:|']
    for year, row in report.get('screen', {}).items():
        def fmt(value):
            return '缺少有效證據' if value is None else f'{value:.6f}'
        lines.append(f"| {year} | {fmt(row['a']['top10_lift'])} | {fmt(row['b']['top10_lift'])} | {fmt(row['a']['pair_accuracy'])} | {fmt(row['b']['pair_accuracy'])} |")
    for key in ('screen_decision', 'rolling_model', 'rolling_decision', 'robustness_decision', 'samples', 'target'):
        if key in report:
            lines += ['', key + '：' + json.dumps(report[key], ensure_ascii=False, default=str)]
    for label, row in report.get('strategies', {}).items():
        lines += ['', f"{label}：" + json.dumps(row, ensure_ascii=False, default=str)]
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    _write(directory.parent.parent / 'latest.json', {'contract_sha256': report['contract_sha256'],
                                                    'summary': str((directory / 'summary.json').relative_to(ROOT)).replace('\\','/')})


def collect_status(study_id='co_history', *, project_root=ROOT):
    from services.breakout_quality.trade_path_target_source import registered_trade_path_source
    study = LEARNING_STUDIES[study_id]
    source = registered_trade_path_source(study['profile'])
    return {'study': study_id, 'model': study['model_id'], 'dependencies': [
        {'name': d.name, **d.inspect(project_root=project_root, seed=int(study['screen_seed']))}
        for d in source.dependencies]}


def _prepare(profile, seed, root):
    from services.breakout_quality.trade_path_target_source import registered_trade_path_source
    source = registered_trade_path_source(profile)
    for dependency in sorted(source.dependencies, key=lambda d: d.priority):
        state = dependency.inspect(project_root=root, seed=seed)
        action = str(state['action'])
        print(f"[Study dependency] {dependency.name} seed={seed} {action}", flush=True)
        if action == 'BLOCKED':
            raise RuntimeError(str(state['reason']))
        if action != 'REUSE':
            dependency.execute(project_root=root, seed=seed)


def _load(profile, seed, root):
    from services.breakout_quality.continuous_ranker_pipeline import load_continuous_ranker_data
    return load_continuous_ranker_data(
        filter_id='breakout_quality_v1', model_architecture='inception_time_v1',
        experiment_profile=profile, preload_feature_bank=False, allow_stale_source=False,
        project_root=root, seed=seed)


def _fit(bundle, *, seed, start, end, directory):
    from services.breakout_quality.point_in_time_scores import build_selection_point_in_time_scores
    from filters.breakout_quality.paths import SELECTION_POINT_IN_TIME_SCORE_FILENAME, SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
    from config.breakout_quality import BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT
    code = build_selection_point_in_time_scores(
        filter_id='breakout_quality_v1', model_architecture='inception_time_v1',
        experiment_profile=bundle.profile.name, seed=seed, score_start_date=start,
        score_end_date=end, fold_months=12, resume=True,
        point_in_time_dir_override=str(directory), prepared_bundle=bundle,
        checkpoint_cache_root=str(ROOT / BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT))
    if code:
        raise RuntimeError(f'Canonical PIT producer failed: {code}')
    return {'score_path': directory / SELECTION_POINT_IN_TIME_SCORE_FILENAME,
            'manifest_path': directory / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME}


def _evaluation(bundle, paths, *, year):
    from filters.breakout_quality.daily_ranker_data import select_breakout_candidate_group_ids
    frame = pd.read_csv(paths['score_path'], dtype={'ticker': str}, parse_dates=['date'])
    group_dates = pd.to_datetime(bundle.group_table['date'])
    mask = np.asarray(bundle.target_valid) & (group_dates.dt.year == year).to_numpy()
    ids = select_breakout_candidate_group_ids(bundle, np.flatnonzero(mask), allow_stale_source=False)
    truth = bundle.group_table.iloc[ids][['ticker','date']].copy()
    truth['ticker'] = truth['ticker'].astype(str)
    truth['target'] = bundle.raw_target[ids]
    joined = truth.merge(frame[['ticker','date','breakout_quality_score']], on=['ticker','date'], validate='one_to_one', how='left')
    if joined['breakout_quality_score'].isna().any():
        raise ValueError('Missing canonical PIT scores in common breakout evaluation')
    return joined.rename(columns={'breakout_quality_score': 'score'})


def _verify_original_truth(a, b):
    cols = ['ticker','date','label_eval_end_date']
    first_original = pd.to_datetime(a.group_table.loc[a.target_valid, 'date']).min()
    left = a.group_table.loc[pd.to_datetime(a.group_table['date']) >= first_original, cols].copy()
    right = b.group_table[cols].copy()
    left['target'], right['target'] = a.raw_target[left.index], b.raw_target
    left['valid'], right['valid'] = a.target_valid[left.index], b.target_valid
    original = left.merge(right, on=['ticker','date'], suffixes=('_a','_b'), validate='one_to_one', how='left', indicator=True)
    if (not (original['_merge'] == 'both').all()
            or not np.array_equal(original['valid_a'], original['valid_b'])
            or not np.allclose(original['target_a'], original['target_b'], rtol=0, atol=0, equal_nan=True)
            or not original['label_eval_end_date_a'].fillna(pd.Timestamp.min).equals(
                original['label_eval_end_date_b'].fillna(pd.Timestamp.min))):
        raise ValueError('Early history changed an original stock-day, label or maturity')


def _episode_inventory(bundle, *, teacher, mask, cutoff):
    """Use the canonical pending-order lifecycle, not ticker/date deduplication."""
    from filters.breakout_quality.daily_trade40_target import compute_counterfactual_trade40_batch
    keys = []
    from filters.breakout_quality.daily_trade40_target import collect_canonical_exchange_sessions
    from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs
    csv_inputs, _duplicates = discover_dataset_csv_inputs(ROOT, bundle.summary['dataset'])
    calendar = collect_canonical_exchange_sessions(csv_inputs, benchmark_ticker='0050', horizon_sessions=None)
    selected = bundle.group_table.loc[mask]
    for ticker, rows in selected.groupby('ticker', observed=True, sort=False):
        frame = bundle.feature_bank.canonical_price_frame(int(rows['ticker_id'].iloc[0]))
        frame = frame.loc[frame.index <= pd.Timestamp(cutoff)]
        observed = {}
        compute_counterfactual_trade40_batch(
            frame, rows['source_pos'].to_numpy(dtype=int), ticker=str(ticker),
            schedule=teacher.schedule, exchange_calendar=calendar,
            horizon_sessions=None, research_cutoff=cutoff,
            signal_state_reset_date=teacher.oos_start_date,
            episode_keys_out=observed, episode_keys_only=True)
        if len(observed) != len(rows):
            raise ValueError('A matured FIT row lacks its canonical original signal')
        keys.extend(observed.values())
    counts = pd.Series(keys,dtype=str).value_counts()
    return {'mature_rows': int(mask.sum()), 'original_signals': int(len(counts)),
            'rows_per_signal_mean': float(counts.mean()) if len(counts) else None,
            'rows_per_signal_max': int(counts.max()) if len(counts) else None,
            'largest_signal_row_share': float(counts.max()/counts.sum()) if len(counts) else None}


def _replay(study, *, seed, start, end, paths_a, paths_b, directory):
    from core.strategy_compare_policy import get_strategy_comparison_settings
    from core.strategy_comparison import StrategyComparisonArm
    from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
    from services.research.strategy_compare_execution import run_strategy_compare_active_arm
    from services.breakout_quality.formal_mintrade40_training_source import _current_p16_seed_artifact
    base = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
    candidate = StrategyComparisonArm(arm_id=study['arm_id'], enabled=True, **STRATEGY_COMPARE_ARMS[study['arm_id']])
    control = base.arms[study['control_arm']]
    dl = replace(base.dl_sources[control.dl_id], dl_id=candidate.dl_id, experiment_profile=study['profile'])
    settings = replace(base, arms={'C96': base.arms['C96'], control.arm_id: control, candidate.arm_id: candidate},
                       dl_sources={**base.dl_sources, candidate.dl_id: dl})
    params, _, _, _ = _current_p16_seed_artifact(ROOT, seed, authorized_build=True)
    result = {}
    for key, arm, paths in (('a',control,paths_a),('b',candidate,paths_b)):
        payload = run_strategy_compare_active_arm(
            settings=settings, arm=arm, project_root=ROOT, params_path=params,
            output_dir=directory / arm.arm_id, comparison_start=start, comparison_end=end,
            param_evaluation_mode='rolling', score_overrides={arm.dl_id: paths},
            expected_seed_override=seed, quiet=True, capture_execution_diagnostics=True)
        observation = payload.get('score_ranking') or payload.get('quality_filter')
        if not isinstance(observation, dict):
            raise ValueError('Canonical Strategy Compare lacks active economic observation')
        result[key] = _economic_observation(observation)
    return result


def _economic_observation(observation):
    # These are the canonical Strategy Compare fields, not recomputed formulas.
    return {'romd': float(observation['return_over_max_drawdown']), 'mdd': float(observation['max_drawdown_pct']),
            'return': float(observation['total_return_pct'])}


def run_study(study_id='co_history'):
    study = dict(LEARNING_STUDIES[study_id])
    contract_sha = canonical_json_sha256(study)
    directory = ROOT / STUDY_OUTPUT_ROOT / study_id / contract_sha[:16]
    report = {'schema_version': 1, 'study': study, 'contract_sha256': contract_sha,
              'status': 'RUNNING', 'screen': {}, 'strategies': {}}
    _publish(directory, report)
    try:
        seed = int(study['screen_seed'])
        _prepare(study['profile'], seed, ROOT)
        a, b = _load(study['control_profile'], seed, ROOT), _load(study['profile'], seed, ROOT)
        _verify_original_truth(a, b)
        # All sample counts are restricted to the first deployment FIT cutoff.
        from core.breakout_quality_policy import get_breakout_quality_rolling_test_mode
        operational_start = pd.Timestamp(get_breakout_quality_rolling_test_mode('rolling').score_start_date)
        fit_inventory_cutoff = str((operational_start - pd.Timedelta(days=1)).date())
        dates = pd.to_datetime(b.group_table['date'])
        maturity = pd.to_datetime(b.group_table['label_eval_end_date'])
        first_original = pd.to_datetime(a.group_table.loc[a.target_valid, 'date']).min()
        new = b.target_valid & (dates < first_original).to_numpy() & (maturity < operational_start).to_numpy()
        report['samples'] = {'added_mature_rows': int(new.sum()),
                             'added_tickers': int(b.group_table.loc[new,'ticker'].nunique()),
                             'added_dates': int(b.group_table.loc[new,'date'].nunique()),
                             'original_truth_unchanged': True}
        from services.breakout_quality.formal_mintrade40_training_source import load_formal_mintrade40_training_source
        teacher = load_formal_mintrade40_training_source(
            project_root=ROOT,seed=seed,history_extension=study['history_extension'])
        report['samples']['added_episode_inventory'] = _episode_inventory(
            b,teacher=teacher,mask=new,cutoff=fit_inventory_cutoff)
        years = tuple(study['screen_years'])
        screen_paths = {}
        for label, bundle in (('a',a),('b',b)):
            screen_paths[label] = _fit(bundle, seed=seed, start=f'{min(years)}-01-01', end=f'{max(years)}-12-31', directory=directory / 'fits' / f'seed_{seed}' / label / 'screen')
        for year in years:
            report['screen'][str(year)] = paired_rank_metrics(
                _evaluation(a,screen_paths['a'],year=year), _evaluation(b,screen_paths['b'],year=year))
        report['screen_decision'] = screen_gate(report['screen'], expected_years=years, **study['screen_gate'])
        _publish(directory,report)
        if report['screen_decision']['status'] == 'INCOMPLETE':
            report['status'] = 'INCOMPLETE'
            report['reason'] = '篩選缺少有效共同樣本，不能判定研究成敗。'
            return report
        from core.breakout_quality_policy import get_breakout_quality_rolling_test_mode
        mode = get_breakout_quality_rolling_test_mode('rolling')
        start = str(mode.score_start_date)
        from config.market_data import RESEARCH_REQUIRED_CUTOFF
        end = str(RESEARCH_REQUIRED_CUTOFF)
        paths = {}
        for label, bundle in (('a',a),('b',b)):
            paths[label] = _fit(bundle, seed=seed, start=start, end=end, directory=directory / 'fits' / f'seed_{seed}' / label / 'rolling')
        report['rolling_model'] = {
            str(year): paired_rank_metrics(_evaluation(a,paths['a'],year=year), _evaluation(b,paths['b'],year=year))
            for year in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1)
        }
        report['strategies']['rolling_seed42'] = _replay(study, seed=seed, start=start, end=end,
            paths_a=paths['a'], paths_b=paths['b'], directory=directory / 'strategies' / f'seed_{seed}')
        report['rolling_decision'] = strategy_gate([report['strategies']['rolling_seed42']], minimum_wins=1)
        _publish(directory,report)
        if report['rolling_decision']['status'] != 'PASS' or report['screen_decision']['status'] != 'PASS':
            report['status'] = 'REJECT' if 'REJECT' in {report['rolling_decision']['status'],report['screen_decision']['status']} else 'INCOMPLETE'
            report['reason'] = '單 seed 模型與策略 Rolling 已完成；未通過後續 Robustness 條件。'
            return report
        from core.training_policy import resolve_robustness_benchmark_seeds
        robustness = []
        for replicate in resolve_robustness_benchmark_seeds():
            _prepare(study['profile'], int(replicate), ROOT)
            pair = {}
            for label, profile in (('a',study['control_profile']),('b',study['profile'])):
                bundle = _load(profile,int(replicate),ROOT)
                pair[label] = _fit(bundle,seed=int(replicate),start=start,end=end,directory=directory/'fits'/f'seed_{replicate}'/label/'rolling')
            outcome = _replay(study,seed=int(replicate),start=start,end=end,paths_a=pair['a'],paths_b=pair['b'],directory=directory/'strategies'/f'seed_{replicate}')
            report['strategies'][f'robustness_{replicate}'] = outcome
            robustness.append(outcome)
            _publish(directory,report)
        report['robustness_decision'] = strategy_gate(robustness, **study['robustness_gate'])
        if report['robustness_decision']['status'] == 'PASS':
            from services.audit.runner import run_enabled_audits
            from core.audit_policy import get_active_audit_module_id
            run_enabled_audits(get_active_audit_module_id(), project_root=ROOT, method_id='target_ground_truth_strategy_romd')
            report['target'] = {'status': 'EXECUTED_CANONICAL_SAME_TARGET', 'control_arm': study['control_arm'],
                                'note': 'Target、Exact 與評分母體相同；新增訓練歷史沒有新 Oracle 真值。'}
        report['status'] = report['robustness_decision']['status']
        return report
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        report['status'] = 'BLOCKED'
        report['reason'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        _publish(directory,report)
        print((directory / 'summary.md').read_text(encoding='utf-8'), flush=True)


def main(args=()):
    action = str(args[0]) if args else 'status'
    if action == 'run':
        try:
            result = run_study()
            return 0 if result['status'] in {'PASS','REJECT'} else 1
        finally:
            from services.research.target_dl_overview import collect_target_dl_overview, render_target_dl_overview
            print(render_target_dl_overview(collect_target_dl_overview(ROOT)))
    if action == 'status':
        status = collect_status()
        for dependency in status['dependencies']:
            if dependency.get('path'):
                dependency['path'] = str(Path(dependency['path']).relative_to(ROOT)).replace('\\','/')
        print(json.dumps(status,ensure_ascii=False,indent=2,default=str))
        return 0
    if action == 'latest':
        pointer = ROOT / STUDY_OUTPUT_ROOT / 'latest.json'
        if not pointer.is_file():
            print('尚無受控研究結果。')
        else:
            path = ROOT / json.loads(pointer.read_text(encoding='utf-8'))['summary']
            print(path.with_suffix('.md').read_text(encoding='utf-8'))
        return 0
    raise ValueError('study 支援 run、status、latest')


def menu():
    from core.console_report import render_menu_item
    while True:
        print('\n=== 受控學習實驗 ===')
        for number, label in ((1,'執行／接續受控研究'),(2,'查看依賴狀態'),(3,'查看最新結果'),(0,'返回')):
            print(render_menu_item(number,label))
        try:
            choice = input('👉 請選擇：').strip()
        except EOFError:
            return 0
        if choice == '0':
            return 0
        if choice in {'','1','2','3'}:
            main([{'':'run','1':'run','2':'status','3':'latest'}[choice]])

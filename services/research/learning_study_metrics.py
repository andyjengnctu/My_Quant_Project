"""Controlled, common-cohort study statistics and prespecified progression gates."""
import numpy as np
import pandas as pd


def paired_rank_metrics(control, treatment, *, top_k=10):
    keys = ['ticker', 'date']
    for name, frame in (('control', control), ('treatment', treatment)):
        if frame.duplicated(keys).any():
            raise ValueError(name + ' has duplicate stock-days')
    a = control[keys + ['score', 'target']]
    b = treatment[keys + ['score', 'target']]
    joined = a.merge(b, on=keys, suffixes=('_a', '_b'), validate='one_to_one', how='outer', indicator=True)
    if not (joined['_merge'] == 'both').all():
        raise ValueError('Controlled evaluation must use identical stock-days')
    if not np.allclose(joined['target_a'], joined['target_b'], rtol=0, atol=0, equal_nan=True):
        raise ValueError('Original-year label truth changed in the treatment')
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna(subset=['score_a','score_b','target_a'])
    from services.breakout_quality.ranker_training import daily_rank_metrics, daily_top_k_metrics
    from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
    truth = joined['target_a'].to_numpy(dtype=float)
    dates = joined['date'].to_numpy()
    percentiles = build_same_date_percentile_targets(truth, np.ones(len(truth),bool), dates)
    result = {'groups': len(joined), 'dates': int(joined['date'].nunique())}
    for suffix in ('a','b'):
        scores = joined['score_'+suffix].to_numpy(dtype=float)
        rank = daily_rank_metrics(dates,scores,truth)
        top = daily_top_k_metrics(dates,scores,truth,percentiles,top_k=top_k,boundary_width=3)
        result[suffix] = {'pair_accuracy':rank['pairwise_concordance'],
                          'pair_count':rank['comparable_pair_count'],
                          'competition_days':top['top_k_date_count'],
                          'top10_lift':top['top_k_raw_target_lift']}
    return result


def screen_gate(years, *, expected_years, minimum_positive_years, pair_delta_min=0.0):
    if set(map(int, years)) != set(map(int, expected_years)):
        return {'status': 'INCOMPLETE', 'reason': 'missing yearly evidence'}
    deltas, pairs = [], []
    for row in years.values():
        if any(row[arm][key] is None for arm in ('a','b') for key in ('top10_lift','pair_accuracy')):
            return {'status': 'INCOMPLETE', 'reason': 'insufficient competition or comparable pairs'}
        deltas.append(row['b']['top10_lift'] - row['a']['top10_lift'])
        pairs.append(row['b']['pair_accuracy'] - row['a']['pair_accuracy'])
    passed = np.mean(deltas) > 0 and sum(x > 0 for x in deltas) >= minimum_positive_years and np.mean(pairs) >= pair_delta_min
    return {'status': 'PASS' if passed else 'REJECT', 'mean_top10_lift_delta': float(np.mean(deltas)),
            'positive_years': int(sum(x > 0 for x in deltas)), 'mean_pair_accuracy_delta': float(np.mean(pairs))}


def strategy_gate(rows, *, minimum_wins, mean_mdd_delta_max=0.0):
    if not rows:
        return {'status': 'INCOMPLETE'}
    delta = np.asarray([r['b']['romd'] - r['a']['romd'] for r in rows], float)
    mdd = np.asarray([r['b']['mdd'] - r['a']['mdd'] for r in rows], float)
    if not np.isfinite(np.r_[delta, mdd]).all():
        return {'status': 'INCOMPLETE'}
    passed = delta.mean() > 0 and np.median(delta) > 0 and (delta > 0).sum() >= minimum_wins and mdd.mean() <= mean_mdd_delta_max
    return {'status': 'PASS' if passed else 'REJECT', 'mean_delta_romd': float(delta.mean()),
            'median_delta_romd': float(np.median(delta)), 'wins': int((delta > 0).sum()),
            'mean_delta_mdd': float(mdd.mean())}

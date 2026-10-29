"""One-time, read-only full-cycle Positive-NetPnL executed-trade attribution.

Every reference is recalculated by the EXISTING single-stock Min path owner
from the same information-date order state; its NetPnL must equal the signed
raw-label value used in the official gated Portfolio Replay.  No retrospective
threshold fitting or new buy/sell/fee logic is allowed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT
from core.capital_policy import resolve_single_backtest_sizing_capital
from core.entry_plans import build_normal_entry_plan
from core.extended_signals import (
    create_signal_tracking_state, should_clear_extended_signal,
    build_extended_entry_plan_from_signal,
)
from filters.breakout_quality.daily_trade40_target import (
    build_formal_min_execution_schedule, resolve_formal_min_param_records,
    resolve_historical_min_period,
)
from filters.breakout_quality.trade_path_label import simulate_realized_trade_path_label
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame
from filters.breakout_quality.mfe_safety_geometry import normalize_ticker
from services.audit.positive_gate_evidence import TARGET, read_evidence
from services.audit.oracle_replay_artifacts import OracleReplayStore
from services.audit.target_oracle_romd import (
    _contract, _robustness_artifacts, _prepare_oracle_parameters,
)
from filters.breakout_quality.trade_path_label import build_signal_cache


def _evidence_inventory(definition, root: Path):
    from core.strategy_compare_policy import get_strategy_multi_seed_robustness_settings
    settings, _, _ = _contract_from_reference(root)
    robust = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    seeds = {int(value) for value in robust.resolved_seeds}
    expected = {}
    path_root = root / AUDIT_OUTPUT_ROOT / definition.source['oracle_output_subdir'] / 'positive_gate_replay_evidence'
    cache = OracleReplayStore(root, definition.source['oracle_output_subdir'])
    for seed in sorted(seeds):
        folder = path_root / str(seed)
        candidates = list(folder.glob('*.json')) if folder.is_dir() else []
        accepted = []
        for path in candidates:
            item = read_evidence(path)
            if int(item['seed']) != seed or item['target'] != TARGET:
                raise ValueError('positive gate evidence seed/target mismatch')
            observation, _ = cache.resolve_completed(item['replay_identity'])
            if observation is None:
                continue
            if (int(observation['trades']) != int(item['summary']['trades']) or
                    not math.isclose(float(observation['win_rate_pct']),
                                     float(item['summary']['win_rate_pct']), abs_tol=1e-8) or
                    not math.isclose(float(observation['romd']),
                                     float(item['summary']['romd']), abs_tol=1e-8)):
                raise ValueError('positive gate Audit sidecar differs from canonical replay')
            accepted.append((path, item))
        if len(accepted) != 1:
            raise ValueError(f'Positive-NetPnL seed {seed}: expected one current completed Replay evidence, got {len(accepted)}')
        expected[seed] = accepted[0]
    return expected


def _contract_from_reference(root: Path):
    from core.audit_policy import get_reusable_audit_definitions
    source = next(x for x in get_reusable_audit_definitions('breakout_quality') if x.report_id == 'target_oracle_romd')
    return _contract(source)


def collect_status(definition, *, project_root: Path):
    root = Path(project_root).resolve()
    try:
        inventory = _evidence_inventory(definition, root)
        return {'status': 'READY', 'reason': '',
                'source': {'path': str(next(iter(inventory.values()))[0]),
                           'display': f'{len(inventory)} signed positive-gate Replay seeds'}}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {'status': 'BLOCKED', 'reason': str(exc),
                'source': {'display': 'positive-gate completed Replay evidence missing or mismatched'}}


def _reference_trade(frame, *, ticker, info_day, schedule, cutoff):
    """Reconstruct the canonical single-ticker signal *state* without cash competition.

    This Audit-only observer delegates all economic behavior to the same Min
    simulator and asserts the signed original outcome before comparing paths.
    """
    dates = pd.DatetimeIndex(frame.index).normalize()
    info_pos = int(dates.get_indexer([pd.Timestamp(info_day).normalize()])[0])
    if info_pos < 0 or info_pos + 1 >= len(dates):
        raise ValueError(f'reference signal-day missing: {ticker} {info_day}')
    caches = {}
    def current_cache(day):
        active = resolve_historical_min_period(schedule, pd.Timestamp(day).normalize())
        if active is None:
            raise ValueError(f'reference P16 parameter missing: {ticker} {day}')
        if active.params_signature not in caches:
            caches[active.params_signature] = build_signal_cache(frame, active.params, ticker=ticker)
        return caches[active.params_signature]
    state = None
    opens, highs, lows, closes, volumes = (frame[col].to_numpy(dtype=np.float64, copy=False)
                                            for col in ('Open', 'High', 'Low', 'Close', 'Volume'))
    for pos in range(info_pos + 1):
        tomorrow = dates[pos + 1]
        period = resolve_historical_min_period(schedule, tomorrow)
        if period is None:
            state = None
            continue
        cache = current_cache(tomorrow)
        if bool(cache[1][pos]):
            state = create_signal_tracking_state(
                cache[3][pos], cache[0][pos], period.params, ticker=ticker,
                security_profile=frame.attrs.get('security_profile'), signal_date=dates[pos],
            )
        if pos == info_pos:
            break
        if state is None:
            continue
        params = state['_params_obj']
        capital = resolve_single_backtest_sizing_capital(params, params.initial_capital)
        if should_clear_extended_signal(
            state, lows[pos + 1], highs[pos + 1],
            t_open=opens[pos + 1], t_close=closes[pos + 1], t_volume=volumes[pos + 1],
            y_close=closes[pos], y_high=highs[pos], y_atr=cache[0][pos],
            y_ind_sell=cache[2][pos], sizing_capital=capital, current_date=tomorrow,
            params=params, copy_shadow_position=False,
        ):
            state = None
    if state is None:
        raise ValueError(f'positive raw outcome with no canonical signal: {ticker} {info_day}')
    origin_day = pd.Timestamp(state['signal_date']).normalize()
    origin_pos = int(dates.get_indexer([origin_day])[0])
    params = state['_params_obj']
    def signal_cache(day):
        return current_cache(day)
    observation = simulate_realized_trade_path_label(
        frame, ticker=ticker, signal_pos=origin_pos, params=params,
        teacher_effective_date=str(origin_day.date()),
        precomputed_signals=current_cache(dates[info_pos + 1]),
        horizon_end_date=pd.Timestamp(cutoff), decision_pos=info_pos,
        starting_signal_state=state, active_signal_cache_resolver=signal_cache,
    )
    # To obtain reference quantity, use canonical entry-plan owners. Pending
    # shadow-state updates on pre-fill sessions use canonical continuation code.
    ref_qty = None
    if observation.fill_date is not None:
        fill_pos = int(dates.get_indexer([pd.Timestamp(observation.fill_date)])[0])
        working = deepcopy(state, memo={id(params): params})
        capital = resolve_single_backtest_sizing_capital(params, params.initial_capital)
        for pos in range(info_pos + 1, fill_pos + 1):
            cache = current_cache(dates[pos])
            if pos == origin_pos + 1:
                plan = build_normal_entry_plan(
                    float(working['orig_limit']), float(working['orig_atr']),
                    capital, params, ticker=ticker,
                    security_profile=frame.attrs.get('security_profile'), trade_date=dates[pos],
                )
            else:
                plan = build_extended_entry_plan_from_signal(
                    working, capital, params, y_close=closes[pos - 1], ticker=ticker,
                    security_profile=frame.attrs.get('security_profile'), trade_date=dates[pos],
                )
            if pos == fill_pos:
                ref_qty = None if plan is None else int(plan.get('qty') or 0)
                break
            if should_clear_extended_signal(
                working, lows[pos], highs[pos], t_open=opens[pos], t_close=closes[pos],
                t_volume=volumes[pos], y_close=closes[pos - 1], y_high=highs[pos - 1],
                y_atr=cache[0][pos - 1], y_ind_sell=cache[2][pos - 1],
                sizing_capital=capital, current_date=dates[pos], params=params,
                copy_shadow_position=False,
            ):
                raise ValueError('canonical reference signal expired before recorded fill')
    return observation, str(origin_day.date()), ref_qty


def _match_fills(rows, fills):
    # Match actual executed entry from the Portfolio's *execution* snapshots,
    # never use the number of positions alone to infer sizing differences.
    grouped = defaultdict(list)
    for item in fills:
        key = (str(item['ticker']), str(item['trade_date'])[:10])
        grouped[key].append(item)
    answer = []
    for trade in rows:
        key = (str(trade['ticker']), str(trade['entry_trade_date'])[:10])
        values = grouped.get(key) or []
        if len(values) != 1:
            raise ValueError(f'cannot uniquely identify actual filled buy: {key} ({len(values)})')
        answer.append(values.pop())
    if any(group for group in grouped.values()):
        raise ValueError('Portfolio filled a trade that is absent from closed Round-Trips')
    return answer


def run_formal_audit(definition, *, project_root: Path, quiet: bool = False):
    root = Path(project_root).resolve()
    inventory = _evidence_inventory(definition, root)
    from core.audit_policy import get_reusable_audit_definitions
    oracle_def = next(x for x in get_reusable_audit_definitions('breakout_quality') if x.report_id == 'target_oracle_romd')
    settings, base_arm, _ = _contract(oracle_def)
    robustness, paths = _robustness_artifacts(oracle_def, root)
    prepared, _ = _prepare_oracle_parameters(
        oracle_def, root=root, parameter_paths=paths,
        start=str(settings.start_date), end=str(settings.end_date), base_arm=base_arm,
    )
    schedules = {}
    for seed, param_sha, kind, base_params, _ in prepared:
        schedule, _ = build_formal_min_execution_schedule(
            resolve_formal_min_param_records(kind, base_params), seed=seed, param_sha256=param_sha,
        )
        schedules[seed] = schedule
    csv_inputs, duplicate = discover_dataset_csv_inputs(root, settings.dataset)
    if duplicate:
        raise ValueError('Audit market source contains duplicate ticker')
    paths_by_ticker = {normalize_ticker(ticker): path for ticker, path in csv_inputs}
    frames = {}
    breakdown = {}
    detail = []
    for seed, (_path, doc) in sorted(inventory.items()):
        trades = doc['rows']
        fills = _match_fills(trades, doc['fills'])
        counts = Counter()
        overlaps = Counter()
        for trade, fill in zip(trades, fills):
            ticker = str(trade['ticker'])
            if ticker not in frames:
                frames[ticker] = load_dataset_frame(paths_by_ticker[ticker], ticker, min_rows=1)
            obs, origin, ref_qty = _reference_trade(
                frames[ticker], ticker=ticker,
                info_day=trade['breakout_quality_score_date'],
                schedule=schedules[seed], cutoff=str(settings.end_date),
            )
            reference_pnl = obs.realized_net_pnl
            if (reference_pnl is None or
                    not math.isclose(float(reference_pnl), float(trade['target_raw_net_pnl']), abs_tol=0.001)):
                raise ValueError(f'canonical reference outcome differs from signed positive gate raw PnL: {ticker}')
            actual_pnl = float(trade['pnl'])
            category = 'profit' if actual_pnl > 0 else 'loss' if actual_pnl < 0 else 'zero'
            # Event difference includes origin, fill date and entry price.
            event_different = (
                origin != str(trade['signal_date'])[:10] or
                str(obs.fill_date) != str(trade['entry_trade_date'])[:10] or
                obs.entry_price is None or trade['entry_price'] is None or
                not math.isclose(float(obs.entry_price), float(trade['entry_price']), abs_tol=1e-8)
            )
            actual_qty = int(fill.get('filled_qty') or 0)
            if ref_qty is None or actual_qty <= 0:
                raise ValueError('canonical sizing evidence missing for filled trade')
            sizing_different = ref_qty != actual_qty
            exit_different = (
                str(obs.exit_date) != str(trade['exit_date'])[:10] or
                obs.exit_price is None or trade['exit_price'] is None or
                not math.isclose(float(obs.exit_price), float(trade['exit_price']), abs_tol=1e-8)
            )
            null_bought = not float(trade['target_raw_net_pnl']) > 0.0
            zero_pnl = actual_pnl == 0.0
            flags = {
                'null_or_nonpositive_bought': null_bought,
                'different_trade_event': event_different,
                'different_quantity': sizing_different,
                'different_exit_path': exit_different,
                'zero_actual_pnl': zero_pnl,
            }
            if null_bought:
                raise ValueError('Gate FAIL: a NULL/nonpositive target was executed')
            counts[f'result_{category}'] += 1
            for key, value in flags.items():
                if value:
                    counts[key] += 1
                    if category != 'profit':
                        counts[f'nonwin_{key}'] += 1
            if category != 'profit':
                pattern = '+'.join(sorted(key for key, value in flags.items() if value and key != 'zero_actual_pnl')) or 'none_of_four'
                overlaps[pattern] += 1
            detail.append({
                'seed': seed, 'ticker': ticker,
                'information_date': trade['breakout_quality_score_date'],
                'actual_entry_date': trade['entry_trade_date'],
                'actual_signal_date': trade['signal_date'],
                'reference_signal_date': origin,
                'reference_fill_date': obs.fill_date,
                'reference_entry_price': obs.entry_price,
                'actual_entry_price': trade['entry_price'],
                'reference_quantity': ref_qty,
                'actual_quantity': actual_qty,
                'reference_exit_date': obs.exit_date,
                'actual_exit_date': trade['exit_date'],
                'reference_exit_price': obs.exit_price,
                'actual_exit_price': trade['exit_price'],
                'reference_exit_reason': obs.exit_reason,
                'actual_exit_type': trade['exit_type'],
                'target_net_pnl': reference_pnl, 'actual_net_pnl': actual_pnl,
                'actual_result': category, **flags,
            })
        breakdown[str(seed)] = {'round_trips': len(trades),
                                'win_rate_pct': doc['summary']['win_rate_pct'],
                                'counts': dict(sorted(counts.items())),
                                'nonwin_overlaps': dict(sorted(overlaps.items()))}
    total = len(detail)
    bad = sum(row['actual_result'] != 'profit' for row in detail)
    audited_factors = (
        'null_or_nonpositive_bought', 'different_trade_event',
        'different_quantity', 'different_exit_path', 'zero_actual_pnl',
    )
    by_result = {
        outcome: {factor: sum(row['actual_result'] == outcome and bool(row[factor])
                              for row in detail) for factor in audited_factors}
        for outcome in ('profit', 'loss', 'zero')
    }
    unresolved = sum(
        row['actual_result'] != 'profit' and not any(row[factor] for factor in (
            'null_or_nonpositive_bought', 'different_trade_event',
            'different_quantity', 'different_exit_path',
        ))
        for row in detail
    )
    audit = {'schema': 1, 'source_target': TARGET, 'source_seed_receipts': {
        str(seed): doc['receipt_sha256'] for seed, (_path, doc) in inventory.items()},
        'round_trips': total, 'nonwinning_round_trips': bad,
        'nonwinning_ratio_pct': 100 * bad / total if total else 0.0,
        'nonwins_without_observed_event_quantity_exit_difference': unresolved,
        'factor_by_actual_result': by_result,
        'by_seed': breakdown,
        'factor_totals': {factor: sum(bool(row[factor]) for row in detail)
                          for factor in ('null_or_nonpositive_bought', 'different_trade_event',
                                         'different_quantity', 'different_exit_path', 'zero_actual_pnl')},
        'nonwin_factor_totals': {factor: sum(bool(row[factor]) and row['actual_result'] != 'profit' for row in detail)
                                for factor in ('null_or_nonpositive_bought', 'different_trade_event',
                                               'different_quantity', 'different_exit_path', 'zero_actual_pnl')},
    }
    import csv
    out = root / AUDIT_OUTPUT_ROOT / definition.output_subdir / 'latest'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'trades.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        if detail:
            writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
            writer.writeheader()
            writer.writerows(detail)
    lines = [
        '# Full-cycle NetPnL > 0 Direct — 一次性 Audit（NON-PIT）',
        '',
        f'完整 Round-Trips：{total}；未勝出：{bad}；比例：{audit["nonwinning_ratio_pct"]:.2f}%',
        '',
        '| 因素 | 全部交易筆數 | 未勝出交易筆數 |', '|---|---:|---:|',
    ]
    labels = {
        'null_or_nonpositive_bought': '1. NULL/非正值仍買入（必須為0）',
        'different_trade_event': '2. 訊號／實際成交事件不同',
        'different_quantity': '3. 參考與實際成交股數不同',
        'different_exit_path': '4. 出場日期／價格不同',
        'zero_actual_pnl': '5. 實際帳本零損益',
    }
    for key, label in labels.items():
        lines.append(f'| {label} | {audit["factor_totals"][key]} | {audit["nonwin_factor_totals"][key]} |')
    lines += [
        '', '## 實際結果 × 五因素（各欄可重疊，不能橫向加總）', '',
        '| 實際結果 | 交易數 | NULL/非正值 | 事件不同 | 股數不同 | 出場不同 | 零損益 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for outcome, label in (('profit', '獲利'), ('loss', '虧損'), ('zero', '零損益')):
        parts = by_result[outcome]
        count = sum(row['actual_result'] == outcome for row in detail)
        lines.append(f'| {label} | {count} | ' + ' | '.join(str(parts[factor]) for factor in audited_factors) + ' |')
    lines += [
        '', f'未勝出且沒有觀察到 NULL／事件／股數／出場差異：{unresolved} 筆（包含可能為零損益者；不得視為已完成因果解釋）。',
        '', '## 四 Seed 明細', '', '| Seed | 交易 | 勝率 | 虧損 | 零損益 |',
        '|---|---:|---:|---:|---:|',
    ]
    for seed, item in breakdown.items():
        c = item['counts']
        lines.append(f'| {seed} | {item["round_trips"]} | {item["win_rate_pct"]:.2f}% | {c.get("result_loss",0)} | {c.get("result_zero",0)} |')
    lines += ['', '## 未勝出因素交集（互斥 pattern）', '']
    for seed, item in breakdown.items():
        lines.append(f'- Seed {seed}: {json.dumps(item["nonwin_overlaps"], ensure_ascii=False, sort_keys=True)}')
    lines += ['', '同一筆可能同時具有訊號、股數及出場差異；以上是觀察到的差異，不宣稱單一因果。',
              '參考 PnL 經原正式單股引擎重播並與已簽章 raw NetPnL 核對；Portfolio 勝負仍採正式 closed_trade_rows。',
              '本 Audit 使用未來資訊，僅診斷，不得作為 PIT 可執行條件。', '']
    (out / 'audit.md').write_text('\n'.join(lines), encoding='utf-8')
    if not quiet:
        print('\n'.join(lines), flush=True)
    return {'status': 'COMPLETED', 'report': audit, 'output': str(out / 'audit.md')}

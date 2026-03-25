import os
import sys
import json
import bisect
import argparse
from pathlib import Path

# # (AI註: 讓直接執行 tools/quantify_overfitting.py 時，也能正確 import 專案根目錄下的 core)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import optuna
from optuna.trial import TrialState

from core.v16_config import (
    V16StrategyParams,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_TRADE_WIN_RATE,
    MIN_FULL_YEAR_RETURN_PCT,
    MAX_PORTFOLIO_MDD_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
)
from core.v16_data_utils import (
    sanitize_ohlcv_dataframe,
    get_required_min_rows,
    get_max_required_min_rows,
    discover_unique_csv_inputs,
)
from core.v16_log_utils import write_issue_log, format_exception_summary
from core.v16_portfolio_engine import (
    prep_stock_data_and_trades,
    pack_prepared_stock_data,
    run_portfolio_timeline,
    calc_portfolio_score,
)


# # (AI註: 預設值集中管理，避免 magic number 散落)
DEFAULT_DATA_DIR = "tw_stock_data_vip"
DEFAULT_DB_PATH = "models/v16_portfolio_ai_10pos_overnight.db"
DEFAULT_STUDY_NAME = "v16_portfolio_optimization_overnight"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_BENCHMARK_TICKER = "0050"
DEFAULT_MAX_POSITIONS = 10
DEFAULT_ENABLE_ROTATION = False
DEFAULT_TOP_K = 20
DEFAULT_MIN_TRAIN_YEARS = 3
DEFAULT_TEST_YEARS = 1
DEFAULT_MAX_FOLDS = 6
DEFAULT_START_YEAR = 2015

# # (AI註: 複合 overfitting 指數的權重；這是診斷指標，不是交易邏輯)
OVERFIT_WEIGHT_PBO = 0.40
OVERFIT_WEIGHT_SCORE_DECAY = 0.30
OVERFIT_WEIGHT_RANK_INSTABILITY = 0.20
OVERFIT_WEIGHT_OOS_FAIL = 0.10
OVERFIT_SCORE_EPS = 1e-12


def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))


# # (AI註: 單一真理來源 - 從 Optuna trial params 重建策略參數，避免另一份 mapping 漂移)
def build_params_from_trial_params(trial_params):
    params = V16StrategyParams()
    for key, value in trial_params.items():
        if hasattr(params, key):
            setattr(params, key, value)
    return params


def load_study_candidates(db_path, study_name, top_k):
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    valid_trials = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
        and t.value is not None
        and t.value > -9000
    ]
    valid_trials.sort(key=lambda t: t.value, reverse=True)

    selected = []
    for trial in valid_trials[:top_k]:
        selected.append({
            "candidate_id": f"trial_{trial.number + 1:06d}",
            "trial_number": trial.number + 1,
            "optimizer_score": float(trial.value),
            "params": build_params_from_trial_params(trial.params),
        })
    return selected


def load_all_raw_data(data_dir, min_rows_needed):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到資料夾: {data_dir}")

    raw_cache = {}
    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    load_issue_lines = list(duplicate_file_issue_lines)

    if not csv_inputs:
        raise FileNotFoundError(f"{data_dir} 中沒有任何 csv 檔")

    for ticker, file_path in csv_inputs:
        try:
            raw_df = pd.read_csv(file_path)
            if len(raw_df) < min_rows_needed:
                load_issue_lines.append(
                    f"[資料不足] {ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {min_rows_needed} 筆"
                )
                continue

            clean_df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
            raw_cache[ticker] = clean_df

            if sanitize_stats['dropped_row_count'] > 0:
                load_issue_lines.append(
                    f"[清洗] {ticker}: 清洗移除 {sanitize_stats['dropped_row_count']} 列 "
                    f"(異常OHLCV={sanitize_stats['invalid_row_count']}, 重複日期={sanitize_stats['duplicate_date_count']})"
                )
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if is_insufficient_data_error(exc):
                load_issue_lines.append(f"[資料不足] {ticker}: {type(exc).__name__}: {exc}")
                continue
            raise RuntimeError(
                f"overfitting 原始資料載入失敗: ticker={ticker} | {format_exception_summary(exc)}"
            ) from exc

    if not raw_cache:
        raise RuntimeError("未能成功載入任何可用股票資料")

    log_path = write_issue_log("overfitting_raw_load_issues", load_issue_lines) if load_issue_lines else None
    return raw_cache, log_path


# # (AI註: 預先為單一候選參數生成全市場 prepared data，後續所有 fold 共用，避免重覆運算)
def prepare_candidate_market_data(raw_cache, params):
    all_dfs_fast = {}
    all_trade_logs = {}
    master_dates = set()
    issue_lines = []

    min_rows_needed = get_required_min_rows(params)

    for ticker, raw_df in raw_cache.items():
        try:
            if len(raw_df) < min_rows_needed:
                issue_lines.append(
                    f"[資料不足] {ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {min_rows_needed} 筆"
                )
                continue

            df = raw_df.copy()
            if len(df) < min_rows_needed:
                issue_lines.append(
                    f"[資料不足] {ticker}: 複製後資料列數不足 ({len(df)})，至少需要 {min_rows_needed} 筆"
                )
                continue

            prepared_df, logs = prep_stock_data_and_trades(df, params)
            packed = pack_prepared_stock_data(prepared_df)
            all_dfs_fast[ticker] = packed
            all_trade_logs[ticker] = sorted(logs, key=lambda x: x['exit_date'])
            master_dates.update(packed['dates'])
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if is_insufficient_data_error(exc):
                issue_lines.append(f"[資料不足] {ticker}: {type(exc).__name__}: {exc}")
                continue
            raise RuntimeError(
                f"overfitting 候選資料準備失敗: ticker={ticker} | {format_exception_summary(exc)}"
            ) from exc

    if not all_dfs_fast:
        raise RuntimeError("此組參數沒有任何可用標的可進行回測")

    sorted_dates = sorted(master_dates)
    return all_dfs_fast, all_trade_logs, sorted_dates, issue_lines


# # (AI註: 只截到 fold end，保留更早歷史，讓指標暖機與 PIT 歷史績效統計不失真)
def truncate_packed_market_data_to_end_date(fast_df, end_date):
    dates = fast_df['dates']
    last_pos = bisect.bisect_right(dates, end_date) - 1
    if last_pos < 0:
        return None

    cutoff = last_pos + 1
    truncated = {
        '_packed_market_data': True,
        'dates': dates[:cutoff],
        'date_to_pos': {dt: i for i, dt in enumerate(dates[:cutoff])},
    }

    for key, value in fast_df.items():
        if key in ('_packed_market_data', 'dates', 'date_to_pos'):
            continue
        truncated[key] = value[:cutoff]

    return truncated


def truncate_trade_logs_to_end_date(trade_logs, end_date):
    if not trade_logs:
        return []
    cutoff = bisect.bisect_right([x['exit_date'] for x in trade_logs], end_date)
    return trade_logs[:cutoff]


def truncate_sorted_dates_to_end_date(sorted_dates, end_date):
    cutoff = bisect.bisect_right(sorted_dates, end_date)
    return sorted_dates[:cutoff]


def calc_positive_decay_ratio(train_value, test_value):
    denom = max(abs(train_value), OVERFIT_SCORE_EPS)
    raw = (train_value - test_value) / denom
    return float(max(0.0, raw))


def safe_descending_percentile(rank_series, candidate_id):
    n = len(rank_series)
    if n <= 1:
        return 1.0
    rank_value = float(rank_series.loc[candidate_id])
    return float((n - rank_value) / (n - 1))


def spearman_rank_correlation_desc(train_scores, test_scores):
    if len(train_scores) < 2:
        return 0.0

    train_s = pd.Series(train_scores, dtype=float)
    test_s = pd.Series(test_scores, dtype=float)

    if train_s.nunique(dropna=False) <= 1 or test_s.nunique(dropna=False) <= 1:
        return 0.0

    train_rank = train_s.rank(ascending=False, method='average')
    test_rank = test_s.rank(ascending=False, method='average')
    corr = train_rank.corr(test_rank, method='pearson')
    if pd.isna(corr):
        return 0.0
    return float(corr)


# # (AI註: 與 optimizer 硬門檻完全一致，避免 overfitting 報告與訓練篩選口徑漂移)
def evaluate_filter_pass(metrics):
    if metrics['mdd'] > MAX_PORTFOLIO_MDD_PCT:
        return False, f"回撤過大 ({metrics['mdd']:.2f}%)"
    if metrics['annual_trades'] < MIN_ANNUAL_TRADES:
        return False, f"年化交易次數過低 ({metrics['annual_trades']:.2f})"
    if metrics['reserved_buy_fill_rate'] < MIN_BUY_FILL_RATE:
        return False, f"保留後買進成交率過低 ({metrics['reserved_buy_fill_rate']:.2f}%)"
    if metrics['annual_return_pct'] <= 0:
        return False, f"年化報酬率非正 ({metrics['annual_return_pct']:.2f}%)"
    if metrics['full_year_count'] <= 0:
        return False, "無完整年度可驗證 min{r_y}"
    if metrics['min_full_year_return_pct'] <= MIN_FULL_YEAR_RETURN_PCT:
        return False, f"完整年度最差報酬過低 ({metrics['min_full_year_return_pct']:.2f}%)"
    if metrics['win_rate'] < MIN_TRADE_WIN_RATE:
        return False, f"實戰勝率偏低 ({metrics['win_rate']:.2f}%)"
    if metrics['m_win_rate'] < MIN_MONTHLY_WIN_RATE:
        return False, f"月勝率偏低 ({metrics['m_win_rate']:.2f}%)"
    if metrics['r_squared'] < MIN_EQUITY_CURVE_R_SQUARED:
        return False, f"曲線過度震盪 (R²={metrics['r_squared']:.3f})"
    return True, "PASS"


def evaluate_period(
    all_dfs_fast,
    all_trade_logs,
    sorted_dates,
    params,
    start_year,
    end_year,
    max_positions,
    enable_rotation,
    benchmark_ticker,
):
    end_date = pd.Timestamp(f"{end_year}-12-31")
    truncated_dates = truncate_sorted_dates_to_end_date(sorted_dates, end_date)
    if len(truncated_dates) < 2:
        raise ValueError(f"{end_year} 年底前沒有足夠日期可回測")

    truncated_dfs = {}
    truncated_logs = {}
    for ticker, fast_df in all_dfs_fast.items():
        t_df = truncate_packed_market_data_to_end_date(fast_df, end_date)
        if t_df is None or len(t_df['dates']) < 2:
            continue
        truncated_dfs[ticker] = t_df
        truncated_logs[ticker] = truncate_trade_logs_to_end_date(all_trade_logs.get(ticker, []), end_date)

    if not truncated_dfs:
        raise RuntimeError(f"{end_year} 年底前沒有任何可用標的")

    benchmark_data = truncated_dfs.get(benchmark_ticker)
    pf_profile = {}
    (
        tot_ret,
        mdd,
        trade_count,
        final_eq,
        avg_exp,
        max_exp,
        bm_ret,
        bm_mdd,
        win_rate,
        pf_ev,
        pf_payoff,
        total_missed,
        total_missed_sells,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        normal_trade_count,
        extended_trade_count,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
    ) = run_portfolio_timeline(
        truncated_dfs,
        truncated_logs,
        truncated_dates,
        start_year,
        params,
        max_positions,
        enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=True,
        profile_stats=pf_profile,
        verbose=False,
    )

    score = calc_portfolio_score(
        tot_ret,
        mdd,
        m_win_rate,
        r_sq,
        annual_return_pct=annual_return_pct,
    )
    full_year_count = int(pf_profile.get('full_year_count', 0))
    min_full_year_return_pct = float(pf_profile.get('min_full_year_return_pct', 0.0))
    filter_pass, filter_reason = evaluate_filter_pass({
        'mdd': float(mdd),
        'annual_trades': float(annual_trades),
        'reserved_buy_fill_rate': float(reserved_buy_fill_rate),
        'annual_return_pct': float(annual_return_pct),
        'full_year_count': full_year_count,
        'min_full_year_return_pct': min_full_year_return_pct,
        'win_rate': float(win_rate),
        'm_win_rate': float(m_win_rate),
        'r_squared': float(r_sq),
    })

    return {
        'score': float(score),
        'tot_ret': float(tot_ret),
        'mdd': float(mdd),
        'trade_count': int(trade_count),
        'win_rate': float(win_rate),
        'pf_ev': float(pf_ev),
        'pf_payoff': float(pf_payoff),
        'final_eq': float(final_eq),
        'avg_exp': float(avg_exp),
        'max_exp': float(max_exp),
        'bm_ret': float(bm_ret),
        'bm_mdd': float(bm_mdd),
        'total_missed': int(total_missed),
        'total_missed_sells': int(total_missed_sells),
        'r_squared': float(r_sq),
        'm_win_rate': float(m_win_rate),
        'bm_r_squared': float(bm_r_sq),
        'bm_m_win_rate': float(bm_m_win_rate),
        'normal_trade_count': int(normal_trade_count),
        'extended_trade_count': int(extended_trade_count),
        'annual_trades': float(annual_trades),
        'reserved_buy_fill_rate': float(reserved_buy_fill_rate),
        'annual_return_pct': float(annual_return_pct),
        'bm_annual_return_pct': float(bm_annual_return_pct),
        'full_year_count': full_year_count,
        'min_full_year_return_pct': min_full_year_return_pct,
        'filter_pass': bool(filter_pass),
        'filter_reason': filter_reason,
    }


def build_expanding_folds(available_years, start_year, min_train_years, test_years, max_folds):
    years = sorted(set(int(y) for y in available_years if int(y) >= int(start_year)))
    if not years:
        raise ValueError("依目前設定找不到可用年份")

    anchor_start = years[0]
    last_year = years[-1]
    folds = []
    train_end = anchor_start + min_train_years - 1
    fold_id = 1

    while (train_end + test_years) <= last_year:
        test_start = train_end + 1
        test_end = test_start + test_years - 1
        folds.append({
            'fold_id': fold_id,
            'train_start': anchor_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })
        fold_id += 1
        train_end += 1

    if max_folds is not None and max_folds > 0 and len(folds) > max_folds:
        folds = folds[-max_folds:]

    if not folds:
        raise ValueError(
            f"無法建立 walk-forward folds；請檢查 start_year={start_year}, "
            f"min_train_years={min_train_years}, test_years={test_years}"
        )
    return folds


def classify_overfit_level(score_0_100):
    if score_0_100 < 25.0:
        return "LOW"
    if score_0_100 < 50.0:
        return "MILD"
    if score_0_100 < 75.0:
        return "HIGH"
    return "EXTREME"


def summarize_overfitting(fold_summary_df):
    pbo_rate = float(fold_summary_df['pbo_event'].mean()) if not fold_summary_df.empty else 0.0
    mean_score_decay = float(fold_summary_df['selected_score_decay'].mean()) if not fold_summary_df.empty else 0.0
    mean_rank_corr = float(fold_summary_df['rank_corr'].mean()) if not fold_summary_df.empty else 0.0
    selected_oos_pass_rate = float(fold_summary_df['selected_oos_pass'].mean()) if not fold_summary_df.empty else 0.0

    rank_instability = 1.0 - max(0.0, mean_rank_corr)
    oos_fail_rate = 1.0 - selected_oos_pass_rate
    composite = 100.0 * (
        OVERFIT_WEIGHT_PBO * pbo_rate +
        OVERFIT_WEIGHT_SCORE_DECAY * mean_score_decay +
        OVERFIT_WEIGHT_RANK_INSTABILITY * rank_instability +
        OVERFIT_WEIGHT_OOS_FAIL * oos_fail_rate
    )

    return {
        'pbo_rate': pbo_rate,
        'mean_selected_score_decay': mean_score_decay,
        'mean_selected_return_decay': float(fold_summary_df['selected_return_decay'].mean()) if not fold_summary_df.empty else 0.0,
        'mean_rank_corr': mean_rank_corr,
        'selected_oos_pass_rate': selected_oos_pass_rate,
        'composite_overfit_score_0_100': float(composite),
        'overfit_level': classify_overfit_level(composite),
    }


def run_overfitting_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)

    candidates = load_study_candidates(args.db_path, args.study_name, args.top_k)
    if not candidates:
        raise RuntimeError("找不到任何可分析的 Optuna 完成 trial")
    print(f"✅ 候選參數載入完成: {len(candidates)} 組")

    raw_min_rows_needed = get_max_required_min_rows([candidate['params'] for candidate in candidates])
    raw_cache, raw_log_path = load_all_raw_data(args.data_dir, min_rows_needed=raw_min_rows_needed)
    print(f"✅ 原始資料載入完成: {len(raw_cache)} 檔 (最低需求 {raw_min_rows_needed} 列)")
    if raw_log_path:
        print(f"⚠️ 原始資料清洗摘要: {raw_log_path}")

    available_years = sorted({int(dt.year) for df in raw_cache.values() for dt in df.index})
    folds = build_expanding_folds(
        available_years=available_years,
        start_year=args.start_year,
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        max_folds=args.max_folds,
    )
    print(f"✅ Walk-forward folds 建立完成: {len(folds)} 組")

    candidate_fold_rows = []
    candidate_issue_logs = []

    for idx, candidate in enumerate(candidates, start=1):
        candidate_id = candidate['candidate_id']
        params = candidate['params']
        print(f"\n[{idx}/{len(candidates)}] 準備候選參數: {candidate_id}")

        all_dfs_fast, all_trade_logs, sorted_dates, issue_lines = prepare_candidate_market_data(raw_cache, params)
        if issue_lines:
            log_path = write_issue_log(f"overfitting_candidate_{candidate_id}", issue_lines)
            if log_path:
                candidate_issue_logs.append(log_path)

        for fold in folds:
            train_metrics = evaluate_period(
                all_dfs_fast, all_trade_logs, sorted_dates, params,
                start_year=fold['train_start'], end_year=fold['train_end'],
                max_positions=args.max_positions,
                enable_rotation=args.enable_rotation,
                benchmark_ticker=args.benchmark_ticker,
            )
            test_metrics = evaluate_period(
                all_dfs_fast, all_trade_logs, sorted_dates, params,
                start_year=fold['test_start'], end_year=fold['test_end'],
                max_positions=args.max_positions,
                enable_rotation=args.enable_rotation,
                benchmark_ticker=args.benchmark_ticker,
            )

            candidate_fold_rows.append({
                'candidate_id': candidate_id,
                'trial_number': candidate['trial_number'],
                'optimizer_score': candidate['optimizer_score'],
                'fold_id': fold['fold_id'],
                'train_start': fold['train_start'],
                'train_end': fold['train_end'],
                'test_start': fold['test_start'],
                'test_end': fold['test_end'],
                'train_score': train_metrics['score'],
                'train_ret_pct': train_metrics['tot_ret'],
                'train_mdd_pct': train_metrics['mdd'],
                'train_trade_count': train_metrics['trade_count'],
                'train_win_rate_pct': train_metrics['win_rate'],
                'train_ev': train_metrics['pf_ev'],
                'train_r_squared': train_metrics['r_squared'],
                'train_m_win_rate_pct': train_metrics['m_win_rate'],
                'train_annual_return_pct': train_metrics['annual_return_pct'],
                'train_full_year_count': train_metrics['full_year_count'],
                'train_filter_pass': train_metrics['filter_pass'],
                'test_score': test_metrics['score'],
                'test_ret_pct': test_metrics['tot_ret'],
                'test_mdd_pct': test_metrics['mdd'],
                'test_trade_count': test_metrics['trade_count'],
                'test_win_rate_pct': test_metrics['win_rate'],
                'test_ev': test_metrics['pf_ev'],
                'test_r_squared': test_metrics['r_squared'],
                'test_m_win_rate_pct': test_metrics['m_win_rate'],
                'test_annual_return_pct': test_metrics['annual_return_pct'],
                'test_full_year_count': test_metrics['full_year_count'],
                'test_filter_pass': test_metrics['filter_pass'],
                'test_filter_reason': test_metrics['filter_reason'],
                'score_decay_ratio': calc_positive_decay_ratio(train_metrics['score'], test_metrics['score']),
                'return_decay_ratio': calc_positive_decay_ratio(train_metrics['tot_ret'], test_metrics['tot_ret']),
            })

    candidate_fold_df = pd.DataFrame(candidate_fold_rows)
    if candidate_fold_df.empty:
        raise RuntimeError("沒有產生任何 fold 分析結果")

    fold_summary_rows = []
    for fold_id, fold_df in candidate_fold_df.groupby('fold_id', sort=True):
        fold_df = fold_df.sort_values(['train_score', 'candidate_id'], ascending=[False, True]).copy()
        selected_row = fold_df.iloc[0].copy()

        train_rank = fold_df['train_score'].rank(ascending=False, method='average')
        test_rank = fold_df['test_score'].rank(ascending=False, method='average')
        fold_df.index = fold_df['candidate_id']
        train_rank.index = fold_df.index
        test_rank.index = fold_df.index

        selected_candidate_id = selected_row['candidate_id']
        selected_test_rank_pct = safe_descending_percentile(test_rank, selected_candidate_id)
        selected_train_rank_pct = safe_descending_percentile(train_rank, selected_candidate_id)
        rank_corr = spearman_rank_correlation_desc(
            fold_df['train_score'].to_numpy(dtype=float),
            fold_df['test_score'].to_numpy(dtype=float),
        )

        pbo_event = bool(selected_test_rank_pct < 0.50)
        fold_summary_rows.append({
            'fold_id': int(fold_id),
            'train_start': int(selected_row['train_start']),
            'train_end': int(selected_row['train_end']),
            'test_start': int(selected_row['test_start']),
            'test_end': int(selected_row['test_end']),
            'candidate_count': int(len(fold_df)),
            'selected_candidate_id': selected_candidate_id,
            'selected_trial_number': int(selected_row['trial_number']),
            'selected_optimizer_score': float(selected_row['optimizer_score']),
            'selected_train_score': float(selected_row['train_score']),
            'selected_test_score': float(selected_row['test_score']),
            'selected_train_ret_pct': float(selected_row['train_ret_pct']),
            'selected_test_ret_pct': float(selected_row['test_ret_pct']),
            'selected_score_decay': float(selected_row['score_decay_ratio']),
            'selected_return_decay': float(selected_row['return_decay_ratio']),
            'selected_train_rank_pct': float(selected_train_rank_pct),
            'selected_test_rank_pct': float(selected_test_rank_pct),
            'selected_oos_pass': bool(selected_row['test_filter_pass']),
            'selected_oos_fail_reason': str(selected_row['test_filter_reason']),
            'rank_corr': float(rank_corr),
            'pbo_event': pbo_event,
        })

    fold_summary_df = pd.DataFrame(fold_summary_rows).sort_values('fold_id').reset_index(drop=True)
    summary = summarize_overfitting(fold_summary_df)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    candidate_csv = os.path.join(args.output_dir, f"overfitting_candidate_fold_metrics_{ts}.csv")
    fold_csv = os.path.join(args.output_dir, f"overfitting_fold_summary_{ts}.csv")
    report_json = os.path.join(args.output_dir, f"overfitting_report_{ts}.json")

    candidate_fold_df.sort_values(
        ['fold_id', 'train_score', 'candidate_id'],
        ascending=[True, False, True]
    ).to_csv(candidate_csv, index=False, encoding='utf-8-sig')
    fold_summary_df.to_csv(fold_csv, index=False, encoding='utf-8-sig')

    report_payload = {
        'methodology': {
            'selection_universe': 'Top-K Optuna completed trials by in-sample optimizer score',
            'fold_mode': 'anchored_expanding_walk_forward',
            'top_k': int(args.top_k),
            'start_year': int(args.start_year),
            'min_train_years': int(args.min_train_years),
            'test_years': int(args.test_years),
            'max_folds': int(args.max_folds),
            'max_positions': int(args.max_positions),
            'enable_rotation': bool(args.enable_rotation),
            'benchmark_ticker': args.benchmark_ticker,
            'composite_formula': {
                'pbo_weight': OVERFIT_WEIGHT_PBO,
                'score_decay_weight': OVERFIT_WEIGHT_SCORE_DECAY,
                'rank_instability_weight': OVERFIT_WEIGHT_RANK_INSTABILITY,
                'oos_fail_weight': OVERFIT_WEIGHT_OOS_FAIL,
            },
        },
        'summary': summary,
        'folds': fold_summary_df.to_dict(orient='records'),
        'candidate_issue_logs': candidate_issue_logs,
        'raw_load_issue_log': raw_log_path,
        'outputs': {
            'candidate_fold_metrics_csv': candidate_csv,
            'fold_summary_csv': fold_csv,
            'report_json': report_json,
        },
    }
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    return report_payload


def build_arg_parser():
    parser = argparse.ArgumentParser(description="量化評估 Optuna 交易策略 overfitting 程度")
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--db-path', default=DEFAULT_DB_PATH)
    parser.add_argument('--study-name', default=DEFAULT_STUDY_NAME)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--benchmark-ticker', default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument('--max-positions', type=int, default=DEFAULT_MAX_POSITIONS)
    parser.add_argument('--enable-rotation', action='store_true', default=DEFAULT_ENABLE_ROTATION)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--start-year', type=int, default=DEFAULT_START_YEAR)
    parser.add_argument('--min-train-years', type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument('--test-years', type=int, default=DEFAULT_TEST_YEARS)
    parser.add_argument('--max-folds', type=int, default=DEFAULT_MAX_FOLDS)
    return parser


def print_console_summary(report_payload):
    summary = report_payload['summary']
    print("\n================================================================================")
    print("📊 Overfitting 量化摘要")
    print("================================================================================")
    print(f"PBO rate                 : {summary['pbo_rate'] * 100:.2f}%")
    print(f"Selected score decay     : {summary['mean_selected_score_decay'] * 100:.2f}%")
    print(f"Selected return decay    : {summary['mean_selected_return_decay'] * 100:.2f}%")
    print(f"Mean rank correlation    : {summary['mean_rank_corr']:.4f}")
    print(f"Selected OOS pass rate   : {summary['selected_oos_pass_rate'] * 100:.2f}%")
    print(f"Composite overfit score  : {summary['composite_overfit_score_0_100']:.2f} / 100")
    print(f"Overfit level            : {summary['overfit_level']}")
    print("================================================================================")
    print(f"Fold summary CSV         : {report_payload['outputs']['fold_summary_csv']}")
    print(f"Candidate metrics CSV    : {report_payload['outputs']['candidate_fold_metrics_csv']}")
    print(f"JSON report              : {report_payload['outputs']['report_json']}")


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    report = run_overfitting_analysis(args)
    print_console_summary(report)
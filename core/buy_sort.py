import math

from core.config import get_buy_sort_method


BUY_LIMIT_OVERAGE_SORT_METHOD = 'BUY_LIMIT_OVERAGE_THEN_PROJ_COST'
ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD = 'ENTRY_TYPE_THEN_PROJ_COST'

BREAKOUT_QUALITY_RANKING_POLICY_SCORE = 'score'
BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED = 'capital-adjusted-score'
BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET = 'capital-bucket-then-score'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY = 'resource-aware-binary'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET = 'resource-aware-binary-basket'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS = 'resource-aware-continuous'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING = (
    'resource-aware-continuous-capital-preserving'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL = (
    'resource-aware-continuous-max-dl'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT = (
    'resource-aware-continuous-max-dl-feasible-ascent'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD = (
    'resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT = (
    'resource-aware-continuous-expected-pnl-feasible-ascent'
)
RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES = frozenset({
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
})
SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES = (
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    *sorted(RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES),
)
BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT = 3


def _as_finite_float(value, *, default):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric_value):
        return default
    return numeric_value


def calc_buy_limit_overage_pct(prev_close, limit_price):
    resolved_prev_close = _as_finite_float(prev_close, default=math.inf)
    resolved_limit_price = _as_finite_float(limit_price, default=math.nan)
    if not math.isfinite(resolved_prev_close) or not math.isfinite(resolved_limit_price) or resolved_limit_price <= 0:
        return math.inf
    return max(0.0, (resolved_prev_close / resolved_limit_price - 1.0) * 100.0)


def calc_buy_limit_overage_pct_from_row(row):
    if row is None:
        return math.inf
    explicit_value = row.get('buy_limit_overage_pct')
    if explicit_value is not None:
        return _as_finite_float(explicit_value, default=math.inf)
    if row.get('prev_close') is not None or row.get('limit_price') is not None or row.get('limit_px') is not None:
        return calc_buy_limit_overage_pct(
            row.get('prev_close'),
            row.get('limit_price') if row.get('limit_price') is not None else row.get('limit_px'),
        )
    return _as_finite_float(row.get('sort_value'), default=math.inf)


def calc_entry_type_priority(candidate_type):
    normalized = str(candidate_type or '').strip().lower()
    if normalized in {'normal', 'buy', 'reentry'}:
        return 0
    if normalized in {'extended', 'extended_tbd', 'continuation'}:
        return 1
    return 2


def calc_entry_type_priority_from_row(row):
    if row is None:
        return 2
    return calc_entry_type_priority(
        row.get('type')
        or row.get('entry_source')
        or row.get('kind')
    )


def _descending_text_key(value):
    text = str(value or '')
    return tuple(-ord(ch) for ch in text)


def is_buy_limit_overage_sort(method=None):
    active_method = get_buy_sort_method() if method is None else method
    return active_method == BUY_LIMIT_OVERAGE_SORT_METHOD


def calc_buy_sort_value(method, ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0, prev_close=None, limit_price=None):
    if method == 'EV':
        return float(ev)
    if method == 'PROJ_COST':
        return float(proj_cost)
    if method == 'HIST_WIN_X_TRADES':
        return float(win_rate) * float(trade_count)
    if method == 'ASSET_GROWTH':
        return float(asset_growth_pct)
    if method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return calc_buy_limit_overage_pct(prev_close, limit_price)
    if method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return calc_buy_limit_overage_pct(prev_close, limit_price)
    raise ValueError(f"未知的 BUY_SORT_METHOD: {method}")


def calc_active_buy_sort_value(ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0, prev_close=None, limit_price=None):
    return calc_buy_sort_value(
        get_buy_sort_method(),
        ev,
        proj_cost,
        win_rate,
        trade_count,
        asset_growth_pct,
        prev_close=prev_close,
        limit_price=limit_price,
    )


def get_buy_sort_title(method=None):
    active_method = get_buy_sort_method() if method is None else method
    if active_method == 'EV':
        return '按期望值 (EV) 由大到小排序'
    if active_method == 'PROJ_COST':
        return '按預估投入資金由大到小排序'
    if active_method == 'HIST_WIN_X_TRADES':
        return '按歷史勝率 × 交易次數由大到小排序'
    if active_method == 'ASSET_GROWTH':
        return '按資產成長由大到小排序'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return '按買入限價超出幅度由小到大，再按預估投入資金排序'
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return '按新突破/Re-entry優先，再按買入限價超出幅度由小到大排序'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def get_buy_sort_metric_label(method=None):
    active_method = get_buy_sort_method() if method is None else method
    if active_method == 'EV':
        return 'EV'
    if active_method == 'PROJ_COST':
        return '預估投入'
    if active_method == 'HIST_WIN_X_TRADES':
        return '勝率×次數'
    if active_method == 'ASSET_GROWTH':
        return '資產成長'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return '超限幅'
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return '超限幅'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def format_buy_sort_metric_value(value, method=None):
    active_method = get_buy_sort_method() if method is None else method
    numeric_value = float(value)
    if active_method in {BUY_LIMIT_OVERAGE_SORT_METHOD, ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD} and not math.isfinite(numeric_value):
        return 'N/A'
    if active_method == 'EV':
        return f'{numeric_value:.2f}R'
    if active_method == 'PROJ_COST':
        return f'{numeric_value:,.0f}'
    if active_method == 'HIST_WIN_X_TRADES':
        return f'{numeric_value:.2f}'
    if active_method == 'ASSET_GROWTH':
        return f'{numeric_value:.2f}%'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return f'{numeric_value:.2f}%'
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return f'{numeric_value:.2f}%'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def calc_projected_capital_metrics(*, proj_cost, sizing_capital, max_position_cap_pct):
    """Derive capital deployment from the canonical pre-trade sizing result."""

    cost = _as_finite_float(proj_cost, default=math.nan)
    capital = _as_finite_float(sizing_capital, default=math.nan)
    cap_fraction = _as_finite_float(max_position_cap_pct, default=math.nan)
    if (
        not math.isfinite(cost)
        or not math.isfinite(capital)
        or not math.isfinite(cap_fraction)
        or cost < 0.0
        or capital <= 0.0
        or cap_fraction <= 0.0
    ):
        return None, None
    projected_fraction = cost / capital
    deployment_rate = min(1.0, max(0.0, projected_fraction / cap_fraction))
    return float(projected_fraction), float(deployment_rate)


def _ranking_enabled_for_rows(rows):
    flags = {bool(item.get("use_breakout_quality_ranking", False)) for item in rows}
    if len(flags) > 1:
        raise ValueError("同一候選集合的 use_breakout_quality_ranking 不一致")
    return bool(flags and True in flags)


def resolve_breakout_quality_ranking_policy(rows):
    policies = {
        str(
            item.get("breakout_quality_ranking_policy")
            or BREAKOUT_QUALITY_RANKING_POLICY_SCORE
        ).strip()
        for item in rows
    }
    if len(policies) > 1:
        raise ValueError("同一候選集合的 breakout_quality_ranking_policy 不一致")
    policy = next(iter(policies), BREAKOUT_QUALITY_RANKING_POLICY_SCORE)
    if policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"不支援的 breakout-quality ranking policy: {policy!r}")
    return policy


def _quality_score_parts(item):
    rank_payload = item.get("breakout_quality_rank")
    explicitly_unavailable = (
        isinstance(rank_payload, dict)
        and not bool(rank_payload.get("available", False))
    )
    score = _as_finite_float(item.get("breakout_quality_score"), default=math.nan)
    if explicitly_unavailable or not math.isfinite(score):
        return False, 0.0
    return True, float(score)


def _quality_score_desc_key(item):
    available, score = _quality_score_parts(item)
    if not available:
        # (AI註: 缺少PIT Score不是REJECT，也不得填0；排在有效Score後再完整沿用原buy-sort。)
        return (1, 0.0)
    return (0, -score)


def _capital_deployment_rate(item):
    explicit = _as_finite_float(
        item.get("projected_capital_deployment_rate"), default=math.nan
    )
    if math.isfinite(explicit) and 0.0 <= explicit <= 1.0:
        return float(explicit)
    projected_fraction, deployment_rate = calc_projected_capital_metrics(
        proj_cost=item.get("proj_cost"),
        sizing_capital=item.get("sizing_capital"),
        max_position_cap_pct=item.get("max_position_cap_pct"),
    )
    if projected_fraction is None or deployment_rate is None:
        raise ValueError(
            "capital-aware ranking候選缺少正式sizing資本欄位："
            f"ticker={item.get('ticker')}, proj_cost={item.get('proj_cost')}, "
            f"sizing_capital={item.get('sizing_capital')}, "
            f"max_position_cap_pct={item.get('max_position_cap_pct')}"
        )
    return float(deployment_rate)


def _linear_quantile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile values不可為空")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_breakout_quality_ranking_prefixes(rows):
    """Build deterministic quality-ranking prefixes for one candidate set."""

    rows = list(rows or [])
    if not rows or not _ranking_enabled_for_rows(rows):
        return {id(item): () for item in rows}
    policy = resolve_breakout_quality_ranking_policy(rows)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_SCORE:
        return {id(item): _quality_score_desc_key(item) for item in rows}
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED:
        prefixes = {}
        for item in rows:
            available, score = _quality_score_parts(item)
            prefixes[id(item)] = (
                (1, 0.0)
                if not available
                else (0, -(score * _capital_deployment_rate(item)))
            )
        return prefixes
    if policy in RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES:
        # (AI註: Resource-aware quality只在盤前資源瓶頸判定後介入；候選建立階段必須完整保留原Min ROOS順序。)
        return {id(item): () for item in rows}
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET:
        available_rows = [item for item in rows if _quality_score_parts(item)[0]]
        rates = [_capital_deployment_rate(item) for item in available_rows]
        if rates:
            lower_boundary = _linear_quantile(rates, 1.0 / BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT)
            upper_boundary = _linear_quantile(rates, 2.0 / BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT)
        else:
            lower_boundary = upper_boundary = 0.0
        prefixes = {}
        for item in rows:
            available, score = _quality_score_parts(item)
            if not available:
                prefixes[id(item)] = (1, 0, 0.0)
                continue
            rate = _capital_deployment_rate(item)
            bucket_priority = 0 if rate >= upper_boundary else 1 if rate >= lower_boundary else 2
            prefixes[id(item)] = (0, bucket_priority, -score)
        return prefixes
    raise ValueError(f"不支援的 breakout-quality ranking policy: {policy!r}")


def sort_candidate_rows(rows, method=None):
    active_method = get_buy_sort_method() if method is None else method
    quality_ranking = _ranking_enabled_for_rows(rows) if rows else False
    ranking_policy = (
        resolve_breakout_quality_ranking_policy(rows) if quality_ranking else None
    )
    resource_aware_quality = (
        ranking_policy in RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES
    )
    quality_prefixes = (
        build_breakout_quality_ranking_prefixes(rows)
        if quality_ranking and not resource_aware_quality
        else {}
    )
    quality_ranking_for_sort = bool(quality_ranking and not resource_aware_quality)

    def quality_prefix(item):
        return quality_prefixes.get(id(item), ())

    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        if quality_ranking_for_sort:
            rows.sort(
                key=lambda item: (
                    *quality_prefix(item),
                    _as_finite_float(item.get('sort_value'), default=math.inf),
                    -_as_finite_float(item.get('proj_cost'), default=0.0),
                    _descending_text_key(item.get('ticker')),
                )
            )
        else:
            rows.sort(
                key=lambda item: (
                    _as_finite_float(item.get('sort_value'), default=math.inf),
                    -_as_finite_float(item.get('proj_cost'), default=0.0),
                    _descending_text_key(item.get('ticker')),
                )
            )
        return rows
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        if quality_ranking_for_sort:
            rows.sort(
                key=lambda item: (
                    *quality_prefix(item),
                    calc_entry_type_priority_from_row(item),
                    calc_buy_limit_overage_pct_from_row(item),
                    str(item.get('ticker') or ''),
                )
            )
        else:
            rows.sort(
                key=lambda item: (
                    calc_entry_type_priority_from_row(item),
                    calc_buy_limit_overage_pct_from_row(item),
                    str(item.get('ticker') or ''),
                )
            )
        return rows
    if quality_ranking_for_sort:
        rows.sort(
            key=lambda item: (
                *quality_prefix(item),
                -_as_finite_float(item.get('sort_value'), default=-math.inf),
                _descending_text_key(item.get('ticker')),
            )
        )
    else:
        rows.sort(
            key=lambda item: (
                _as_finite_float(item.get('sort_value'), default=-math.inf),
                str(item.get('ticker') or ''),
            ),
            reverse=True,
        )
    return rows


def is_sort_value_better(candidate_value, incumbent_value, method=None):
    active_method = get_buy_sort_method() if method is None else method
    smaller_is_better = active_method in {BUY_LIMIT_OVERAGE_SORT_METHOD, ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD}
    candidate_numeric = _as_finite_float(candidate_value, default=math.inf if smaller_is_better else -math.inf)
    incumbent_numeric = _as_finite_float(incumbent_value, default=math.inf if smaller_is_better else -math.inf)
    if smaller_is_better:
        return candidate_numeric < incumbent_numeric
    return candidate_numeric > incumbent_numeric

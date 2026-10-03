import bisect


FAST_STATIC_FLOAT_FIELDS = ('Open', 'High', 'Low', 'Close', 'Volume')
FAST_DYNAMIC_FLOAT_FIELDS = ('ATR', 'buy_limit')
FAST_DYNAMIC_BOOL_FIELDS = ('is_setup', 'ind_sell_signal')
FAST_DYNAMIC_INT_FIELDS = ()
FAST_FLOAT_FIELDS = FAST_STATIC_FLOAT_FIELDS + FAST_DYNAMIC_FLOAT_FIELDS
FAST_BOOL_FIELDS = FAST_DYNAMIC_BOOL_FIELDS
FAST_INT_FIELDS = FAST_DYNAMIC_INT_FIELDS


def is_packed_market_data(data):
    return isinstance(data, dict) and data.get('_packed_market_data', False) is True


def get_fast_security_profile(data):
    if is_packed_market_data(data):
        return data.get('security_profile')
    return None


def get_fast_dates(data):
    if is_packed_market_data(data):
        return data['dates']
    return tuple(sorted(data.keys()))


def get_fast_pos(data, date):
    if is_packed_market_data(data):
        return data['date_to_pos'].get(date, -1)
    if date not in data:
        return -1
    d_list = get_fast_dates(data)
    return bisect.bisect_left(d_list, date)


def has_fast_date(data, date):
    if is_packed_market_data(data):
        return date in data['date_to_pos']
    return date in data


def get_fast_value(data, field, *, pos=None, date=None):
    if is_packed_market_data(data):
        if pos is None:
            pos = data['date_to_pos'].get(date, -1)
        if pos < 0:
            raise KeyError(date)
        value = data[field][pos]
        if field in FAST_BOOL_FIELDS:
            return bool(value)
        if field in FAST_INT_FIELDS:
            return int(value)
        return float(value)
    if date is None:
        raise KeyError('dict market data 需要 date')
    return data[date][field]


def get_fast_close(data, *, pos=None, date=None):
    return get_fast_value(data, 'Close', pos=pos, date=date)


def get_fast_normal_signal_values(data, y_pos):
    """Return buy_limit, ATR, close and signal date for one completed bar."""
    if is_packed_market_data(data):
        return (
            float(data['buy_limit'][y_pos]),
            float(data['ATR'][y_pos]),
            float(data['Close'][y_pos]),
            data['dates'][y_pos],
        )
    dates = get_fast_dates(data)
    y_date = dates[y_pos]
    row = data[y_date]
    return row['buy_limit'], row['ATR'], row['Close'], y_date


def get_fast_extended_signal_values(data, t_pos):
    """Return previous close and information date for an extended candidate day."""
    y_pos = int(t_pos) - 1
    if is_packed_market_data(data):
        return float(data['Close'][y_pos]), data['dates'][y_pos]
    dates = get_fast_dates(data)
    y_date = dates[y_pos]
    return data[y_date]['Close'], y_date


def get_fast_entry_session_values(data, t_pos, y_pos=None):
    """Return current OHLCV plus previous close with one packed-data dispatch."""
    y_pos = int(t_pos) - 1 if y_pos is None else int(y_pos)
    if is_packed_market_data(data):
        return (
            float(data['Open'][t_pos]),
            float(data['High'][t_pos]),
            float(data['Low'][t_pos]),
            float(data['Close'][t_pos]),
            float(data['Volume'][t_pos]),
            float(data['Close'][y_pos]),
        )
    dates = get_fast_dates(data)
    t_row = data[dates[t_pos]]
    y_row = data[dates[y_pos]]
    return (
        t_row['Open'], t_row['High'], t_row['Low'], t_row['Close'],
        t_row['Volume'], y_row['Close'],
    )


def get_fast_previous_management_values(data, t_pos):
    """Return only completed previous-bar fields needed before today's execution."""
    y_pos = int(t_pos) - 1
    if is_packed_market_data(data):
        return (
            float(data['ATR'][y_pos]),
            bool(data['ind_sell_signal'][y_pos]),
            float(data['Close'][y_pos]),
            float(data['High'][y_pos]),
        )
    dates = get_fast_dates(data)
    y_row = data[dates[y_pos]]
    return y_row['ATR'], bool(y_row['ind_sell_signal']), y_row['Close'], y_row['High']


def get_fast_current_high_low_values(data, t_pos):
    """Return current High/Low only; used after pre-market decisions are frozen."""
    if is_packed_market_data(data):
        return float(data['High'][t_pos]), float(data['Low'][t_pos])
    dates = get_fast_dates(data)
    t_row = data[dates[t_pos]]
    return t_row['High'], t_row['Low']


def get_fast_current_session_values(data, t_pos):
    """Return current OHLCV without re-reading any previous-bar fields."""
    if is_packed_market_data(data):
        return (
            float(data['Open'][t_pos]),
            float(data['High'][t_pos]),
            float(data['Low'][t_pos]),
            float(data['Close'][t_pos]),
            float(data['Volume'][t_pos]),
        )
    dates = get_fast_dates(data)
    t_row = data[dates[t_pos]]
    return t_row['Open'], t_row['High'], t_row['Low'], t_row['Close'], t_row['Volume']


def get_fast_management_session_values(data, t_pos):
    """Return all current/previous fields required by canonical position management."""
    y_pos = int(t_pos) - 1
    if is_packed_market_data(data):
        return (
            float(data['ATR'][y_pos]),
            bool(data['ind_sell_signal'][y_pos]),
            float(data['Close'][y_pos]),
            float(data['High'][y_pos]),
            float(data['Open'][t_pos]),
            float(data['High'][t_pos]),
            float(data['Low'][t_pos]),
            float(data['Close'][t_pos]),
            float(data['Volume'][t_pos]),
        )
    dates = get_fast_dates(data)
    t_row = data[dates[t_pos]]
    y_row = data[dates[y_pos]]
    return (
        y_row['ATR'], bool(y_row['ind_sell_signal']), y_row['Close'], y_row['High'],
        t_row['Open'], t_row['High'], t_row['Low'], t_row['Close'], t_row['Volume'],
    )


def get_fast_live_shadow_cleanup_values(data, t_pos):
    """Return only fields needed by the common live-shadow cleanup path.

    For a non-indicator day the canonical fast cleanup only consumes current
    high/low and previous high/ATR.  The remaining execution fields are read
    lazily by the caller only when an indicator exit is present.
    """
    y_pos = int(t_pos) - 1
    if is_packed_market_data(data):
        return (
            float(data['High'][t_pos]),
            float(data['Low'][t_pos]),
            float(data['High'][y_pos]),
            float(data['ATR'][y_pos]),
            bool(data['ind_sell_signal'][y_pos]),
        )
    dates = get_fast_dates(data)
    t_row = data[dates[t_pos]]
    y_row = data[dates[y_pos]]
    return t_row['High'], t_row['Low'], y_row['High'], y_row['ATR'], bool(y_row['ind_sell_signal'])


def get_fast_extended_day_market_values(data, date):
    """Return the PIT-safe extended-candidate market snapshot for one day.

    Candidate construction is a pre-market phase.  This helper intentionally reads
    only the completed previous bar plus the current-date position.  It MUST NOT
    touch today's OHLCV.  The fifth tuple slot is reserved for a current-session
    snapshot that may be attached later by the execution phase, after those values
    are legitimately consumed.

    Returns ``(t_pos, previous_values, information_date, security_profile,
    current_session_values)`` where ``previous_values`` matches
    :func:`get_fast_previous_management_values`.  ``current_session_values`` is
    always ``None`` on this pre-market path.
    """
    if is_packed_market_data(data):
        t_pos = int(data['date_to_pos'].get(date, -1))
        if t_pos <= 0:
            return t_pos, None, None, data.get('security_profile'), None
        y_pos = t_pos - 1
        previous_values = (
            float(data['ATR'][y_pos]),
            bool(data['ind_sell_signal'][y_pos]),
            float(data['Close'][y_pos]),
            float(data['High'][y_pos]),
        )
        return (
            t_pos,
            previous_values,
            data['dates'][y_pos],
            data.get('security_profile'),
            None,
        )

    t_pos = get_fast_pos(data, date)
    if t_pos <= 0:
        return t_pos, None, None, None, None
    dates = get_fast_dates(data)
    return (
        int(t_pos),
        get_fast_previous_management_values(data, t_pos),
        dates[t_pos - 1],
        get_fast_security_profile(data),
        None,
    )


# # (AI註: benchmark 起算必須與模擬起點對齊；若起點當日無資料，取起點當日或之前最近一筆收盤價，避免之後才開始算 benchmark)
def get_fast_close_on_or_before(data, date):
    dates = get_fast_dates(data)
    pos = bisect.bisect_right(dates, date) - 1
    if pos < 0:
        return None, None

    anchor_date = dates[pos]
    return get_fast_close(data, date=anchor_date), anchor_date

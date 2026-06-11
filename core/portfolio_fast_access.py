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


# # (AI註: benchmark 起算必須與模擬起點對齊；若起點當日無資料，取起點當日或之前最近一筆收盤價，避免之後才開始算 benchmark)
def get_fast_close_on_or_before(data, date):
    dates = get_fast_dates(data)
    pos = bisect.bisect_right(dates, date) - 1
    if pos < 0:
        return None, None

    anchor_date = dates[pos]
    return get_fast_close(data, date=anchor_date), anchor_date

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Optimizer-only native no-event position-management probe.
 *
 * This kernel intentionally handles only the common bar where there is no
 * pending exit and no indicator sell.  It computes the exact management state
 * that the Python SSOT would reach and tells the caller whether STOP/TP is hit.
 * If an exit is hit, Python falls back to the canonical transition without
 * mutating state; only the no-event case is committed from this result.
 *
 * No fast-math is allowed.  Monetary/price state stays integer milli-units.
 */

static inline int64_t price_to_milli_double(double value) {
    double scaled = value * 1000.0;
    if (scaled >= 0.0) {
        return (int64_t)floor(scaled + 0.5);
    }
    return (int64_t)ceil(scaled - 0.5);
}

static int resolve_tick_milli(double value, int is_fund, int64_t *tick_out) {
    if (!isfinite(value)) {
        return 0;
    }

    if (is_fund) {
        const double threshold = 50.0;
        if (fabs(value - threshold) <= 1e-12) {
            return 0; /* Python Decimal fallback boundary. */
        }
        *tick_out = (value < threshold) ? 10 : 50;
        return 1;
    }

    static const double thresholds[] = {1.0, 10.0, 50.0, 100.0, 500.0, 1000.0};
    static const int64_t ticks[] = {1, 10, 50, 100, 500, 1000};
    for (int i = 0; i < 6; ++i) {
        if (fabs(value - thresholds[i]) <= 1e-12) {
            return 0; /* Preserve Python Decimal boundary behavior via fallback. */
        }
        if (value < thresholds[i]) {
            *tick_out = ticks[i];
            return 1;
        }
    }
    *tick_out = 5000;
    return 1;
}

static int round_up_price_to_tick_milli(double value, int is_fund, int64_t *result_out) {
    int64_t tick_milli = 0;
    if (!resolve_tick_milli(value, is_fund, &tick_milli)) {
        return 0;
    }
    const double scaled = value * 1000.0;
    const double ratio = scaled / (double)tick_milli;
    const double rounded_ratio = ceil(ratio - 1e-12);
    if (!isfinite(rounded_ratio)) {
        return 0;
    }
    *result_out = (int64_t)rounded_ratio * tick_milli;
    return 1;
}

static int get_double_buffer(PyObject *obj, Py_buffer *view) {
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0) {
        return 0;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(double) || view->format == NULL || view->format[0] != 'd') {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError, "expected contiguous float64 buffer");
        return 0;
    }
    return 1;
}


static int get_writable_double_buffer(PyObject *obj, Py_buffer *view) {
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG | PyBUF_FORMAT | PyBUF_WRITABLE) != 0) {
        return 0;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(double) || view->format == NULL || view->format[0] != 'd') {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError, "expected writable contiguous float64 buffer");
        return 0;
    }
    return 1;
}

/*
 * Exact shifted rolling maximum used by optimizer breakout features.
 * out[j] = max(source[j-window:j]) for j >= window, NaN otherwise.
 * A NaN anywhere inside the full window preserves pandas min_periods=window
 * semantics by producing NaN for that output.  The operation uses only
 * comparisons, so finite outputs are bit-identical to the source values.
 */
static PyObject *native_rolling_max_shift1_into(PyObject *self, PyObject *args) {
    PyObject *source_obj, *out_obj;
    Py_ssize_t window;
    Py_buffer source_view = {0}, out_view = {0};
    Py_ssize_t *deque = NULL;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOn", &source_obj, &out_obj, &window)) {
        return NULL;
    }
    if (window <= 0) {
        PyErr_SetString(PyExc_ValueError, "window must be positive");
        return NULL;
    }
    if (!get_double_buffer(source_obj, &source_view)) goto fail;
    if (!get_writable_double_buffer(out_obj, &out_view)) goto fail;

    const Py_ssize_t n = source_view.len / source_view.itemsize;
    const Py_ssize_t n_out = out_view.len / out_view.itemsize;
    if (n_out != n) {
        PyErr_SetString(PyExc_ValueError, "source and output lengths must match");
        goto fail;
    }

    const double *source = (const double *)source_view.buf;
    double *out = (double *)out_view.buf;
    for (Py_ssize_t i = 0; i < n; ++i) {
        out[i] = NAN;
    }
    if (n == 0 || window >= n) {
        PyBuffer_Release(&out_view);
        PyBuffer_Release(&source_view);
        Py_RETURN_NONE;
    }

    deque = (Py_ssize_t *)PyMem_Malloc((size_t)n * sizeof(Py_ssize_t));
    if (deque == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    Py_ssize_t head = 0, tail = 0;
    Py_ssize_t nan_count = 0;

    for (Py_ssize_t i = 0; i < n; ++i) {
        const double value = source[i];
        if (isnan(value)) {
            nan_count += 1;
        } else {
            while (tail > head && source[deque[tail - 1]] <= value) {
                tail -= 1;
            }
            deque[tail++] = i;
        }

        if (i >= window) {
            const Py_ssize_t outgoing = i - window;
            if (isnan(source[outgoing])) {
                nan_count -= 1;
            }
            if (tail > head && deque[head] == outgoing) {
                head += 1;
            }
        }

        const Py_ssize_t out_index = i + 1;
        if (out_index < n && out_index >= window && nan_count == 0 && tail > head) {
            out[out_index] = source[deque[head]];
        }
    }

    PyMem_Free(deque);
    PyBuffer_Release(&out_view);
    PyBuffer_Release(&source_view);
    Py_RETURN_NONE;

fail:
    if (deque != NULL) PyMem_Free(deque);
    if (out_view.obj) PyBuffer_Release(&out_view);
    if (source_view.obj) PyBuffer_Release(&source_view);
    return NULL;
}



static int get_writable_byte_buffer(PyObject *obj, Py_buffer *view) {
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG | PyBUF_FORMAT | PyBUF_WRITABLE) != 0) {
        return 0;
    }
    if (view->itemsize != 1) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError, "expected writable contiguous 1-byte buffer");
        return 0;
    }
    return 1;
}

/* Fused TradingView SuperTrend direction kernel.  It avoids materializing
 * hl2/basic upper/basic lower/final band arrays.  The reference direction
 * recursion and comparison order are preserved exactly; unsupported/nonfinite
 * market inputs return False so Python can execute the canonical fallback. */
static PyObject *native_supertrend_direction_into(PyObject *self, PyObject *args) {
    PyObject *high_obj, *low_obj, *close_obj, *atr_obj, *out_obj;
    double multiplier;
    Py_buffer high_view = {0}, low_view = {0}, close_view = {0}, atr_view = {0}, out_view = {0};
    (void)self;

    if (!PyArg_ParseTuple(args, "OOOOOd", &high_obj, &low_obj, &close_obj, &atr_obj, &out_obj, &multiplier)) {
        return NULL;
    }
    if (!isfinite(multiplier)) {
        PyErr_SetString(PyExc_ValueError, "multiplier must be finite");
        return NULL;
    }
    if (!get_double_buffer(high_obj, &high_view)) goto fail;
    if (!get_double_buffer(low_obj, &low_view)) goto fail;
    if (!get_double_buffer(close_obj, &close_view)) goto fail;
    if (!get_double_buffer(atr_obj, &atr_view)) goto fail;
    if (!get_writable_byte_buffer(out_obj, &out_view)) goto fail;

    Py_ssize_t n = high_view.len / high_view.itemsize;
    if (low_view.len / low_view.itemsize != n || close_view.len / close_view.itemsize != n ||
        atr_view.len / atr_view.itemsize != n || out_view.len / out_view.itemsize != n) {
        PyErr_SetString(PyExc_ValueError, "supertrend buffer lengths must match");
        goto fail;
    }

    const double *H = (const double *)high_view.buf;
    const double *L = (const double *)low_view.buf;
    const double *C = (const double *)close_view.buf;
    const double *ATR = (const double *)atr_view.buf;
    signed char *OUT = (signed char *)out_view.buf;
    for (Py_ssize_t i = 0; i < n; ++i) OUT[i] = 1;
    if (n == 0) {
        PyBuffer_Release(&out_view); PyBuffer_Release(&atr_view); PyBuffer_Release(&close_view);
        PyBuffer_Release(&low_view); PyBuffer_Release(&high_view);
        Py_RETURN_TRUE;
    }

    Py_ssize_t first = -1;
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!isnan(ATR[i])) { first = i; break; }
    }
    if (first < 0) {
        PyBuffer_Release(&out_view); PyBuffer_Release(&atr_view); PyBuffer_Release(&close_view);
        PyBuffer_Release(&low_view); PyBuffer_Release(&high_view);
        Py_RETURN_TRUE;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!isfinite(H[i]) || !isfinite(L[i]) || !isfinite(C[i])) {
            PyBuffer_Release(&out_view); PyBuffer_Release(&atr_view); PyBuffer_Release(&close_view);
            PyBuffer_Release(&low_view); PyBuffer_Release(&high_view);
            Py_RETURN_FALSE;
        }
        if (i >= first && !isfinite(ATR[i])) {
            PyBuffer_Release(&out_view); PyBuffer_Release(&atr_view); PyBuffer_Release(&close_view);
            PyBuffer_Release(&low_view); PyBuffer_Release(&high_view);
            Py_RETURN_FALSE;
        }
    }

    double hl2 = (H[first] + L[first]) / 2.0;
    double final_ub_prev = hl2 + multiplier * ATR[first];
    double final_lb_prev = hl2 - multiplier * ATR[first];
    signed char direction_prev = 1;
    OUT[first] = 1;

    for (Py_ssize_t i = first + 1; i < n; ++i) {
        hl2 = (H[i] + L[i]) / 2.0;
        const double basic_ub = hl2 + multiplier * ATR[i];
        const double basic_lb = hl2 - multiplier * ATR[i];
        const double final_lb = (C[i - 1] > final_lb_prev)
            ? (basic_lb > final_lb_prev ? basic_lb : final_lb_prev)
            : basic_lb;
        const double final_ub = (C[i - 1] < final_ub_prev)
            ? (basic_ub < final_ub_prev ? basic_ub : final_ub_prev)
            : basic_ub;
        signed char direction = direction_prev;
        if (direction_prev == -1 && C[i] < final_lb_prev) {
            direction = 1;
        } else if (direction_prev == 1 && C[i] > final_ub_prev) {
            direction = -1;
        }
        OUT[i] = direction;
        direction_prev = direction;
        final_lb_prev = final_lb;
        final_ub_prev = final_ub;
    }

    PyBuffer_Release(&out_view); PyBuffer_Release(&atr_view); PyBuffer_Release(&close_view);
    PyBuffer_Release(&low_view); PyBuffer_Release(&high_view);
    Py_RETURN_TRUE;

fail:
    if (out_view.obj) PyBuffer_Release(&out_view);
    if (atr_view.obj) PyBuffer_Release(&atr_view);
    if (close_view.obj) PyBuffer_Release(&close_view);
    if (low_view.obj) PyBuffer_Release(&low_view);
    if (high_view.obj) PyBuffer_Release(&high_view);
    return NULL;
}


static int get_byte_buffer(PyObject *obj, Py_buffer *view) {
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0) {
        return 0;
    }
    if (view->itemsize != 1) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError, "expected contiguous 1-byte bool/uint8 buffer");
        return 0;
    }
    return 1;
}

static PyObject *native_scan_position_span(PyObject *self, PyObject *args) {
    PyObject *high_obj, *low_obj, *atr_obj, *sell_obj;
    Py_ssize_t start_j, end_j;
    long long previous_high_ll, initial_stop_ll, trailing_stop_ll, tp_half_ll;
    int qty, sold_half;
    double atr_times_trail, tp_percent;
    int is_fund;
    Py_buffer high_view = {0}, low_view = {0}, atr_view = {0}, sell_view = {0};
    (void)self;

    if (!PyArg_ParseTuple(
            args, "OOOOnnLLLLipddp",
            &high_obj, &low_obj, &atr_obj, &sell_obj,
            &start_j, &end_j,
            &previous_high_ll, &initial_stop_ll, &trailing_stop_ll, &tp_half_ll,
            &qty, &sold_half, &atr_times_trail, &tp_percent, &is_fund)) {
        return NULL;
    }
    if (qty <= 0 || start_j < 1 || end_j < start_j || !isfinite(atr_times_trail) || !isfinite(tp_percent)) {
        PyErr_SetString(PyExc_ValueError, "invalid native span arguments");
        return NULL;
    }
    if (!get_double_buffer(high_obj, &high_view)) goto fail;
    if (!get_double_buffer(low_obj, &low_view)) goto fail;
    if (!get_double_buffer(atr_obj, &atr_view)) goto fail;
    if (!get_byte_buffer(sell_obj, &sell_view)) goto fail;

    Py_ssize_t n_high = high_view.len / high_view.itemsize;
    Py_ssize_t n_low = low_view.len / low_view.itemsize;
    Py_ssize_t n_atr = atr_view.len / atr_view.itemsize;
    Py_ssize_t n_sell = sell_view.len / sell_view.itemsize;
    Py_ssize_t n = n_high;
    if (n_low < n) n = n_low;
    if (n_atr < n) n = n_atr;
    if (n_sell < n) n = n_sell;
    if (end_j > n) end_j = n;

    const double *H = (const double *)high_view.buf;
    const double *L = (const double *)low_view.buf;
    const double *ATR = (const double *)atr_view.buf;
    const unsigned char *SELL = (const unsigned char *)sell_view.buf;

    int64_t highest_high = (int64_t)previous_high_ll;
    int64_t trailing_stop = (int64_t)trailing_stop_ll;
    int64_t initial_stop = (int64_t)initial_stop_ll;
    int64_t sl_milli = initial_stop > trailing_stop ? initial_stop : trailing_stop;
    int half_qty = 0;
    if (tp_percent > 0.0) {
        double raw_half = floor(((double)qty) * tp_percent);
        if (isfinite(raw_half) && raw_half > 0.0 && raw_half < (double)qty) half_qty = (int)raw_half;
    }

    Py_ssize_t j = start_j;
    for (; j < end_j; ++j) {
        const double y_atr = ATR[j - 1];
        if (!isfinite(y_atr)) {
            continue; /* canonical backtest skips this bar before management */
        }
        if (SELL[j - 1] != 0) {
            break; /* indicator sell: Python owns the executable transition */
        }
        const double t_high = H[j];
        const double t_low = L[j];
        if (!isfinite(t_high) || !isfinite(t_low)) {
            break;
        }

        int64_t candidate_highest = highest_high;
        int64_t candidate_trailing = trailing_stop;
        const double y_high = H[j - 1];
        if (isfinite(y_high)) {
            int64_t y_high_milli = price_to_milli_double(y_high);
            if (y_high_milli > highest_high) {
                candidate_highest = y_high_milli;
                const double trail_reference = ((double)candidate_highest) / 1000.0;
                const double trail_offset = y_atr * atr_times_trail;
                const double raw_trail = trail_reference - trail_offset;
                int64_t candidate_trail_milli = 0;
                if (!round_up_price_to_tick_milli(raw_trail, is_fund, &candidate_trail_milli)) {
                    break; /* exact Python Decimal boundary path */
                }
                if (candidate_trail_milli > candidate_trailing) {
                    candidate_trailing = candidate_trail_milli;
                }
            }
        }
        int64_t candidate_sl = initial_stop > candidate_trailing ? initial_stop : candidate_trailing;
        const int stop_hit = price_to_milli_double(t_low) <= candidate_sl;
        int tp_hit = 0;
        if (!stop_hit && !sold_half && half_qty > 0) {
            tp_hit = price_to_milli_double(t_high) >= (int64_t)tp_half_ll;
        }
        if (stop_hit || tp_hit) {
            break; /* leave this event bar to canonical Python, pre-bar state intact */
        }

        highest_high = candidate_highest;
        trailing_stop = candidate_trailing;
        sl_milli = candidate_sl;
    }

    PyBuffer_Release(&sell_view);
    PyBuffer_Release(&atr_view);
    PyBuffer_Release(&low_view);
    PyBuffer_Release(&high_view);
    return Py_BuildValue(
        "(nLLL)",
        j,
        (long long)highest_high,
        (long long)trailing_stop,
        (long long)sl_milli
    );

fail:
    if (sell_view.obj) PyBuffer_Release(&sell_view);
    if (atr_view.obj) PyBuffer_Release(&atr_view);
    if (low_view.obj) PyBuffer_Release(&low_view);
    if (high_view.obj) PyBuffer_Release(&high_view);
    return NULL;
}



/*
 * Optimizer-only breakout re-entry watch scanner.
 *
 * The Python SSOT checks one completed bar at a time while a stopped position
 * waits for a reclaim confirmation.  In optimizer no-log mode the intermediate
 * failed checks have no side effects beyond bars_checked/last_checked_date.
 * This scanner skips those failed checks and stops before the first bar that
 * must be handled by Python (normal setup or valid reclaim trigger).  If the
 * watch window expires without either event, it reports the consumed checks so
 * Python can retire the watch state directly.
 */
static PyObject *native_scan_reentry_watch_span(PyObject *self, PyObject *args) {
    PyObject *close_obj, *atr_obj, *buy_obj;
    Py_ssize_t start_j, end_j;
    int bars_checked, window_bars;
    double confirm_price;
    Py_buffer close_view = {0}, atr_view = {0}, buy_view = {0};
    (void)self;

    if (!PyArg_ParseTuple(
            args, "OOOnniid",
            &close_obj, &atr_obj, &buy_obj,
            &start_j, &end_j,
            &bars_checked, &window_bars,
            &confirm_price)) {
        return NULL;
    }
    if (start_j < 1 || end_j < start_j || bars_checked < 0 || window_bars <= 0 ||
        !isfinite(confirm_price) || confirm_price <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "invalid native reentry scan arguments");
        return NULL;
    }
    if (!get_double_buffer(close_obj, &close_view)) goto fail;
    if (!get_double_buffer(atr_obj, &atr_view)) goto fail;
    if (!get_byte_buffer(buy_obj, &buy_view)) goto fail;

    Py_ssize_t n_close = close_view.len / close_view.itemsize;
    Py_ssize_t n_atr = atr_view.len / atr_view.itemsize;
    Py_ssize_t n_buy = buy_view.len / buy_view.itemsize;
    Py_ssize_t n = n_close;
    if (n_atr < n) n = n_atr;
    if (n_buy < n) n = n_buy;
    if (end_j > n) end_j = n;

    const double *CLOSE = (const double *)close_view.buf;
    const double *ATR = (const double *)atr_view.buf;
    const unsigned char *BUY = (const unsigned char *)buy_view.buf;

    int remaining = window_bars - bars_checked;
    if (remaining <= 0) {
        PyBuffer_Release(&buy_view);
        PyBuffer_Release(&atr_view);
        PyBuffer_Release(&close_view);
        return Py_BuildValue("(nii)", start_j, 0, 1);
    }

    Py_ssize_t j = start_j;
    int consumed = 0;
    int expired = 0;
    for (; j < end_j && remaining > 0; ++j) {
        const Py_ssize_t y = j - 1;

        /* Re-entry check occurs before normal setup handling on the same bar;
           stop so Python preserves that exact ordering when a setup exists. */
        if (BUY[y] != 0) {
            break;
        }

        const double close_price = CLOSE[y];
        const double atr = ATR[y];
        const int trigger = isfinite(close_price) && close_price > 0.0 &&
                            close_price >= confirm_price && !isnan(atr);
        if (trigger) {
            break;
        }

        consumed += 1;
        remaining -= 1;
        if (remaining <= 0) {
            expired = 1;
            j += 1; /* return the first unconsumed outer-loop index */
            break;
        }
    }

    PyBuffer_Release(&buy_view);
    PyBuffer_Release(&atr_view);
    PyBuffer_Release(&close_view);
    return Py_BuildValue("(nii)", j, consumed, expired);

fail:
    if (buy_view.obj) PyBuffer_Release(&buy_view);
    if (atr_view.obj) PyBuffer_Release(&atr_view);
    if (close_view.obj) PyBuffer_Release(&close_view);
    return NULL;
}


static inline int64_t fee_nearest_milli(int64_t gross_milli, int64_t fee_ppm, int64_t min_fee_milli) {
    int64_t fee = (gross_milli * fee_ppm + 500000LL) / 1000000LL;
    return fee > min_fee_milli ? fee : min_fee_milli;
}

static inline int64_t fee_floor_twd_milli(int64_t gross_milli, int64_t fee_ppm, int64_t min_fee_milli) {
    int64_t fee_twd = (gross_milli * fee_ppm) / 1000000000LL;
    int64_t fee_milli = fee_twd * 1000LL;
    return fee_milli > min_fee_milli ? fee_milli : min_fee_milli;
}

static inline int64_t tax_floor_twd_milli(int64_t gross_milli, int64_t tax_ppm) {
    int64_t tax_twd = (gross_milli * tax_ppm) / 1000000000LL;
    return tax_twd * 1000LL;
}

static inline int64_t economic_fee_milli(int64_t gross_milli, int64_t fee_ppm, int64_t min_fee_milli, int floor_twd_mode) {
    return floor_twd_mode
        ? fee_floor_twd_milli(gross_milli, fee_ppm, min_fee_milli)
        : fee_nearest_milli(gross_milli, fee_ppm, min_fee_milli);
}

static int qty_is_valid_exact(
    int64_t qty,
    int64_t buy_price_milli,
    int64_t stop_price_milli,
    int64_t cap_milli,
    int64_t max_risk_milli,
    int64_t max_position_cap_milli,
    int64_t buy_fee_ppm,
    int64_t sell_fee_ppm,
    int64_t tax_ppm,
    int64_t min_fee_milli,
    int floor_twd_mode,
    int64_t broker_fee_ppm,
    int64_t broker_min_fee_milli
) {
    if (qty <= 0) return 0;
    const int64_t gross_buy = buy_price_milli * qty;
    const int64_t buy_fee = economic_fee_milli(gross_buy, buy_fee_ppm, min_fee_milli, floor_twd_mode);
    const int64_t broker_buy_fee = fee_floor_twd_milli(gross_buy, broker_fee_ppm, broker_min_fee_milli);
    const int64_t net_buy_total = gross_buy + buy_fee;
    const int64_t cash_buy_total = gross_buy + broker_buy_fee;

    const int64_t gross_sell = stop_price_milli * qty;
    const int64_t sell_fee = economic_fee_milli(gross_sell, sell_fee_ppm, min_fee_milli, floor_twd_mode);
    const int64_t tax = tax_floor_twd_milli(gross_sell, tax_ppm);
    const int64_t net_sell_total = gross_sell - sell_fee - tax;
    const int64_t actual_risk = net_buy_total - net_sell_total;

    return cash_buy_total <= cap_milli
        && cash_buy_total <= max_position_cap_milli
        && actual_risk <= max_risk_milli;
}

static PyObject *native_calc_position_size_exact(PyObject *self, PyObject *args) {
    long long buy_price_milli_ll, stop_price_milli_ll, cap_milli_ll;
    long long max_risk_milli_ll, max_position_cap_milli_ll;
    long long buy_fee_ppm_ll, sell_fee_ppm_ll, tax_ppm_ll, min_fee_milli_ll;
    int floor_twd_mode;
    long long broker_fee_ppm_ll, broker_min_fee_milli_ll;
    (void)self;

    if (!PyArg_ParseTuple(
            args,
            "LLLLLLLLLpLL",
            &buy_price_milli_ll,
            &stop_price_milli_ll,
            &cap_milli_ll,
            &max_risk_milli_ll,
            &max_position_cap_milli_ll,
            &buy_fee_ppm_ll,
            &sell_fee_ppm_ll,
            &tax_ppm_ll,
            &min_fee_milli_ll,
            &floor_twd_mode,
            &broker_fee_ppm_ll,
            &broker_min_fee_milli_ll)) {
        return NULL;
    }

    const int64_t buy_price_milli = (int64_t)buy_price_milli_ll;
    const int64_t stop_price_milli = (int64_t)stop_price_milli_ll;
    const int64_t cap_milli = (int64_t)cap_milli_ll;
    const int64_t max_risk_milli = (int64_t)max_risk_milli_ll;
    const int64_t max_position_cap_milli = (int64_t)max_position_cap_milli_ll;
    const int64_t buy_fee_ppm = (int64_t)buy_fee_ppm_ll;
    const int64_t sell_fee_ppm = (int64_t)sell_fee_ppm_ll;
    const int64_t tax_ppm = (int64_t)tax_ppm_ll;
    const int64_t min_fee_milli = (int64_t)min_fee_milli_ll;
    const int64_t broker_fee_ppm = (int64_t)broker_fee_ppm_ll;
    const int64_t broker_min_fee_milli = (int64_t)broker_min_fee_milli_ll;

    if (buy_price_milli <= 0 || stop_price_milli <= 0 || cap_milli <= 0 || max_risk_milli < 0 || max_position_cap_milli <= 0) {
        return PyLong_FromLong(0);
    }

    const int64_t est_buy_fee_per_share = buy_fee_ppm <= 0
        ? 0
        : (buy_price_milli * buy_fee_ppm + 999999LL) / 1000000LL;
    const int64_t combined_sell_ppm = sell_fee_ppm + tax_ppm;
    const int64_t est_sell_cost_per_share = combined_sell_ppm <= 0
        ? 0
        : (stop_price_milli * combined_sell_ppm + 999999LL) / 1000000LL;
    const int64_t est_entry_unit = buy_price_milli + est_buy_fee_per_share;
    const int64_t est_exit_unit = stop_price_milli - est_sell_cost_per_share;
    const int64_t risk_per_share = est_entry_unit - est_exit_unit;
    if (risk_per_share <= 0 || est_entry_unit <= 0) {
        return PyLong_FromLong(0);
    }

    int64_t q_cap = cap_milli / est_entry_unit;
    int64_t q_risk = max_risk_milli / risk_per_share;
    int64_t q_pos = max_position_cap_milli / est_entry_unit;
    int64_t qty = q_cap < q_risk ? q_cap : q_risk;
    if (q_pos < qty) qty = q_pos;
    if (qty <= 0) return PyLong_FromLong(0);

    if (qty_is_valid_exact(
            qty, buy_price_milli, stop_price_milli, cap_milli, max_risk_milli,
            max_position_cap_milli, buy_fee_ppm, sell_fee_ppm, tax_ppm,
            min_fee_milli, floor_twd_mode, broker_fee_ppm, broker_min_fee_milli)) {
        return PyLong_FromLongLong((long long)qty);
    }

    int64_t probe_qty = qty;
    for (int i = 0; i < 8; ++i) {
        probe_qty -= 1;
        if (probe_qty <= 0) return PyLong_FromLong(0);
        if (qty_is_valid_exact(
                probe_qty, buy_price_milli, stop_price_milli, cap_milli, max_risk_milli,
                max_position_cap_milli, buy_fee_ppm, sell_fee_ppm, tax_ppm,
                min_fee_milli, floor_twd_mode, broker_fee_ppm, broker_min_fee_milli)) {
            return PyLong_FromLongLong((long long)probe_qty);
        }
    }

    int64_t lo = 0;
    int64_t hi = probe_qty - 1;
    while (lo < hi) {
        int64_t mid = (lo + hi + 1) / 2;
        if (qty_is_valid_exact(
                mid, buy_price_milli, stop_price_milli, cap_milli, max_risk_milli,
                max_position_cap_milli, buy_fee_ppm, sell_fee_ppm, tax_ppm,
                min_fee_milli, floor_twd_mode, broker_fee_ppm, broker_min_fee_milli)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return PyLong_FromLongLong((long long)lo);
}

static PyMethodDef NativeOptimizerMethods[] = {
    {
        "supertrend_direction_into",
        native_supertrend_direction_into,
        METH_VARARGS,
        "Fill an int8 direction buffer with the fused exact-order SuperTrend recursion."
    },
    {
        "rolling_max_shift1_into",
        native_rolling_max_shift1_into,
        METH_VARARGS,
        "Fill a float64 output buffer with the exact shifted rolling maximum."
    },
    {
        "scan_position_span",
        native_scan_position_span,
        METH_VARARGS,
        "Scan a no-event optimizer position span and return the first bar that requires canonical Python handling."
    },
    {
        "scan_reentry_watch_span",
        native_scan_reentry_watch_span,
        METH_VARARGS,
        "Skip optimizer re-entry watch bars until setup, reclaim trigger, or window expiry."
    },
    {
        "calc_position_size_exact",
        native_calc_position_size_exact,
        METH_VARARGS,
        "Compute canonical exact-accounting position size using integer milli/ppm arithmetic."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nativeoptimizermodule = {
    PyModuleDef_HEAD_INIT,
    "_native_optimizer_hotpath",
    "Native optimizer-only position-management fast path.",
    -1,
    NativeOptimizerMethods
};

PyMODINIT_FUNC PyInit__native_optimizer_hotpath(void) {
    return PyModule_Create(&nativeoptimizermodule);
}

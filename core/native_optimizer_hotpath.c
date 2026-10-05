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



/* Rolling mean with a one-bar shift: out[j] = mean(source[j-window:j]).
 * NaNs preserve pandas rolling(min_periods=window) semantics. */
static PyObject *native_rolling_mean_shift1_into(PyObject *self, PyObject *args) {
    PyObject *source_obj, *out_obj;
    Py_ssize_t window;
    Py_buffer source_view = {0}, out_view = {0};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOn", &source_obj, &out_obj, &window)) return NULL;
    if (window <= 0) { PyErr_SetString(PyExc_ValueError, "window must be positive"); return NULL; }
    if (!get_double_buffer(source_obj, &source_view)) goto fail;
    if (!get_writable_double_buffer(out_obj, &out_view)) goto fail;
    const Py_ssize_t n = source_view.len / source_view.itemsize;
    if (out_view.len / out_view.itemsize != n) { PyErr_SetString(PyExc_ValueError, "source and output lengths must match"); goto fail; }
    const double *src = (const double *)source_view.buf;
    double *out = (double *)out_view.buf;
    for (Py_ssize_t i=0;i<n;++i) out[i]=NAN;
    if (n==0 || window>=n) { PyBuffer_Release(&out_view); PyBuffer_Release(&source_view); Py_RETURN_NONE; }
    long double sum=0.0L;
    Py_ssize_t nan_count=0;
    for (Py_ssize_t i=0;i<n;++i) {
        double v=src[i];
        if (isnan(v)) nan_count++; else sum += (long double)v;
        if (i>=window) {
            double old=src[i-window];
            if (isnan(old)) nan_count--; else sum -= (long double)old;
        }
        Py_ssize_t out_index=i+1;
        if (out_index<n && out_index>=window && nan_count==0) out[out_index]=(double)(sum/(long double)window);
    }
    PyBuffer_Release(&out_view); PyBuffer_Release(&source_view); Py_RETURN_NONE;
fail:
    if (out_view.obj) PyBuffer_Release(&out_view);
    if (source_view.obj) PyBuffer_Release(&source_view);
    return NULL;
}

/* Rolling mean and population std (ddof=0) over the unshifted window.
 * Designed for optimizer feature generation; NaN semantics match pandas with
 * min_periods=window. */
static PyObject *native_rolling_mean_std0_into(PyObject *self, PyObject *args) {
    PyObject *source_obj, *mean_obj, *std_obj;
    Py_ssize_t window;
    Py_buffer source_view={0}, mean_view={0}, std_view={0};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOn", &source_obj, &mean_obj, &std_obj, &window)) return NULL;
    if (window <= 0) { PyErr_SetString(PyExc_ValueError, "window must be positive"); return NULL; }
    if (!get_double_buffer(source_obj,&source_view)) goto fail;
    if (!get_writable_double_buffer(mean_obj,&mean_view)) goto fail;
    if (!get_writable_double_buffer(std_obj,&std_view)) goto fail;
    const Py_ssize_t n=source_view.len/source_view.itemsize;
    if (mean_view.len/mean_view.itemsize!=n || std_view.len/std_view.itemsize!=n) { PyErr_SetString(PyExc_ValueError,"buffer lengths must match"); goto fail; }
    const double *src=(const double*)source_view.buf;
    double *mean_out=(double*)mean_view.buf, *std_out=(double*)std_view.buf;
    for (Py_ssize_t i=0;i<n;++i) { mean_out[i]=NAN; std_out[i]=NAN; }
    if (n==0 || window>n) { PyBuffer_Release(&std_view); PyBuffer_Release(&mean_view); PyBuffer_Release(&source_view); Py_RETURN_NONE; }
    long double sum=0.0L, sumsq=0.0L;
    Py_ssize_t nan_count=0;
    for (Py_ssize_t i=0;i<n;++i) {
        double v=src[i];
        if (isnan(v)) nan_count++; else { long double lv=(long double)v; sum+=lv; sumsq+=lv*lv; }
        if (i>=window) {
            double old=src[i-window];
            if (isnan(old)) nan_count--; else { long double lo=(long double)old; sum-=lo; sumsq-=lo*lo; }
        }
        if (i+1>=window && nan_count==0) {
            long double mean=sum/(long double)window;
            long double var=sumsq/(long double)window - mean*mean;
            if (var<0.0L && var>-1e-18L) var=0.0L;
            if (var<0.0L) var=0.0L;
            mean_out[i]=(double)mean;
            std_out[i]=sqrt((double)var);
        }
    }
    PyBuffer_Release(&std_view); PyBuffer_Release(&mean_view); PyBuffer_Release(&source_view); Py_RETURN_NONE;
fail:
    if (std_view.obj) PyBuffer_Release(&std_view);
    if (mean_view.obj) PyBuffer_Release(&mean_view);
    if (source_view.obj) PyBuffer_Release(&source_view);
    return NULL;
}



/*
 * Exact adjust=False EWM recurrence from an already-resolved seed.
 *
 * The Python caller owns canonical seed semantics (EMA first valid value or
 * RMA/SMA seed) and only uses this kernel for a contiguous non-NaN tail.  This
 * keeps the pandas/fallback edge-case contract unchanged while removing the
 * Series/EWM allocation overhead from the optimizer hot path.
 */
static PyObject *native_ewm_adjust_false_from_seed_into(PyObject *self, PyObject *args) {
    PyObject *source_obj, *out_obj;
    Py_ssize_t start_index;
    double seed, alpha;
    Py_buffer source_view = {0}, out_view = {0};
    (void)self;

    if (!PyArg_ParseTuple(args, "OOndd", &source_obj, &out_obj, &start_index, &seed, &alpha)) {
        return NULL;
    }
    if (!(alpha > 0.0 && alpha <= 1.0) || !isfinite(alpha)) {
        PyErr_SetString(PyExc_ValueError, "alpha must be finite and in (0, 1]");
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
    if (start_index < 0 || start_index >= n) {
        PyErr_SetString(PyExc_ValueError, "start_index out of range");
        goto fail;
    }

    const double *source = (const double *)source_view.buf;
    double *out = (double *)out_view.buf;
    const double one_minus_alpha = 1.0 - alpha;
    const double normalize_weight = one_minus_alpha + alpha;

    for (Py_ssize_t i = 0; i < start_index; ++i) {
        out[i] = NAN;
    }
    out[start_index] = seed;
    for (Py_ssize_t i = start_index + 1; i < n; ++i) {
        const double value = source[i];
        if (isnan(value)) {
            PyBuffer_Release(&out_view);
            PyBuffer_Release(&source_view);
            Py_RETURN_FALSE;
        }
        if (out[i - 1] != value) {
            out[i] = ((one_minus_alpha * out[i - 1]) + (alpha * value)) / normalize_weight;
        } else {
            out[i] = out[i - 1];
        }
    }

    PyBuffer_Release(&out_view);
    PyBuffer_Release(&source_view);
    Py_RETURN_TRUE;

fail:
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

static int64_t calc_position_size_exact_core(
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
    if (buy_price_milli <= 0 || stop_price_milli <= 0 || cap_milli <= 0 || max_risk_milli < 0 || max_position_cap_milli <= 0) {
        return 0;
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
        return 0;
    }

    int64_t q_cap = cap_milli / est_entry_unit;
    int64_t q_risk = max_risk_milli / risk_per_share;
    int64_t q_pos = max_position_cap_milli / est_entry_unit;
    int64_t qty = q_cap < q_risk ? q_cap : q_risk;
    if (q_pos < qty) qty = q_pos;
    if (qty <= 0) return 0;

    if (qty_is_valid_exact(
            qty, buy_price_milli, stop_price_milli, cap_milli, max_risk_milli,
            max_position_cap_milli, buy_fee_ppm, sell_fee_ppm, tax_ppm,
            min_fee_milli, floor_twd_mode, broker_fee_ppm, broker_min_fee_milli)) {
        return qty;
    }

    int64_t probe_qty = qty;
    for (int i = 0; i < 8; ++i) {
        probe_qty -= 1;
        if (probe_qty <= 0) return 0;
        if (qty_is_valid_exact(
                probe_qty, buy_price_milli, stop_price_milli, cap_milli, max_risk_milli,
                max_position_cap_milli, buy_fee_ppm, sell_fee_ppm, tax_ppm,
                min_fee_milli, floor_twd_mode, broker_fee_ppm, broker_min_fee_milli)) {
            return probe_qty;
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
    return lo;
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

    const int64_t qty = calc_position_size_exact_core(
        buy_price_milli, stop_price_milli, cap_milli, max_risk_milli,
        max_position_cap_milli, buy_fee_ppm, sell_fee_ppm, tax_ppm,
        min_fee_milli, floor_twd_mode, broker_fee_ppm, broker_min_fee_milli
    );
    return PyLong_FromLongLong((long long)qty);
}



/*
 * Native continuation helpers.
 *
 * These kernels deliberately accept Python state objects but keep the whole
 * common transition inside one C call.  This removes repeated Python helper
 * calls/dict orchestration while preserving Python as the owner of uncommon
 * and unsupported cases.  Returning None means "fall back to canonical
 * Python without any mutation".
 */

static PyObject *dict_get_borrowed(PyObject *dict_obj, const char *key) {
    if (!PyDict_Check(dict_obj)) return NULL;
    return PyDict_GetItemString(dict_obj, key);
}

static int object_to_double_noerr(PyObject *obj, double *out) {
    if (obj == NULL || obj == Py_None) return 0;
    double value = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    *out = value;
    return 1;
}

static int object_to_int64_noerr(PyObject *obj, int64_t *out) {
    if (obj == NULL || obj == Py_None) return 0;
    long long value = PyLong_AsLongLong(obj);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    *out = (int64_t)value;
    return 1;
}

static int dict_int64_default(PyObject *dict_obj, const char *key, int64_t fallback, int64_t *out) {
    PyObject *obj = dict_get_borrowed(dict_obj, key);
    if (obj == NULL || obj == Py_None) {
        *out = fallback;
        return 1;
    }
    if (!object_to_int64_noerr(obj, out)) {
        return 0;
    }
    return 1;
}

static int profile_is_fund(PyObject *security_profile, int *is_fund_out) {
    if (!PyDict_Check(security_profile)) return 0;
    PyObject *tick_profile = PyDict_GetItemString(security_profile, "tick_profile");
    if (!PyUnicode_Check(tick_profile)) return 0;
    int eq_fund = PyUnicode_CompareWithASCIIString(tick_profile, "fund_two_tier") == 0;
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    if (eq_fund) {
        *is_fund_out = 1;
        return 1;
    }
    int eq_stock = PyUnicode_CompareWithASCIIString(tick_profile, "stock_six_tier") == 0;
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    if (!eq_stock) return 0;
    *is_fund_out = 0;
    return 1;
}

static int get_tick_milli_for_ratio_native(int64_t numerator_milli, int64_t denominator, int is_fund, int64_t *tick_out) {
    if (denominator <= 0) return 0;
    if (is_fund) {
        *tick_out = (numerator_milli < 50000LL * denominator) ? 10LL : 50LL;
        return 1;
    }
    static const int64_t thresholds[] = {1000LL, 10000LL, 50000LL, 100000LL, 500000LL, 1000000LL};
    static const int64_t ticks[] = {1LL, 10LL, 50LL, 100LL, 500LL, 1000LL};
    for (int i = 0; i < 6; ++i) {
        if (numerator_milli < thresholds[i] * denominator) {
            *tick_out = ticks[i];
            return 1;
        }
    }
    *tick_out = 5000LL;
    return 1;
}

static int calc_limit_down_milli_native(int64_t reference_milli, int is_fund, int64_t *out) {
    if (reference_milli <= 0) return 0;
    const int64_t numerator = reference_milli * 90LL;
    const int64_t denominator = 100LL;
    int64_t tick_milli = 0;
    if (!get_tick_milli_for_ratio_native(numerator, denominator, is_fund, &tick_milli)) return 0;
    const int64_t step = denominator * tick_milli;
    *out = ((numerator + step - 1LL) / step) * tick_milli;
    return 1;
}

static int set_dict_int64(PyObject *dict_obj, const char *key, int64_t value) {
    PyObject *obj = PyLong_FromLongLong((long long)value);
    if (obj == NULL) return 0;
    int rc = PyDict_SetItemString(dict_obj, key, obj);
    Py_DECREF(obj);
    return rc == 0;
}

static int set_dict_double(PyObject *dict_obj, const char *key, double value) {
    PyObject *obj = PyFloat_FromDouble(value);
    if (obj == NULL) return 0;
    int rc = PyDict_SetItemString(dict_obj, key, obj);
    Py_DECREF(obj);
    return rc == 0;
}

static int reset_exec_contexts_if_needed(PyObject *shadow_position) {
    PyObject *contexts = dict_get_borrowed(shadow_position, "_last_exec_contexts");
    if (contexts != NULL && PyList_Check(contexts) && PyList_GET_SIZE(contexts) == 0) {
        return 1;
    }
    PyObject *empty = PyList_New(0);
    if (empty == NULL) return 0;
    int rc = PyDict_SetItemString(shadow_position, "_last_exec_contexts", empty);
    Py_DECREF(empty);
    return rc == 0;
}

static int unicode_equals_ascii(PyObject *obj, const char *text) {
    if (!PyUnicode_Check(obj)) return 0;
    int rc = PyUnicode_CompareWithASCIIString(obj, text);
    if (rc == 0) return 1;
    if (PyErr_Occurred()) PyErr_Clear();
    return 0;
}

static int indicator_sell_is_executable_native(
    double t_open,
    double t_high,
    double t_low,
    double t_close,
    double t_volume,
    double y_close,
    int is_fund
) {
    if (!isfinite(t_volume) || t_volume <= 0.0) return 0;
    if (!(isfinite(t_open) && isfinite(t_high) && isfinite(t_low) && isfinite(t_close) && isfinite(y_close))) {
        return 1;
    }
    const int64_t y_close_milli = price_to_milli_double(y_close);
    int64_t limit_down_milli = 0;
    if (!calc_limit_down_milli_native(y_close_milli, is_fund, &limit_down_milli)) {
        return 1;
    }
    const int locked_down =
        price_to_milli_double(t_open) == limit_down_milli &&
        price_to_milli_double(t_high) == limit_down_milli &&
        price_to_milli_double(t_low) == limit_down_milli &&
        price_to_milli_double(t_close) == limit_down_milli;
    return locked_down ? 0 : 1;
}

static int get_attr_double_native(PyObject *obj, const char *name, double *out) {
    if (obj == NULL || obj == Py_None) return 0;
    PyObject *attr = PyObject_GetAttrString(obj, name);
    if (attr == NULL) { PyErr_Clear(); return 0; }
    double value = PyFloat_AsDouble(attr);
    Py_DECREF(attr);
    if (PyErr_Occurred() || !isfinite(value)) { PyErr_Clear(); return 0; }
    *out = value;
    return 1;
}

static int rate_to_ppm_double_native(double value, int64_t *out) {
    if (!isfinite(value)) return 0;
    const double scaled = value * 1000000.0;
    if (!isfinite(scaled) || scaled > (double)INT64_MAX || scaled < (double)INT64_MIN) return 0;
    *out = scaled >= 0.0 ? (int64_t)floor(scaled + 0.5) : (int64_t)ceil(scaled - 0.5);
    return 1;
}

static int money_to_milli_double_native(double value, int64_t *out) {
    if (!isfinite(value)) return 0;
    const double scaled = value * 1000.0;
    if (!isfinite(scaled) || scaled > (double)INT64_MAX || scaled < (double)INT64_MIN) return 0;
    *out = scaled >= 0.0 ? (int64_t)floor(scaled + 0.5) : (int64_t)ceil(scaled - 0.5);
    return 1;
}

static int mul_nonnegative_i64_checked(int64_t a, int64_t b, int64_t *out) {
    if (a < 0 || b < 0) return 0;
    if (a != 0 && b > INT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int unicode_equals_ascii_native(PyObject *obj, const char *text, int *out) {
    if (!PyUnicode_Check(obj)) return 0;
    int cmp = PyUnicode_CompareWithASCIIString(obj, text);
    if (cmp == -1 && PyErr_Occurred()) { PyErr_Clear(); return 0; }
    *out = (cmp == 0);
    return 1;
}

/* Optimizer-only cached mark-to-market batch.  Unsupported/date-dependent
 * cases are returned to the canonical Python exact-accounting path. */
static PyObject *native_mark_to_market_cached_batch(PyObject *self, PyObject *args) {
    PyObject *portfolio, *tickers_obj, *default_params;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOO", &portfolio, &tickers_obj, &default_params)) return NULL;
    if (!PyDict_Check(portfolio)) {
        PyErr_SetString(PyExc_TypeError, "expected portfolio dict input");
        return NULL;
    }
    PyObject *tickers = PySequence_Fast(tickers_obj, "tickers must be a sequence");
    if (tickers == NULL) return NULL;
    PyObject *fallback = PyList_New(0);
    if (fallback == NULL) { Py_DECREF(tickers); return NULL; }

    PyObject *last_params = NULL;
    int64_t last_sell_fee_ppm = 0, last_stock_tax_ppm = 0, last_min_fee_milli = 0;
    int last_fee_floor_twd = 0, have_last_params = 0;
    int64_t total_net_milli = 0;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(tickers);
    PyObject **items = PySequence_Fast_ITEMS(tickers);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *ticker = items[i];
        PyObject *position = PyDict_GetItemWithError(portfolio, ticker);
        if (position == NULL) {
            if (PyErr_Occurred()) goto error;
            continue;
        }
        if (!PyDict_Check(position)) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }

        int64_t qty = 0;
        if (!dict_int64_default(position, "qty", 0, &qty) || qty <= 0) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        PyObject *last_px_obj = dict_get_borrowed(position, "last_px");
        if (last_px_obj == NULL || last_px_obj == Py_None) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }
        double last_px = PyFloat_AsDouble(last_px_obj);
        if (PyErr_Occurred() || !isfinite(last_px)) {
            PyErr_Clear();
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }

        PyObject *security_profile = dict_get_borrowed(position, "security_profile");
        int is_fund = 0;
        if (!profile_is_fund(security_profile, &is_fund)) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }
        PyObject *family_obj = PyDict_GetItemString(security_profile, "family");
        PyObject *broad_type_obj = PyDict_GetItemString(security_profile, "broad_type");
        if (!PyUnicode_Check(family_obj)) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }

        int family_stock = 0, family_reit = 0, family_etf = 0, family_etn = 0;
        if (!unicode_equals_ascii_native(family_obj, "stock", &family_stock) ||
            !unicode_equals_ascii_native(family_obj, "reit", &family_reit) ||
            !unicode_equals_ascii_native(family_obj, "etf", &family_etf) ||
            !unicode_equals_ascii_native(family_obj, "etn", &family_etn)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        if (family_etf) {
            int broad_bond = 0;
            if (broad_type_obj != NULL && PyUnicode_Check(broad_type_obj)) {
                if (!unicode_equals_ascii_native(broad_type_obj, "bond", &broad_bond)) {
                    if (PyList_Append(fallback, ticker) != 0) goto error;
                    continue;
                }
            }
            if (broad_bond) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }
        }

        PyObject *params_obj = dict_get_borrowed(position, "_entry_params_obj");
        if (params_obj == NULL || params_obj == Py_None) params_obj = default_params;
        int64_t sell_fee_ppm = 0, stock_tax_ppm = 0, min_fee_milli = 0;
        int fee_floor_twd = 0;
        if (have_last_params && params_obj == last_params) {
            sell_fee_ppm = last_sell_fee_ppm;
            stock_tax_ppm = last_stock_tax_ppm;
            min_fee_milli = last_min_fee_milli;
            fee_floor_twd = last_fee_floor_twd;
        } else {
            double sell_fee = 0.0, tax_rate = 0.0, min_fee = 0.0;
            if (!get_attr_double_native(params_obj, "sell_fee", &sell_fee) ||
                !get_attr_double_native(params_obj, "tax_rate", &tax_rate) ||
                !get_attr_double_native(params_obj, "min_fee", &min_fee) ||
                !rate_to_ppm_double_native(sell_fee, &sell_fee_ppm) ||
                !rate_to_ppm_double_native(tax_rate, &stock_tax_ppm) ||
                !money_to_milli_double_native(min_fee, &min_fee_milli)) {
                if (PyList_Append(fallback, ticker) != 0) goto error;
                continue;
            }
            PyObject *rounding_obj = PyObject_GetAttrString(params_obj, "charge_rounding_mode");
            if (rounding_obj == NULL) {
                PyErr_Clear();
                fee_floor_twd = 0;
            } else if (rounding_obj == Py_None) {
                fee_floor_twd = 0;
                Py_DECREF(rounding_obj);
            } else if (PyUnicode_Check(rounding_obj)) {
                int is_floor = 0, is_nearest = 0;
                int ok_floor = unicode_equals_ascii_native(rounding_obj, "floor_twd", &is_floor);
                int ok_nearest = unicode_equals_ascii_native(rounding_obj, "nearest_milli", &is_nearest);
                Py_DECREF(rounding_obj);
                if (!ok_floor || !ok_nearest || (!is_floor && !is_nearest)) {
                    if (PyList_Append(fallback, ticker) != 0) goto error;
                    continue;
                }
                fee_floor_twd = is_floor ? 1 : 0;
            } else {
                Py_DECREF(rounding_obj);
                if (PyList_Append(fallback, ticker) != 0) goto error;
                continue;
            }
            last_params = params_obj;
            last_sell_fee_ppm = sell_fee_ppm;
            last_stock_tax_ppm = stock_tax_ppm;
            last_min_fee_milli = min_fee_milli;
            last_fee_floor_twd = fee_floor_twd;
            have_last_params = 1;
        }

        int64_t tick_milli = 0;
        if (!resolve_tick_milli(last_px, is_fund, &tick_milli)) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }
        const double ratio = (last_px * 1000.0) / (double)tick_milli;
        const double rounded_ratio = floor(ratio + 1e-12);
        if (!isfinite(rounded_ratio) || rounded_ratio < 0.0 || rounded_ratio > (double)(INT64_MAX / tick_milli)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        const int64_t exec_price_milli = (int64_t)rounded_ratio * tick_milli;
        int64_t gross_milli = 0;
        if (!mul_nonnegative_i64_checked(exec_price_milli, qty, &gross_milli)) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }

        int64_t tax_ppm = stock_tax_ppm;
        if (family_reit) tax_ppm = 0;
        else if (family_etf || family_etn) tax_ppm = 1000;
        else if (!family_stock) tax_ppm = stock_tax_ppm;

        int64_t fee_product = 0, tax_product = 0;
        if (!mul_nonnegative_i64_checked(gross_milli, sell_fee_ppm, &fee_product) ||
            !mul_nonnegative_i64_checked(gross_milli, tax_ppm, &tax_product)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        int64_t sell_fee_milli = 0;
        if (fee_floor_twd) {
            sell_fee_milli = (fee_product / 1000000000LL) * 1000LL;
        } else {
            if (fee_product > INT64_MAX - 500000LL) { if (PyList_Append(fallback, ticker) != 0) goto error; continue; }
            sell_fee_milli = (fee_product + 500000LL) / 1000000LL;
        }
        if (sell_fee_milli < min_fee_milli) sell_fee_milli = min_fee_milli;
        const int64_t tax_milli = (tax_product / 1000000000LL) * 1000LL;
        const int64_t net_milli = gross_milli - sell_fee_milli - tax_milli;
        if ((net_milli > 0 && total_net_milli > INT64_MAX - net_milli) ||
            (net_milli < 0 && total_net_milli < INT64_MIN - net_milli)) {
            Py_DECREF(tickers);
            Py_DECREF(fallback);
            Py_RETURN_NONE;
        }
        total_net_milli += net_milli;
    }

    PyObject *result = Py_BuildValue("(LO)", (long long)total_net_milli, fallback);
    Py_DECREF(fallback);
    Py_DECREF(tickers);
    return result;
error:
    Py_DECREF(tickers);
    Py_DECREF(fallback);
    return NULL;
}

static int resolve_atr_times_trail_native(PyObject *params_obj, double *out) {
    if (params_obj == NULL || params_obj == Py_None) return 0;
    PyObject *trail_attr = PyObject_GetAttrString(params_obj, "atr_times_trail");
    if (trail_attr == NULL) {
        PyErr_Clear();
        return 0;
    }
    double value = PyFloat_AsDouble(trail_attr);
    Py_DECREF(trail_attr);
    if (PyErr_Occurred() || !isfinite(value)) {
        PyErr_Clear();
        return 0;
    }
    *out = value;
    return 1;
}

/* Return 0=keep, 1=clear, 2=canonical Python fallback, -1=Python error. */
static int cleanup_live_shadow_core_native(
    PyObject *signal_state,
    PyObject *shadow_position,
    double t_low,
    double t_high,
    double t_open,
    double t_close,
    double t_volume,
    double y_close,
    double y_high,
    double y_atr,
    int y_ind_sell,
    double atr_times_trail
) {
    if (!PyDict_Check(signal_state) || !PyDict_Check(shadow_position) || !isfinite(atr_times_trail)) {
        return 2;
    }

    PyObject *security_profile = dict_get_borrowed(shadow_position, "security_profile");
    int is_fund = 0;
    if (!profile_is_fund(security_profile, &is_fund)) return 2;

    int64_t entry_fill_milli = 0;
    if (!dict_int64_default(shadow_position, "entry_fill_price_milli", 0, &entry_fill_milli)) return 2;
    int64_t previous_high_milli = entry_fill_milli;
    if (!dict_int64_default(shadow_position, "highest_high_since_entry_milli", entry_fill_milli, &previous_high_milli)) return 2;
    int64_t previous_trailing_milli = 0, previous_sl_milli = 0, initial_stop_milli = 0;
    if (!dict_int64_default(shadow_position, "trailing_stop_milli", 0, &previous_trailing_milli) ||
        !dict_int64_default(shadow_position, "sl_milli", 0, &previous_sl_milli) ||
        !dict_int64_default(shadow_position, "initial_stop_milli", 0, &initial_stop_milli)) {
        return 2;
    }

    PyObject *pending_exit = dict_get_borrowed(shadow_position, "pending_exit_action");
    int pending_terminal = unicode_equals_ascii(pending_exit, "STOP") || unicode_equals_ascii(pending_exit, "TP_HALF");

    int64_t highest_high_milli = previous_high_milli;
    int64_t trailing_milli = previous_trailing_milli;
    int64_t sl_milli = previous_sl_milli;
    int made_new_high = 0;

    if (!pending_terminal && isfinite(y_high)) {
        const int64_t y_high_milli = price_to_milli_double(y_high);
        if (y_high_milli > previous_high_milli) {
            highest_high_milli = y_high_milli;
            made_new_high = 1;
        }
    }

    if (!pending_terminal && made_new_high && isfinite(y_atr)) {
        const double candidate_trail = ((double)highest_high_milli / 1000.0) - (y_atr * atr_times_trail);
        int64_t candidate_trail_milli = 0;
        if (!round_up_price_to_tick_milli(candidate_trail, is_fund, &candidate_trail_milli)) {
            return 2;
        }
        if (candidate_trail_milli > trailing_milli) trailing_milli = candidate_trail_milli;
    }
    if (!pending_terminal) {
        sl_milli = initial_stop_milli > trailing_milli ? initial_stop_milli : trailing_milli;
    }

    int indicator_clear = 0;
    int should_clear = pending_terminal;
    if (!should_clear && y_ind_sell) {
        indicator_clear = indicator_sell_is_executable_native(
            t_open, t_high, t_low, t_close, t_volume, y_close, is_fund
        );
        if (indicator_clear) should_clear = 1;
    }

    if (!should_clear) {
        PyObject *sold_half_obj = dict_get_borrowed(shadow_position, "sold_half");
        int sold_half = sold_half_obj == NULL ? 0 : PyObject_IsTrue(sold_half_obj);
        if (sold_half < 0) {
            PyErr_Clear();
            return 2;
        }
        if (sold_half) {
            should_clear = 1;
        } else if (isfinite(t_low) && price_to_milli_double(t_low) <= sl_milli) {
            should_clear = 1;
        } else {
            PyObject *target_obj = dict_get_borrowed(shadow_position, "tp_half_milli");
            if (target_obj != NULL && target_obj != Py_None && isfinite(t_high)) {
                int64_t target_milli = 0;
                if (!object_to_int64_noerr(target_obj, &target_milli)) return 2;
                if (price_to_milli_double(t_high) >= target_milli) should_clear = 1;
            }
        }
    }

    if (!reset_exec_contexts_if_needed(shadow_position)) return -1;

    if (!pending_terminal) {
        if (highest_high_milli != previous_high_milli &&
            !set_dict_int64(shadow_position, "highest_high_since_entry_milli", highest_high_milli)) return -1;
        if (trailing_milli != previous_trailing_milli &&
            !set_dict_int64(shadow_position, "trailing_stop_milli", trailing_milli)) return -1;
        if (sl_milli != previous_sl_milli &&
            !set_dict_int64(shadow_position, "sl_milli", sl_milli)) return -1;
        if (highest_high_milli != previous_high_milli || trailing_milli != previous_trailing_milli || sl_milli != previous_sl_milli) {
            if (!set_dict_double(shadow_position, "highest_high_since_entry", (double)highest_high_milli / 1000.0) ||
                !set_dict_double(shadow_position, "trailing_stop", (double)trailing_milli / 1000.0) ||
                !set_dict_double(shadow_position, "sl", (double)sl_milli / 1000.0)) return -1;
        }
        if (sl_milli != previous_sl_milli) {
            PyObject *sl_obj = dict_get_borrowed(shadow_position, "sl");
            if (sl_obj == NULL) {
                sl_obj = PyFloat_FromDouble((double)sl_milli / 1000.0);
                if (sl_obj == NULL) return -1;
                int rc = PyDict_SetItemString(signal_state, "continuation_invalidation_barrier", sl_obj);
                Py_DECREF(sl_obj);
                if (rc != 0) return -1;
            } else if (PyDict_SetItemString(signal_state, "continuation_invalidation_barrier", sl_obj) != 0) {
                return -1;
            }
        }
    }

    if (indicator_clear && PyDict_SetItemString(signal_state, "shadow_position", Py_None) != 0) return -1;
    return should_clear ? 1 : 0;
}

static PyObject *native_cleanup_live_shadow_inplace(PyObject *self, PyObject *args) {
    PyObject *signal_state, *shadow_position, *params_obj;
    double t_low, t_high, t_open, t_close, t_volume, y_close, y_high, y_atr;
    int y_ind_sell;
    (void)self;
    if (!PyArg_ParseTuple(
            args, "OOddddddddpO",
            &signal_state, &shadow_position,
            &t_low, &t_high, &t_open, &t_close, &t_volume, &y_close, &y_high, &y_atr,
            &y_ind_sell, &params_obj)) {
        return NULL;
    }
    double atr_times_trail = 0.0;
    if (!resolve_atr_times_trail_native(params_obj, &atr_times_trail)) Py_RETURN_NONE;
    int rc = cleanup_live_shadow_core_native(
        signal_state, shadow_position,
        t_low, t_high, t_open, t_close, t_volume, y_close, y_high, y_atr,
        y_ind_sell, atr_times_trail
    );
    if (rc < 0) return NULL;
    if (rc == 2) Py_RETURN_NONE;
    if (rc == 1) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/*
 * Batch form used by optimizer cleanup.  Each item is:
 * (ticker, signal_state, shadow_position,
 *  t_low, t_high, t_open, t_close, t_volume, y_close, y_high, y_atr,
 *  y_ind_sell, params_obj)
 *
 * Clear states are removed from active_extended_signals in C.  Rare unsupported
 * threshold/profile cases are returned as item indexes for canonical Python fallback.
 */
static PyObject *native_cleanup_live_shadow_batch(PyObject *self, PyObject *args) {
    PyObject *active_signals, *items_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &active_signals, &items_obj)) return NULL;
    if (!PyDict_Check(active_signals) || !PySequence_Check(items_obj)) {
        PyErr_SetString(PyExc_TypeError, "expected active-signals dict and cleanup item sequence");
        return NULL;
    }

    PyObject *fallback_indexes = PyList_New(0);
    if (fallback_indexes == NULL) return NULL;
    PyObject *fast_items = PySequence_Fast(items_obj, "cleanup items must be a sequence");
    if (fast_items == NULL) {
        Py_DECREF(fallback_indexes);
        return NULL;
    }

    PyObject *last_params = NULL;
    double last_atr_times_trail = 0.0;
    int have_last_params = 0;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast_items);
    PyObject **items = PySequence_Fast_ITEMS(fast_items);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = items[i];
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 13) {
            PyObject *idx = PyLong_FromSsize_t(i);
            if (idx == NULL || PyList_Append(fallback_indexes, idx) != 0) {
                Py_XDECREF(idx); Py_DECREF(fast_items); Py_DECREF(fallback_indexes); return NULL;
            }
            Py_DECREF(idx);
            continue;
        }
        PyObject *ticker = PyTuple_GET_ITEM(item, 0);
        PyObject *signal_state = PyTuple_GET_ITEM(item, 1);
        PyObject *shadow_position = PyTuple_GET_ITEM(item, 2);
        PyObject *params_obj = PyTuple_GET_ITEM(item, 12);

        double vals[8];
        int parse_ok = 1;
        for (int j = 0; j < 8; ++j) {
            vals[j] = PyFloat_AsDouble(PyTuple_GET_ITEM(item, 3 + j));
            if (PyErr_Occurred()) { PyErr_Clear(); parse_ok = 0; break; }
        }
        int y_ind_sell = 0;
        if (parse_ok) {
            y_ind_sell = PyObject_IsTrue(PyTuple_GET_ITEM(item, 11));
            if (y_ind_sell < 0) { PyErr_Clear(); parse_ok = 0; }
        }
        if (!parse_ok) {
            PyObject *idx = PyLong_FromSsize_t(i);
            if (idx == NULL || PyList_Append(fallback_indexes, idx) != 0) {
                Py_XDECREF(idx); Py_DECREF(fast_items); Py_DECREF(fallback_indexes); return NULL;
            }
            Py_DECREF(idx);
            continue;
        }

        double atr_times_trail = 0.0;
        if (have_last_params && params_obj == last_params) {
            atr_times_trail = last_atr_times_trail;
        } else {
            if (!resolve_atr_times_trail_native(params_obj, &atr_times_trail)) {
                PyObject *idx = PyLong_FromSsize_t(i);
                if (idx == NULL || PyList_Append(fallback_indexes, idx) != 0) {
                    Py_XDECREF(idx); Py_DECREF(fast_items); Py_DECREF(fallback_indexes); return NULL;
                }
                Py_DECREF(idx);
                continue;
            }
            last_params = params_obj;
            last_atr_times_trail = atr_times_trail;
            have_last_params = 1;
        }

        int rc = cleanup_live_shadow_core_native(
            signal_state, shadow_position,
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7],
            y_ind_sell, atr_times_trail
        );
        if (rc < 0) {
            Py_DECREF(fast_items); Py_DECREF(fallback_indexes); return NULL;
        }
        if (rc == 2) {
            PyObject *idx = PyLong_FromSsize_t(i);
            if (idx == NULL || PyList_Append(fallback_indexes, idx) != 0) {
                Py_XDECREF(idx); Py_DECREF(fast_items); Py_DECREF(fallback_indexes); return NULL;
            }
            Py_DECREF(idx);
            continue;
        }
        if (rc == 1) {
            if (PyDict_DelItem(active_signals, ticker) != 0) {
                if (PyErr_ExceptionMatches(PyExc_KeyError)) PyErr_Clear();
                else { Py_DECREF(fast_items); Py_DECREF(fallback_indexes); return NULL; }
            }
        }
    }

    Py_DECREF(fast_items);
    return fallback_indexes;
}


static int packed_float64_at(PyObject *packed_data, const char *field, Py_ssize_t pos, double *out) {
    if (!PyDict_Check(packed_data) || pos < 0) return 0;
    PyObject *packed_flag = PyDict_GetItemString(packed_data, "_packed_market_data");
    if (packed_flag == NULL || packed_flag != Py_True) return 0;
    PyObject *array_obj = PyDict_GetItemString(packed_data, field);
    if (array_obj == NULL) return 0;
    Py_buffer view;
    if (PyObject_GetBuffer(array_obj, &view, PyBUF_CONTIG_RO) != 0) {
        PyErr_Clear();
        return 0;
    }
    int ok = 0;
    if (view.itemsize == (Py_ssize_t)sizeof(double) &&
        pos < (view.len / (Py_ssize_t)sizeof(double))) {
        memcpy(out, ((const char *)view.buf) + pos * (Py_ssize_t)sizeof(double), sizeof(double));
        ok = 1;
    }
    PyBuffer_Release(&view);
    return ok;
}

static int append_fallback_ticker(PyObject *fallback_tickers, PyObject *ticker) {
    return PyList_Append(fallback_tickers, ticker) == 0;
}

/*
 * Optimizer-only fused cleanup/data-prep path.
 *
 * Stage-2 moved the live-shadow state transition into C, but Python still paid
 * the dominant orchestration cost for every active continuation: cache lookup,
 * packed-market scalar extraction, float conversion and temporary tuple/list
 * construction.  This entry point consumes the already PIT-safe pre-market
 * cache and packed market arrays directly.  Only unsupported/non-live states
 * are returned to Python for the canonical fallback loop.
 */
static PyObject *native_cleanup_live_shadow_cached_batch(PyObject *self, PyObject *args) {
    PyObject *active_signals, *portfolio, *market_cache, *tickers_obj, *all_dfs_fast, *default_params;
    (void)self;
    if (!PyArg_ParseTuple(
            args, "OOOOOO",
            &active_signals, &portfolio, &market_cache, &tickers_obj,
            &all_dfs_fast, &default_params)) {
        return NULL;
    }
    if (!PyDict_Check(active_signals) || !PyDict_Check(portfolio) ||
        !PyDict_Check(market_cache) || !PyDict_Check(all_dfs_fast)) {
        PyErr_SetString(PyExc_TypeError, "expected dict state/cache inputs");
        return NULL;
    }

    PyObject *tickers = PySequence_Fast(tickers_obj, "tickers must be a sequence");
    if (tickers == NULL) return NULL;
    PyObject *fallback_tickers = PyList_New(0);
    PyObject *delete_tickers = PyList_New(0);
    if (fallback_tickers == NULL || delete_tickers == NULL) {
        Py_XDECREF(fallback_tickers); Py_XDECREF(delete_tickers); Py_DECREF(tickers);
        return NULL;
    }

    PyObject *last_params = NULL;
    double last_atr_times_trail = 0.0;
    int have_last_params = 0;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(tickers);
    PyObject **ticker_items = PySequence_Fast_ITEMS(tickers);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *ticker = ticker_items[i];
        PyObject *signal_state = PyDict_GetItemWithError(active_signals, ticker);
        if (signal_state == NULL) {
            if (PyErr_Occurred()) goto error;
            continue;
        }
        int in_portfolio = PyDict_Contains(portfolio, ticker);
        if (in_portfolio < 0) goto error;
        if (in_portfolio) {
            if (PyList_Append(delete_tickers, ticker) != 0) goto error;
            continue;
        }
        if (!PyDict_Check(signal_state)) {
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
            continue;
        }

        PyObject *shadow_position = dict_get_borrowed(signal_state, "shadow_position");
        int64_t shadow_qty = 0;
        if (!PyDict_Check(shadow_position) ||
            !dict_int64_default(shadow_position, "qty", 0, &shadow_qty) || shadow_qty <= 0) {
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
            continue;
        }

        PyObject *cached = PyDict_GetItemWithError(market_cache, ticker);
        if (cached == NULL) {
            if (PyErr_Occurred()) goto error;
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
            continue;
        }
        if (!PyTuple_Check(cached) || PyTuple_GET_SIZE(cached) < 4) {
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
            continue;
        }
        int64_t t_pos_i64 = 0;
        if (!object_to_int64_noerr(PyTuple_GET_ITEM(cached, 0), &t_pos_i64) || t_pos_i64 <= 0) {
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
            continue;
        }
        PyObject *previous_values = PyTuple_GET_ITEM(cached, 1);
        if (!PyTuple_Check(previous_values) || PyTuple_GET_SIZE(previous_values) != 4) {
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
            continue;
        }
        double y_atr = NAN, y_close = NAN, y_high = NAN;
        if (!object_to_double_noerr(PyTuple_GET_ITEM(previous_values, 0), &y_atr) ||
            !object_to_double_noerr(PyTuple_GET_ITEM(previous_values, 2), &y_close) ||
            !object_to_double_noerr(PyTuple_GET_ITEM(previous_values, 3), &y_high)) {
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
            continue;
        }
        int y_ind_sell = PyObject_IsTrue(PyTuple_GET_ITEM(previous_values, 1));
        if (y_ind_sell < 0) { PyErr_Clear(); if (!append_fallback_ticker(fallback_tickers, ticker)) goto error; continue; }

        double t_open = NAN, t_high = NAN, t_low = NAN, t_close = NAN, t_volume = NAN;
        int have_current = 0;
        if (PyTuple_GET_SIZE(cached) > 4) {
            PyObject *current = PyTuple_GET_ITEM(cached, 4);
            if (current != Py_None && PyTuple_Check(current) && PyTuple_GET_SIZE(current) == 5) {
                have_current =
                    object_to_double_noerr(PyTuple_GET_ITEM(current, 0), &t_open) &&
                    object_to_double_noerr(PyTuple_GET_ITEM(current, 1), &t_high) &&
                    object_to_double_noerr(PyTuple_GET_ITEM(current, 2), &t_low) &&
                    object_to_double_noerr(PyTuple_GET_ITEM(current, 3), &t_close) &&
                    object_to_double_noerr(PyTuple_GET_ITEM(current, 4), &t_volume);
            }
        }
        if (!have_current) {
            PyObject *packed_data = PyDict_GetItemWithError(all_dfs_fast, ticker);
            if (packed_data == NULL) {
                if (PyErr_Occurred()) goto error;
                if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
                continue;
            }
            Py_ssize_t t_pos = (Py_ssize_t)t_pos_i64;
            if (!packed_float64_at(packed_data, "High", t_pos, &t_high) ||
                !packed_float64_at(packed_data, "Low", t_pos, &t_low)) {
                if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
                continue;
            }
            if (y_ind_sell) {
                if (!packed_float64_at(packed_data, "Open", t_pos, &t_open) ||
                    !packed_float64_at(packed_data, "Close", t_pos, &t_close) ||
                    !packed_float64_at(packed_data, "Volume", t_pos, &t_volume)) {
                    if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
                    continue;
                }
            } else {
                t_open = t_close = t_volume = y_close = NAN;
            }
        } else if (!y_ind_sell) {
            /* Canonical fast cleanup deliberately ignores these execution fields
             * on non-indicator days.  Keep the same NaN contract. */
            t_open = t_close = t_volume = y_close = NAN;
        }

        PyObject *params_obj = dict_get_borrowed(signal_state, "_params_obj");
        if (params_obj == NULL || params_obj == Py_None) params_obj = default_params;
        double atr_times_trail = 0.0;
        if (have_last_params && params_obj == last_params) {
            atr_times_trail = last_atr_times_trail;
        } else {
            if (!resolve_atr_times_trail_native(params_obj, &atr_times_trail)) {
                if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
                continue;
            }
            last_params = params_obj;
            last_atr_times_trail = atr_times_trail;
            have_last_params = 1;
        }

        int rc = cleanup_live_shadow_core_native(
            signal_state, shadow_position,
            t_low, t_high, t_open, t_close, t_volume,
            y_close, y_high, y_atr, y_ind_sell, atr_times_trail
        );
        if (rc < 0) goto error;
        if (rc == 2) {
            if (!append_fallback_ticker(fallback_tickers, ticker)) goto error;
        } else if (rc == 1) {
            if (PyList_Append(delete_tickers, ticker) != 0) goto error;
        }
    }

    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(delete_tickers); ++i) {
        PyObject *ticker = PyList_GET_ITEM(delete_tickers, i);
        if (PyDict_DelItem(active_signals, ticker) != 0) {
            if (PyErr_ExceptionMatches(PyExc_KeyError)) PyErr_Clear();
            else goto error;
        }
    }
    Py_DECREF(delete_tickers);
    Py_DECREF(tickers);
    return fallback_tickers;

error:
    Py_DECREF(delete_tickers);
    Py_DECREF(fallback_tickers);
    Py_DECREF(tickers);
    return NULL;
}

static PyObject *native_resolve_live_extended_candidate_core(PyObject *self, PyObject *args) {
    PyObject *signal_state, *shadow_position, *ticker_obj, *security_profile;
    double y_close;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOdOO", &signal_state, &shadow_position, &y_close, &ticker_obj, &security_profile)) {
        return NULL;
    }
    if (!PyDict_Check(signal_state) || !PyDict_Check(shadow_position)) {
        Py_RETURN_NONE;
    }
    int64_t qty = 0;
    if (!dict_int64_default(shadow_position, "qty", 0, &qty) || qty <= 0) {
        Py_RETURN_NONE;
    }
    int is_fund = 0;
    if (!profile_is_fund(security_profile, &is_fund)) {
        Py_RETURN_NONE;
    }

    PyObject *pending_exit = dict_get_borrowed(shadow_position, "pending_exit_action");
    if (pending_exit != NULL && pending_exit != Py_None) {
        Py_RETURN_FALSE;
    }

    PyObject *entry_atr_obj = dict_get_borrowed(signal_state, "orig_atr");
    PyObject *limit_obj = dict_get_borrowed(signal_state, "orig_limit");
    PyObject *sl_obj = dict_get_borrowed(shadow_position, "sl");
    PyObject *trail_obj = dict_get_borrowed(shadow_position, "trailing_stop");
    PyObject *target_obj = dict_get_borrowed(shadow_position, "tp_half");
    double entry_atr, limit_price, init_sl, init_trail, target_price;
    if (!object_to_double_noerr(entry_atr_obj, &entry_atr) ||
        !object_to_double_noerr(limit_obj, &limit_price) ||
        !object_to_double_noerr(sl_obj, &init_sl) ||
        !object_to_double_noerr(trail_obj, &init_trail) ||
        !object_to_double_noerr(target_obj, &target_price)) {
        Py_RETURN_FALSE;
    }
    if (!(isfinite(entry_atr) && isfinite(limit_price) && isfinite(init_sl) && isfinite(init_trail) && isfinite(target_price) && isfinite(y_close))) {
        Py_RETURN_FALSE;
    }

    const int64_t limit_milli = price_to_milli_double(limit_price);
    const int64_t stop_milli = price_to_milli_double(init_sl);
    const int64_t y_close_milli = price_to_milli_double(y_close);
    if (limit_milli <= 0 || y_close_milli <= 0) {
        Py_RETURN_FALSE;
    }
    int64_t limit_down_milli = 0;
    if (!calc_limit_down_milli_native(y_close_milli, is_fund, &limit_down_milli)) {
        Py_RETURN_NONE;
    }
    if (limit_milli < limit_down_milli) {
        Py_RETURN_FALSE;
    }

    PyObject *entry_ref_obj = dict_get_borrowed(signal_state, "entry_ref_price");
    PyObject *owned_entry_ref_nan = NULL;
    if (entry_ref_obj == NULL) {
        owned_entry_ref_nan = PyFloat_FromDouble(NAN);
        if (owned_entry_ref_nan == NULL) return NULL;
        entry_ref_obj = owned_entry_ref_nan;
    }
    PyObject *signal_date_obj = dict_get_borrowed(signal_state, "signal_date");
    PyObject *max_qty_obj = dict_get_borrowed(signal_state, "max_qty");
    PyObject *source_obj = dict_get_borrowed(signal_state, "source");
    PyObject *source_order_obj = dict_get_borrowed(signal_state, "source_entry_order_id");

    if (signal_date_obj == NULL) signal_date_obj = Py_None;
    if (max_qty_obj == NULL) max_qty_obj = Py_None;
    if (source_order_obj == NULL) source_order_obj = Py_None;

    PyObject *entry_source = NULL;
    int source_truth = source_obj == NULL ? 0 : PyObject_IsTrue(source_obj);
    if (source_truth < 0) {
        PyErr_Clear();
        Py_XDECREF(owned_entry_ref_nan);
        Py_RETURN_NONE;
    }
    if (source_truth) entry_source = PyObject_Str(source_obj);
    else entry_source = PyUnicode_FromString("extended");
    if (entry_source == NULL) {
        Py_XDECREF(owned_entry_ref_nan);
        return NULL;
    }

    PyObject *result = PyTuple_New(14);
    if (result == NULL) {
        Py_DECREF(entry_source);
        Py_XDECREF(owned_entry_ref_nan);
        return NULL;
    }
#define SET_TUPLE_INCREF(idx, obj) do { PyObject *_o = (obj); Py_INCREF(_o); PyTuple_SET_ITEM(result, (idx), _o); } while (0)
    SET_TUPLE_INCREF(0, limit_obj);
    SET_TUPLE_INCREF(1, sl_obj);
    SET_TUPLE_INCREF(2, trail_obj);
    SET_TUPLE_INCREF(3, target_obj);
    SET_TUPLE_INCREF(4, entry_atr_obj);
    SET_TUPLE_INCREF(5, entry_ref_obj);
    if (owned_entry_ref_nan != NULL) { Py_DECREF(owned_entry_ref_nan); owned_entry_ref_nan = NULL; }
    SET_TUPLE_INCREF(6, signal_date_obj);
    SET_TUPLE_INCREF(7, max_qty_obj);
    PyTuple_SET_ITEM(result, 8, entry_source); /* steals */
    SET_TUPLE_INCREF(9, source_order_obj);
    SET_TUPLE_INCREF(10, shadow_position);
    PyTuple_SET_ITEM(result, 11, PyLong_FromLongLong((long long)limit_milli));
    PyTuple_SET_ITEM(result, 12, PyLong_FromLongLong((long long)stop_milli));
    PyTuple_SET_ITEM(result, 13, PyLong_FromLongLong((long long)y_close_milli));
#undef SET_TUPLE_INCREF
    if (PyErr_Occurred()) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}


static int parse_int_tuple_item(PyObject *seq, Py_ssize_t index, int64_t *out) {
    if (!PyTuple_Check(seq) || index < 0 || index >= PyTuple_GET_SIZE(seq)) return 0;
    return object_to_int64_noerr(PyTuple_GET_ITEM(seq, index), out);
}

static int64_t apply_board_lot_preferred_qty_native(
    int64_t limit_price_milli,
    int64_t qty,
    int64_t min_entry_notional_milli
) {
    if (qty <= 0 || limit_price_milli <= 0) return 0;
    const int64_t board_lot_qty = (qty / 1000LL) * 1000LL;
    if (board_lot_qty > 0 &&
        (min_entry_notional_milli <= 0 || limit_price_milli * board_lot_qty >= min_entry_notional_milli)) {
        return board_lot_qty;
    }
    if (min_entry_notional_milli <= 0 || limit_price_milli * qty >= min_entry_notional_milli) {
        return qty;
    }
    return 0;
}

static PyObject *native_resolve_live_extended_candidate_batch_sized(PyObject *self, PyObject *args) {
    PyObject *items_obj;
    long long broker_fee_ppm_ll, broker_min_fee_milli_ll;
    (void)self;
    if (!PyArg_ParseTuple(args, "OLL", &items_obj, &broker_fee_ppm_ll, &broker_min_fee_milli_ll)) {
        return NULL;
    }

    PyObject *items = PySequence_Fast(items_obj, "items must be a sequence");
    if (items == NULL) return NULL;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(items);
    PyObject *out = PyList_New(n);
    if (out == NULL) { Py_DECREF(items); return NULL; }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(items, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 7) {
            Py_INCREF(Py_None); PyList_SET_ITEM(out, i, Py_None); continue;
        }
        PyObject *ticker = PyTuple_GET_ITEM(item, 0);
        PyObject *signal_state = PyTuple_GET_ITEM(item, 1);
        PyObject *shadow_position = PyTuple_GET_ITEM(item, 2);
        PyObject *y_close_obj = PyTuple_GET_ITEM(item, 3);
        PyObject *security_profile = PyTuple_GET_ITEM(item, 4);
        PyObject *capital_context = PyTuple_GET_ITEM(item, 5);
        PyObject *fee_context = PyTuple_GET_ITEM(item, 6);
        double y_close = PyFloat_AsDouble(y_close_obj);
        if (PyErr_Occurred()) {
            PyErr_Clear(); Py_INCREF(Py_None); PyList_SET_ITEM(out, i, Py_None); continue;
        }

        PyObject *core_args = PyTuple_New(5);
        if (core_args == NULL) { Py_DECREF(items); Py_DECREF(out); return NULL; }
#define SET_CORE_ARG(idx, obj) do { PyObject *_o=(obj); Py_INCREF(_o); PyTuple_SET_ITEM(core_args,(idx),_o); } while(0)
        SET_CORE_ARG(0, signal_state);
        SET_CORE_ARG(1, shadow_position);
        PyTuple_SET_ITEM(core_args, 2, PyFloat_FromDouble(y_close));
        SET_CORE_ARG(3, ticker);
        SET_CORE_ARG(4, security_profile);
#undef SET_CORE_ARG
        if (PyErr_Occurred()) { Py_DECREF(core_args); Py_DECREF(items); Py_DECREF(out); return NULL; }
        PyObject *core = native_resolve_live_extended_candidate_core(NULL, core_args);
        Py_DECREF(core_args);
        if (core == NULL) { Py_DECREF(items); Py_DECREF(out); return NULL; }
        if (core == Py_None || core == Py_False) {
            PyList_SET_ITEM(out, i, core); /* steals owned core reference */
            continue;
        }
        if (!PyTuple_Check(core) || PyTuple_GET_SIZE(core) < 14 ||
            !PyTuple_Check(capital_context) || PyTuple_GET_SIZE(capital_context) < 3 ||
            !PyTuple_Check(fee_context) || PyTuple_GET_SIZE(fee_context) < 6) {
            Py_DECREF(core);
            Py_INCREF(Py_None); PyList_SET_ITEM(out, i, Py_None); continue;
        }

        int64_t cap_milli, max_risk_milli, max_position_cap_milli;
        int64_t buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli, rounding_mode_code, min_entry_notional_milli;
        if (!parse_int_tuple_item(capital_context,0,&cap_milli) ||
            !parse_int_tuple_item(capital_context,1,&max_risk_milli) ||
            !parse_int_tuple_item(capital_context,2,&max_position_cap_milli) ||
            !parse_int_tuple_item(fee_context,0,&buy_fee_ppm) ||
            !parse_int_tuple_item(fee_context,1,&sell_fee_ppm) ||
            !parse_int_tuple_item(fee_context,2,&tax_ppm) ||
            !parse_int_tuple_item(fee_context,3,&min_fee_milli) ||
            !parse_int_tuple_item(fee_context,4,&rounding_mode_code) ||
            !parse_int_tuple_item(fee_context,5,&min_entry_notional_milli) ||
            rounding_mode_code < 0) {
            Py_DECREF(core);
            Py_INCREF(Py_None); PyList_SET_ITEM(out, i, Py_None); continue;
        }

        int64_t limit_milli = PyLong_AsLongLong(PyTuple_GET_ITEM(core,11));
        int64_t stop_milli = PyLong_AsLongLong(PyTuple_GET_ITEM(core,12));
        if (PyErr_Occurred()) {
            PyErr_Clear(); Py_DECREF(core); Py_INCREF(Py_None); PyList_SET_ITEM(out, i, Py_None); continue;
        }
        int64_t qty = calc_position_size_exact_core(
            limit_milli, stop_milli, cap_milli, max_risk_milli, max_position_cap_milli,
            buy_fee_ppm, sell_fee_ppm, tax_ppm, min_fee_milli, rounding_mode_code == 1,
            (int64_t)broker_fee_ppm_ll, (int64_t)broker_min_fee_milli_ll
        );
        PyObject *max_qty_obj = PyTuple_GET_ITEM(core,7);
        if (max_qty_obj != Py_None) {
            int64_t max_qty = 0;
            if (!object_to_int64_noerr(max_qty_obj, &max_qty)) {
                Py_DECREF(core); Py_INCREF(Py_None); PyList_SET_ITEM(out, i, Py_None); continue;
            }
            if (max_qty > 0 && qty > max_qty) qty = max_qty;
        }
        qty = apply_board_lot_preferred_qty_native(limit_milli, qty, min_entry_notional_milli);
        if (qty <= 0) {
            Py_DECREF(core); Py_INCREF(Py_False); PyList_SET_ITEM(out, i, Py_False); continue;
        }

        const int64_t gross_buy = limit_milli * qty;
        const int64_t broker_fee = fee_floor_twd_milli(
            gross_buy, (int64_t)broker_fee_ppm_ll, (int64_t)broker_min_fee_milli_ll
        );
        const int64_t proj_cost_milli = gross_buy + broker_fee;
        double limit_price = PyFloat_AsDouble(PyTuple_GET_ITEM(core,0));
        if (PyErr_Occurred() || !isfinite(limit_price) || limit_price <= 0.0 || !isfinite(y_close)) {
            PyErr_Clear(); Py_DECREF(core); Py_INCREF(Py_None); PyList_SET_ITEM(out, i, Py_None); continue;
        }
        double sort_value = (y_close / limit_price - 1.0) * 100.0;
        if (sort_value < 0.0) sort_value = 0.0;

        PyObject *result = PyTuple_New(13);
        if (result == NULL) { Py_DECREF(core); Py_DECREF(items); Py_DECREF(out); return NULL; }
#define COPY_CORE(dst, src) do { PyObject *_o=PyTuple_GET_ITEM(core,(src)); Py_INCREF(_o); PyTuple_SET_ITEM(result,(dst),_o); } while(0)
        COPY_CORE(0,0); /* limit */
        COPY_CORE(1,1); /* sl */
        COPY_CORE(2,2); /* trail */
        COPY_CORE(3,3); /* target */
        COPY_CORE(4,4); /* atr */
        COPY_CORE(5,6); /* signal date */
        COPY_CORE(6,7); /* max qty */
        COPY_CORE(7,8); /* entry source */
        COPY_CORE(8,11); /* limit milli */
        COPY_CORE(9,12); /* stop milli */
#undef COPY_CORE
        PyTuple_SET_ITEM(result,10,PyLong_FromLongLong((long long)qty));
        PyTuple_SET_ITEM(result,11,PyLong_FromLongLong((long long)proj_cost_milli));
        PyTuple_SET_ITEM(result,12,PyFloat_FromDouble(sort_value));
        Py_DECREF(core);
        if (PyErr_Occurred()) { Py_DECREF(result); Py_DECREF(items); Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i, result);
    }
    Py_DECREF(items);
    return out;
}



static int packed_bool1_at(PyObject *packed_data, const char *field, Py_ssize_t pos, int *out) {
    if (!PyDict_Check(packed_data) || pos < 0) return 0;
    PyObject *packed_flag = PyDict_GetItemString(packed_data, "_packed_market_data");
    if (packed_flag == NULL || packed_flag != Py_True) return 0;
    PyObject *array_obj = PyDict_GetItemString(packed_data, field);
    if (array_obj == NULL) return 0;
    Py_buffer view;
    if (PyObject_GetBuffer(array_obj, &view, PyBUF_CONTIG_RO) != 0) {
        PyErr_Clear();
        return 0;
    }
    int ok = 0;
    if (view.itemsize == 1 && pos < view.len) {
        const unsigned char raw = *(((const unsigned char *)view.buf) + pos);
        *out = raw ? 1 : 0;
        ok = 1;
    }
    PyBuffer_Release(&view);
    return ok;
}

/*
 * Optimizer-only existing-position common no-event fast path.
 *
 * The canonical Python settlement path is retained for any day that can create
 * an observable execution event (pending exit, indicator exit, STOP or TP), for
 * non-packed/legacy market data, or for positions carrying a private data
 * context.  This native batch only advances the very common no-event day:
 * previous-bar trailing state + last close cache.  It intentionally performs no
 * cash/accounting operation.
 */
static PyObject *native_settle_position_noevent_cached_batch(PyObject *self, PyObject *args) {
    PyObject *portfolio, *tickers_obj, *all_dfs_fast, *today, *default_params;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOO", &portfolio, &tickers_obj, &all_dfs_fast, &today, &default_params)) {
        return NULL;
    }
    if (!PyDict_Check(portfolio) || !PyDict_Check(all_dfs_fast)) {
        PyErr_SetString(PyExc_TypeError, "expected portfolio/all_dfs_fast dict inputs");
        return NULL;
    }
    PyObject *tickers = PySequence_Fast(tickers_obj, "tickers must be a sequence");
    if (tickers == NULL) return NULL;
    PyObject *fallback = PyList_New(0);
    if (fallback == NULL) { Py_DECREF(tickers); return NULL; }

    PyObject *last_params = NULL;
    double last_atr_times_trail = 0.0;
    double last_tp_percent = 0.0;
    int have_last_params = 0;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(tickers);
    PyObject **items = PySequence_Fast_ITEMS(tickers);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *ticker = items[i];
        PyObject *position = PyDict_GetItemWithError(portfolio, ticker);
        if (position == NULL) {
            if (PyErr_Occurred()) goto error;
            continue;
        }
        if (!PyDict_Check(position)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        int64_t qty = 0;
        if (!dict_int64_default(position, "qty", 0, &qty)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        if (qty <= 0) continue;

        PyObject *entry_context = dict_get_borrowed(position, "_entry_context");
        if (entry_context != NULL && PyDict_Check(entry_context)) {
            PyObject *private_data = PyDict_GetItemString(entry_context, "all_dfs_fast");
            if (private_data != NULL && private_data != Py_None) {
                if (PyList_Append(fallback, ticker) != 0) goto error;
                continue;
            }
        }

        PyObject *pending = dict_get_borrowed(position, "pending_exit_action");
        if (pending != NULL && pending != Py_None) {
            int pending_truth = PyObject_IsTrue(pending);
            if (pending_truth < 0) { PyErr_Clear(); pending_truth = 1; }
            if (pending_truth) {
                if (PyList_Append(fallback, ticker) != 0) goto error;
                continue;
            }
        }

        PyObject *packed = PyDict_GetItemWithError(all_dfs_fast, ticker);
        if (packed == NULL) {
            if (PyErr_Occurred()) goto error;
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        if (!PyDict_Check(packed) || PyDict_GetItemString(packed, "_packed_market_data") != Py_True) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        PyObject *date_to_pos = PyDict_GetItemString(packed, "date_to_pos");
        if (!PyDict_Check(date_to_pos)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        PyObject *pos_obj = PyDict_GetItemWithError(date_to_pos, today);
        if (pos_obj == NULL) {
            if (PyErr_Occurred()) goto error;
            continue;
        }
        int64_t t_pos_i64 = 0;
        if (!object_to_int64_noerr(pos_obj, &t_pos_i64) || t_pos_i64 <= 0) continue;
        const Py_ssize_t t_pos = (Py_ssize_t)t_pos_i64;
        const Py_ssize_t y_pos = t_pos - 1;

        double y_atr, y_high, t_high, t_low, t_close;
        int y_ind_sell = 0;
        if (!packed_float64_at(packed, "ATR", y_pos, &y_atr) ||
            !packed_float64_at(packed, "High", y_pos, &y_high) ||
            !packed_float64_at(packed, "High", t_pos, &t_high) ||
            !packed_float64_at(packed, "Low", t_pos, &t_low) ||
            !packed_float64_at(packed, "Close", t_pos, &t_close) ||
            !packed_bool1_at(packed, "ind_sell_signal", y_pos, &y_ind_sell)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        if (!(isfinite(y_atr) && isfinite(y_high) && isfinite(t_high) && isfinite(t_low) && isfinite(t_close))) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        if (y_ind_sell) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }

        PyObject *params_obj = dict_get_borrowed(position, "_entry_params_obj");
        if (params_obj == NULL || params_obj == Py_None) params_obj = default_params;
        double atr_times_trail = 0.0;
        double tp_percent = 0.0;
        if (have_last_params && params_obj == last_params) {
            atr_times_trail = last_atr_times_trail;
            tp_percent = last_tp_percent;
        } else {
            if (!resolve_atr_times_trail_native(params_obj, &atr_times_trail) ||
                !get_attr_double_native(params_obj, "tp_percent", &tp_percent)) {
                if (PyList_Append(fallback, ticker) != 0) goto error;
                continue;
            }
            last_params = params_obj;
            last_atr_times_trail = atr_times_trail;
            last_tp_percent = tp_percent;
            have_last_params = 1;
        }

        PyObject *security_profile = dict_get_borrowed(position, "security_profile");
        int is_fund = 0;
        if (!profile_is_fund(security_profile, &is_fund)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        int64_t entry_fill_milli = 0, previous_high_milli = 0, previous_trailing_milli = 0;
        int64_t previous_sl_milli = 0, initial_stop_milli = 0, tp_half_milli = 0;
        if (!dict_int64_default(position, "entry_fill_price_milli", 0, &entry_fill_milli) ||
            !dict_int64_default(position, "highest_high_since_entry_milli", entry_fill_milli, &previous_high_milli) ||
            !dict_int64_default(position, "trailing_stop_milli", 0, &previous_trailing_milli) ||
            !dict_int64_default(position, "sl_milli", 0, &previous_sl_milli) ||
            !dict_int64_default(position, "initial_stop_milli", 0, &initial_stop_milli) ||
            !dict_int64_default(position, "tp_half_milli", 0, &tp_half_milli)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }
        PyObject *sold_half_obj = dict_get_borrowed(position, "sold_half");
        int sold_half = sold_half_obj == NULL ? 0 : PyObject_IsTrue(sold_half_obj);
        if (sold_half < 0) { PyErr_Clear(); if (PyList_Append(fallback, ticker) != 0) goto error; continue; }

        int64_t highest_high_milli = previous_high_milli;
        int64_t trailing_milli = previous_trailing_milli;
        if (price_to_milli_double(y_high) > previous_high_milli) {
            highest_high_milli = price_to_milli_double(y_high);
            const double candidate_trail = ((double)highest_high_milli / 1000.0) - (y_atr * atr_times_trail);
            int64_t candidate_trail_milli = 0;
            if (!round_up_price_to_tick_milli(candidate_trail, is_fund, &candidate_trail_milli)) {
                if (PyList_Append(fallback, ticker) != 0) goto error;
                continue;
            }
            if (candidate_trail_milli > trailing_milli) trailing_milli = candidate_trail_milli;
        }
        const int64_t sl_milli = initial_stop_milli > trailing_milli ? initial_stop_milli : trailing_milli;
        const int64_t t_low_milli = price_to_milli_double(t_low);
        const int64_t t_high_milli = price_to_milli_double(t_high);
        int64_t half_qty = 0;
        if (tp_percent > 0.0) {
            const double raw_half = floor(((double)qty) * tp_percent);
            if (isfinite(raw_half) && raw_half > 0.0 && raw_half < (double)qty) {
                half_qty = (int64_t)raw_half;
            }
        }
        if (t_low_milli <= sl_milli ||
            (!sold_half && half_qty > 0 && tp_half_milli > 0 && t_high_milli >= tp_half_milli)) {
            if (PyList_Append(fallback, ticker) != 0) goto error;
            continue;
        }

        if (highest_high_milli != previous_high_milli &&
            !set_dict_int64(position, "highest_high_since_entry_milli", highest_high_milli)) goto error;
        if (trailing_milli != previous_trailing_milli &&
            !set_dict_int64(position, "trailing_stop_milli", trailing_milli)) goto error;
        if (sl_milli != previous_sl_milli &&
            !set_dict_int64(position, "sl_milli", sl_milli)) goto error;
        if (PyDict_DelItemString(position, "_last_exec_contexts") != 0 &&
            !PyErr_ExceptionMatches(PyExc_KeyError)) goto error;
        PyErr_Clear();
        if (!set_dict_double(position, "last_px", t_close)) goto error;
    }

    Py_DECREF(tickers);
    return fallback;
error:
    Py_DECREF(tickers);
    Py_DECREF(fallback);
    return NULL;
}

static PyMethodDef NativeOptimizerMethods[] = {
    {
        "supertrend_direction_into",
        native_supertrend_direction_into,
        METH_VARARGS,
        "Fill an int8 direction buffer with the fused exact-order SuperTrend recursion."
    },
    {
        "rolling_mean_shift1_into",
        native_rolling_mean_shift1_into,
        METH_VARARGS,
        "Fill a float64 output buffer with the shifted rolling mean."
    },
    {
        "rolling_mean_std0_into",
        native_rolling_mean_std0_into,
        METH_VARARGS,
        "Fill float64 mean/std buffers with rolling population statistics."
    },
    {
        "rolling_max_shift1_into",
        native_rolling_max_shift1_into,
        METH_VARARGS,
        "Fill a float64 output buffer with the exact shifted rolling maximum."
    },
    {
        "ewm_adjust_false_from_seed_into",
        native_ewm_adjust_false_from_seed_into,
        METH_VARARGS,
        "Fill a float64 output buffer with the exact adjust=False EWM recurrence from a caller-provided seed."
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
    {
        "cleanup_live_shadow_inplace",
        native_cleanup_live_shadow_inplace,
        METH_VARARGS,
        "Advance one optimizer continuation live-shadow state in place; None requests canonical Python fallback."
    },
    {
        "cleanup_live_shadow_batch",
        native_cleanup_live_shadow_batch,
        METH_VARARGS,
        "Advance optimizer live-shadow cleanup items in one native batch and delete terminal signals in place."
    },
    {
        "cleanup_live_shadow_cached_batch",
        native_cleanup_live_shadow_cached_batch,
        METH_VARARGS,
        "Advance cached optimizer live-shadow states directly from packed market data; return canonical fallback tickers."
    },
    {
        "mark_to_market_cached_batch",
        native_mark_to_market_cached_batch,
        METH_VARARGS,
        "Return exact cached mark-to-market net liquidation for supported optimizer positions plus Python fallbacks."
    },
    {
        "settle_position_noevent_cached_batch",
        native_settle_position_noevent_cached_batch,
        METH_VARARGS,
        "Advance optimizer existing-position no-event settlement days directly from packed market data; return canonical fallback tickers."
    },
    {
        "resolve_live_extended_candidate_core",
        native_resolve_live_extended_candidate_core,
        METH_VARARGS,
        "Resolve live continuation frozen geometry, daily orderability, and milli values in one optimizer call."
    },
    {
        "resolve_live_extended_candidate_batch_sized",
        native_resolve_live_extended_candidate_batch_sized,
        METH_VARARGS,
        "Resolve and exact-size optimizer live continuation candidates in one native batch."
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

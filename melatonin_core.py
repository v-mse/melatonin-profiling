"""
melatonin_core.py
=================
App part 1 ("I already have measurements"): fit the bounded GP to a user's melatonin samples and report
onset, offset, peak time, peak value and AUC with uncertainties.

Same model as the research notebook (melatonin_clean_curvature_v2.ipynb):
  * the GP is fitted on a latent scale z = softplus^-1(y) (or log / none, from the model file),
    min-max normalised, kernel Constant * (RBF + ExpSineSquared) with bounds learned from the kernel profiles,
  * every sample gets its own noise variance (delta method): assay noise inflated by `alpha_mult` plus a part
    proportional to the measured value (`alpha_cv`); no noise kernel,
  * onset / offset: threshold rule of the notebook (`dlmo` / `dlmoff`), on the notebook's grid.

The app needs this file and `app_model.json` (exported once by the notebook). It never sees the reference profiles.

Time axis: hours relative to habitual sleep onset (0 = sleep onset, negative = before).

    model = load_app_model('app_model.json')                    # once at start-up
    res = analyze_measurements('measurements.csv', '23:30', model)
    res['curve_table'], res['metrics_table'], res['warnings']   # DataFrames / list of str
    plot_fit(res), plot_metrics(res)                            # matplotlib figures
"""
import io
import json
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, ExpSineSquared

trapz = getattr(np, 'trapezoid', None) or np.trapz        # numpy 1.x / 2.x

SUPPORTED_FORMAT = 3

# Defaults = notebook settings. load_app_model overwrites them with the values in app_model.json.
CONFIG = {
    'noise_std': 0.3,                  # assay noise sd (ng/L)
    'transform': 'softplus',           # 'softplus', 'log' or None
    'softplus_beta': 1.0,
    'y_floor': 0.01,                   # ng/L: values below are set to this before the inverse transform
    'alpha_mult': 7,                   # inflates the assay-noise variance of the GP (calibration of the posterior width)
    'alpha_cv': 0.07,                  # extra noise sd proportional to the measured value
    'window': (-12.0, 15.0),           # model window (h rel. to sleep onset)
    'grid_step': 0.5,
    'baseline_window': (-10.0, -5.0),  # onset / offset threshold: baseline points are taken in here
    'n_baseline': 3,
    'baseline': 'equidistant',         # 'equidistant' or 'random'
    'n_sd': 2,                         # threshold = baseline mean + n_sd * max(sd, noise_std)
    'onset_skip_h': 3.0,               # onset is searched from grid start + this
    'offset_skip_h': 2.0,              # offset is searched from peak + this
    'auc_window': None,                # None = whole model window
    'peak_window': (0.0, 8.0),         # the peak is searched in here (None = whole model window)
    'n_draws': 100,                    # posterior curves for the uncertainties
    'n_restarts': 5,
}
PARAMS = ['constant', 'rbf_length', 'per_length', 'per_period']
GENERIC_BOUNDS = {'constant': [1e-3, 1e3, 1.0], 'rbf_length': [1.0, 100.0, 5.68],     # only without a model file
                  'per_length': [1e-2, 1e2, 1.0], 'per_period': [20.0, 28.0, 24.0]}
UNITS = {'ng/L': 1.0, 'pg/mL': 1.0, 'pmol/L': 232.28 / 1000, 'nmol/L': 232.28}      # -> ng/L
METRICS = [('onset', 'Melatonin onset', 'h', True), ('offset', 'Melatonin offset', 'h', True),
           ('peak_time', 'Peak time', 'h', True), ('peak_value', 'Peak value', 'ng/L', False),
           ('auc', 'AUC', 'ng·h/L', False)]


# ======================================================================================= model file

def load_app_model(path_or_dict):
    """Read app_model.json (written by the notebook) and set CONFIG from it. Returns the model dict."""
    if isinstance(path_or_dict, dict):
        model = path_or_dict
    else:
        with open(path_or_dict) as f:
            model = json.load(f)
    version = model.get('format_version', 1)
    if version != SUPPORTED_FORMAT:
        raise ValueError(f'app_model.json has format version {version}, this app needs {SUPPORTED_FORMAT}. '
                         'Re-export it with the current notebook.')
    for key, value in model['config'].items():
        CONFIG[key] = tuple(value) if isinstance(value, list) else value
    return model


def model_grid(model=None):
    """The notebook's common grid (onset / offset depend on its step, so the app uses exactly this grid)."""
    if model is not None:
        return np.asarray(model['grid']['hours'], float)
    lo, hi = CONFIG['window']
    step = CONFIG['grid_step']
    return np.arange(lo, hi + step / 2, step)


def model_bounds(model=None):
    return {k: tuple(v) for k, v in (GENERIC_BOUNDS if model is None else model['bounds']).items()}


# ======================================================================================= transform (notebook)

def to_latent(y):
    """ng/L -> latent scale (the GP is fitted on this scale)."""
    if CONFIG['transform'] is None:
        return np.asarray(y, float)
    y = np.maximum(np.asarray(y, float), CONFIG['y_floor'])
    if CONFIG['transform'] == 'softplus':
        b = CONFIG['softplus_beta']
        return y + np.log1p(-np.exp(-b * y)) / b
    if CONFIG['transform'] == 'log':
        return np.log(y)
    raise ValueError("transform must be 'softplus', 'log' or None")


def from_latent(z):
    """latent scale -> ng/L (always >= 0 for softplus / log)."""
    if CONFIG['transform'] is None:
        return np.asarray(z, float)
    if CONFIG['transform'] == 'softplus':
        b = CONFIG['softplus_beta']
        return np.logaddexp(0, b * np.asarray(z, float)) / b
    if CONFIG['transform'] == 'log':
        return np.exp(z)
    raise ValueError("transform must be 'softplus', 'log' or None")


def latent_slope(y):
    """dz/dy at y: latent noise sd = assay noise sd * dz/dy (delta method)."""
    if CONFIG['transform'] is None:
        return np.ones_like(np.asarray(y, float))
    y = np.maximum(np.asarray(y, float), CONFIG['y_floor'])
    if CONFIG['transform'] == 'softplus':
        return 1 / (1 - np.exp(-CONFIG['softplus_beta'] * y))
    return 1 / y


def minmax(z):
    z = np.asarray(z, float)
    z_min, z_max = z.min(), z.max()
    if z_max == z_min:
        raise ValueError('All melatonin values are identical; nothing to fit.')
    return (z - z_min) / (z_max - z_min), z_min, z_max


# ======================================================================================= GP (notebook)

def build_kernel(b):
    """Constant * (RBF + Periodic). b = {hyperparameter: (min, max, start value)}."""
    return ConstantKernel(b['constant'][2], b['constant'][:2]) * (
        RBF(b['rbf_length'][2], b['rbf_length'][:2])
        + ExpSineSquared(b['per_length'][2], b['per_period'][2],
                         length_scale_bounds=b['per_length'][:2],
                         periodicity_bounds=b['per_period'][:2]))


def _fit_gp(x, y, kernel, seed, n_restarts):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if len(x) < 2:
        raise ValueError('Need at least 2 measurements (3 or more recommended).')
    z_norm, z_min, z_max = minmax(to_latent(y))
    # noise variance of every sample on the normalised latent scale (delta method)
    alpha = ((CONFIG['alpha_mult'] * CONFIG['noise_std'] ** 2 + (CONFIG['alpha_cv'] * y) ** 2) ** 0.5
             * latent_slope(y) / (z_max - z_min)) ** 2
    gpr = GaussianProcessRegressor(kernel, alpha=alpha, n_restarts_optimizer=n_restarts, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gpr.fit(x.reshape(-1, 1), z_norm)
    return {'gpr': gpr, 'x': x, 'y': y, 'z_min': z_min, 'z_max': z_max}


def predict_curve(fit, grid):
    """Posterior on grid: latent mean / sd; 'mean' and the 95% band in ng/L (monotone transform -> exact)."""
    mean_n, std_n = fit['gpr'].predict(grid.reshape(-1, 1), return_std=True)
    scale = fit['z_max'] - fit['z_min']
    z, s = mean_n * scale + fit['z_min'], std_n * scale
    return {'grid': grid, 'z_mean': z, 'z_std': s, 'mean': from_latent(z),
            'lower': from_latent(z - 1.96 * s), 'upper': from_latent(z + 1.96 * s)}


def fit_sparse(x, y, bounds, seed=0, grid=None):
    fit = _fit_gp(x, y, build_kernel(bounds), seed, CONFIG['n_restarts'])
    fit.update(predict_curve(fit, model_grid() if grid is None else grid))
    return fit


def posterior_draws(fit, n_draws=None, seed=0):
    """Posterior curves, shape (len(grid), n_draws), in ng/L."""
    n_draws = n_draws or CONFIG['n_draws']
    s = fit['gpr'].sample_y(fit['grid'].reshape(-1, 1), n_samples=n_draws, random_state=seed)
    return from_latent(s * (fit['z_max'] - fit['z_min']) + fit['z_min'])


# ======================================================================================= metrics (notebook)

def _baseline_points(h, window, n_baseline, baseline, seed):
    in_win = np.flatnonzero((h >= window[0]) & (h <= window[1]))
    if len(in_win) < n_baseline:
        return list(range(min(n_baseline, len(h))))
    if baseline == 'equidistant':
        return in_win[np.round(np.linspace(0, len(in_win) - 1, n_baseline)).astype(int)].tolist()
    if baseline == 'random':
        return sorted(np.random.default_rng(seed).choice(in_win, n_baseline, replace=False).tolist())
    raise ValueError("baseline must be 'equidistant' or 'random'")


def _threshold(h, y, seed=0):
    pick = _baseline_points(h, CONFIG['baseline_window'], CONFIG['n_baseline'], CONFIG['baseline'], seed)
    base = y[pick]
    return pick, base.mean() + CONFIG['n_sd'] * max(base.std(ddof=1), CONFIG['noise_std'])


def onset_time(h, y, seed=0):
    """First rise above the threshold that stays above (2 grid points), after the baseline points."""
    pick, threshold = _threshold(h, y, seed)
    first = max(pick[-1] + 1, int(np.searchsorted(h, h[0] + CONFIG['onset_skip_h'])))
    cand = np.flatnonzero((y[:-1] >= threshold) & (y[1:] >= threshold))
    cand = cand[cand >= max(first, 1)]
    if not len(cand):
        return np.nan
    i = cand[0]
    return h[i - 1] + (threshold - y[i - 1]) / (y[i] - y[i - 1]) * (h[i] - h[i - 1])


def offset_time(h, y, seed=0):
    """First fall below the threshold that stays below, from offset_skip_h after the peak."""
    _, threshold = _threshold(h, y, seed)
    t_min = h[np.argmax(y)] + CONFIG['offset_skip_h']
    cand = np.flatnonzero((y[1:-1] <= threshold) & (y[2:] <= threshold) & (y[:-2] > threshold)) + 1
    cand = cand[cand >= max(1, int(np.searchsorted(h, t_min)))]
    if not len(cand):
        return np.nan
    i = cand[0]
    return h[i - 1] + (y[i - 1] - threshold) / (y[i - 1] - y[i]) * (h[i] - h[i - 1])


def threshold_of(grid, curve):
    """(baseline mask, threshold) of a curve - for plots."""
    pick, thr = _threshold(np.asarray(grid, float), np.asarray(curve, float))
    mask = np.zeros(len(grid), bool)
    mask[pick] = True
    return mask, thr


def curve_metrics(grid, curve, auc_window=None):
    grid, curve = np.asarray(grid, float), np.asarray(curve, float)
    if CONFIG['peak_window'] is None:
        i = np.argmax(curve)
    else:
        in_window = np.flatnonzero((grid >= CONFIG['peak_window'][0]) & (grid <= CONFIG['peak_window'][1]))
        i = in_window[np.argmax(curve[in_window])]
    m = np.ones(len(grid), bool) if auc_window is None else (grid >= auc_window[0]) & (grid <= auc_window[1])
    return {'onset': onset_time(grid, curve), 'offset': offset_time(grid, curve),
            'peak_time': grid[i], 'peak_value': curve[i], 'auc': trapz(curve[m], grid[m])}


def draw_metrics(grid, draws, auc_window=None):
    rows = [curve_metrics(grid, draws[:, j], auc_window) for j in range(draws.shape[1])]
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def summarize(values):
    v = np.asarray(values, float)
    ok = v[np.isfinite(v)]
    if len(ok) == 0:
        return {'mean': np.nan, 'sd': np.nan, 'lo': np.nan, 'hi': np.nan, 'valid': 0.0}
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return {'mean': ok.mean(), 'sd': ok.std(ddof=1) if len(ok) > 1 else 0.0, 'lo': lo, 'hi': hi,
            'valid': len(ok) / len(v)}


# ======================================================================================= clock times

def parse_clock_hours(text):
    """'23:30', '23:30:00', '11:30 PM', '23.5' -> 23.5 (hours after midnight)."""
    s = str(text).strip()
    m = re.fullmatch(r'(\d{1,2})[:.h](\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?', s)
    if m:
        h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if m.group(4):
            if not 1 <= h <= 12:
                raise ValueError(f'Invalid clock time: {text!r}')
            h = h % 12 + (12 if m.group(4).lower() == 'pm' else 0)
        if h > 24 or mi > 59 or se > 59:
            raise ValueError(f'Invalid clock time: {text!r}')
        return h + mi / 60 + se / 3600
    m = re.fullmatch(r'(\d{1,2})\s*([AaPp][Mm])', s)
    if m:
        return int(m.group(1)) % 12 + (12 if m.group(2).lower() == 'pm' else 0)
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        raise ValueError(f'Cannot read clock time {text!r}; use e.g. 23:30.') from None


def hours_to_hhmm(h):
    if h is None or not np.isfinite(h):
        return ''
    minutes = int(round(h * 60)) % (24 * 60)
    return f'{minutes // 60:02d}:{minutes % 60:02d}'


# ======================================================================================= user csv

def _read_text(file):
    if isinstance(file, (bytes, bytearray)):
        return bytes(file).decode('utf-8-sig', errors='replace')
    if hasattr(file, 'read'):
        content = file.read()
        return content.decode('utf-8-sig', errors='replace') if isinstance(content, bytes) else content
    with open(file, encoding='utf-8-sig', errors='replace') as f:
        return f.read()


def _parse_time(text):
    """-> ('clock', hours after midnight) or ('datetime', datetime)."""
    s = text.strip()
    if re.search(r'\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}', s):
        return 'datetime', pd.to_datetime(s, dayfirst=not re.match(r'\d{4}-', s)).to_pydatetime()
    return 'clock', parse_clock_hours(s)


def read_user_csv(file, sleep_time, unit='ng/L'):
    """Two columns, chronological: time, melatonin. Header optional; delimiter , ; or tab; decimal comma allowed.
    Returns (DataFrame Hour / Melatonin / Clock, sleep time in clock hours, warnings)."""
    if unit not in UNITS:
        raise ValueError(f'Unknown unit {unit!r}. Available: {list(UNITS)}')
    sleep_h = parse_clock_hours(sleep_time) % 24
    lines = [ln for ln in _read_text(file).splitlines() if ln.strip()]
    if not lines:
        raise ValueError('The file is empty.')
    sample = '\n'.join(lines[:20])
    delim = ';' if ';' in sample else '\t' if '\t' in sample else ','

    warns, kinds, times, values, skipped = [], [], [], [], []
    for k, line in enumerate(lines):
        parts = [p.strip().strip('"') for p in line.split(delim)]
        parts = [p for p in parts if p != '']
        if delim == ',' and len(parts) == 3 and re.fullmatch(r'\d+', parts[2]):   # "21:00,2,5" = decimal comma
            parts = [parts[0], parts[1] + '.' + parts[2]]
        try:
            if len(parts) < 2:
                raise ValueError
            kind, t = _parse_time(parts[0])
            v = float(parts[1].replace(',', '.'))
            if not np.isfinite(v):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            if k > 0 or len(parts) < 2:
                skipped.append(k + 1)
            continue                                                   # first unreadable row = header
        kinds.append(kind); times.append(t); values.append(v)
    if skipped:
        warns.append(f'{len(skipped)} unreadable row(s) skipped (line {", ".join(map(str, skipped[:10]))}'
                     + (' ...' if len(skipped) > 10 else '') + ').')
    if len(values) < 2:
        raise ValueError('Fewer than 2 readable measurements (time, melatonin) found.')
    if len(set(kinds)) > 1:
        raise ValueError('Mixed time formats: use either clock times or full date-times in all rows.')

    if kinds[0] == 'datetime':
        t0 = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
        clock = np.array([(t - t0).total_seconds() / 3600 for t in times])
        order = np.argsort(clock, kind='stable')
        clock, values = clock[order], np.asarray(values)[order]
    else:
        clock = np.array(times, float)
        for i in range(1, len(clock)):                                 # clock goes back -> next day
            while clock[i] < clock[i - 1]:
                clock[i:] += 24
        values = np.asarray(values)
    sleep_abs = sleep_h + 24 * np.round((np.median(clock) - sleep_h) / 24)   # the sleep onset closest to the samples
    hours = clock - sleep_abs

    df = pd.DataFrame({'Hour': hours, 'Melatonin': values * UNITS[unit], 'Clock': [hours_to_hhmm(c) for c in clock]})
    df = df.groupby('Hour', as_index=False).agg({'Melatonin': 'mean', 'Clock': 'first'})
    if len(df) < len(hours):
        warns.append('Samples with the same time were averaged.')
    if (df['Melatonin'] < 0).any():
        warns.append('Negative melatonin values were set to the detection floor for the fit.')
    if len(df) < 3:
        warns.append('Only 2 samples: the curve is mostly determined by the model; 3 or more are recommended.')
    lo, hi = CONFIG['window']
    outside = df[(df['Hour'] < lo) | (df['Hour'] > hi)]
    if len(outside):
        warns.append(f'{len(outside)} sample(s) lie outside the model window ({hours_to_hhmm(lo + sleep_h)} - '
                     f'{hours_to_hhmm(hi + sleep_h)}); they are used in the fit, but results are only reported '
                     'inside the window. Check the time of sleep.')
    return df, sleep_h, warns


# ======================================================================================= analysis

def fit_measurements(hours, values, model, seed=0):
    """Fit + posterior draws for samples at `hours` (rel. to sleep onset), values in ng/L. Returns (fit, draws)."""
    fit = fit_sparse(hours, values, model_bounds(model), seed=seed, grid=model_grid(model))
    return fit, posterior_draws(fit, seed=seed)


def analyze_measurements(file, sleep_time, model, unit='ng/L', seed=0):
    """Whole part-1 workflow. Returns a dict:
    data, curve_table, metrics_table, warnings (for the UI) and sleep_h, fit, draws, metric_draws, point (internals)."""
    warns = []
    if model is None:
        warns.append('No app_model.json loaded: generic kernel bounds are used instead of bounds learned from '
                     'dense reference profiles. Do not use these results.')
    df, sleep_h, w = read_user_csv(file, sleep_time, unit)
    warns += w

    fit, draws = fit_measurements(df['Hour'].values, df['Melatonin'].values, model, seed)
    grid = fit['grid']
    metric_draws = draw_metrics(grid, draws, CONFIG['auc_window'])
    point = curve_metrics(grid, fit['mean'], CONFIG['auc_window'])     # metrics of the fitted (median) curve

    curve_table = pd.DataFrame({
        'hour_rel_sleep': grid, 'clock_time': [hours_to_hhmm(h + sleep_h) for h in grid],
        'mean': fit['mean'], 'sd': draws.std(axis=1, ddof=1), 'lower95': fit['lower'], 'upper95': fit['upper']})

    rows = []
    for key, label, unit_, is_time in METRICS:
        s = summarize(metric_draws[key])
        row = {'metric': label, 'unit': unit_, 'estimate': point[key], 'posterior_mean': s['mean'], 'sd': s['sd'],
               'ci95_low': s['lo'], 'ci95_high': s['hi'], 'valid_fraction': s['valid']}
        if is_time:
            row.update(estimate_clock=hours_to_hhmm(point[key] + sleep_h),
                       ci95_low_clock=hours_to_hhmm(s['lo'] + sleep_h), ci95_high_clock=hours_to_hhmm(s['hi'] + sleep_h))
        rows.append(row)
    metrics_table = pd.DataFrame(rows)

    weak = metrics_table.loc[metrics_table['valid_fraction'] < 0.9, 'metric']
    if len(weak):
        warns.append('No clear threshold crossing in more than 10% of the posterior curves for: '
                     + ', '.join(weak) + '. Treat these intervals with caution.')
    bw = CONFIG['baseline_window']
    if not ((df['Hour'] >= bw[0] - 1) & (df['Hour'] <= bw[1] + 1)).any():
        warns.append(f'No sample near the baseline window ({hours_to_hhmm(bw[0] + sleep_h)} - '
                     f'{hours_to_hhmm(bw[1] + sleep_h)}): the onset / offset threshold is extrapolated, so their '
                     'intervals are wide.')
    return {'data': df, 'curve_table': curve_table, 'metrics_table': metrics_table, 'warnings': warns,
            'sleep_h': sleep_h, 'fit': fit, 'draws': draws, 'metric_draws': metric_draws, 'point': point}


# ======================================================================================= plots

def _clock_axis(ax, x, sleep_h):
    first = np.ceil((x.min() + sleep_h) / 2) * 2 - sleep_h            # ticks on even clock hours
    ticks = np.arange(first, x.max() + 0.1, 2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([hours_to_hhmm(t + sleep_h) for t in ticks])
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel('Clock time')


def plot_fit(res, ax=None, clock_axis=True):
    """Fitted curve, 95% band and the samples."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 4.2))
    ct, d = res['curve_table'], res['data']
    x = ct['hour_rel_sleep'].values
    ax.fill_between(x, ct['lower95'], ct['upper95'], color='tab:blue', alpha=0.2, label='95% band')
    ax.plot(x, ct['mean'], color='tab:blue', label='fitted curve')
    ax.scatter(d['Hour'], d['Melatonin'], color='k', zorder=5, label='measurements')
    ax.axvline(0, color='gray', ls=':', lw=1)
    ax.set_ylabel('Melatonin (ng/L)')
    if clock_axis:
        _clock_axis(ax, x, res['sleep_h'])
    else:
        ax.set_xlabel('Hours relative to sleep onset')
    ax.legend(fontsize=8, loc='upper left')
    return fig


def plot_metrics(res, ax=None, clock_axis=True):
    """plot_fit plus onset / offset / peak (estimate and 95% interval), peak value and threshold."""
    fig = plot_fit(res, ax, clock_axis)
    ax = fig.axes[0] if ax is None else ax
    mt = res['metrics_table'].set_index('metric')
    for key, label, color in [('onset', 'Melatonin onset', 'tab:red'), ('offset', 'Melatonin offset', 'tab:green'),
                              ('peak_time', 'Peak time', 'tab:purple')]:
        r = mt.loc[label]
        found = np.isfinite(r['estimate'])
        if np.isfinite(r['ci95_low']):
            ax.axvspan(r['ci95_low'], r['ci95_high'], color=color, alpha=0.12,
                       label=None if found else f"{label}: not in fitted curve; 95% of curves "
                                                f"[{r.get('ci95_low_clock', '')} - {r.get('ci95_high_clock', '')}]")
        if found:
            ax.axvline(r['estimate'], color=color, ls='--', lw=1.5,
                       label=f"{label} {r.get('estimate_clock', '')} [{r.get('ci95_low_clock', '')} - "
                             f"{r.get('ci95_high_clock', '')}]")
    pv = mt.loc['Peak value']
    ax.errorbar([res['point']['peak_time']], [pv['estimate']], fmt='o', color='tab:purple', capsize=4, zorder=6,
                yerr=[[max(pv['estimate'] - pv['ci95_low'], 0)], [max(pv['ci95_high'] - pv['estimate'], 0)]])
    _, thr = threshold_of(res['curve_table']['hour_rel_sleep'].values, res['curve_table']['mean'].values)
    ax.axhline(thr, color='gray', ls='-.', lw=1, label=f'threshold {thr:.2f} ng/L')
    auc = mt.loc['AUC']
    ax.set_title(f"AUC {auc['estimate']:.0f} ng·h/L [{auc['ci95_low']:.0f} - {auc['ci95_high']:.0f}]", fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    return fig

"""
melatonin_recommend.py
======================
App part 2 ("I don't have measurements yet"): recommended sampling times.

The recommended times are those of MPCW (melatonin profile curvature-weighted sampling), restricted to the time
windows of the chosen preset (e.g. no sampling during sleep). MPCW is one design for all metrics: it depends on the
number of samples n and on the preset only. Everything is only LOOKED UP in the table model['recommend'] that the
notebook computed offline (section "UI"); nothing is fitted here. Needs melatonin_core.py and app_model.json.

    import melatonin_core as mc, melatonin_recommend as rec
    model = mc.load_app_model('app_model.json')
    # A) I can afford n samples
    a = rec.recommend_for_budget(model, '23:30', n_samples=6, metrics=['onset', 'auc'], preset='any_time')
    # B) I need this accuracy (minutes for onset / offset / peak_time, percent for peak_value / auc)
    b = rec.recommend_for_accuracy(model, '23:30', {'onset': 30, 'auc': 10}, preset='any_time')
    a['times'], a['accuracy'], a['error_vs_n'], a['message']      # DataFrames / text
    rec.plot_recommended_times(model, b)

Errors in the table come from the simulation study: the test profiles were sampled with each design under the same
window restriction (random seeds for the equidistant and QMC designs), the profile was reconstructed and its metrics
compared with the dense reference.
The MPCW template and the kernel bounds are built from the kernel profiles only, so the test profiles are new
profiles for the method and the errors are honest estimates for a new person.
  expected_error = mean over the test profiles of the absolute error,
  error_80pct    = 80th percentile over the test profiles (the error that 80% of the profiles stay below).
"""
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import melatonin_core as mc

METRIC_KEYS = ['onset', 'offset', 'peak_time', 'peak_value', 'auc']
METRIC_LABELS = {'onset': 'Melatonin onset', 'offset': 'Melatonin offset', 'peak_time': 'Peak time',
                 'peak_value': 'Peak value', 'auc': 'AUC'}
TIME_METRICS = ['onset', 'offset', 'peak_time']     # stored in hours -> shown in minutes; others: fraction -> %
PRESET_LABELS = {'any_time': 'Any time', 'evening_only': 'Evening only (before sleep)',
                 'not_during_sleep': 'Not during sleep (0 to +8 h)'}
COMPARE = [('equidistant', 'Equidistant'), ('qmc', 'QMC')]
INCLUDE_QMC = False       # True: the tables of a result also get qmc_* columns (the interface expects only the equidistant ones)


def _unit(key):
    return 'min' if key in TIME_METRICS else '%'


def _to_ui(key, value):
    if value is None:
        return np.nan
    return value * 60 if key in TIME_METRICS else value * 100


def _from_ui(key, value):
    return value / 60 if key in TIME_METRICS else value / 100


def _rec(model):
    if model is None or 'recommend' not in model:
        raise ValueError('app_model.json has no recommendation table; run the notebook section "UI" and export again.')
    return model['recommend']


def list_presets(model):
    """{preset: allowed sampling windows [(from, to), ...] in hours relative to sleep onset}"""
    return {name: [tuple(w) for w in windows] for name, windows in _rec(model)['presets'].items()}


def available_n(model):
    return [int(n) for n in _rec(model)['n_values']]


def _check(model, preset, metrics):
    rec = _rec(model)
    if preset not in rec['presets']:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(rec['presets'])}")
    metrics = list(METRIC_KEYS if metrics is None else metrics)
    if not metrics:
        raise ValueError('Choose at least one metric.')
    for m in metrics:
        if m not in METRIC_KEYS:
            raise ValueError(f"Unknown metric '{m}'. Available: {METRIC_KEYS}")
    return [m for m in METRIC_KEYS if m in metrics]                  # canonical order, no duplicates


def _entry(model, preset, n):
    entries = _rec(model)['entries'][preset]
    if str(n) not in entries:
        raise ValueError(f'No recommendation for n = {n}. Available: {available_n(model)}')
    return entries[str(n)]


def _result(model, sleep_time, preset, n, metrics, message, reached=None):
    sleep_h = mc.parse_clock_hours(sleep_time) % 24
    entry = _entry(model, preset, n)
    times = np.asarray(entry['times'], float)
    times_df = pd.DataFrame({'sample': np.arange(1, len(times) + 1), 'hour_rel_sleep': times,
                             'clock_time': [mc.hours_to_hhmm(t + sleep_h) for t in times]})
    rows = []
    for j, key in enumerate(METRIC_KEYS):
        row = {'metric': METRIC_LABELS[key], 'unit': _unit(key), 'optimised_for': key in metrics,
               'expected_error': _to_ui(key, entry['mean_err'][j]),
               'error_80pct': _to_ui(key, entry['p80_err'][j])}
        for name, _ in COMPARE:
            if name in entry and (name == 'equidistant' or INCLUDE_QMC):
                row[f'{name}_expected_error'] = _to_ui(key, entry[name]['mean_err'][j])
                row[f'{name}_error_80pct'] = _to_ui(key, entry[name]['p80_err'][j])
        rows.append(row)
    vs_n = []
    for nn in available_n(model):
        e = _entry(model, preset, nn)
        for key in metrics:
            j = METRIC_KEYS.index(key)
            row = {'n': nn, 'metric': METRIC_LABELS[key], 'unit': _unit(key), 'expected_error': _to_ui(key, e['mean_err'][j])}
            for name, _ in COMPARE:
                if name in e and (name == 'equidistant' or INCLUDE_QMC):
                    row[f'{name}_expected_error'] = _to_ui(key, e[name]['mean_err'][j])
            vs_n.append(row)
    out = {'times': times_df, 'accuracy': pd.DataFrame(rows), 'error_vs_n': pd.DataFrame(vs_n), 'message': message,
           'n': n, 'preset': preset, 'metrics': metrics, 'sleep_h': sleep_h}
    if reached is not None:
        out['reached'] = reached
    return out


def recommend_for_budget(model, sleep_time, n_samples, metrics=None, preset='any_time'):
    """Mode A: n samples are affordable; the MPCW times for n in the windows of the preset and the accuracy to expect.
    metrics (optional): the metrics that matter; they are marked in the table and shown in error_vs_n (default: all).
    The times themselves do not depend on the metrics."""
    metrics = _check(model, preset, metrics)
    n = int(n_samples)
    msg = (f'{n} samples, accuracy shown for {", ".join(METRIC_LABELS[m] for m in metrics)} '
           f'({PRESET_LABELS.get(preset, preset)}).')
    return _result(model, sleep_time, preset, n, metrics, msg)


def recommend_for_accuracy(model, sleep_time, targets, preset='any_time'):
    """Mode B: targets {metric: max expected error} in minutes (onset / offset / peak_time) or percent
    (peak_value / auc). Smallest n whose expected errors meet every target and keep meeting them for all larger n."""
    metrics = _check(model, preset, targets)
    goal = {m: _from_ui(m, float(targets[m])) for m in metrics}
    ns = available_n(model)

    def meets(n):
        err = _entry(model, preset, n)['mean_err']
        return all(err[METRIC_KEYS.index(m)] is not None and err[METRIC_KEYS.index(m)] <= goal[m] for m in metrics)

    ok = [meets(n) for n in ns]
    stays = [all(ok[i:]) for i in range(len(ns))]
    if any(stays):
        n = ns[stays.index(True)]
        msg = f'{n} samples reach the target accuracy ({PRESET_LABELS.get(preset, preset)}).'
        return _result(model, sleep_time, preset, n, metrics, msg, reached=True)
    n = ns[-1]
    msg = (f'The target accuracy is not reached with up to {n} samples in this time window. '
           f'Showing the times for {n} samples; consider a wider sampling window or a looser target.')
    return _result(model, sleep_time, preset, n, metrics, msg, reached=False)


def plot_recommended_times(model, rec, ax=None):
    """Recommended times on the clock axis, over the sampling density."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 2.8))
    x = np.asarray(model['density']['hours'], float)
    p = np.asarray(model['density']['p'], float)
    ax.fill_between(x, 0, p / p.max(), color='lightgray', label='curvature of a typical profile')
    for k, (a, b) in enumerate(list_presets(model)[rec['preset']]):
        ax.axvspan(a, b, color='tab:green', alpha=0.08, label='allowed' if k == 0 else None)
    t = rec['times']['hour_rel_sleep'].values
    ax.vlines(t, 0, 1.05, color='tab:blue', lw=2, label='recommended times')
    for k, (h, c) in enumerate(zip(t, rec['times']['clock_time'])):
        ax.text(h, 1.08 + 0.12 * (k % 2), c, ha='center', fontsize=7, color='tab:blue')
    ax.axvline(0, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    mc._clock_axis(ax, x, rec['sleep_h'])
    ax.set_title('\n'.join(textwrap.wrap(rec['message'], 100)), fontsize=9)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig

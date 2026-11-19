"""Independent measured-cost audit for schema-v3 NavRL NPZ traces.

No simulator, policy, critic prediction, or old outcome is imported. Unknown
schemas, duplicate episode keys, wrong costs, inconsistent masks and unequal
case panels fail closed. Synthetic fixtures are software checks, never evidence
of physical performance or budget compliance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

METRICS = (
    'finite_first_event', 'first_event_unsafe_absorbing',
    'env_done_terminated_absorbing_zero_tail',
    'env_done_terminated_absorbing_tail_upper',
)
REQUIRED = {
    'trace_schema_version', 'safety_cost', 'clearance_m', 'collision',
    'reached_goal', 'terminated', 'truncated', 'done', 'final_steps',
    'discount', 'evaluation_seed', 'training_seed', 'condition',
    'checkpoint_name', 'checkpoint_sha256', 'source_sha256', 'preferred_clearance_m', 'temperature_m',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def scalar(data, key, cast):
    a = np.asarray(data[key])
    if a.shape != ():
        raise ValueError(f'{key} must be scalar')
    return cast(a.item())


def mask(data, key, shape):
    a = np.asarray(data[key])
    if a.shape != shape or not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        raise ValueError(f'Invalid Boolean mask {key}; expected {shape}')
    return a.astype(bool)


def measured_returns(costs, terminal, gamma):
    """Exact backwards-compatible normalized recursion over supplied prefix."""
    c = np.asarray(costs, dtype=float)
    finite = float(np.dot((1 - gamma) * gamma ** np.arange(len(c)), c))
    absorbing = (float(np.dot((1 - gamma) * gamma ** np.arange(len(c) - 1), c[:-1]))
                 + gamma ** (len(c) - 1)) if terminal else finite
    return finite, absorbing


def read_trace(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open('rb') as stream, np.load(stream, allow_pickle=False) as data:
        missing = REQUIRED - set(data.files)
        if missing:
            raise ValueError(f'{path}: missing trace keys: {sorted(missing)}')
        if scalar(data, 'trace_schema_version', int) != 3:
            raise ValueError('Only raw schema-v3 traces are accepted; old means cannot reconstruct returns')
        c = np.asarray(data['safety_cost'], dtype=float)
        d = np.asarray(data['clearance_m'], dtype=float)
        if c.ndim != 2 or min(c.shape) < 1 or d.shape != c.shape:
            raise ValueError('cost/clearance must be same nonempty [env,time] shape')
        if not np.isfinite(c).all() or not np.isfinite(d).all() or not ((c >= 0) & (c <= 1)).all():
            raise ValueError('Nonfinite/out-of-range measured cost or clearance')
        masks = {k: mask(data, k, c.shape) for k in ('collision', 'reached_goal', 'terminated', 'truncated', 'done')}
        if not np.array_equal(masks['done'], masks['terminated'] | masks['truncated']):
            raise ValueError('done != terminated OR truncated')
        if (masks['collision'] & ~masks['terminated']).any():
            raise ValueError('Collision did not set unsafe terminated mask')
        gamma = scalar(data, 'discount', float)
        preferred = scalar(data, 'preferred_clearance_m', float)
        temp = scalar(data, 'temperature_m', float)
        if not 0 <= gamma < 1 or temp <= 0:
            raise ValueError('Invalid cost parameters')
        # Stable sigmoid; collision clamp must reproduce raw logged cost.
        z = np.clip((preferred - d) / temp, -700, 700)
        reconstructed = np.maximum(1 / (1 + np.exp(-z)), masks['collision'].astype(float))
        maxerr = float(np.max(np.abs(reconstructed - c)))
        if maxerr > 2e-6:
            raise ValueError(f'Cost formula mismatch max_abs_error={maxerr}')
        name = scalar(data, 'checkpoint_name', str)
        weight_hash = scalar(data, 'checkpoint_sha256', str)
        sources = json.loads(scalar(data, 'source_sha256', str))
        if not re.fullmatch(r'[0-9a-f]{64}', weight_hash) or not isinstance(sources, dict) or not sources:
            raise ValueError('Invalid checkpoint/source provenance')
        if any(not re.fullmatch(r'[0-9a-f]{64}', str(v)) for v in sources.values()):
            raise ValueError('Invalid source hash manifest')
        train = scalar(data, 'training_seed', int)
        evalseed = scalar(data, 'evaluation_seed', int)
        condition = scalar(data, 'condition', str)
        suffix = re.search(r'_s(\d+)$', name)
        if suffix and int(suffix.group(1)) != train:
            raise ValueError('training_seed does not agree with checkpoint name')
        if train < -1:
            raise ValueError('training_seed must be >=0, or -1 for unique frozen checkpoint')
        method = re.sub(r'_s\d+$', '', name) if train >= 0 else name
        finals = np.asarray(data['final_steps'])
        if finals.shape != (c.shape[0],) or not np.issubdtype(finals.dtype, np.integer):
            raise ValueError('Invalid final_steps vector')
        rows = []
        for env in range(c.shape[0]):
            def first(which):
                indices = np.flatnonzero(which[env])
                return int(indices[0]) if len(indices) else c.shape[1]
            end_done = min(first(masks['done']), c.shape[1] - 1)
            end_event = min(first(masks['reached_goal']), first(masks['terminated']), end_done)
            if int(finals[env]) != end_event:
                raise ValueError(f'final_steps inconsistent with independent first-event reconstruction: env={env}')
            event_terminal = bool(masks['terminated'][env, end_event])
            env_terminal = bool(masks['terminated'][env, end_done])
            finite, event_abs = measured_returns(c[env, :end_event + 1], event_terminal, gamma)
            _, env_abs = measured_returns(c[env, :end_done + 1], env_terminal, gamma)
            # A timeout/truncation is NOT unsafe absorption. Its unseen
            # continuation is bounded, never replaced by a critic prediction.
            censored = not env_terminal
            tail = gamma ** (end_done + 1) if censored else 0.0
            rows.append({
                'method': method, 'checkpoint_name': name, 'condition': condition,
                'checkpoint_sha256': weight_hash, 'source_sha256': json.dumps(sources, sort_keys=True),
                'training_seed': train, 'evaluation_seed': evalseed, 'env_index': env,
                'finite_first_event': finite,
                'first_event_unsafe_absorbing': event_abs,
                'env_done_terminated_absorbing_zero_tail': env_abs,
                'env_done_terminated_absorbing_tail_upper': min(1., env_abs + tail),
                'first_event_step': end_event, 'first_env_done_step': end_done,
                'first_event_unsafe': event_terminal,
                'env_done_unsafe': env_terminal, 'env_done_censored': censored,
                'env_done_truncated': bool(masks['truncated'][env, end_done]),
                'unobserved_tail_upper': tail, 'cost_formula_max_abs_error': maxerr,
                'source_trace': str(path), 'discount': gamma,
                'preferred_clearance_m': preferred, 'temperature_m': temp,
            })
        return rows


def _betacf(a, b, x):
    """Continued fraction for regularized incomplete beta, double precision."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1., 1 - qab * x / qap
    if abs(d) < 1e-300: d = 1e-300
    d = 1 / d
    h = d
    for m in range(1, 500):
        aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
        d, c = 1 + aa * d, 1 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d, c = 1 + aa * d, 1 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14: return h
    raise ArithmeticError('Incomplete beta failed to converge')


def _ibeta(a, b, x):
    if x <= 0: return 0.
    if x >= 1: return 1.
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2): return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def t_upper_quantile(prob, df):
    if not .5 < prob < 1 or df < 1: raise ValueError('Invalid t quantile')
    def cdf(t): return 1 - .5 * _ibeta(df / 2, .5, df / (df + t * t))
    lo, hi = 0., 1.
    while cdf(hi) < prob: hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if cdf(mid) < prob: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def summarize_panel(rows, metric, *, bootstrap=20000, seed=20261020, alpha=.05, budget=.05):
    train_ids = sorted({r['training_seed'] for r in rows})
    eval_ids = sorted({r['evaluation_seed'] for r in rows})
    if -1 in train_ids and len(train_ids) != 1:
        raise ValueError('Frozen baseline was mixed with training replicates')
    panels = {}
    for r in rows:
        key = (r['training_seed'], r['evaluation_seed'])
        panels.setdefault(key, {})
        if r['env_index'] in panels[key]: raise ValueError('Duplicate episode key')
        panels[key][r['env_index']] = float(r[metric])
    expected_envs = set(next(iter(panels.values())))
    grid = np.empty((len(train_ids), len(eval_ids)))
    seed_rows = []
    for i, tr in enumerate(train_ids):
        for j, ev in enumerate(eval_ids):
            cell = panels.get((tr, ev))
            if cell is None or set(cell) != expected_envs:
                raise ValueError('Incomplete/unbalanced train x eval x env panel')
            grid[i, j] = np.mean(list(cell.values()))
        seed_rows.append({'training_seed': tr, 'mean': float(grid[i].mean()),
                          'eval_seeds': len(eval_ids), 'episodes': len(eval_ids) * len(expected_envs)})
    rng = np.random.default_rng(seed)
    # Crossed resampling: each evaluation-seed draw is shared across sampled
    # training seeds, preserving common fixed test-case dependence.
    ix_t = rng.integers(0, len(train_ids), (bootstrap, len(train_ids)))
    ix_e = rng.integers(0, len(eval_ids), (bootstrap, len(eval_ids)))
    samples = grid[ix_t[:, :, None], ix_e[:, None, :]].mean(axis=(1, 2))
    mean = float(grid.mean())
    boot_ucb = float(np.quantile(samples, 1 - alpha))
    # Sensitivity 1: seed-level Student-t, conditional on fixed eval panel.
    seed_means = grid.mean(axis=1)
    if train_ids != [-1] and len(train_ids) >= 2:
        t_ucb = min(1., mean + t_upper_quantile(1 - alpha, len(train_ids) - 1)
                    * float(seed_means.std(ddof=1)) / math.sqrt(len(train_ids)))
        t_assumption = 'independent approximately normal training-seed means, conditional on fixed evaluation panel'
    else:
        t_ucb, t_assumption = None, 'no independent training-seed replication for this frozen checkpoint'
    # Sensitivity 2: bounded-return Hoeffding on independent cluster means.
    # This is deliberately conservative; episode repetition cannot inflate n.
    ncluster = len(train_ids) if train_ids != [-1] else len(eval_ids)
    hoeffding = min(1., mean + math.sqrt(math.log(1 / alpha) / (2 * ncluster)))
    conservative_crossed = None
    if train_ids != [-1]:
        conservative_crossed = min(1., mean + math.sqrt(math.log(2 / alpha) / (2 * len(train_ids)))
                                  + math.sqrt(math.log(2 / alpha) / (2 * len(eval_ids))))
    verdict = ('OBSERVED_MEAN_EXCEEDS_BUDGET' if mean > budget else
               'APPROX_BOOTSTRAP_UCB_SUPPORTS_BUDGET' if boot_ucb <= budget else 'INSUFFICIENT_PRECISION')
    return {
        'metric': metric, 'mean': mean, 'budget': budget, 'mean_minus_budget': mean - budget,
        'bootstrap_ucb_one_sided': boot_ucb,
        'bootstrap_ci_two_sided': [float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))],
        'confidence': 1 - alpha, 'bootstrap_replicates': bootstrap,
        'bootstrap_scheme': 'crossed_training_seed_and_shared_eval_seed_fixed_environment_cases',
        'seed_t_ucb_conditional': t_ucb, 'seed_t_assumption': t_assumption,
        'bounded_hoeffding_ucb_conditional': hoeffding,
        'bounded_hoeffding_independent_clusters': ncluster,
        'bounded_crossed_union_ucb': conservative_crossed,
        'bounded_assumption': 'independent train seeds and/or independent eval randomization clusters; fixed cases within each block',
        'bootstrap_verdict': verdict,
        'all_reported_sensitivities_support_budget': all(x <= budget for x in [boot_ucb, hoeffding] + ([] if t_ucb is None else [t_ucb]) + ([] if conservative_crossed is None else [conservative_crossed])),
        'training_seed_count': 0 if train_ids == [-1] else len(train_ids),
        'evaluation_seed_count': len(eval_ids), 'environment_case_count': len(expected_envs),
        'unique_episode_count': len(rows), 'per_training_seed': seed_rows,
        'interpretation': 'Approximate bootstrap and sensitivity bounds are reported separately; critic predictions are never used.',
    }

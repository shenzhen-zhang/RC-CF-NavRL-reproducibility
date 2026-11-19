"""Prospective fixed-qualification checks; full arrays are always audited.

No comparison derives its domain from observed error or from replay attrition.
The reference prefix fixes qualification before any candidate execution. All
exogenous state, RNG and fixed parameters are unmasked exact controls.
"""
from __future__ import annotations
import hashlib
import numpy as np
from qualified_state_registry import (PROTOCOL, FIELD_REGISTRY, NUM_ENVS,
    REGISTRY_SHA256, PREFIX_EVENTS, OUTCOME_EVENTS)


class ReplayCheckError(ValueError):
    def __init__(self, message, control):
        super().__init__(message)
        self.control = control


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def _mask(value):
    value = np.asarray(value)
    if value.shape != (NUM_ENVS,) or value.dtype != np.dtype('bool'):
        raise ValueError('Frozen qualification mask requires bool[N64]')
    return value


def validate_state_schema(state):
    if set(state) != set(FIELD_REGISTRY):
        raise ValueError('State fields differ from explicit 61-field registry: ' +
            repr({'unknown': sorted(set(state)-set(FIELD_REGISTRY)),
                  'missing': sorted(set(FIELD_REGISTRY)-set(state))}))
    for key, spec in FIELD_REGISTRY.items():
        value = np.asarray(state[key])
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
            raise ValueError('Frozen replay shape/dtype differs: ' + key)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite full replay state: ' + key)


def check_replay(reference, current, *, valid_mask=None, tolerance=1e-4, empty_query=False):
    """Return every full/qualified difference, then fail if any required gate fails."""
    if not np.isfinite(tolerance) or tolerance < 0 or tolerance > 1e-4:
        raise ValueError('Physical tolerance must remain at most 1e-4')
    if empty_query and tolerance != 0:
        raise ValueError('Empty query requires zero tolerance for all 61 fields')
    valid = np.ones(NUM_ENVS, dtype=bool) if empty_query else _mask(valid_mask)
    validate_state_schema(reference)
    validate_state_schema(current)
    errors, diagnostics, failures = {}, {}, []
    for key, spec in FIELD_REGISTRY.items():
        expected, observed = np.asarray(reference[key]), np.asarray(current[key])
        delta = np.abs(expected.astype(np.float64)-observed.astype(np.float64))
        maximum = float(delta.max(initial=0))
        exact = bool(np.array_equal(expected, observed))
        item = {'domain': 'all_exact' if empty_query else spec['domain'],
                'shape': list(expected.shape), 'dtype': str(expected.dtype),
                'exact_equal_all': exact, 'maximum_absolute_error_all': maximum}
        if spec['domain'] == 'qualified_env_axis0':
            per_env = delta.reshape(NUM_ENVS, -1).max(axis=1)
            qualified = float(per_env[valid].max(initial=0))
            unqualified = float(per_env[~valid].max(initial=0))
            item.update(maximum_absolute_error_qualified=qualified,
                        maximum_absolute_error_unqualified=unqualified,
                        changed_envs=np.flatnonzero(per_env > 0).tolist(),
                        qualified_envs_over_tolerance=np.flatnonzero((per_env > tolerance) & valid).tolist(),
                        unqualified_envs_over_tolerance=np.flatnonzero((per_env > tolerance) & ~valid).tolist())
            error = maximum if empty_query else qualified
            failed = (not exact) if empty_query else error > tolerance
        else:
            # Fixed parameters may differ between envs, but their entire tensor
            # must equal the reference; no qualification mask applies.
            error = maximum
            failed = not exact
        item['passed'] = not failed
        errors[key], diagnostics[key] = error, item
        if failed:
            failures.append(key)
    control = {'passed': not failures, 'protocol': PROTOCOL,
        'domain': 'all_61_fields_exact' if empty_query else 'fixed_reference_prefix_qualified_envs',
        'registry_sha256': REGISTRY_SHA256, 'tolerance': float(tolerance),
        'maximum_absolute_error': max(errors.values(), default=0.), 'per_field_error': errors,
        'reference_hashes': {k: array_hash(v) for k, v in reference.items()},
        'current_hashes': {k: array_hash(v) for k, v in current.items()},
        'qualified_mask_sha256': array_hash(valid), 'qualified_count': int(valid.sum()),
        'empty_qualification': bool(not valid.any()),
        'per_field_diagnostics': diagnostics, 'failed_fields': failures}
    if failures:
        raise ReplayCheckError('Qualified replay control failed: ' + ', '.join(failures), control)
    return control


def validate_prefix_events(events, prefix_steps=None):
    if set(events) != set(PREFIX_EVENTS):
        raise ValueError('Prefix event coverage must include done/goal/termination/truncation/collision')
    arrays = {key: np.asarray(value) for key, value in events.items()}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError('Prefix event shapes differ')
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[0] != NUM_ENVS or (prefix_steps is not None and shape[1] != prefix_steps):
        raise ValueError('Prefix event history must cover all N64 transitions')
    if any(value.dtype != np.dtype('bool') for value in arrays.values()):
        raise ValueError('Prefix events require explicit Boolean dtype')
    if not np.array_equal(arrays['done'], arrays['terminated'] | arrays['truncated']):
        raise ValueError('Prefix done differs from terminated|truncated')
    if (arrays['collision'] & ~arrays['terminated']).any():
        raise ValueError('Prefix collision lacks unsafe termination')
    return arrays


def qualification_from_events(events, prefix_steps=None):
    events = validate_prefix_events(events, prefix_steps)
    disqualified = events['done'] | events['reached_goal']
    eligible = ~disqualified.any(axis=1)
    first = np.full(NUM_ENVS, -1, dtype=np.int64)
    if disqualified.shape[1]:
        first[~eligible] = disqualified[~eligible].argmax(axis=1)
    return eligible, first


def check_prefix_qualification(reference_events, replay_events, valid_mask):
    """Replay attrition cannot narrow the domain fixed by reference history."""
    valid = _mask(valid_mask)
    ref_valid, ref_first = qualification_from_events(reference_events)
    replay_valid, replay_first = qualification_from_events(replay_events, reference_events['done'].shape[1])
    if not np.array_equal(ref_valid, valid):
        raise ValueError('Provided qualification mask differs from complete reference prefix events')
    lost = np.flatnonzero(valid & ~replay_valid).tolist()
    control = {'passed': not lost, 'protocol': PROTOCOL,
        'qualification_rule': 'no_done_or_reached_goal_in_reference_prefix',
        'qualified_mask_sha256': array_hash(valid), 'qualified_count': int(valid.sum()),
        'empty_qualification': bool(not valid.any()), 'lost_reference_qualified_envs': lost,
        'all_reference_qualified_remain_qualified': not lost,
        'reference_prefix_event_hashes': {k: array_hash(v) for k,v in reference_events.items()},
        'replay_prefix_event_hashes': {k: array_hash(v) for k,v in replay_events.items()},
        'reference_first_event_sha256': array_hash(ref_first),
        'replay_first_event_sha256': array_hash(replay_first),
        'reference_excluded_envs': np.flatnonzero(~valid).tolist(),
        'replay_excluded_envs': np.flatnonzero(~replay_valid).tolist()}
    if lost:
        raise ReplayCheckError('Reference-qualified environments became ineligible during replay prefix', control)
    return control


def validate_outcome_schema(outcome):
    shapes = {'positions': (64,50,3), 'dynamic_positions': (50,80,3), 'costs': (64,50),
        'observed_clearance': (64,50), 'collision': (64,), 'unsafe': (64,),
        'first_unsafe_step': (64,), 'progress_m': (64,)}
    if set(outcome) != set(shapes) | {'events'}:
        raise ValueError('Repeated outcome coverage changed')
    for key, shape in shapes.items():
        value = np.asarray(outcome[key])
        dtype = 'bool' if key in {'collision','unsafe'} else 'int64' if key == 'first_unsafe_step' else 'float32'
        if value.shape != shape or str(value.dtype) != dtype or not np.isfinite(value).all():
            raise ValueError('Frozen full outcome shape/dtype/finiteness invalid: ' + key)
    events = outcome['events']
    if set(events) != set(OUTCOME_EVENTS):
        raise ValueError('Repeated branch event coverage changed')
    for key in OUTCOME_EVENTS:
        value = np.asarray(events[key])
        if value.shape != (64,50) or value.dtype != np.dtype('bool'):
            raise ValueError('Frozen full outcome event schema invalid: ' + key)
    if not np.array_equal(events['done'], events['terminated'] | events['truncated']):
        raise ValueError('Outcome done disagrees with termination|truncation')
    if (events['collision'] & ~events['terminated']).any():
        raise ValueError('Outcome collision lacks unsafe termination')
    alive = np.ones((64,50), dtype=bool)
    alive[:,1:] = ~np.maximum.accumulate(events['terminated'][:,:-1],axis=1)
    if not np.array_equal(alive,events['valid_before_transition']):
        raise ValueError('Outcome alive-before-transition events are inconsistent')
    unsafe = events['terminated'].any(axis=1)
    first = np.full(64,-1,dtype=np.int64)
    first[unsafe] = events['terminated'][unsafe].argmax(axis=1)
    if not np.array_equal(outcome['first_unsafe_step'],first) or not np.array_equal(outcome['unsafe'],unsafe):
        raise ValueError('Outcome first-unsafe/unsafe aggregate disagrees with full events')
    if not np.array_equal(outcome['collision'],(events['collision'] & alive).any(axis=1)):
        raise ValueError('Outcome collision aggregate disagrees with full events')


def check_repeated_outcome(reference, current, *, valid_mask, tolerance=1e-4):
    """Compare all 50 recorded transitions on the fixed qualification domain.

    No post-unsafe time mask is introduced. Shared obstacle trajectories are
    unmasked and exact. All full arrays, including excluded envs, must be finite.
    """
    valid = _mask(valid_mask)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-4:
        raise ValueError('Repeated-outcome tolerance exceeds frozen bound')
    fields = {'positions', 'dynamic_positions', 'costs', 'observed_clearance',
              'collision', 'unsafe', 'first_unsafe_step', 'progress_m', 'events'}
    if set(reference) != fields or set(current) != fields:
        raise ValueError('Repeated outcome coverage changed')
    validate_outcome_schema(reference)
    validate_outcome_schema(current)
    errors, diagnostics, failures, hashes = {}, {}, [], {'reference': {}, 'current': {}}
    for key in sorted(fields - {'events'}):
        x, y = np.asarray(reference[key]), np.asarray(current[key])
        if x.shape != y.shape or x.dtype != y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid full repeated branch array: ' + key)
        global_field = key == 'dynamic_positions'
        if not global_field and (not x.shape or x.shape[0] != NUM_ENVS):
            raise ValueError('Repeated outcome explicit env axis differs: ' + key)
        delta = np.abs(x.astype(np.float64)-y.astype(np.float64))
        discrete = key in {'collision', 'unsafe', 'first_unsafe_step'}
        if global_field:
            err = float(delta.max(initial=0))
            failed = not np.array_equal(x,y)
            item = {'domain': 'all_exact', 'maximum_absolute_error_all': err}
        else:
            per_env = delta.reshape(NUM_ENVS,-1).max(axis=1)
            err = float(per_env[valid].max(initial=0))
            failed = not np.array_equal(x[valid],y[valid]) if discrete else err > tolerance
            item = {'domain': 'fixed_reference_prefix_qualified_envs',
                'maximum_absolute_error_all': float(per_env.max(initial=0)),
                'maximum_absolute_error_qualified': err,
                'maximum_absolute_error_unqualified': float(per_env[~valid].max(initial=0)),
                'changed_envs': np.flatnonzero(per_env > 0).tolist()}
        errors[key], diagnostics[key] = err, item
        hashes['reference'][key], hashes['current'][key] = array_hash(x), array_hash(y)
        if failed:
            failures.append(key)
    if set(reference['events']) != set(OUTCOME_EVENTS) or set(current['events']) != set(OUTCOME_EVENTS):
        raise ValueError('Repeated branch event coverage changed')
    for key in OUTCOME_EVENTS:
        x, y = np.asarray(reference['events'][key]), np.asarray(current['events'][key])
        if x.dtype != np.dtype('bool') or y.dtype != x.dtype or x.shape != y.shape or x.ndim != 2 or x.shape[0] != NUM_ENVS:
            raise ValueError('Repeated branch event schema invalid: ' + key)
        name = 'events/' + key
        errors[name] = float(np.max(x[valid] != y[valid], initial=False))
        diagnostics[name] = {'domain': 'fixed_reference_prefix_qualified_envs',
            'maximum_absolute_error_all': float(np.max(x != y, initial=False)),
            'maximum_absolute_error_qualified': errors[name],
            'maximum_absolute_error_unqualified': float(np.max(x[~valid] != y[~valid], initial=False)),
            'changed_envs': np.flatnonzero((x != y).any(axis=1)).tolist()}
        hashes['reference'][name], hashes['current'][name] = array_hash(x), array_hash(y)
        if errors[name]:
            failures.append(name)
    control = {'passed': not failures, 'protocol': PROTOCOL, 'tolerance': float(tolerance),
        'domain': 'fixed_reference_prefix_qualified_envs_all_recorded_horizon_steps',
        'maximum_absolute_error': max(errors.values(), default=0.), 'per_field_error': errors,
        'per_field_diagnostics': diagnostics, 'hashes': hashes,
        'qualified_mask_sha256': array_hash(valid), 'qualified_count': int(valid.sum()),
        'failed_fields': failures}
    if failures:
        raise ReplayCheckError('Repeated candidate control failed: ' + ', '.join(failures), control)
    return control

"""Analyze physically realized candidate outcomes, clustered by source/eval.

Teacher scores are clearance-derived risk scores, not calibrated collision
probabilities. No teacher label is used as a realized collision outcome.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import importlib.util
import sys


def _load_qualified_file(name):
    path=Path(__file__).resolve().with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('_prospective_qualified_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


# A runner can have the historical project's replay_checks in sys.modules.
# Bind these particular files, without replacing the caller's legacy modules.
_registry=_load_qualified_file('qualified_state_registry')
_previous_registry=sys.modules.get('qualified_state_registry')
sys.modules['qualified_state_registry']=_registry
try:_checks=_load_qualified_file('replay_checks')
finally:
    if _previous_registry is None:sys.modules.pop('qualified_state_registry',None)
    else:sys.modules['qualified_state_registry']=_previous_registry
_usd=_load_qualified_file('usd_isolation')
PROTOCOL=_registry.PROTOCOL;FIELD_REGISTRY=_registry.FIELD_REGISTRY
REGISTRY_SHA256=_registry.REGISTRY_SHA256;QUALIFICATION_RULE=_registry.QUALIFICATION_RULE
PREFIX_EVENTS=_registry.PREFIX_EVENTS;OUTCOME_EVENTS=_registry.OUTCOME_EVENTS

def number(value):
    return float(value) if value is not None and np.isfinite(value) else None

def weighted_auc_ap(score, label, weight):
    order = np.argsort(score, kind='stable')
    s, y, w = score[order], label[order], weight[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1]
    positive = np.add.reduceat(w*y, starts)
    negative = np.add.reduceat(w*(1-y), starts)
    p, n = positive.sum(), negative.sum()
    if p <= 0 or n <= 0: return None, None
    auc = np.sum(positive*(np.cumsum(negative)-negative+.5*negative))/(p*n)
    pp, nn = positive[::-1], negative[::-1]
    ap = np.sum(pp*np.cumsum(pp)/np.maximum(np.cumsum(pp+nn), 1e-15))/p
    return float(auc), float(ap)

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _boolean(value, field):
    if value is True or value == 'True': return True
    if value is False or value == 'False': return False
    raise ValueError('Invalid Boolean CSV/control field: '+field)


def _array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def _passed_control(value, tolerance, label):
    if not isinstance(value,dict) or value.get('passed') is not True:
        raise ValueError('Replay control did not pass: '+label)
    logged=float(value.get('tolerance',float('nan')))
    error=float(value.get('maximum_absolute_error',float('nan')))
    if not np.isfinite([logged,error]).all() or logged<0 or logged>tolerance or error<0 or error>logged:
        raise ValueError('Replay control tolerance/error invalid: '+label)


PHYSICAL_STATE_FIELDS = frozenset([
    'drone/full_state', 'physical/joint_positions', 'physical/joint_velocities',
    *[f'physical/{link}/world_{field}' for link in ['base_link','rotors_view']
      for field in ['position','orientation','velocity']],
])
RANDOM_STATE_FIELDS = frozenset(['random/torch_cpu','random/cuda_0',
    'random/numpy_keys','random/numpy_position','random/python_sha'])


def _sha256(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _passed_state_control(value, tolerance, label):
    _passed_control(value,tolerance,label)
    refs=value.get('reference_hashes');errors=value.get('per_field_error')
    required=PHYSICAL_STATE_FIELDS|RANDOM_STATE_FIELDS
    if not isinstance(refs,dict) or not required<=refs.keys() or not all(_sha256(v) for v in refs.values()):
        raise ValueError('Replay physical/random reference coverage invalid: '+label)
    if not isinstance(errors,dict) or errors.keys()!=refs.keys():
        raise ValueError('Replay per-field error coverage invalid: '+label)
    values=np.asarray(list(errors.values()),dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>float(value['tolerance'])).any():
        raise ValueError('Replay per-field error exceeds tolerance: '+label)
    if float(value['maximum_absolute_error'])!=float(values.max()):
        raise ValueError('Replay maximum error differs from per-field errors: '+label)
    if any(float(error)!=0. for key,error in errors.items() if key.startswith('random/')):
        raise ValueError('Replay random stream differs: '+label)


def _validate_lidar_clock(value):
    if not isinstance(value,dict) or set(value)!={'_timestamp','_timestamp_last_update','_is_outdated'}:
        raise ValueError('Initialization lidar clock coverage invalid')
    for name,record in value.items():
        if not isinstance(record,dict) or not _sha256(record.get('sha256')):
            raise ValueError('Initialization lidar clock hash invalid')
        shape=record.get('shape')
        if not isinstance(shape,list) or not shape or any(type(x) is not int or x<1 for x in shape):
            raise ValueError('Initialization lidar clock shape invalid')
        bounds=np.asarray([record.get('minimum',np.nan),record.get('maximum',np.nan)],dtype=float)
        if not np.isfinite(bounds).all() or bounds[0]>bounds[1] or bounds[0]<0:
            raise ValueError('Initialization lidar clock bounds invalid')
        if name=='_is_outdated' and not np.isin(bounds,[0,1]).all():
            raise ValueError('Initialization lidar outdated flag invalid')


def _simulation_clock_delta_matches(before,after,ticks,physics_dt):
    """Legacy Isaac callback clock uses a float32 timestep quantum.

    SimulationContext accumulates the callback step_size, while get_physics_dt
    returns the double reciprocal of integer Hz. Fixed-runtime metadata audits
    identify the callback quantum as float32(physics_dt). Allow only binary64
    arithmetic ULPs here; physical states and integer ticks keep their gates.
    """
    quantum=float(np.float32(physics_dt));expected=ticks*quantum
    before=float(before);after=float(after)
    if not all(math.isfinite(x) for x in [before,after,expected]) or after<before:
        return False
    bound=8*(math.ulp(before)+math.ulp(after)+math.ulp(expected))
    return abs((after-before)-expected)<=bound


def _validate_initialization(metadata, eval_seeds, anchors, count, timing):
    """Audit every prepare call, including its unscored physical clock ticks."""
    if type(metadata.get('articulation_signature_version')) is not int or metadata['articulation_signature_version']!=2:
        raise ValueError('Articulation signature version 2 is required')
    runs=len(eval_seeds)*len(anchors)*(count+2)
    if type(metadata.get('control_runs')) is not int or metadata['control_runs']!=runs:
        raise ValueError('Initialization control_runs count differs from frozen panel')
    records=metadata.get('initialization_diagnostics')
    if not isinstance(records,list) or len(records)!=len(eval_seeds)+runs:
        raise ValueError('Initialization diagnostics must cover every prepare call')
    dt=timing['physics_dt_s'];horizon=timing['horizon_steps']
    # The bank reaches the last anchor; each candidate then executes its entire
    # saved prefix and local horizon, even after an unsafe event.
    schedule=[]
    for seed in eval_seeds:
        schedule.append((seed,max(anchors)))
        schedule.extend((seed,a+horizon) for a in sorted(anchors) for _ in range(count+2))
    previous=None;previous_gap=None;fixed_joints=None
    for index,(record,(seed,next_gap)) in enumerate(zip(records,schedule),1):
        if not isinstance(record,dict) or type(record.get('reset_index')) is not int or record['reset_index']!=index or type(record.get('evaluation_seed')) is not int or record['evaluation_seed']!=seed:
            raise ValueError('Initialization reset index/evaluation panel differs')
        d=record.get('diagnostics')
        if not isinstance(d,dict) or d.get('articulation_reset_protocol')!='legacy_isaac_three_commit_full_reset_v1':
            raise ValueError('Initialization articulation reset protocol invalid')
        for key,expected in [('articulation_signature_version',2),('unscored_physics_steps',3),
                ('native_canonical_reset_passes',4),('native_env_reset_calls',8),
                ('measured_transition_count',0),('dynamic_obstacle_step_count',0)]:
            if type(d.get(key)) is not int or d[key]!=expected:
                raise ValueError('Initialization reset count/progress invalid: '+key)
        for key in ['joint_readback_matches_fixed_defaults','lidar_refreshed_before_rollout',
                    'double_reset_after_restore','rng_reseeded_after_refresh']:
            if d.get(key) is not True:raise ValueError('Initialization required readback/RNG flag invalid: '+key)
        if not np.isclose(float(d.get('physics_dt_s',np.nan)),dt,rtol=0,atol=1e-12):
            raise ValueError('Initialization physics dt differs from measured timing')
        before,after=d.get('simulation_step_before'),d.get('simulation_step_after')
        times=np.asarray([d.get('simulation_time_before_s',np.nan),d.get('simulation_time_after_s',np.nan)],dtype=float)
        if type(before) is not int or type(after) is not int or before<0 or after-before!=3 or not np.isfinite(times).all() or times[0]<0 or not _simulation_clock_delta_matches(times[0],times[1],3,dt):
            raise ValueError('Initialization three-tick simulation clock invalid')
        if previous is not None and (before-previous[0]!=previous_gap or not _simulation_clock_delta_matches(previous[1],times[0],previous_gap,dt)):
            raise ValueError('Initialization clock gap differs from complete prefix/horizon schedule')
        joints=tuple(d.get('initial_joint_'+field+'_sha256') for field in ['positions','velocities'])
        if not all(_sha256(v) for v in joints) or (fixed_joints is not None and joints!=fixed_joints):
            raise ValueError('Initialization fixed joint defaults changed or lack valid hashes')
        fixed_joints=joints
        _validate_lidar_clock(d.get('lidar_clock_before'));_validate_lidar_clock(d.get('lidar_clock_after'))
        cycles=d.get('unscored_commit_diagnostics')
        if not isinstance(cycles,list) or len(cycles)!=3:
            raise ValueError('Initialization requires three commit diagnostics')
        for cycle,c in enumerate(cycles,1):
            if not isinstance(c,dict) or type(c.get('cycle')) is not int or c['cycle']!=cycle or type(c.get('simulation_step_before')) is not int or c['simulation_step_before']!=before+cycle-1 or type(c.get('simulation_step_after')) is not int or c['simulation_step_after']!=before+cycle or not _simulation_clock_delta_matches(times[0],float(c.get('simulation_time_after_commit_s',np.nan)),cycle,dt):
                raise ValueError('Initialization commit tick/clock invalid')
            if c.get('task_progress_zero') is not True or type(c.get('dynamic_obstacle_step_count')) is not int or c['dynamic_obstacle_step_count']!=0:
                raise ValueError('Initialization commit leaked task/obstacle progress')
            for key in ['after_commit_physical_state_sha256','after_full_reset_physical_state_sha256']:
                hashes=c.get(key)
                if not isinstance(hashes,dict) or hashes.keys()!=PHYSICAL_STATE_FIELDS or not all(_sha256(v) for v in hashes.values()):
                    raise ValueError('Initialization commit requires all nine physical state hashes')
            if tuple(c['after_full_reset_physical_state_sha256']['physical/joint_'+field] for field in ['positions','velocities'])!=joints:
                raise ValueError('Initialization full-reset joint hashes differ from fixed defaults')
            _validate_lidar_clock(c.get('lidar_clock_after_reset'))
        if cycles[-1]['lidar_clock_after_reset']!=d['lidar_clock_after']:
            raise ValueError('Initialization final lidar clock differs from last reset')
        previous=(after,times[1]);previous_gap=next_gap
    return {'articulation_signature_version':2,'prepare_calls':len(records),'unscored_physics_steps':3*len(records)}


def _unique_nonnegative(values, field):
    if not isinstance(values,list) or not values or any(type(x) is not int or x<0 for x in values) or len(set(values))!=len(values):
        raise ValueError('Expected unique nonnegative integers: '+field)
    return values


def _load_masks(data, candidate, n, horizon):
    result={}
    for key in ['collision','terminated','truncated','done','reached_goal','valid_before_transition']:
        name=f'candidate_{candidate}_event_{key}'
        value=np.asarray(data[name])
        if value.shape!=(n,horizon) or not np.isin(value,[0,1]).all():
            raise ValueError('Invalid raw event mask: '+name)
        result[key]=value.astype(bool)
    if not np.array_equal(result['done'],result['terminated']|result['truncated']):
        raise ValueError('Raw done mask disagrees with terminated|truncated')
    if (result['collision']&~result['terminated']).any():
        raise ValueError('Raw collision did not terminate unsafe state')
    valid=np.ones((n,horizon),dtype=bool)
    if horizon>1:valid[:,1:]=~np.maximum.accumulate(result['terminated'][:,:-1],axis=1)
    if not np.array_equal(valid,result['valid_before_transition']):
        raise ValueError('Raw local-estimand alive mask is inconsistent')
    return result


def _validate_actual_timing(metadata, config):
    """Use recorded simulator timing; configured dt may be quantized by Isaac."""
    timing=metadata.get('actual_timing')
    fields=['configured_physics_dt_s','physics_dt_s','control_dt_s',
            'requested_teacher_horizon_s','realized_horizon_s']
    if not isinstance(timing,dict) or any(key not in timing for key in fields):
        raise ValueError('Measured actual_timing metadata is required')
    values={key:float(timing[key]) for key in fields}
    if not np.isfinite(list(values.values())).all() or min(values.values())<=0:
        raise ValueError('Invalid positive finite actual_timing values')
    sim=config['sim'];requested=float(config['algo']['method']['teacher']['horizon_s'])
    if int(sim['substeps'])!=1:raise ValueError('Teacher fixed horizon requires substeps=1')
    if not np.isclose(values['configured_physics_dt_s'],float(sim['dt']),rtol=0,atol=1e-12):
        raise ValueError('actual_timing configured dt differs from resolved configuration')
    if not np.isclose(values['physics_dt_s'],values['control_dt_s'],rtol=0,atol=1e-12):
        raise ValueError('actual_timing physics/control dt differ with substeps=1')
    if not np.isclose(values['requested_teacher_horizon_s'],requested,rtol=0,atol=1e-12):
        raise ValueError('actual_timing requested horizon differs from teacher definition')
    dt=values['control_dt_s'];horizon=timing.get('horizon_steps')
    if type(horizon) is not int or horizon<1 or horizon!=int(round(requested/dt)) or timing.get('rounding')!='nearest_control_step':
        raise ValueError('actual_timing must use the nearest positive control-step horizon')
    realized=values['realized_horizon_s']
    if not np.isclose(realized,horizon*dt,rtol=0,atol=1e-10) or abs(realized-requested)>.5*dt+1e-10:
        raise ValueError('actual_timing realized horizon/nearest-step error is inconsistent')
    floor_steps=int(np.floor(requested/dt+1e-10))
    # Only recorded steps may be reused; never extrapolate an unobserved tail.
    if floor_steps>horizon:raise ValueError('Floor horizon exceeds recorded trajectory')
    return {**values,'horizon_steps':horizon,'rounding':timing['rounding'],
            'floor_control_horizon_steps':floor_steps,'floor_control_horizon_s':floor_steps*dt,
            'requested_within_recorded_bracket':floor_steps*dt-1e-10<=requested<=realized+1e-10}


def _validate_qualified_extension(metadata):
    identity=metadata.get('extension_identity')
    if not isinstance(identity,dict) or identity.get('protocol')!='TEACHER_VALIDATION_EXTENSION_QUALIFIED_ANCHOR_V1':
        raise ValueError('Prospective qualified extension identity is required')
    here=Path(__file__).resolve().parent
    path=here/'sealed/extension.json'
    if not _sha256(identity.get('sha256')) or _digest(path)!=identity['sha256']:
        raise ValueError('Qualified extension manifest SHA differs')
    # The evidence export normalized workstation locations to recorded identifiers.
    manifest=json.loads(path.read_text(encoding='utf-8').replace('/root/rc_cf_revision_20261020/', 'recorded://'))
    if any(manifest.get(k)!=identity.get(k) for k in ['protocol','parent_plan','qualified_sources']):
        raise ValueError('Qualified extension identity differs from its sealed manifest')
    parent=identity['parent_plan']
    if parent.get('sha256')!='8142cb94b8ccf9918795df73c60646a8610e6522e2fd70b1537e44293288dac9' or _digest(here/'sealed/study_plan.json')!=parent['sha256']:
        raise ValueError('Qualified extension must retain the frozen parent plan')
    sources=identity.get('qualified_sources')
    required={'collect_teacher_replay.py','replay_checks.py','qualified_state_registry.py','usd_isolation.py','analyze_teacher_replay.py'}
    if not isinstance(sources,dict) or set(sources)!=required:
        raise ValueError('Qualified extension source coverage differs')
    for name,item in sources.items():
        p=(here/'sealed' if name=='analyze_teacher_replay.py' else here)/name
        if item['path']!='recorded://runtime/teacher_qualified_replay2/'+name or not _sha256(item.get('sha256')) or _digest(p)!=item['sha256'] or metadata.get('source_sha256',{}).get(item['path'])!=item['sha256']:
            raise ValueError('Qualified actual source path/hash differs: '+name)
    return identity


def _qualified_arrays(data,prefix):
    expected={prefix+key.replace('/','__') for key in FIELD_REGISTRY}
    if {k for k in data.files if k.startswith(prefix)}!=expected:
        raise ValueError('All 61 raw qualified state fields are required: '+prefix)
    state={key:np.asarray(data[prefix+key.replace('/','__')]) for key in FIELD_REGISTRY}
    _checks.validate_state_schema(state)
    return state


def _qualified_prefix(data,prefix,anchor):
    expected={prefix+key for key in PREFIX_EVENTS}
    if {k for k in data.files if k.startswith(prefix)}!=expected:
        raise ValueError('Complete raw prefix event coverage is required')
    events={key:np.asarray(data[prefix+key]) for key in PREFIX_EVENTS}
    mask,first=_checks.qualification_from_events(events,prefix_steps=anchor)
    return events,mask,first


def _qualified_outcome(data,prefix):
    expected={prefix+key for key in ['positions','dynamic_positions','costs','observed_clearance','collision','unsafe','first_unsafe_step','progress_m']}
    expected.update(prefix+'event_'+key for key in OUTCOME_EVENTS)
    if {key for key in data.files if key.startswith(prefix)}!=expected:
        raise ValueError('Unknown/missing full outcome field coverage: '+prefix)
    result={key:np.asarray(data[prefix+key]) for key in ['positions','dynamic_positions','costs','observed_clearance','collision','unsafe','first_unsafe_step','progress_m']}
    result['events']={key:np.asarray(data[prefix+'event_'+key]) for key in OUTCOME_EVENTS}
    _checks.validate_outcome_schema(result)
    return result


def _validate_qualified_protocol(csv_path,metadata,controls):
    if metadata.get('teacher_validation_protocol')!=PROTOCOL:
        raise ValueError('Prospective qualified-anchor protocol is required; historical failures cannot be adopted')
    identity=_validate_qualified_extension(metadata)
    protocol_file=csv_path.with_name('protocol_identity.json')
    expected_identity={'teacher_validation_protocol':PROTOCOL,'extension_identity':identity,
        'registry_sha256':REGISTRY_SHA256,'qualification_rule':QUALIFICATION_RULE,'prospective_only':True}
    if json.loads(protocol_file.read_text(encoding='utf-8'))!=expected_identity:
        raise ValueError('Qualified protocol identity sidecar differs')
    q=metadata.get('qualified_replay')
    if not isinstance(q,dict) or any(q.get(k)!=v for k,v in {'protocol':PROTOCOL,'registry_sha256':REGISTRY_SHA256,'qualification_rule':QUALIFICATION_RULE,'num_envs':64}.items()):
        raise ValueError('Qualified registry/domain identity differs')
    for key in ['usd_isolation_unchanged_at_completion','excluded_envs_are_saved_and_audited_for_full_schema_and_finiteness','no_post_unsafe_time_mask_for_repetition','old_all_environment_failures_remain_rejected']:
        if q.get(key) is not True:raise ValueError('Qualified protocol evidence flag missing: '+key)
    config=metadata['config']
    if config['env']['num_envs']!=64 or config['env_dyn']['num_obstacles']!=80 or metadata['actual_timing']['horizon_steps']!=50:
        raise ValueError('Qualified protocol requires N64/D80 and the complete 50-step horizon')
    isolation=q.get('usd_isolation');final=q.get('usd_isolation_at_completion')
    if not isinstance(isolation,dict) or set(isolation)!={'snapshot','validation'} or isolation!=final:
        raise ValueError('Actual USD initial/final isolation evidence is missing or changed')
    if _usd.validate_isolation(isolation['snapshot'])!=isolation['validation']:
        raise ValueError('Actual USD isolation proof differs from independent audit')
    if json.loads(csv_path.with_name('usd_isolation.json').read_text(encoding='utf-8'))!=isolation:
        raise ValueError('Actual USD isolation sidecar differs')
    spec=metadata['spec'];blocks={(int(e),int(a)) for e in spec['evaluation_seeds'] for a in spec['anchor_steps']}
    names={f'seed{e}_anchor{a}.npz' for e,a in blocks}
    if set(metadata.get('raw_npz_sha256',{}))!=names:
        raise ValueError('Qualified raw NPZ SHA inventory is incomplete')
    control_index={}
    for c in controls:
        key=tuple(int(c[k]) for k in ['evaluation_seed','anchor_step','order_position'])
        if key in control_index:raise ValueError('Duplicate qualified replay control')
        control_index[key]=c
    if control_index.keys()!={(e,a,o) for e,a in blocks for o in range(15)}:
        raise ValueError('Incomplete 13+2 qualified replay controls')
    records=[];prefix_banks={}
    for e,a in sorted(blocks):
        raw=csv_path.with_name(f'seed{e}_anchor{a}.npz')
        if _digest(raw)!=metadata['raw_npz_sha256'][raw.name]:raise ValueError('Qualified raw NPZ SHA differs')
        with np.load(raw,allow_pickle=False) as data:
            reference=_qualified_arrays(data,'anchor_state__')
            after_empty=_qualified_arrays(data,'empty_query_after__')
            empty=_checks.check_replay(reference,after_empty,tolerance=0.,empty_query=True)
            reference_events,valid,first=_qualified_prefix(data,'reference_prefix_event_',a)
            commands=np.asarray(data['command_prefix_world'])
            if commands.dtype!=np.dtype('float32') or commands.shape!=(a,64,1,3) or not np.isfinite(commands).all():
                raise ValueError('Qualified reference complete command prefix shape/dtype differs')
            if e in prefix_banks:
                previous_steps,previous_commands,previous_events=prefix_banks[e]
                if not np.array_equal(commands[:previous_steps],previous_commands) or any(not np.array_equal(reference_events[k][:,:previous_steps],previous_events[k]) for k in PREFIX_EVENTS):
                    raise ValueError('Reference anchors do not share one complete command/event prefix bank')
            prefix_banks[e]=(a,commands.copy(),{k:v.copy() for k,v in reference_events.items()})
            if data['valid_anchor_mask'].dtype!=np.dtype('bool') or not np.array_equal(data['valid_anchor_mask'],valid):
                raise ValueError('Qualification mask differs from complete reference prefix; intersection is forbidden')
            actual_first=data['reference_first_disqualifying_step']
            if actual_first.dtype!=np.dtype('int64') or not np.array_equal(actual_first,first):
                raise ValueError('Reference first disqualifying event differs')
            record={'evaluation_seed':e,'anchor_step':a,'planned_envs':64,'qualified_count':int(valid.sum()),
                'excluded_envs':np.flatnonzero(~valid).tolist(),'qualified_mask_sha256':_array_hash(valid),
                'reference_prefix_event_hashes':{k:_array_hash(v) for k,v in reference_events.items()},
                'reference_first_event_sha256':_array_hash(first),'empty_estimate':bool(not valid.any()),
                'qualification_fixed_before_candidates':True}
            records.append(record)
            outcomes={candidate:_qualified_outcome(data,f'candidate_{candidate}_') for candidate in range(13)}
            tolerance=float(spec.get('replay_tolerance',1e-4))
            for order,candidate in enumerate(list(range(13))+[12,11]):
                control=control_index[e,a,order]
                current=_qualified_arrays(data,f'replay_{order}_state__')
                events,_,first_replay=_qualified_prefix(data,f'replay_{order}_prefix_event_',a)
                recorded_first=data[f'replay_{order}_first_disqualifying_step']
                if recorded_first.dtype!=np.dtype('int64') or not np.array_equal(recorded_first,first_replay):
                    raise ValueError('Replay first disqualifying event differs')
                qualified=_checks.check_prefix_qualification(reference_events,events,valid)
                replay=_checks.check_replay(reference,current,valid_mask=valid,tolerance=tolerance)
                for name,computed in [('state_replay',replay),('empty_control',empty),('qualification',qualified)]:
                    if control.get(name)!=computed:raise ValueError('Qualified raw/recomputed control differs: '+name)
                if order>=13:
                    repeated=_qualified_outcome(data,f'repetition_{order}_')
                    result=_checks.check_repeated_outcome(outcomes[candidate],repeated,valid_mask=valid,tolerance=tolerance)
                    if control.get('outcome_repetition')!=result:
                        raise ValueError('Full fixed-domain 50-step repetition control differs')
    if q.get('qualification_by_anchor')!=records or json.loads(csv_path.with_name('qualification_summary.json').read_text(encoding='utf-8'))!=records:
        raise ValueError('Qualification summary differs from independently reconstructed prefix eligibility')
    if not sum(r['qualified_count'] for r in records):raise ValueError('All anchor qualification domains empty; no statistical estimate')
    return {'protocol':PROTOCOL,'registry_sha256':REGISTRY_SHA256,'qualification_by_anchor':records,
            'usd_isolation_validation':isolation['validation'],'all_61_fields_recomputed':True,
            'unqualified_state_saved_and_finite':True,'repeat_domain':'fixed_reference_qualification_all_50_steps',
            'artifact_sha256':{str(csv_path.with_name(name).resolve()):_digest(csv_path.with_name(name))
                for name in ['protocol_identity.json','usd_isolation.json','qualification_summary.json']}}


def validate_run(csv_path, metadata, rows, controls):
    qualified_validation=_validate_qualified_protocol(csv_path,metadata,controls)
    spec,config=metadata['spec'],metadata['config']
    eval_seeds=_unique_nonnegative(spec['evaluation_seeds'],'evaluation_seeds')
    anchors=_unique_nonnegative(spec['anchor_steps'],'anchor_steps')
    n=int(config['env']['num_envs'])
    count=int(config['algo']['method']['counterfactual_count'])+1
    if count!=13 or n<1:raise ValueError('Formal teacher study requires 12 teacher candidates plus one nominal')
    tolerance=float(spec.get('replay_tolerance',1e-4))
    if not np.isfinite(tolerance) or not 0<=tolerance<=1e-4:
        raise ValueError('Frozen replay tolerance exceeds 1e-4')
    timing=_validate_actual_timing(metadata,config)
    horizon=timing['horizon_steps'];duration=timing['realized_horizon_s']
    floor_steps=timing['floor_control_horizon_steps']
    if any(a<1 or a+horizon>=int(config['env']['max_episode_length']) for a in anchors):
        raise ValueError('Anchor/horizon may cross environment truncation')
    expected_blocks={(e,a) for e in eval_seeds for a in anchors}
    initialization=_validate_initialization(metadata,eval_seeds,anchors,count,timing)
    control_index={}
    expected_order=list(range(count))+[count-1,count-2]
    for control in controls:
        key=(int(control['evaluation_seed']),int(control['anchor_step']),int(control['order_position']))
        if key in control_index:raise ValueError('Duplicate replay control position')
        if key[:2] not in expected_blocks or not 0<=key[2]<len(expected_order):
            raise ValueError('Unexpected replay control block/order')
        if int(control['candidate'])!=expected_order[key[2]]:raise ValueError('Replay candidate order differs from frozen scheme')
        _passed_state_control(control.get('state_replay'),tolerance,'state replay')
        _passed_state_control(control.get('empty_control'),0.,'empty/no-action')
        if control.get('common_future_obstacle_trajectory') is not True:
            raise ValueError('Common future obstacle trajectory not validated')
        if key[2]>=count:
            _passed_control(control.get('outcome_repetition'),tolerance,'nominal/selected outcome repetition')
        elif 'outcome_repetition' in control:raise ValueError('Unexpected repetition control')
        control_index[key]=control
    expected_controls={(e,a,o) for e,a in expected_blocks for o in range(count+2)}
    if control_index.keys()!=expected_controls:raise ValueError('Incomplete 13+2 replay control coverage')
    indexed={}
    for row in rows:
        key=tuple(int(row[k]) for k in ['evaluation_seed','anchor_step','env_index','candidate_index'])
        if key in indexed:raise ValueError('Duplicate CSV anchor/candidate row')
        indexed[key]=row
    expected_rows=set();raw_hashes={};valid_total=0;attrition=[]
    for e,a in sorted(expected_blocks):
        raw_path=csv_path.with_name(f'seed{e}_anchor{a}.npz')
        if not raw_path.is_file():raise ValueError('Missing planned raw NPZ anchor block: '+str(raw_path))
        raw_hashes[str(raw_path)]=_digest(raw_path)
        with np.load(raw_path,allow_pickle=False) as data:
            valid=np.asarray(data['valid_anchor_mask'])
            commands=np.asarray(data['candidate_commands_local'])
            pred_clear=np.asarray(data['teacher_clearances']);pred_risk=np.asarray(data['teacher_risks'])
            if valid.shape!=(n,) or not np.isin(valid,[0,1]).all():raise ValueError('Invalid NPZ anchor validity mask')
            valid=valid.astype(bool);valid_total+=int(valid.sum())
            attrition.append({"evaluation_seed":e,"anchor_step":a,"planned_environment_cases":n,"valid_anchor_states":int(valid.sum()),"excluded_after_goal_or_done":n-int(valid.sum())})
            if commands.shape!=(n,count,3) or not np.isfinite(commands).all():raise ValueError('Invalid NPZ candidate command shape')
            if pred_clear.shape!=(n,count) or np.isnan(pred_clear).any() or np.isneginf(pred_clear).any():raise ValueError('Invalid teacher clearance array')
            if pred_risk.shape!=(n,count) or not np.isfinite(pred_risk).all() or not ((pred_risk>=0)&(pred_risk<=1)).all():raise ValueError('Invalid teacher score array')
            expected_risk=1/(1+np.exp(np.clip((pred_clear-.35)/.15,-80,80)))
            if not np.allclose(pred_risk,expected_risk,rtol=0,atol=2e-6):raise ValueError('Teacher score does not match logged clearance')
            limit=float(config['algo']['actor']['action_limit'])
            if (np.linalg.norm(commands[:,:count-1],axis=-1)>limit+2e-6).any() or (np.abs(commands[:,-1])>limit+2e-6).any():
                raise ValueError('Pool norm cap / actual nominal component limit inconsistent')
            prefix=np.asarray(data['command_prefix_world'])
            if prefix.shape[0]!=a or prefix.size!=a*n*3 or not np.isfinite(prefix).all():raise ValueError('Invalid saved complete command prefix')
            state_keys={k for k in data.files if k.startswith('anchor_state__')}
            if not {'anchor_state__'+key.replace('/','__') for key in PHYSICAL_STATE_FIELDS|RANDOM_STATE_FIELDS}<=state_keys:
                raise ValueError('Raw anchor lacks required physical/random state fields')
            for key in PHYSICAL_STATE_FIELDS:
                value=np.asarray(data['anchor_state__'+key.replace('/','__')])
                if value.ndim<2 or value.shape[0]!=n or not value.size or not np.isfinite(value).all():
                    raise ValueError('Invalid raw articulated physical state: '+key)
            for position in range(count+2):
                for kind in ['state_replay','empty_control']:
                    refs=control_index[e,a,position][kind]['reference_hashes']
                    if {'anchor_state__'+key.replace('/','__') for key in refs}!=state_keys:
                        raise ValueError('Replay reference hash state coverage differs from saved NPZ')
                    for key,expected_hash in refs.items():
                        if _array_hash(data['anchor_state__'+key.replace('/','__')])!=expected_hash:
                            raise ValueError('Replay reference state hash differs from saved NPZ')
            target=np.asarray(data['anchor_state__environment__target_pos']).reshape(n,-1,3)
            drone=np.asarray(data['anchor_state__drone__full_state'])
            if target.shape!=(n,1,3) or drone.shape[0]!=n or drone.shape[-1]<13:raise ValueError('Invalid anchor drone/goal state')
            start=drone[...,:3].reshape(n,-1,3)
            if start.shape!=(n,1,3) or not np.isfinite(start).all() or not np.isfinite(target).all():raise ValueError('Invalid anchor start/goal geometry')
            initial_distance=np.linalg.norm(target[:,0]-start[:,0],axis=-1)
            arrays={};masks={};common_obstacles=None
            for candidate in range(count):
                one={}
                for field,shape in [('positions',(n,horizon,3)),('costs',(n,horizon)),('observed_clearance',(n,horizon))]:
                    array=np.asarray(data[f'candidate_{candidate}_{field}'])
                    if array.shape!=shape or not np.isfinite(array).all():raise ValueError('Invalid raw continuous branch array: '+field)
                    one[field]=array
                obstacles=np.asarray(data[f'candidate_{candidate}_dynamic_positions'])
                if obstacles.ndim!=3 or obstacles.shape[0]!=horizon or obstacles.shape[-1]!=3 or not np.isfinite(obstacles).all():raise ValueError('Invalid raw obstacle trajectory')
                if common_obstacles is None:common_obstacles=obstacles
                elif not np.array_equal(common_obstacles,obstacles):raise ValueError('Different candidates changed common obstacle trajectory')
                masks[candidate]=_load_masks(data,candidate,n,horizon)
                if masks[candidate]['truncated'].any():raise ValueError('Unexpected truncated event inside local horizon')
                predicted_cost=np.maximum(1/(1+np.exp(np.clip((one['observed_clearance']-.35)/.15,-80,80))),masks[candidate]['collision'].astype(float))
                if not np.allclose(one['costs'],predicted_cost,rtol=0,atol=2e-6):raise ValueError('Raw cost/clearance/collision formula mismatch')
                arrays[candidate]=one
                for env in np.flatnonzero(valid):
                    key=(e,a,int(env),candidate);expected_rows.add(key)
                    if key not in indexed:continue
                    row=indexed[key]
                    role=('nominal' if candidate==count-1 else 'teacher_selected' if candidate==count-2 else 'speed_capped_nominal_pool' if candidate==0 else 'fixed_lattice')
                    if row['candidate_role']!=role:raise ValueError('CSV candidate role does not match preserved pool/nominal map')
                    command=np.array([float(row['candidate_local_'+axis]) for axis in 'xyz'])
                    if not np.allclose(command,commands[env,candidate],rtol=0,atol=1e-7):raise ValueError('CSV candidate command differs from NPZ')
                    pc=float(row['teacher_predicted_clearance_m']);pr=float(row['teacher_predicted_risk'])
                    if not (pc==pred_clear[env,candidate] or np.isclose(pc,pred_clear[env,candidate],rtol=0,atol=2e-6)) or not np.isclose(pr,pred_risk[env,candidate],rtol=0,atol=2e-6):raise ValueError('CSV teacher prediction differs from NPZ')
                    if _boolean(row['teacher_predicted_feasible'],'feasible')!=(pc>=0) or _boolean(row['teacher_clearance_is_infinite'],'infinite')!=bool(np.isinf(pc)):raise ValueError('CSV teacher feasibility/finite flag inconsistent')
                    termination=masks[candidate]['terminated'][env]
                    where=np.flatnonzero(termination);first=int(where[0]) if len(where) else -1
                    stop=first+1 if first>=0 else horizon
                    collision=bool(masks[candidate]['collision'][env,:stop].any())
                    if _boolean(row['realized_collision'],'collision')!=collision or _boolean(row['realized_unsafe'],'unsafe')!=(first>=0) or int(row['realized_first_unsafe_step'])!=first:raise ValueError('CSV realized event differs from raw masks')
                    progress=initial_distance[env]-np.linalg.norm(target[env,0]-one['positions'][env,stop-1])
                    values={'realized_observation_min_clearance_m':one['observed_clearance'][env,:stop].min(),
                            'realized_mean_step_cost':one['costs'][env,:stop].mean(),'realized_progress_m':progress,
                            'horizon_steps':horizon,'horizon_s':duration}
                    for field,value in values.items():
                        atol=(1e-10 if field=='horizon_s' else 0. if field=='horizon_steps' else max(tolerance,2e-6))
                        if not np.isfinite(float(row[field])) or not np.isclose(float(row[field]),value,rtol=0,atol=atol):
                            raise ValueError('CSV/raw numeric branch outcome differs: '+field)
                    if 'requested_horizon_s' in row and not np.isclose(float(row['requested_horizon_s']),timing['requested_teacher_horizon_s'],rtol=0,atol=1e-12):
                        raise ValueError('CSV requested teacher horizon differs from actual_timing')
                    # Derived from raw event masks, not trusted CSV labels.
                    floor_termination=termination[:floor_steps]
                    floor_where=np.flatnonzero(floor_termination)
                    floor_stop=int(floor_where[0])+1 if len(floor_where) else floor_steps
                    row['floor_control_realized_collision']=str(bool(masks[candidate]['collision'][env,:floor_stop].any()))
                    row['floor_control_realized_unsafe']=str(bool(len(floor_where)))
                    row['actual_timing']=timing
            # An exact repeated command at one anchor must have consistent
            # outcomes BEFORE any command deduplication reduces row counts.
            for env in np.flatnonzero(valid):
                seen={}
                for candidate in range(count):
                    key=np.asarray(commands[env,candidate],dtype=np.float32).tobytes()
                    if key in seen:
                        previous=seen[key]
                        for field in arrays[candidate]:
                            if not np.allclose(arrays[candidate][field][env],arrays[previous][field][env],rtol=0,atol=tolerance):
                                raise ValueError('Duplicate command changed raw continuous trajectory/outcome')
                        for field in masks[candidate]:
                            if not np.array_equal(masks[candidate][field][env],masks[previous][field][env]):raise ValueError('Duplicate command changed raw event masks')
                    else:seen[key]=candidate
    if indexed.keys()!=expected_rows:raise ValueError('CSV does not cover exactly 13 candidates per valid NPZ anchor')
    if int(metadata['rows'])!=len(rows) or int(metadata['unique_anchor_states'])!=valid_total:
        raise ValueError('Metadata row/anchor counts differ from verified NPZ coverage')
    return {'qualified_replay':qualified_validation,'planned_blocks':len(expected_blocks),'controls':len(controls),'valid_anchor_states':valid_total,
            'actual_timing':timing,'initialization':initialization,'anchor_attrition':attrition,'raw_npz_sha256':raw_hashes,
            'controls_sha256':_digest(csv_path.with_name('replay_controls.json'))}


def read(root, allow_pilot=False):
    groups=defaultdict(list);sources={};source_hashes={};seen_blocks=set();designs={};timing_designs={}
    source_eval_sets=defaultdict(lambda:defaultdict(set))
    for csv_path in sorted(Path(root).rglob('candidate_outcomes.csv')):
        metadata_path=csv_path.with_name('metadata.json')
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        status=metadata['status'];spec=metadata['spec']
        if status=='QUALIFIED_PHYSICAL_REPLAY_PILOT_COMPLETE' and not allow_pilot:
            raise ValueError('Pilot output rejected from formal analysis; use --allow-pilot explicitly')
        if status not in {'QUALIFIED_PHYSICAL_REPLAY_FORMAL_COMPLETE','QUALIFIED_PHYSICAL_REPLAY_PILOT_COMPLETE'}:
            raise ValueError('Unvalidated physical branch run: '+str(csv_path))
        phase='formal' if status=='QUALIFIED_PHYSICAL_REPLAY_FORMAL_COMPLETE' else 'pilot'
        if spec.get('phase','pilot')!=phase:raise ValueError('Metadata phase/status mismatch')
        source=spec.get('source_name',csv_path.parent.name)
        family=spec.get('family',spec['policy_kind']);condition=spec.get('condition','unspecified')
        weight_hash=metadata['checkpoint_sha256']
        if not isinstance(source,str) or not source or not isinstance(weight_hash,str) or len(weight_hash)!=64 or any(c not in '0123456789abcdef' for c in weight_hash) or spec.get('checkpoint_sha256')!=weight_hash:
            raise ValueError('Invalid source/checkpoint identity or spec hash mismatch')
        if source in source_hashes and source_hashes[source]!=weight_hash:raise ValueError('Source name binds multiple checkpoint hashes')
        source_hashes[source]=weight_hash
        design=tuple(spec['anchor_steps'])
        if condition in designs and designs[condition]!=design:raise ValueError('Shared source/evaluation/anchor design differs within condition')
        designs[condition]=design
        source_eval_sets[condition][source].update(spec['evaluation_seeds'])
        for e in spec['evaluation_seeds']:
            for a in spec['anchor_steps']:
                key=(source,int(e),int(a))
                if key in seen_blocks:raise ValueError('Duplicate source/seed/anchor block across files')
                seen_blocks.add(key)
        with csv_path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        controls=json.loads(csv_path.with_name('replay_controls.json').read_text(encoding='utf-8'))
        validation=validate_run(csv_path,metadata,rows,controls)
        timing_signature=json.dumps(validation['actual_timing'],sort_keys=True)
        if condition in timing_designs and timing_designs[condition]!=timing_signature:
            raise ValueError('Measured timing differs across sources within one condition')
        timing_designs[condition]=timing_signature
        sources[str(csv_path)]={'sha256':_digest(csv_path),'checkpoint_sha256':weight_hash,
            'metadata_sha256':_digest(metadata_path),'source':source,'family':family,'condition':condition,
            'phase':phase,'validation':validation,'teacher_nominal_input':metadata.get('teacher_nominal_input'),
            'local_outcome_estimand':metadata.get('local_outcome_estimand')}
        for row in rows:
            row.update({'source':source,'family':family,'condition':condition})
            groups[condition,family].append(row)
    if not groups:raise ValueError('No completed physical replay candidate files')
    for condition,panels in source_eval_sets.items():
        reference=next(iter(panels.values()))
        if any(seeds!=reference for seeds in panels.values()):
            raise ValueError('Unbalanced evaluation-seed panels across sources within condition: '+condition)
    return groups,sources

def prepare(rows):
    anchors = defaultdict(list)
    seen_rows=set()
    for row in rows:
        row_key=(row['source'],int(row['evaluation_seed']),int(row['anchor_step']),int(row['env_index']),int(row['candidate_index']))
        if row_key in seen_rows:raise ValueError('Duplicate CSV anchor/candidate row')
        seen_rows.add(row_key)
        anchors[(row['source'], int(row['evaluation_seed']), int(row['anchor_step']), int(row['env_index']))].append(row)
    candidates, paired = [], []
    for key, group in sorted(anchors.items()):
        group.sort(key=lambda row: int(row['candidate_index']))
        nominal = next(r for r in group if r['candidate_role']=='nominal')
        selected = next(r for r in group if r['candidate_role']=='teacher_selected')
        cluster = key[:2]
        seen = set()
        unique = []
        for row in group:
            command = np.asarray([float(row[f'candidate_local_{axis}']) for axis in 'xyz'], dtype=np.float32).tobytes()
            if command in seen: continue
            seen.add(command)
            unique.append(row)
            predicted = float(row['teacher_predicted_clearance_m'])
            realized = float(row['realized_observation_min_clearance_m'])
            candidates.append((cluster, float(row['teacher_predicted_risk']), float(row['realized_collision']=='True'),
                               float(row['teacher_predicted_feasible']=='True'), predicted, realized,
                               float(row['realized_unsafe']=='True')))
        scores = np.array([float(row['teacher_predicted_risk']) for row in unique])
        clear = np.array([float(row['realized_observation_min_clearance_m']) for row in unique])
        i,j = np.triu_indices(len(unique),1)
        predicted_delta = scores[i]-scores[j]
        clearance_delta = clear[i]-clear[j]
        eligible = (np.abs(predicted_delta)>.05)&(np.abs(clearance_delta)>.02)
        rank_accuracy = float(np.mean(predicted_delta[eligible]*(-clearance_delta[eligible])>0)) if eligible.any() else 0.
        paired.append((cluster, float(selected['realized_collision']=='True')-float(nominal['realized_collision']=='True'),
            float(selected['realized_observation_min_clearance_m'])-float(nominal['realized_observation_min_clearance_m']),
            float(selected['realized_progress_m'])-float(nominal['realized_progress_m']),rank_accuracy,float(eligible.any()),
            float(selected['realized_unsafe']=='True')-float(nominal['realized_unsafe']=='True')))
    clusters = sorted({r[0] for r in candidates})
    ids = {key:i for i,key in enumerate(clusters)}
    c = np.array([[ids[r[0]], *r[1:]] for r in candidates], dtype=float)
    p = np.array([[ids[r[0]], *r[1:]] for r in paired], dtype=float)
    return clusters, c, p, len(anchors), len(rows)

def calculate(c, p, cluster_weights):
    w = cluster_weights[c[:,0].astype(int)]
    pw = cluster_weights[p[:,0].astype(int)]
    def average(values, weights):
        return number(np.sum(values*weights)/weights.sum()) if weights.sum() else None
    auc, ap = weighted_auc_ap(c[:,1], c[:,2], w)
    feasible = w*c[:,3]
    finite = np.isfinite(c[:,4])
    # This calibration target is a measured-clearance proxy, not a binary
    # collision probability and not a long-horizon discounted cost return.
    proxy = 1/(1+np.exp(np.clip((c[:,5]-.35)/.15, -80, 80)))
    return {'collision_score_auroc': auc, 'collision_score_auprc': ap,
            'realized_collision_base_rate': average(c[:,2], w),
            'false_safe_collision_rate': average(c[:,2], feasible),
            'realized_unsafe_base_rate': average(c[:,6], w),
            'false_safe_unsafe_rate': average(c[:,6], feasible),
            'finite_teacher_vs_observed_clearance_discrepancy_mae_m': average(np.where(finite, np.abs(c[:,4]-c[:,5]), 0), w*finite),
            'infinite_teacher_clearance_fraction': average((~finite).astype(float), w),
            'teacher_vs_measured_clearance_proxy_score_discrepancy': average(abs(c[:,1]-proxy), w),
            'within_state_resolvable_clearance_ranking_accuracy':average(p[:,4],pw*p[:,5]),
            'ranking_eligible_anchor_fraction':average(p[:,5],pw),
            'selected_minus_nominal_collision': average(p[:,1], pw),
            'selected_minus_nominal_unsafe': average(p[:,6], pw),
            'selected_minus_nominal_observation_clearance_m': average(p[:,2], pw),
            'selected_minus_nominal_progress_m': average(p[:,3], pw)}

def analyze(rows, replicates, rng):
    clusters, c, p, anchors, raw_rows = prepare(rows)
    sources = sorted({key[0] for key in clusters})
    seeds = sorted({key[1] for key in clusters})
    source_id = {s:i for i,s in enumerate(sources)}
    seed_id = {s:i for i,s in enumerate(seeds)}
    si = np.array([source_id[k[0]] for k in clusters])
    ei = np.array([seed_id[k[1]] for k in clusters])
    estimate = calculate(c,p,np.ones(len(clusters)))
    draws = defaultdict(list)
    event_keys={'collision_score_auroc','collision_score_auprc','realized_collision_base_rate',
                'false_safe_collision_rate','realized_unsafe_base_rate','false_safe_unsafe_rate',
                'selected_minus_nominal_collision','selected_minus_nominal_unsafe'}
    sensitivity=all('floor_control_realized_collision' in r and 'floor_control_realized_unsafe' in r and 'actual_timing' in r for r in rows)
    floor_estimate=None;floor_draws=defaultdict(list)
    if sensitivity:
        floor_rows=[{**r,'realized_collision':r['floor_control_realized_collision'],
                     'realized_unsafe':r['floor_control_realized_unsafe']} for r in rows]
        floor_clusters,fc,fp,_,_=prepare(floor_rows)
        if floor_clusters!=clusters:raise ValueError('Timing sensitivity changed source/eval clusters')
        floor_estimate={key:value for key,value in calculate(fc,fp,np.ones(len(clusters))).items() if key in event_keys}
    for _ in range(replicates):
        sc = np.bincount(rng.integers(len(sources),size=len(sources)), minlength=len(sources))
        ec = np.bincount(rng.integers(len(seeds),size=len(seeds)), minlength=len(seeds))
        weights = sc[si]*ec[ei]
        if weights.sum() == 0: continue
        for key, value in calculate(c,p,weights).items():
            if value is not None: draws[key].append(value)
        if sensitivity:
            for key,value in calculate(fc,fp,weights).items():
                if key in event_keys and value is not None:floor_draws[key].append(value)
    result = {'anchor_states': anchors, 'raw_candidate_rows': raw_rows,
              'unique_state_command_rows': len(c), 'source_checkpoints': len(sources),
              'evaluation_seeds': len(seeds), 'replay_clusters': len(clusters), 'metrics': {}}
    for key, value in estimate.items():
        values = draws[key]
        degenerate_rare_event = key in {'false_safe_collision_rate','realized_collision_base_rate',
                                      'false_safe_unsafe_rate','realized_unsafe_base_rate'} and value in {0.,1.}
        result['metrics'][key] = {'estimate': value,
            'descriptive_cluster_bootstrap_ci95': np.quantile(values,[.025,.975]).tolist() if values and not degenerate_rare_event else None,
            'rare_event_interval_unavailable':degenerate_rare_event,
            'bootstrap_draws_defined':len(values)}
    result['false_safe_interpretation'] = 'Zero observed events do not establish zero risk; bootstrap alone cannot give a nonzero rare-event upper bound.'
    result['clearance_interpretation'] = 'Realized NavRL observation clearance; teacher includes additional static/dynamic margins. Finite MAE excludes infinite predictions and reports their fraction.'
    if sensitivity:
        metrics={}
        for key,value in floor_estimate.items():
            values=floor_draws[key]
            degenerate=key in {'false_safe_collision_rate','realized_collision_base_rate',
                              'false_safe_unsafe_rate','realized_unsafe_base_rate'} and value in {0.,1.}
            metrics[key]={'estimate':value,
                'descriptive_cluster_bootstrap_ci95':np.quantile(values,[.025,.975]).tolist() if values and not degenerate else None,
                'rare_event_interval_unavailable':degenerate,'bootstrap_draws_defined':len(values)}
        timing_records={json.dumps(r['actual_timing'],sort_keys=True) for r in rows}
        result['duration_sensitivity']={
            'method':'Raw trajectory prefix through floor(requested/control_dt) steps; same source/eval bootstrap draws, no new physical execution.',
            'timing_records':[json.loads(s) for s in sorted(timing_records)],
            'floor_control_event_metrics':metrics,
            'nearest_only_collision_fraction':estimate['realized_collision_base_rate']-floor_estimate['realized_collision_base_rate'],
            'nearest_only_unsafe_fraction':estimate['realized_unsafe_base_rate']-floor_estimate['realized_unsafe_base_rate'],
            'interpretation':'The two recorded durations bracket requested time only when requested_within_recorded_bracket is true. This measures sensitivity to the final discrete transition, not a guaranteed continuous-contact incidence bound or a reconstruction of continuous first-contact times.'}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--allow-pilot', action='store_true', help='Explicit exploratory inclusion of pilot outputs')
    args = parser.parse_args()
    groups, sources = read(args.root,allow_pilot=args.allow_pilot)
    rng = np.random.default_rng(20261020)
    result = {'analysis_kind':'prospective_qualified_physical_replay_validation', 'teacher_validation_protocol':PROTOCOL,
              'pilot_inclusion_explicitly_allowed':args.allow_pilot, 'source_files':sources,
              'bootstrap':'source checkpoints and shared evaluation seeds resampled, anchor cases/candidates retained together',
              'limitations':['Teacher scores are not claimed as calibrated collision probabilities.',
                             'Motion-state and repeated-outcome equivalence apply to the fixed reference-prefix qualified environments. All global state, RNG and fixed parameters are unmasked exact controls; excluded trajectories are retained and audited for finiteness and complete diagnostics.',
                             'Historical all-environment replay failures remain rejected; this prospective protocol requires fresh qualified replay evidence and actual USD isolation proof.',
                             'Replayed exposed state is validated; hidden PhysX state is not serialized.',
                             'Local hazards continue past reached-goal states until first unsafe or horizon; goal is not an absorbing endpoint here.',
                             'For F11 anchor trajectories, nominal refers to the frozen base-task mean used for training labels, not the current adapted actor.',
                             'Requested teacher duration is rounded to actual simulator control steps; reported measured duration can differ from 0.8 seconds.',
                             'These are local fixed-command outcomes, not complete episode outcomes.'],
              'groups':{condition+'/'+family:analyze(rows,args.bootstrap,rng) for (condition,family),rows in groups.items()}}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'teacher_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'completed_groups':len(groups),'output':str(args.output)}))

if __name__=='__main__': main()

"""Prespecified paired endpoint and gate-by-Lcf factorial contrasts.

Shared released-baseline episodes are stored once. Training and shared
evaluation-seed clusters are jointly resampled; 64 environment cases remain
fixed within a block. These approximate intervals do not prove equivalence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

METRICS = ['success','collision','path_efficiency','mean_safety_cost']
CONTRASTS = {
    'full_minus_released_NavRL': {'F11_full':1,'released_NavRL_single_checkpoint':-1},
    'full_minus_PPO_L': {'F11_full':1,'E_ppolag_reference':-1},
    'gate_effect_Lcf_on': {'F11_full':1,'F01_no_gate':-1},
    'Lcf_effect_gate_on': {'F11_full':1,'F10_no_Lcf':-1},
    'gate_by_Lcf_interaction': {'F11_full':1,'F01_no_gate':-1,'F10_no_Lcf':-1,'F00_no_gate_no_Lcf':1},
    'gate_effect_Lcf_off': {'F10_no_Lcf':1,'F00_no_gate_no_Lcf':-1},
    'Lcf_effect_gate_off': {'F01_no_gate':1,'F00_no_gate_no_Lcf':-1},
    'freezing_effect': {'F11_full':1,'P_free_nominal':-1},
    'quantile_effect': {'F11_full':1,'Q_scalar':-1},
    'warmup_effect': {'F11_full':1,'W0_no_warmup':-1},
    'teacher_supervision_bundle_effect': {'F11_full':1,'T0_no_teacher':-1},
}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def holm(values):
    values = np.asarray(values,dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Holm inputs must be finite probabilities in [0,1]')
    order = np.argsort(values)
    corrected = np.minimum(1.,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    result = np.empty_like(corrected); result[order] = corrected
    return result

def validate_plan(plan):
    for key in ['training_seeds','evaluation_seeds']:
        values=plan.get(key)
        if not isinstance(values,list) or not values or len(set(values))!=len(values) or any(type(x) is not int or x<0 for x in values):
            raise ValueError(f'Formal plan requires unique nonnegative integer {key}')
    if type(plan.get('num_envs')) is not int or plan['num_envs']<1:
        raise ValueError('Formal plan requires a positive integer num_envs')
    variants=set(plan['variant_configurations'])
    referenced=set().union(*(set(c) for c in CONTRASTS.values()))-{'released_NavRL_single_checkpoint'}
    if variants!=referenced:
        raise ValueError('Formal plan does not match prespecified nine-configuration matrix')
    jobs=[(j['variant'],j['training_seed']) for j in plan['training_jobs']]
    expected=set(itertools.product(variants,plan['training_seeds']))
    if len(jobs)!=len(set(jobs)) or set(jobs)!=expected:
        raise ValueError('Formal plan training jobs are duplicated or incomplete')


def load_panel(paths, plan):
    validate_plan(plan)
    required_names = {f'{j["variant"]}_s{j["training_seed"]}' for j in plan['training_jobs']}
    required_names.add('released_NavRL_single_checkpoint')
    index, sources = {}, {}
    for path in paths:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            rows = list(csv.DictReader(stream))
        sources[str(path.resolve())] = {'sha256':digest(path),'rows':len(rows)}
        for row in rows:
            name = row['mode']
            if name not in required_names: raise ValueError(f'Unexpected method in formal panel: {name}')
            key = (name,int(row['seed']),int(row['env_index']))
            if key in index: raise ValueError(f'Duplicate shared episode: {key}')
            values = np.array([float(row[m]) for m in METRICS])
            if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all() or not np.isin(values[:2],[0,1]).all() or values[:2].sum()>1:
                raise ValueError(f'Invalid primary outcome: {key}')
            distance=float(row['straight_distance_m'])
            if not np.isfinite(distance) or distance<0:
                raise ValueError(f'Invalid initial goal distance: {key}')
            index[key] = (values,distance)
    expected = set(itertools.product(required_names,plan['evaluation_seeds'],range(plan['num_envs'])))
    if index.keys() != expected:
        raise ValueError(f'Formal episode panel incomplete: missing {len(expected-index.keys())}, extra {len(index.keys()-expected)}')
    panels = {}
    for variant in plan['variant_configurations']:
        values = []
        for training_seed in plan['training_seeds']:
            evalseeds=[]
            for evaluation_seed in plan['evaluation_seeds']:
                cases=[]
                for env in range(plan['num_envs']):
                    value,distance = index[(f'{variant}_s{training_seed}',evaluation_seed,env)]
                    base_distance = index[('released_NavRL_single_checkpoint',evaluation_seed,env)][1]
                    if abs(distance-base_distance)>1e-5: raise ValueError('Pair reset goal distance differs')
                    cases.append(value)
                evalseeds.append(cases)
            values.append(evalseeds)
        panels[variant] = np.asarray(values)
    baseline=np.asarray([[index[('released_NavRL_single_checkpoint',ev,env)][0]
                         for env in range(plan['num_envs'])] for ev in plan['evaluation_seeds']])
    panels['released_NavRL_single_checkpoint'] = baseline[None,...]
    return panels,sources

def crossed_samples(block, replicates, seed):
    block=np.asarray(block,dtype=float)
    if type(replicates) is not int or replicates<100 or block.ndim<2 or min(block.shape[:2])<1 or not np.isfinite(block).all():
        raise ValueError('Crossed bootstrap needs finite nonempty clusters and at least 100 replicates')
    rng=np.random.default_rng(seed)
    s,e=block.shape[:2]
    chunks=[]
    for start in range(0,replicates,500):
        b=min(500,replicates-start)
        si=rng.integers(s,size=(b,s))
        ei=rng.integers(e,size=(b,e))
        chunks.append(block[si[:,:,None],ei[:,None,:]].mean(axis=(1,2)))
    return np.concatenate(chunks)

def analyze(panels, plan, replicates=20000):
    from scipy.stats import t
    validate_plan(plan)
    expected_names=set(plan['variant_configurations'])|{'released_NavRL_single_checkpoint'}
    if set(panels)!=expected_names: raise ValueError('Analysis panels do not match frozen matrix')
    for name,data in panels.items():
        expected_shape=(1 if name=='released_NavRL_single_checkpoint' else len(plan['training_seeds']),
                        len(plan['evaluation_seeds']),plan['num_envs'],len(METRICS))
        if np.asarray(data).shape!=expected_shape or not np.isfinite(data).all():
            raise ValueError(f'Analysis panel shape/values differ from plan: {name}')
    names=list(CONTRASTS)
    differences=np.stack([sum(coef*panels[name] for name,coef in CONTRASTS[contrast].items())
                          for contrast in names],axis=-2)
    block=differences.mean(axis=2)  # [training, eval, contrast, metric]
    estimate=block.mean(axis=(0,1))
    draws=crossed_samples(block,replicates,20261020)
    rows=[]
    seed_means=block.mean(axis=1)
    s=len(plan['training_seeds'])
    signs=(2*((np.arange(2**s)[:,None]>>np.arange(s))&1)-1) if s<=16 else None
    for ci,name in enumerate(names):
        for mi,metric in enumerate(METRICS):
            value=estimate[ci,mi]
            low,high=np.quantile(draws[:,ci,mi],[.025,.975])
            error=draws[:,ci,mi]-value
            p=(1+int((np.abs(error)>=abs(value)-1e-15).sum()))/(replicates+1)
            per_seed=seed_means[:,ci,mi]
            se=per_seed.std(ddof=1)/np.sqrt(s) if s>1 else None
            t_ci=None if se is None else (value+t.ppf([.025,.975],s-1)*se).tolist()
            exact=None if signs is None else float(np.mean(np.abs(signs@per_seed/s)>=abs(value)-1e-14))
            if mi>=2: family='secondary_descriptive'
            elif ci<2: family='primary_system_4_tests'
            elif name in {'gate_effect_Lcf_on','Lcf_effect_gate_on','gate_by_Lcf_interaction'}: family='component_factorial_6_tests'
            elif name in {'gate_effect_Lcf_off','Lcf_effect_gate_off'}: family='conditional_off_4_exploratory_tests'
            else: family='matched_factors_8_exploratory_tests'
            rows.append({'contrast':name,'metric':metric,'difference':float(value),
                'ci95_low':float(low),'ci95_high':float(high),'bootstrap_centered_null_p_approx':p,
                'holm_family':family,'holm_adjusted_p_approx':None,
                'conditional_seed_sign_flip_p_sensitivity':exact,
                'conditional_seed_t_ci95_sensitivity':t_ci,
                'training_seeds':s,'evaluation_seeds':len(plan['evaluation_seeds']),
                'fixed_environment_cases':plan['num_envs'],
                'shared_frozen_baseline_unique_episodes':len(plan['evaluation_seeds'])*plan['num_envs'] if ci==0 else 0,
                'referenced_paired_differences':s*len(plan['evaluation_seeds'])*plan['num_envs']})
    for family in {r['holm_family'] for r in rows if r['holm_family']!='secondary_descriptive'}:
        selected=[r for r in rows if r['holm_family']==family]
        for row,value in zip(selected,holm([r['bootstrap_centered_null_p_approx'] for r in selected])):
            row['holm_adjusted_p_approx']=float(value)
    absolute={name:{'mean':dict(zip(METRICS,np.asarray(data).mean(axis=tuple(range(data.ndim-1))).tolist())),
                    'independent_training_checkpoints':0 if name.startswith('released_') else s,
                    'unique_evaluation_episodes':(1 if name.startswith('released_') else s)*len(plan['evaluation_seeds'])*plan['num_envs']}
              for name,data in panels.items()}
    return rows,absolute

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',type=Path,nargs='+',required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=20000)
    args=parser.parse_args()
    if digest(args.plan)!=args.plan.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]: raise ValueError('Plan SHA mismatch')
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    paths=sorted(p for target in args.episodes for p in (target.rglob('episodes.csv') if target.is_dir() else [target]))
    panels,sources=load_panel(paths,plan)
    rows,absolute=analyze(panels,plan,args.bootstrap)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'paired_effects.json').write_text(json.dumps({'source_sha256':digest(__file__),
        'input_files':sources,'plan_sha256':digest(args.plan),'absolute':absolute,'contrasts':rows,
        'inference':'Crossed training and shared evaluation seeds; fixed case blocks; all methods sampled jointly.',
        'limitations':['Approximate centered bootstrap p-values and descriptive 95% CIs are distinguished from Holm correction.',
        'Seed t/sign-flip sensitivities condition on the finite evaluation cases and assume appropriate seed exchangeability/symmetry.',
        'Shared baseline has one checkpoint; no baseline pretraining variance is represented.',
        'Equal initial goal distances do not establish identical physical reset states; collection provenance must verify full matched resets separately.',
        'mean_safety_cost is an arithmetic diagnostic, not the CMDP discounted budget.',
        'No equivalence, no component benefit, or no safety claim follows merely from a confidence interval spanning zero.']},indent=2),encoding='utf-8')
    serial=[]
    for row in rows:
        serial.append({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    with (args.output/'paired_effects.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(serial[0]));writer.writeheader();writer.writerows(serial)
    print(json.dumps({'formal_rows':len(rows),'output':str(args.output)}))

if __name__=='__main__': main()

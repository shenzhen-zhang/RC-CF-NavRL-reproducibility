"""CPU statistical APIs; actual evidence is reachable only through the sealed gate.

No episode/trace is synthesized or saved. Missing endpoint arrays exist only
as internal arguments to linear, marginal sensitivity calculations.
"""
from __future__ import annotations
import csv,hashlib,importlib.util,itertools,json,math,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
PAIRED_SHA='35bd679cf534157c269f410ed803eb8e439443d7cfbe463fe3f188ea8fd4111c'
BUDGET_SHA='bd2ebb26ced59b9fce64ff8390ab4570475a2482af65fda59320ab7d821d8a3e'
REPLICATES=20000;SEED=20261020;ALPHA=.05;BUDGET=.05;GAMMA=.99
PRIMARY=('full_minus_released_NavRL','full_minus_PPO_L')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def load(path,name,expected):
 require(sha(path)==expected,'Original analyzer source SHA differs')
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def case_names(plan):return {j['variant']+'_s'+str(j['training_seed']) for j in plan['training_jobs']}|{'released_NavRL_single_checkpoint'}
def episode_keys(plan):return set(itertools.product(case_names(plan),plan['evaluation_seeds'],range(plan['num_envs'])))
def missing_keys(cases,plan):return {(name,int(ev),env) for name,ev in cases for env in range(plan['num_envs'])}

def load_available(paths,plan,missing_cases,paired):
 """Use the original loader's per-observation validation, preserve its refusal.

 The frozen loader checks every observed value/duplicate before its final
 complete-panel check. Only its exact, prespecified missing-key rejection is
 admitted; the original342/full-matrix gate itself remains failed.
 """
 absent=missing_keys(missing_cases,plan);expected=episode_keys(plan)
 require(absent<=expected and len(absent)==len(missing_cases)*plan['num_envs'],'Invalid/duplicate missing case specification')
 try:paired.load_panel(paths,plan)
 except ValueError as exc:
  require(str(exc)==f'Formal episode panel incomplete: missing {len(absent)}, extra 0','Original outcome validation failed: '+str(exc))
 else:raise ValueError('Expected incomplete original matrix; full-panel gate must remain rejected')
 index={};sources={}
 for path in paths:
  with Path(path).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
  sources[str(Path(path).resolve())]={'sha256':sha(path),'rows':len(rows)}
  for row in rows:
   key=(row['mode'],int(row['seed']),int(row['env_index']))
   require(key not in index,'Duplicate actual available episode')
   index[key]=(np.array([float(row[m]) for m in paired.METRICS]),float(row['straight_distance_m']))
 require(set(index)==expected-absent,'Missing/extra actual episode differs from exact five-case scope')
 for (name,ev,env),(values,distance) in index.items():
  base=index.get(('released_NavRL_single_checkpoint',ev,env));require(base is not None,'Any missing primary released case rejects execution')
  require(abs(distance-base[1])<=1e-5,'Original paired reset goal distance differs')
 panels={}
 for variant in plan['variant_configurations']:
  data=np.full((len(plan['training_seeds']),len(plan['evaluation_seeds']),plan['num_envs'],len(paired.METRICS)),np.nan)
  for ti,tr in enumerate(plan['training_seeds']):
   for ei,ev in enumerate(plan['evaluation_seeds']):
    for env in range(plan['num_envs']):
     key=(variant+'_s'+str(tr),ev,env)
     if key in index:data[ti,ei,env]=index[key][0]
  panels[variant]=data
 panels['released_NavRL_single_checkpoint']=np.asarray([[index[('released_NavRL_single_checkpoint',ev,env)][0] for env in range(plan['num_envs'])] for ev in plan['evaluation_seeds']])[None,...]
 validate_partial_panels(panels,plan,missing_cases,paired)
 return panels,sources

def validate_partial_panels(panels,plan,missing_cases,paired):
 paired.validate_plan(plan);absent=set((str(n),int(e)) for n,e in missing_cases)
 require(len(absent)==len(missing_cases) and absent<=set(itertools.product(case_names(plan),plan['evaluation_seeds'])),'Duplicate or wrong missing case identity')
 require(set(panels)==set(plan['variant_configurations'])|{'released_NavRL_single_checkpoint'},'Exact original method matrix required')
 for name,data in panels.items():
  trains=[-1] if name=='released_NavRL_single_checkpoint' else plan['training_seeds']
  require(np.asarray(data).shape==(len(trains),len(plan['evaluation_seeds']),plan['num_envs'],4),'Original panel shape differs')
  for ti,tr in enumerate(trains):
   checkpoint=name if tr==-1 else name+'_s'+str(tr)
   for ei,ev in enumerate(plan['evaluation_seeds']):
    cell=np.asarray(data[ti,ei]);unknown=(checkpoint,ev) in absent
    if unknown:require(np.isnan(cell).all(),'A failed partial case must not enter observed outcomes')
    else:
     require(np.isfinite(cell).all() and ((cell>=0)&(cell<=1)).all(),'Observed endpoint outside original [0,1] domain')
     require(np.isin(cell[:,:2],[0,1]).all() and (cell[:,:2].sum(axis=1)<=1).all(),'Invalid original success/collision outcomes')

def primary_rows(panels,plan,paired,replicates=REPLICATES):
 """Projection of original analyze rows, original draws/t/signflip/Holm intact.

 Kept explicit rather than constructing fake methods for original analyze.
 Software controls compare all eight dictionaries to original analyze.
 """
 from scipy.stats import t
 paired.validate_plan(plan);s=len(plan['training_seeds']);e=len(plan['evaluation_seeds']);n=plan['num_envs']
 required={'F11_full','E_ppolag_reference','released_NavRL_single_checkpoint'}
 for name in required:
  x=np.asarray(panels[name]);require(x.shape==(1 if name.startswith('released_') else s,e,n,4) and np.isfinite(x).all(),'Any missing primary18 case rejects execution')
 differences=np.stack([sum(coef*panels[name] for name,coef in paired.CONTRASTS[contrast].items()) for contrast in PRIMARY],axis=-2)
 block=differences.mean(axis=2);estimate=block.mean(axis=(0,1));draws=paired.crossed_samples(block,replicates,SEED);seed_means=block.mean(axis=1)
 signs=(2*((np.arange(2**s)[:,None]>>np.arange(s))&1)-1) if s<=16 else None;rows=[]
 for ci,name in enumerate(PRIMARY):
  for mi,metric in enumerate(paired.METRICS):
   value=estimate[ci,mi];low,high=np.quantile(draws[:,ci,mi],[.025,.975]);error=draws[:,ci,mi]-value
   p=(1+int((np.abs(error)>=abs(value)-1e-15).sum()))/(replicates+1);per_seed=seed_means[:,ci,mi];se=per_seed.std(ddof=1)/np.sqrt(s) if s>1 else None
   t_ci=None if se is None else (value+t.ppf([.025,.975],s-1)*se).tolist();exact=None if signs is None else float(np.mean(np.abs(signs@per_seed/s)>=abs(value)-1e-14))
   rows.append({'contrast':name,'metric':metric,'difference':float(value),'ci95_low':float(low),'ci95_high':float(high),'bootstrap_centered_null_p_approx':p,'holm_family':'secondary_descriptive' if mi>=2 else 'primary_system_4_tests','holm_adjusted_p_approx':None,'conditional_seed_sign_flip_p_sensitivity':exact,'conditional_seed_t_ci95_sensitivity':t_ci,'training_seeds':s,'evaluation_seeds':e,'fixed_environment_cases':n,'shared_frozen_baseline_unique_episodes':e*n if ci==0 else 0,'referenced_paired_differences':s*e*n})
 selected=[row for row in rows if row['holm_family']=='primary_system_4_tests']
 for row,value in zip(selected,paired.holm([row['bootstrap_centered_null_p_approx'] for row in selected])):row['holm_adjusted_p_approx']=float(value)
 return rows

def contrast_endpoint_blocks(panels,paired):
 lower=[];upper=[]
 for coefficients in paired.CONTRASTS.values():
  low=0.;high=0.
  for name,coef in coefficients.items():
   data=panels[name];low=low+coef*np.where(np.isnan(data),0. if coef>0 else 1.,data);high=high+coef*np.where(np.isnan(data),1. if coef>0 else 0.,data)
  lower.append(low.mean(axis=2));upper.append(high.mean(axis=2))
 return np.stack(lower,axis=-2),np.stack(upper,axis=-2)

def sensitivity_rows(panels,plan,missing_cases,paired,replicates=REPLICATES):
 validate_partial_panels(panels,plan,missing_cases,paired);lower,upper=contrast_endpoint_blocks(panels,paired)
 # The same frozen generator seed, shape and chunk scheme reproduce exactly
 # the same nonnegative crossed weights for both endpoints and every contrast.
 lowdraw=paired.crossed_samples(lower,replicates,SEED);highdraw=paired.crossed_samples(upper,replicates,SEED)
 require((lowdraw<=highdraw+1e-14).all(),'Sensitivity endpoint draw ordering failed')
 lows=lower.mean(axis=(0,1));highs=upper.mean(axis=(0,1));rows=[]
 for ci,name in enumerate(paired.CONTRASTS):
  for mi,metric in enumerate(paired.METRICS):
   rows.append({'contrast':name,'metric':metric,'finite_original18_point_lower':float(lows[ci,mi]),'finite_original18_point_upper':float(highs[ci,mi]),'bootstrap_sensitivity_envelope_lower_q025':float(np.quantile(lowdraw[:,ci,mi],.025)),'bootstrap_sensitivity_envelope_upper_q975':float(np.quantile(highdraw[:,ci,mi],.975)),'bootstrap_replicates':replicates,'bootstrap_seed':SEED,'training_seeds':len(plan['training_seeds']),'evaluation_seeds':len(plan['evaluation_seeds']),'fixed_environment_cases':plan['num_envs'],'interpretation':'Marginal bounded-missing-outcome sensitivity; not a confirmatory95% coverage guarantee','p_value_or_significance_decision':None,'joint_completion_optimality_claimed':False})
 return rows,{'lower_draw_sha256':hashlib.sha256(lowdraw.tobytes()).hexdigest(),'upper_draw_sha256':hashlib.sha256(highdraw.tobytes()).hexdigest(),'same_crossed_draws':True,'missingness_assumed_random':False,'simultaneous_coverage_guaranteed':False}

def budget_rows(paths,plan,missing_cases,budget,weight_hashes=None):
 """Read genuine schema3 traces for F11/F01/F10 only, preserve four estimands."""
 target={'F11_full','F01_no_gate','F10_no_Lcf'};missing=set((n,int(e)) for n,e in missing_cases)
 expected={(j['variant']+'_s'+str(j['training_seed']),ev,env) for j in plan['training_jobs'] if j['variant'] in target for ev in plan['evaluation_seeds'] for env in range(plan['num_envs']) if (j['variant']+'_s'+str(j['training_seed']),ev) not in missing}
 if weight_hashes is None:
  weights={j['variant']+'_s'+str(j['training_seed']):j['expected_checkpoint'] for j in plan['training_jobs'] if j['variant'] in target}
  weight_hashes={name:budget.digest(path) for name,path in weights.items()}
 require({name for name, _, _ in expected} <= weight_hashes.keys(),'A checkpoint hash is missing for the budget panel')
 rows=[];keys=set()
 for path in paths:
  for row in budget.read_trace(path):
   key=(row['checkpoint_name'],row['evaluation_seed'],row['env_index']);require(key in expected and key not in keys,'Wrong/duplicate budget case or failed partial trace')
   require(row['method'] in target and row['condition']=='canonical_D80','Original budget method/condition differs')
   require(row['checkpoint_sha256']==weight_hashes[row['checkpoint_name']],'Raw trace checkpoint SHA differs')
   require(all(plan['source_sha256'].get(k)==v for k,v in json.loads(row['source_sha256']).items()),'Raw trace source differs from frozen plan')
   require(math.isclose(row['discount'],GAMMA,abs_tol=1e-12) and math.isclose(row['preferred_clearance_m'],.35,abs_tol=1e-12) and math.isclose(row['temperature_m'],.15,abs_tol=1e-12),'Original cost parameters differ')
   require(all(math.isfinite(row[m]) and 0<=row[m]<=1 for m in budget.METRICS),'Normalized measured cost return is not in [0,1]')
   keys.add(key);rows.append(row)
 require(keys==expected,'Any genuine raw F11/F01/F10 trace episode absent')
 full=[row for row in rows if row['method']=='F11_full'];require(len(full)==len(plan['training_seeds'])*len(plan['evaluation_seeds'])*plan['num_envs'],'F11 budget full18 panel incomplete')
 summaries=[budget.summarize_panel(full,m,bootstrap=REPLICATES,seed=SEED,alpha=ALPHA,budget=BUDGET) for m in budget.METRICS]
 point=[];denominator=len(plan['training_seeds'])*len(plan['evaluation_seeds'])*plan['num_envs']
 for method in ('F01_no_gate','F10_no_Lcf'):
  actual=[row for row in rows if row['method']==method];unknown=denominator-len(actual)
  require(unknown==sum(1 for n,e in missing if n.startswith(method+'_s'))*plan['num_envs'],'Wrong original whole-case missing cost count')
  for metric in budget.METRICS:
   total=math.fsum(row[metric] for row in actual);point.append({'method':method,'metric':metric,'observed_unique_episodes':len(actual),'missing_unique_episodes':unknown,'original18_episode_denominator':denominator,'finite_original18_mean_lower':total/denominator,'finite_original18_mean_upper':(total+unknown)/denominator,'sampling_envelope_computed':False,'safety_compliance_claimed':False,'mean_safety_cost_substituted':False})
 return summaries,point

"""Reorganize existing measurements by question; never recompute uncertainty."""
from pathlib import Path
import sys,json,csv,hashlib
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'visuals/reconstruction'))
import build_remaining as b
O=R/'visuals/evidence_design/output';D=R/'visuals/evidence_design/.build'
O.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
b.OUT=O;b.BUILD=D
C=b.C
sources={}
def read(rel):
 p=R/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.suffix=='.json':return json.loads(p.read_text())
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def row(rs,**kw):return b.one(rs,**kw)
def label(v,n=2):return f'{v:+.{n}f}'.replace('-','−')
def bar_interval(f,x,low,high,cap,width):
 segments=[([[x,low],[x,high]],width)]
 segments.extend(([[x-cap,y],[x+cap,y]],width-.2) for y in (low,high))
 for color,outline in [('white',1.8),(C['ink'],0)]:
  for points,weight in segments:
   f.line(points,color,weight+outline)
def haxis(f,x0,x1,lo,hi,top,bottom,ticks,title):
 X=lambda v:x0+(v-lo)/(hi-lo)*(x1-x0)
 for t in ticks:
  if t==0:f.line([[X(t),top],[X(t),bottom]],C['edge'],.9,True)
  f.line([[X(t),bottom],[X(t),bottom+5]],width=.9)
  f.text(b.fmt(t),X(t)-25,bottom+9,50,24,19,C['muted'],'center')
 f.line([[x0,bottom],[x1,bottom]],width=.9)
 f.text(title,x0,bottom+37,x1-x0,25,20,C['muted'],'center')
 return X

def navigation():
 orig=read('evidence/canonical_paired_effects.json');released=read('evidence/primary_complete18.csv')
 f=b.Canvas('RC_CF_NavRL_Navigation_Evidence',430)
 for key,x0,x1,letter in [('success',66,516,'a'),('collision',658,1108,'b')]:
  f.panel(f'({letter}) {key.capitalize()} difference (pp)',x0-35,3,x1-x0+35,32,24)
  f.text('RC-CF-NavRL − comparator',x0,34,x1-x0,22,18,C['muted'],'center')
  X,Y=f.axes(x0,x1,60,316,-16,16,[-15,-10,-5,0,5,10,15],[],[])
  f.line([[x0,Y(0)],[x1,Y(0)]],C['muted'],1.1,zorder=.8)
  # Cohorts have separate categorical positions; bars begin at zero.
  xs=[X(.17),X(.57),X(.85)]
  f.line([[X(.35),60],[X(.35),316]],C['light'],1,True)
  m=orig['metrics'][key]
  values=[('original',100*m['paired_difference'],*[q*100 for q in m['hierarchical_bootstrap_ci95']])]
  for contrast in ['full_minus_PPO_L','full_minus_released_NavRL']:
   rr=row(released,contrast=contrast,metric=key);values.append((contrast,*[float(rr[k])*100 for k in ['difference','ci95_low','ci95_high']]))
  for xx,(comp,v,lo,hi) in zip(xs,values):
   col=C['green'] if comp=='original' else C['blue']
   assert -16<=lo<=v<=hi<=16
   f.rect(xx-30,min(Y(v),Y(0)),60,abs(Y(v)-Y(0)),col)
   bar_interval(f,xx,Y(lo),Y(hi),7,1.7)
   label_y=Y(hi)-28 if v>=0 else min(Y(lo)+4,290)
   f.text(label(v),xx-45,label_y,90,26,22,col,'center',bold=True)
   f.data.append(dict(metric=key,contrast=comp,estimate_pp=v,ci95_pp=[lo,hi]))
  for xx,t in zip(xs,['PPO-L','PPO-L','NavRL']):f.text(t,xx-55,326,110,26,22,C['muted'],'center')
  f.text('Original',x0,365,X(.34)-x0,27,22,C['green'],'center',bold=True)
  f.text('Released checkpoints',X(.38),365,x1-X(.38),27,22,C['blue'],'center',bold=True)
  f.text('5 seeds',x0,397,X(.34)-x0,25,19,C['muted'],'center')
  f.text('10 seeds',X(.38),397,x1-X(.38),25,19,C['muted'],'center')
 f.save()
 (D/'navigation_secondary_in_text.json').write_text(json.dumps({k:v for k,v in orig['metrics'].items() if k not in ['success','collision']},indent=2))


def environmental():
 sums=read('evidence/paper_aligned_summary.csv')
 f=b.Canvas('RC_CF_NavRL_Environmental_Evidence',370)
 families=[('official_navrl','NavRL (fixed)',C['muted'],'triangle'),('ppolag','PPO-Lagrangian',C['amber'],'circle'),('mean_risk','RC-CF-NavRL',C['green'],'diamond')]
 for i,(fam,lab,col,mk) in enumerate(families):
  x=170+i*300;f.line([[x,22],[x+27,22]],col,2.1,fam!='mean_risk');f.marker(x+13,22,mk,col,fam=='mean_risk',5);f.text(lab,x+36,7,215,30,21)
 for key,x0,x1,letter in [('success',85,525,'a'),('collision',670,1110,'b')]:
  f.panel(f'({letter}) {key.capitalize()} rate (%)',x0-24,51,x1-x0+24,32,24)
  X,Y=f.axes(x0,x1,104,295,0,100,[0,25,50,75,100],[1/14,5/14,9/14,13/14],[60,80,100,120])
  f.text('Moving obstacles',x0,338,x1-x0,28,22,C['muted'],'center')
  for (fam,lab,col,mk),offset in zip(families,(-9,0,9)):
   series=[]
   for n in [60,80,100,120]:
    rr=row(sums,condition=f'difficulty_d{n}',family=fam,variant='actor',metric=key)
    v,lo,hi=[100*float(rr[k]) for k in ['five_seed_mean_or_single_checkpoint_value','ci95_low','ci95_high']]
    # Keep density ticks centered while separating overlapping uncertainty marks.
    series.append((X((n-55)/70)+offset,v,lo,hi))
    f.data.append(dict(kind='absolute',condition=f'difficulty_d{n}',metric=key,family=fam,estimate_pct=v,ci95_pct=[lo,hi] if fam!='official_navrl' else None))
   f.line([[xx,Y(v)] for xx,v,_,_ in series],col,2.6 if fam=='mean_risk' else 2,fam!='mean_risk',zorder=1.5)
   for xx,v,lo,hi in series:
    if fam!='official_navrl':f.ci(xx,Y,v,lo,hi,col,mk,fam=='mean_risk')
    else:f.marker(xx,Y(v),mk,col,False,5)
 f.save()


def robustness():
 pair=read('evidence/paper_aligned_paired_effects.csv');ood=read('visuals/data/ood_paired_effects.json')
 f=b.Canvas('RC_CF_NavRL_Robustness_Evidence',410)
 # Separate outcomes; categorical spacing preserves each noise realization.
 conditions=['ood',*[f'noise_mild_n{i}' for i in range(3)],*[f'noise_moderate_n{i}' for i in range(3)]]
 positions=[.055,.30,.42,.54,.735,.845,.955]
 for key,title,col,x0,x1,letter in [
  ('success','Success gain vs PPO-L (pp)',C['blue'],65,518,'a'),
  ('collision','Collision reduction vs PPO-L (pp)',C['green'],658,1094,'b')]:
  f.panel(f'({letter}) {title}',x0-45,3,x1-x0+45,32,24)
  X,Y=f.axes(x0,x1,98,320,0,20,[0,5,10,15,20],[],[])
  for left,right,group in [(0,.14,'Unseen'),(.23,.61,'Mild noise'),(.67,1.0,'Moderate noise')]:
   f.text(group,X(left),51,X(right)-X(left),28,21,C['ink'],'center')
   f.line([[X(left),85],[X(right),85]],C['edge'],.8)
  for pos in [.18,.635]:f.line([[X(pos),98],[X(pos),320]],C['light'],.9,True)
  for cond,pos in zip(conditions,positions):
   if cond=='ood':
    m=ood['metrics'][key];v,lo,hi=100*m['paired_difference'],*[100*q for q in m['three_level_bootstrap_ci95']]
   else:
    rr=row(pair,condition=cond,comparison='mean_risk_vs_ppolag',metric=key);v,lo,hi=b.eff(rr);assert int(rr['paired_episode_count'])==1920
   gain,gl,gh=(v,lo,hi) if key=='success' else (-v,-hi,-lo)
   assert 0<=gl<=gain<=gh<=20
   xx=X(pos)
   f.rect(xx-13,Y(gain),26,Y(0)-Y(gain),col)
   bar_interval(f,xx,Y(gl),Y(gh),5,1.45)
   f.text('Pooled' if cond=='ood' else str(int(cond[-1])+1),xx-32,332,64,27,20,C['muted'],'center')
   f.data.append(dict(kind='paired',condition=cond,group='layout' if cond=='ood' else 'noise',metric=key,estimate_pp=v,ci95_pp=[lo,hi],display_gain_pp=gain,display_ci95_pp=[gl,gh]))
  f.text('Noise realization',X(.23),373,X(1.0)-X(.23),27,20,C['muted'],'center')
 f.save()


def diagnostics():
 rs=read('evidence/mechanism_seed_diagnostics.csv')
 rs=sorted([r for r in rs if r['variant']=='mean_risk'],key=lambda r:int(r['training_seed_index']))
 assert [int(r['training_seed_index']) for r in rs]==list(range(5))
 assert all(int(r['evaluation_seed_count'])==6 for r in rs)
 f=b.Canvas('RC_CF_NavRL_Learned_Mechanisms',370)
 for letter,title,unit,x0,x1,limit,ticks,keys,factor in [
  ('a','Gate response','Mean gate output',64,330,.4,[0,.1,.2,.3,.4],('gate_mean_safe','gate_mean_danger'),1),
  ('b','Command correction','Correction norm (mm/s)',449,715,12,[0,3,6,9,12],('adapter_shift_mean_safe','adapter_shift_mean_danger'),1000)]:
  f.panel(f'({letter}) {title}',x0-40,4,x1-x0+40,32,24)
  f.text(unit,x0-25,44,x1-x0+50,26,20,C['muted'],'center')
  X,Y=f.axes(x0,x1,86,281,0,limit,ticks,[],[])
  for j,(key,lab) in enumerate(zip(keys,['No intervention','Intervention'])):
   xx=X(.21 if j==0 else .79);values=[float(r[key])*factor for r in rs];mean=sum(values)/len(values)
   fill=C['blue'] if j==0 else C['green']
   f.rect(xx-33,Y(mean),66,Y(0)-Y(mean),fill)
   # Fixed seed offsets preserve individual values and their identity across labels.
   for i,(r,v) in enumerate(zip(rs,values)):
    f.marker(xx+(i-2)*11,Y(v),'circle',C['ink'],False,4.7)
    f.data.append(dict(seed=i,metric=key,source_value=float(r[key]),display_value=v,scale=factor))
   f.text(lab,xx-76,294,152,26,20,C['muted'],'center')
  f.text('Teacher label',x0,336,x1-x0,26,20,C['muted'],'center')
 x0,x1=834,1110
 f.panel('(c) Candidate ordering',x0-38,4,x1-x0+38,32,24)
 f.text('Pairwise accuracy',x0-20,44,x1-x0+40,26,20,C['muted'],'center')
 X,Y=f.axes(x0,x1,86,281,0,1,[0,.25,.5,.75,1],[],[])
 f.line([[x0,Y(.5)],[x1,Y(.5)]],C['edge'],1.1,True)
 f.text('Chance',x1-90,Y(.5)-27,90,22,18,C['muted'],'right')
 for i,r in enumerate(rs):
  v=float(r['counterfactual_pairwise_accuracy']);xx=X(.10+.20*i)
  f.marker(xx,Y(v),'diamond',C['green'],True,6.5)
  f.text(str(i),xx-15,294,30,26,20,C['muted'],'center')
  f.data.append(dict(seed=i,metric='counterfactual_pairwise_accuracy',source_value=v,display_value=v,scale=1))
 f.text('Training seed',x0,336,x1-x0,26,20,C['muted'],'center')
 f.save()


if len(sys.argv)>1:
 {'navigation':navigation,'environment':environmental,'robustness':robustness,'diagnostics':diagnostics}[sys.argv[1]]()
else:
 navigation();environmental();robustness();diagnostics()
# A targeted rebuild leaves other source bindings intact.
if (D/'source_trace.json').exists():
 sources={**json.loads((D/'source_trace.json').read_text())['sources'],**sources}
(D/'source_trace.json').write_text(json.dumps(dict(sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy='Existing frozen aggregate estimates and intervals. Mechanism panels use existing mean-risk seed-level diagnostics only; no new inference. Canonical and density snapshots remain distinct.'),indent=2))
print(json.dumps([dict(name=x['name'],marks=len(x['data'])) for x in b.ALL]))

#!/usr/bin/env python3
"""Rebuild remaining manuscript evidence from frozen exports.

No fitted curves, reconstructed uncertainty, or synthetic experimental data.
Native drawing primitives are shared by the PDF/SVG and editable PPTX exports.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'output';BUILD=HERE/'.build'
OUT.mkdir(exist_ok=True,parents=True);BUILD.mkdir(exist_ok=True)
C={'ink':'#26323A','blue':'#486A8D','green':'#237D73','amber':'#B87D42','muted':'#64717A','edge':'#A9B3B9','light':'#E9ECEF'}
plt.rcParams.update({'font.family':'Liberation Sans','font.size':15.75,
 'text.color':C['ink'],'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
 'path.simplify':False,'axes.unicode_minus':True})
SOURCES={};ALL=[]
def source(rel):
 p=ROOT/rel;SOURCES[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(rel):return json.loads(source(rel).read_text())
def rows(rel):
 with source(rel).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def one(rs,**kw):
 r=[r for r in rs if all(r.get(k)==v for k,v in kw.items())]
 assert len(r)==1,(kw,len(r));return r[0]
def eff(r):return [100*float(r[k]) for k in ['difference','ci95_low','ci95_high']]
def fmt(v):return f'{v:g}'.replace('-','−')

class Canvas:
 def __init__(self,name,h):
  self.name=name;self.w=1120;self.h=h;self.primitives=[];self.data=[]
  self.fig=plt.figure(figsize=(self.w/96,h/96),dpi=96,facecolor='white')
  self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,self.w),ylim=(h,0));self.ax.set_axis_off()
 def text(self,t,x,y,w,h=28,size=21,color=None,align='left',bold=False,italic=False,font=None):
  color=color or C['ink'];self.primitives.append(dict(type='text',text=t,x=x,y=y,w=w,h=h,size=size,color=color,align=align,bold=bold,italic=italic,font=font or 'Liberation Sans'))
  tx=x if align=='left' else x+w if align=='right' else x+w/2
  self.ax.text(tx,y+h/2,t,fontsize=.75*size,color=color,ha=align,va='center',fontweight='bold' if bold else 'normal',fontstyle='italic' if italic else 'normal',fontfamily=font or 'Liberation Sans',clip_on=False)
 def panel(self,t,x,y,w,h=28,size=22):
  letter,title=t.split(') ',1)
  self.text(letter.lstrip('('),x,y,24,h,size,bold=True)
  self.text(title,x+28,y,w-28,h,size)
 def line(self,points,color=None,width=1,dash=False,zorder=2):
  color=color or C['edge'];points=np.asarray(points).tolist()
  self.primitives.append(dict(type='line',points=points,color=color,width=width,dash=dash))
  self.ax.add_line(Line2D([p[0] for p in points],[p[1] for p in points],color=color,linewidth=width*.75,linestyle=(0,(4,4)) if dash else '-',solid_capstyle='butt',zorder=zorder))
 def marker(self,x,y,kind='circle',color=None,filled=False,r=5):
  color=color or C['green'];self.primitives.append(dict(type='marker',x=x,y=y,kind=kind,color=color,filled=filled,r=r))
  mk={'circle':'o','diamond':'D','triangle':'^','square':'s','cross':'x'}[kind]
  self.ax.plot(x,y,mk,ms=2*r*.75/(2**.5 if kind=='diamond' else 1),mfc=color if filled else 'white',mec=color,mew=1.55*.75,clip_on=False)
 def rect(self,x,y,w,h,fill,stroke=None,width=.8):
  self.primitives.append(dict(type='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke or fill,width=width))
  self.ax.add_patch(Rectangle((x,y),w,h,fc=fill,ec=stroke or fill,lw=width*.75))
 def image(self,rel,x,y,w,h):
  p=source(rel);self.primitives.append(dict(type='image',path=str(p),x=x,y=y,w=w,h=h))
  self.ax.imshow(plt.imread(p),extent=(x,x+w,y+h,y),aspect='auto')
 def axes(self,x0,x1,top,bottom,ylo,yhi,yticks,xticks,xlabels=None):
  X=lambda v:x0+v*(x1-x0)
  Y=lambda v:bottom-(v-ylo)/(yhi-ylo)*(bottom-top)
  self.line([[x0,top],[x0,bottom],[x1,bottom]])
  for v in yticks:
   yy=Y(v)
   if ylo<v<yhi:self.line([[x0,yy],[x1,yy]],C['light'],.8,zorder=.5)
   self.line([[x0-5,yy],[x0,yy]],width=.8)
   self.text(fmt(v),x0-53,yy-12,42,24,19,C['muted'],'right')
  for v,label in zip(xticks,xlabels if xlabels is not None else xticks):
   xx=X(v);self.line([[xx,bottom],[xx,bottom+5]],width=.8)
   self.text(str(label),xx-52,bottom+10,104,26,20,C['muted'],'center')
  return X,Y
 def ci(self,x,Y,v,lo,hi,color=None,kind='diamond',filled=True):
  color=color or C['green'];assert lo<=v<=hi
  self.line([[x,Y(lo)],[x,Y(hi)]],color,1.9)
  for q in [lo,hi]:self.line([[x-5,Y(q)],[x+5,Y(q)]],color,1.5)
  self.marker(x,Y(v),kind,color,filled,5.7)
 def save(self):
  for ext in ['pdf','svg','png']:
   self.fig.savefig(OUT/f'{self.name}.{ext}',facecolor='white',dpi=600)
  self.fig.savefig(BUILD/f'{self.name}-preview.png',dpi=96*900/self.w,facecolor='white');plt.close(self.fig)
  d=dict(name=self.name,width=self.w,height=self.h,font='Liberation Sans',primitives=self.primitives,data=self.data)
  (BUILD/f'{self.name}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');ALL.append(d)

def environment():
 d=js('visuals/data/d80_snapshot.json');pos=np.asarray(d['dynamic_obstacle_origin']);size=np.asarray(d['dynamic_obstacle_size'])
 assert pos.shape==size.shape==(80,3)
 f=Canvas('RC_CF_NavRL_Environment',514)
 f.text('(a) Canonical D80 environment',8,5,605,32,24)
 f.image('figures/canonical_d80_scene_cropped.png',8,57,612,437)
 f.text('(b) Moving obstacles at initialization',660,5,455,32,24)
 x0,x1,top,bottom=746,1096,93,443
 X=lambda v:x0+(v+21)/42*(x1-x0);Y=lambda v:bottom-(v+21)/42*(bottom-top)
 f.line([[x0,top],[x0,bottom],[x1,bottom],[x1,top],[x0,top]],width=.9)
 for t in [-20,0,20]:
  if t==0:
   f.line([[X(t),top],[X(t),bottom]],C['light'],.8);f.line([[x0,Y(t)],[x1,Y(t)]],C['light'],.8)
  f.text(str(t),X(t)-28,bottom+10,56,26,20,C['muted'],'center')
  f.text(str(t),x0-53,Y(t)-12,42,24,20,C['muted'],'right')
 f.text('y (m)',x0-74,56,75,26,20,C['muted'])
 f.text('x (m)',x0,474,x1-x0,26,21,C['muted'],'center')
 for p,s in zip(pos,size):
  f.rect(X(p[0]-s[0]/2),Y(p[1]+s[1]/2),s[0]/42*350,s[1]/42*350,C['blue'],C['blue'],.8)
 f.data=[dict(kind='moving_obstacle',index=i,center=p.tolist(),size=s.tolist()) for i,(p,s) in enumerate(zip(pos,size))]
 f.save()

def flight():
 records=js('visuals/data/trajectory_paths.json');audit=js('visuals/data/trajectory_capture_audit.json');assert audit['status']=='pass'
 f=Canvas('RC_CF_NavRL_Flight_Progress',584)
 specs=[('NavRL','NavRL',C['muted'],False),('NavRL_Safety_Shield','NavRL + VO',C['muted'],True),('PPO_Lagrangian','PPO-Lagrangian',C['blue'],True),('RC_CF_NavRL','RC-CF-NavRL',C['green'],False)]
 for i,(_,label,col,dash) in enumerate(specs):
  x=[75,305,565,850][i];f.line([[x,22],[x+35,22]],col,2,dash);f.text(label,x+46,8,230,28,21)
 panels=[(80,251,-1,3.5,[-1,0,1,2,3],'(a) Lateral position','Lateral displacement (m)'),(360,510,1.8,3.0,[2,2.4,2.8],'(b) Altitude','Altitude (m)')]
 for j,(top,bottom,lo,hi,ticks,head,ylab) in enumerate(panels):
  f.text(head,80,top-40,600,28,23);f.text(ylab,640,top-40,450,28,21,C['muted'],'right')
  X,Y=f.axes(80,1090,top,bottom,lo,hi,ticks,[0,.25,.5,.75,1],['0','12','24','36','48'])
  f.line([[X(1),top],[X(1),bottom]],C['edge'],.9,True)
  for key,label,col,dash in specs:
   r=next(r for r in records if r['method']==key);pos=np.asarray(r['positions']);start=pos[0];goal=np.asarray(r['target_position'])
   toward=start[1]-pos[:,1];values=pos[:,0]-start[0] if j==0 else pos[:,2]
   pts=[[X(x/48),Y(y)] for x,y in zip(toward,values)]
   f.line(pts,col,2.1 if key=='RC_CF_NavRL' else 1.5,dash)
   ok=bool(r['metadata']['success']);f.marker(*pts[-1],'square' if ok else 'cross',col,ok,5)
   if j==0:f.data.append(dict(method=key,positions=pos.tolist(),end_state_index=r['end_state_index'],terminal_event='goal' if ok else 'collision',source_npz_sha256=r['original_npz_sha256']))
 f.text('Displacement toward the goal (m)',80,545,1010,28,21,C['muted'],'center');f.save()

def forest_axes(f,x0,x1,top,bottom):
 X=lambda v:x0+(v+20)/40*(x1-x0)
 for v in [-20,-10,0,10,20]:
  f.line([[X(v),top],[X(v),bottom]],C['edge'] if v==0 else C['light'],1 if v==0 else .7,v==0)
  f.text(fmt(v),X(v)-23,bottom+9,46,26,19,C['muted'],'center')
 f.line([[x0,bottom],[x1,bottom]],width=.8)
 return X

def hci(f,X,y,v,lo,hi,col,kind='circle'):
 f.line([[X(lo),y],[X(hi),y]],col,1.9)
 for q in [lo,hi]:f.line([[X(q),y-4],[X(q),y+4]],col,1.5)
 f.marker(X(v),y,kind,col,True,5.8)

def robustness():
 pair=rows('evidence/paper_aligned_paired_effects.csv');ood=js('visuals/data/ood_paired_effects.json');assert ood['paired_episode_count']==5760
 f=Canvas('RC_CF_NavRL_Generalization',538)
 ys=[89,181,250,286,322,389,425,461]
 labels=['Three unseen layouts','Clean','Realization 1','Realization 2','Realization 3','Realization 1','Realization 2','Realization 3']
 conds=['ood','difficulty_d120',*[f'noise_mild_n{i}' for i in range(3)],*[f'noise_moderate_n{i}' for i in range(3)]]
 f.text('Layout transfer',4,43,290,26,20,C['muted'])
 f.text('Observation perturbations at D120',4,130,335,26,20,C['muted'])
 f.text('Mild',4,228,130,28,22)
 f.text('Moderate',4,367,130,28,22)
 for k,(y,lab) in enumerate(zip(ys,labels)):
  f.text(lab,4 if k<2 else 142,y-14,326 if k<2 else 180,28,21)
 for col,key in enumerate(['success','collision']):
  x0,x1=(362,684) if col==0 else (786,1108)
  f.text(f'({"ab"[col]}) {key.capitalize()}',x0,4,x1-x0,30,24)
  X=forest_axes(f,x0,x1,55,487)
  for y in [121,216,355]:f.line([[x0,y],[x1,y]],C['light'],.9)
  for cond,y in zip(conds,ys):
   if cond=='ood':
    m=ood['metrics'][key];v,lo,hi=[100*m['paired_difference'],*[100*q for q in m['three_level_bootstrap_ci95']]]
   else:
    r=one(pair,condition=cond,comparison='mean_risk_vs_ppolag',metric=key);v,lo,hi=eff(r);assert int(r['paired_episode_count'])==1920
   hci(f,X,y,v,lo,hi,C['blue'] if cond=='ood' else C['green'],'diamond' if cond=='ood' else 'circle')
   if cond in ['ood','difficulty_d120']:f.text(f'{v:+.2f}'.replace('-','−'),X(v)-45,y-34,90,26,20,C['blue'] if cond=='ood' else C['green'],'center')
   f.data.append(dict(condition=cond,metric=key,estimate_pp=v,ci95_pp=[lo,hi],paired_episodes=5760 if cond=='ood' else 1920))
  f.text('Difference (pp)',x0,513,x1-x0,25,21,C['muted'],'center')
 f.save()

def protection():
 pair=rows('evidence/deployment_shield_paired_effects.csv');absolute=rows('visuals/data/deployment_shield_summary.csv')
 f=Canvas('RC_CF_NavRL_Runtime_Protection',414)
 comps=['mean_risk_official_vo_vs_actor','mean_risk_official_vo_vs_ppolag_official_vo'];ys=[123,177,276,330]
 f.text('Add VO to a fixed RC-CF actor',4,57,334,28,22)
 f.text('Compare actors under shared VO',4,210,334,28,22)
 for y,lab in zip(ys,['D60','D120']*2):f.text(lab,24,y-14,260,28,21,C['muted'])
 for col,key in enumerate(['success','collision']):
  x0,x1=(338,680) if col==0 else (764,1106);f.text(f'({"ab"[col]}) {key.capitalize()}',x0,6,x1-x0,30,24)
  X=forest_axes(f,x0,x1,58,358);f.line([[x0,206],[x1,206]],C['light'],.8)
  for j,comp in enumerate(comps):
   for cond,y in zip(['d60','d120'],ys[j*2:j*2+2]):
    r=one(pair,condition=cond,comparison=comp,metric=key);v,lo,hi=eff(r)
    aa=one(absolute,condition=cond,family='mean_risk',deployment='official_vo',metric=key)
    bb=one(absolute,condition=cond,family='mean_risk' if j==0 else 'ppolag',deployment='actor' if j==0 else 'official_vo',metric=key)
    assert abs(v-100*(float(aa['five_seed_mean'])-float(bb['five_seed_mean'])))<1e-10
    assert int(r['paired_episode_count'])==1920
    hci(f,X,y,v,lo,hi,C['green'] if j==0 else C['blue'],'circle' if j==0 else 'diamond')
    f.text(f'{v:+.2f}'.replace('-','−'),X(v)-43,y-31,86,24,20,C['green'] if j==0 else C['blue'],'center')
    f.data.append(dict(condition=cond,comparison=comp,metric=key,estimate_pp=v,ci95_pp=[lo,hi],source_row=r))
  f.text('Difference (pp)',x0,386,x1-x0,26,21,C['muted'],'center')
 f.save()

if __name__=='__main__':
 environment();flight();robustness();protection()
 manifest={'input_sha256':SOURCES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'figures':[{k:v for k,v in f.items() if k!='primitives'} for f in ALL],
  'data_policy':'Frozen estimates and intervals only. Layout-seed points have no inferred interval. No trajectories smoothed or resampled.',
  'integration':'See sections and verification.json for final manuscript paths.'}
 (HERE/'figure_trace_remaining.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps([{'name':f['name'],'data_records':len(f['data']),'native_objects':len(f['primitives'])} for f in ALL]))

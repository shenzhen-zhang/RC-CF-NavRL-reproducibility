#!/usr/bin/env python3
"""Reproduce the local-teacher validation figure from frozen aggregate estimates.

Signed paired effects from the frozen aggregate CSV. No candidate-level
observations or distributional shapes are reconstructed.
Run: python3 build_effect_matrix.py
"""
from pathlib import Path
import csv, json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'output'
BUILD=HERE/'.build'
OUT.mkdir(exist_ok=True);BUILD.mkdir(exist_ok=True)
SOURCE=ROOT/'evidence/teacher_all_reported_metrics.csv'
QUAL=ROOT/'evidence/teacher_qualification_coverage.csv'
W,H=1120,322
C={'ink':'#26323A','blue':'#486A8D','green':'#237D73','muted':'#64717A','edge':'#A9B3B9','light':'#E9ECEF'}
plt.rcParams.update({'font.family':'Liberation Sans','font.size':15.75,'text.color':C['ink'],
 'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':True,
 'savefig.dpi':300,'path.simplify':False})
fig=plt.figure(figsize=(W/96,H/96),dpi=96,facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.set_axis_off()
primitives=[]
def text(t,x,y,w,h=28,size=21,color=None,align='left',italic=False,bold=False):
 color=color or C['ink']
 p={'type':'text','text':t,'x':x,'y':y,'w':w,'h':h,'size':size,'color':color,'align':align,'italic':italic,'bold':bold};primitives.append(p)
 tx=x if align=='left' else x+w if align=='right' else x+w/2
 ax.text(tx,y+h/2,t,fontsize=size*.75,color=color,ha=align,va='center',fontstyle='italic' if italic else 'normal',fontweight='bold' if bold else 'normal',clip_on=False)
def line(points,color=None,width=1.0,dash=False,zorder=2):
 color=color or C['edge'];primitives.append({'type':'line','points':points,'color':color,'width':width,'dash':dash})
 ax.add_line(Line2D([p[0] for p in points],[p[1] for p in points],color=color,linewidth=width*.75,
                   linestyle=(0,(4,4)) if dash else '-',solid_capstyle='butt',dash_capstyle='butt',zorder=zorder))
def marker(x,y,family,r=4.8):
 color=C['blue'] if family=='baseline' else C['green'];kind='circle' if family=='baseline' else 'diamond'
 primitives.append({'type':'marker','x':x,'y':y,'r':r,'kind':kind,'color':color,'filled':family!='baseline'})
 if kind=='circle':
  ax.plot(x,y,'o',markersize=2*r*.75,markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.5*.75,clip_on=False)
 else:
  # Matplotlib diamond size is its bounding square diagonal, matching the native diamond.
  ax.plot(x,y,'D',markersize=(2*r/2**.5)*.75,markerfacecolor=color,markeredgecolor=color,markeredgewidth=.9*.75,clip_on=False)

with SOURCE.open(encoding='utf-8-sig') as f:all_rows=list(csv.DictReader(f))
with QUAL.open(encoding='utf-8-sig') as f:qrows=list(csv.DictReader(f))
counts={fam:sum(int(r['qualified_count']) for r in qrows if r['family']==fam) for fam in ['baseline','F11']}
assert counts=={'baseline':692,'F11':2784},counts
plotted=[]
def metric(family,key,factor):
 rows=[r for r in all_rows if r['family']==family and r['metric']==key and r['condition']=='canonical_D80' and r['duration']=='nearest_50_steps']
 assert len(rows)==1,(family,key,len(rows))
 r=rows[0];v=[float(r[k])*factor for k in ['estimate','ci95_low','ci95_high']]
 assert v[1]<=v[0]<=v[2]
 return v,r

# Independent outcome panels share only the state-source legend.
rows=[('baseline',144,'Released-policy states','n = 692'),('F11',221,'Adapted-policy states','n = 2,784')]
for family,x,label,n in [('baseline',112,'Released-policy states',692),('F11',647,'Adapted-policy states',2784)]:
 color=C['blue'] if family=='baseline' else C['green']
 primitives.append(dict(type='rect',x=x-9,y=15,w=18,h=12,fill=color,stroke=color,width=0))
 ax.add_patch(Rectangle((x-9,15),18,12,facecolor=color,edgecolor='none'))
 text(label,x+18,5,345,32,23)
configs=[
 ('(a) Collision incidence','Selected − nominal (pp)','selected_minus_nominal_collision',100,(-3.5,.5),[-3,-2,-1,0],24,332),
 ('(b) Minimum clearance','Selected − nominal (cm)','selected_minus_nominal_observation_clearance_m',100,(-3.5,2.5),[-3,-2,-1,0,1,2],408,716),
 ('(c) Goal progress','Selected − nominal (cm)','selected_minus_nominal_progress_m',100,(-3.5,2.5),[-3,-2,-1,0,1,2],792,1100)]
for title,unit,key,factor,limits,ticks,x0,x1 in configs:
 letter,head=title.split(') ',1)
 text(letter.lstrip('('),x0,57,24,32,25,bold=True)
 text(head,x0+30,57,x1-x0-10,32,24)
 X=lambda v:x0+(v-limits[0])/(limits[1]-limits[0])*(x1-x0)
 line([[X(0),106],[X(0),249]],C['edge'],1.2,True,zorder=.5)
 line([[x0,258],[x1,258]],C['edge'],1)
 for t in ticks:
  xx=X(t);line([[xx,258],[xx,263]],C['edge'],1)
  text(str(t).replace('-', '\N{MINUS SIGN}'),xx-20,267,40,25,21,C['muted'],'center')
 text(unit,x0,292,x1-x0,28,20,C['muted'],'center')
 for family,y,_,_ in rows:
  (v,lo,hi),raw=metric(family,key,factor)
  color=C['blue'] if family=='baseline' else C['green']
  assert limits[0]<=lo<=v<=hi<=limits[1]
  # Zero-anchored bars expose the sign and size; dark whiskers retain uncertainty.
  left=min(X(0),X(v));width=abs(X(v)-X(0))
  primitives.append(dict(type='rect',x=left,y=y-11,w=width,h=22,fill=color,stroke=color,width=0))
  ax.add_patch(Rectangle((left,y-11),width,22,facecolor=color,edgecolor='none'))
  for whisker,outline in [('white',1.8),(C['ink'],0)]:
   line([[X(lo),y],[X(hi),y]],whisker,1.6+outline)
   for bound in [lo,hi]:line([[X(bound),y-6],[X(bound),y+6]],whisker,1.4+outline)
  label=f'{v:+.2f}'.replace('-', '\N{MINUS SIGN}')
  text(label,X(v)-45,y-34,90,28,24,color,'center',bold=family!='baseline')
  plotted.append({'family':family,'metric':key,'source_row':raw,'scale':factor,'display_unit':'pp' if 'collision' in key else 'cm',
    'estimate':v,'ci95':[lo,hi],'label':label,'null_value':0})
stem='RC_CF_NavRL_Local_Replay_Estimates'
for ext in ['pdf','svg','png']:
 fig.savefig(OUT/f'{stem}.{ext}',facecolor='white',dpi=600)
fig.savefig(BUILD/'estimates-publication-size.png',dpi=96*784/W,facecolor='white')
plt.close(fig)
manifest={'width':W,'height':H,'font':'Liberation Sans','counts':counts,'duration':'nearest_50_steps','control_steps':50,'replay_seconds':50/62,
 'source_csv':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'statistical_unit':'qualified replay anchor; dependent candidate rows are not independent samples',
 'interval_method':'existing 20000-draw descriptive crossed cluster-bootstrap over source checkpoints and shared evaluation seeds',
 'panels':plotted,'primitives':primitives}
(BUILD/'estimates-layout.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'estimates':len(plotted),'primitives':len(primitives),'size':[W,H]}))

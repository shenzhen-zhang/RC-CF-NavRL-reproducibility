"""Render the audited four-method flight case as shared vector primitives."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import proj3d

HERE=Path(__file__).resolve().parent
R=HERE.parents[1]
sys.path.insert(0,str(R/'visuals/reconstruction'))
import build_remaining as b
b.OUT=HERE/'output';b.BUILD=HERE/'.build'
b.OUT.mkdir(exist_ok=True);b.BUILD.mkdir(exist_ok=True)
D=json.loads((R/'visuals/data/flight_case.json').read_text())
assert D['evaluation_seed']==1183109 and D['env_index']==0
assert D['scene_sha256']=='61d6825f64954ad72f360fb95f80e829f42496550f11423f5976907f6526e9d3'
f=b.Canvas('RC_CF_NavRL_Flight_Case',702)

# Orthographic projection, identical for all four panels. The vertical axis is
# expanded by 14/6 for legibility; every panel uses that same declared scale.
camera=b.plt.figure().add_subplot(projection='3d')
camera.set(xlim=(-22.5,22.5),ylim=(-25.5,25.5),zlim=(0,6))
camera.set_box_aspect((45,51,14));camera.view_init(elev=24,azim=-58)
camera.set_proj_type('ortho');matrix=camera.get_proj()
box=np.array([[x,y,z] for x in [-22.5,22.5] for y in [-25.5,25.5] for z in [0,6]])
pbox=np.array(proj3d.proj_transform(*box.T,matrix)).T
lo=pbox[:,:2].min(axis=0);hi=pbox[:,:2].max(axis=0)
b.plt.close(camera.figure)

def polygon(points,fill,edge='#BAC2C8',width=.5,zorder=1):
    pts=np.asarray(points).tolist()
    f.primitives.append(dict(type='polygon',points=pts,fill=fill,stroke=edge,width=width))
    f.ax.add_patch(Polygon(pts,closed=True,facecolor=fill,edgecolor=edge,linewidth=.75*width,zorder=zorder))

def cross(point,r=5,col='#AF574C'):
    x,y=point
    for stroke,width in [('white',4),(col,2)]:
        for sign in [-1,1]:f.line([[x-r,y-sign*r],[x+r,y+sign*r]],stroke,width,zorder=2.4)

def star(point,r=7):
    x,y=point;pts=[]
    for k in range(10):
        a=-np.pi/2+k*np.pi/5;rad=r if k%2==0 else .43*r
        pts.append([x+rad*np.cos(a),y+rad*np.sin(a)])
    polygon(pts,'white','white',3,zorder=2.3)
    polygon(pts,'#DBB15B','#967331',.8,zorder=2.3)

specs=[('NavRL','NavRL',b.C['muted'],False),
       ('NavRL_Safety_Shield','NavRL + VO',b.C['muted'],True),
       ('PPO_Lagrangian','PPO-Lagrangian',b.C['amber'],False),
       ('RC_CF_NavRL','RC-CF-NavRL',b.C['green'],False)]
for i,(key,name,color,dash) in enumerate(specs):
    record=next(a for a in D['methods'] if a['method']==key)
    left=8+(i%2)*568;top=6+(i//2)*326
    f.panel(f'({chr(97+i)}) {name}',left,top,350,29,22)
    f.text('Collision' if record['outcome']=='Collision' else 'Goal reached',left+345,top+1,182,27,19,b.C['muted'],'right')
    # Preserve the projected aspect, then translate into each panel.
    scale=min(470/(hi[0]-lo[0]),245/(hi[1]-lo[1]))
    x0=left+32+(470-scale*(hi[0]-lo[0]))/2
    y0=top+47+(245-scale*(hi[1]-lo[1]))/2
    def project(points):
        points=np.asarray(points);pr=np.array(proj3d.proj_transform(*points.T,matrix)).T
        return np.column_stack((x0+(pr[:,0]-lo[0])*scale,y0+(hi[1]-pr[:,1])*scale))
    # A light common floor and three coordinate axes replace a full cube grid.
    floor=np.array([[-22.5,-25.5,0],[22.5,-25.5,0],[22.5,25.5,0],[-22.5,25.5,0]])
    polygon(project(floor),'#FAFBFC','#D6DDE2',.65)
    for y in [-20,0,20]:f.line(project([[-22.5,y,0],[22.5,y,0]]),'#E5EAEE',.6)
    for x in [-20,0,20]:f.line(project([[x,-25.5,0],[x,25.5,0]]),'#E5EAEE',.6)
    # The three visible faces of each measured cuboid. All 80 centers and sizes
    # are retained. Pale fills keep trajectories readable without false occlusion.
    faces=[]
    for center,size in zip(record['terminal_obstacle_positions'],record['obstacle_sizes']):
        x,y,z=np.asarray(center)-np.asarray(size)/2
        dx,dy,dz=size;zmax=min(6,z+dz);z=max(0,z)
        if zmax<=z:continue
        verts=np.array([[x,y,z],[x+dx,y,z],[x+dx,y+dy,z],[x,y+dy,z],
                        [x,y,zmax],[x+dx,y,zmax],[x+dx,y+dy,zmax],[x,y+dy,zmax]])
        for inds,fill in [([0,1,5,4],'#DDE2E6'),([1,2,6,5],'#CDD5DB'),([4,5,6,7],'#EBEEF1')]:
            face=verts[inds];depth=np.mean(proj3d.proj_transform(*face.T,matrix)[2])
            faces.append((depth,project(face),fill))
    for _,pts,fill in sorted(faces,key=lambda a:a[0],reverse=True):polygon(pts,fill,'#BBC4CC',.4)
    # Use every stored trajectory point through the event; no decimation.
    path=project(record['positions']);f.line(path,color,2.7,dash)
    start=path[0];goal=project([record['target_position']])[0]
    f.marker(*start,'circle',b.C['ink'],False,5);star(goal)
    if record['outcome']=='Collision':cross(path[-1],5)
    else:f.marker(*path[-1],'diamond',color,True,5)
    # World coordinate axes, with compact direct labels in metres.
    for pts,title,offset in [
            ([[-22.5,-25.5,0],[22.5,-25.5,0]],'x (m)',(0,24)),
            ([[22.5,-25.5,0],[22.5,25.5,0]],'y (m)',(29,12)),
            ([[22.5,25.5,0],[22.5,25.5,6]],'z (m)',(48,-15))]:
        pp=project(pts);f.line(pp,b.C['edge'],.9)
        mid=pp.mean(axis=0);f.text(title,mid[0]-28+offset[0],mid[1]+offset[1],62,22,17,b.C['muted'],'center')
    for x in [-20,0,20]:
        px,py=project([[x,-25.5,0]])[0];f.text(str(x).replace('-','−'),px-20,py+3,40,20,16,b.C['muted'],'center')
    for y in [-20,0,20]:
        px,py=project([[22.5,y,0]])[0];f.text(str(y).replace('-','−'),px+3,py-9,33,20,16,b.C['muted'])
    for z in [0,3,6]:
        px,py=project([[22.5,25.5,z]])[0];f.text(str(z),px+4,py-9,25,20,16,b.C['muted'])
    f.data.append(dict(method=key,trajectory_point_count=len(path),terminal_state_index=record['terminal_state_index'],event_step=record['event_step'],outcome=record['outcome'],source_npz_sha256=record['source_npz_sha256'],obstacles_drawn=80))

for x,title,kind in [(32,'Start','start'),(242,'Goal','goal'),(452,'Collision','collision'),(770,'Moving obstacles','obstacles')]:
    y=680
    if kind=='start':f.marker(x,y,'circle',b.C['ink'],False,5)
    elif kind=='goal':star([x,y])
    elif kind=='collision':cross([x,y])
    elif kind=='success':f.marker(x,y,'diamond',b.C['green'],True,5)
    else:f.rect(x-6,y-7,12,14,'#DDE2E6','#BBC4CC',.6)
    f.text(title,x+16,y-14,235,28,19,b.C['muted'])
f.save()
(HERE/'figure_trace.json').write_text(json.dumps(dict(
    source='visuals/data/flight_case.json',source_sha256=hashlib.sha256((R/'visuals/data/flight_case.json').read_bytes()).hexdigest(),
    script='visuals/flight_case/build_flight_case.py',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    render='Orthographic projection, common view and coordinate limits. z displayed at 14/6 of horizontal scale. All trajectory points retained, no smoothing; exact dynamic-obstacle dimensions, clipped only at the plotted z range.',
    limitations=['One existing fixed density-D80 example, distinct from the canonical primary test.','Static obstacles are omitted. Dynamic obstacles show different method-specific terminal times.','Projected paths are drawn over pale obstacle geometry for visibility; this drawing does not establish collision causes.'],
    records=f.data),indent=2)+'\n')
print('Rendered four audited flight paths and terminal obstacle states.')

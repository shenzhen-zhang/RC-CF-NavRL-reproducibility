#!/usr/bin/env python3
"""Render the existing Isaac Sim scene and its frozen obstacle positions."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
import re
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'visuals/reconstruction'))
import build_remaining as base
from build_remaining import Canvas, C

base.OUT = HERE / 'output'
base.BUILD = HERE / '.build'
base.OUT.mkdir(exist_ok=True)
base.BUILD.mkdir(exist_ok=True)
snapshot = ROOT / 'visuals/data/d80_snapshot.json'
data = json.loads(snapshot.read_text())
positions = np.asarray(data['dynamic_obstacle_origin'])
assert positions.shape == (80, 3)
assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == '61d6825f64954ad72f360fb95f80e829f42496550f11423f5976907f6526e9d3'

sizes = np.asarray(data['dynamic_obstacle_size'])
assert sizes.shape == positions.shape
# Verify the old figure's 80 centers and rectangle dimensions against the snapshot.
reference_svg = HERE / 'reference_d80_distribution.svg'
root = ET.parse(reference_svg).getroot()
cmap = colormaps['viridis']
marks = [p for p in root.findall('.//{http://www.w3.org/2000/svg}path')
         if p.attrib.get('clip-path') == 'url(#pff6c0ae43f)' and 'fill: #' in p.attrib.get('style', '')]
assert len(marks) == 80
for mark, point, size in zip(marks, positions, sizes):
    points = np.asarray(re.findall(r'[-+]?\d*\.?\d+', mark.attrib['d']), dtype=float).reshape(-1, 2)
    center = (points.min(0) + points.max(0)) / 2
    expected = np.array([42.392411 + (point[0] + 20) * 5.134017725,
                         218.024355 - (point[1] + 20) * 5.134017725])
    assert re.search(r'fill: (#[0-9a-f]{6})', mark.attrib['style']).group(1) == to_hex(cmap(point[2] / 4.5))
    assert np.max(np.abs(center - expected)) < 2e-6
    assert np.max(np.abs((points.max(0) - points.min(0)) - size[:2] * 5.134017725)) < 2e-6

f = Canvas('RC_CF_NavRL_Environment_Setup', 512)
f.panel('(a) Isaac Sim environment', 8, 2, 560, 32, 24)
f.image('figures/canonical_d80_scene_cropped.png', 8, 57, 560, 560 * 2428 / 3400)
f.panel('(b) Moving-obstacle initialization', 622, 2, 490, 32, 22)
x0, x1, top, bottom = 650, 1050, 57, 457
X = lambda x: x0 + (x + 21) / 42 * (x1 - x0)
Y = lambda y: bottom - (y + 21) / 42 * (bottom - top)
f.line([[x0, top], [x0, bottom], [x1, bottom], [x1, top], [x0, top]], C['edge'], .9)
for t in [-20, -10, 0, 10, 20]:
    if t in [-10, 0, 10]:
        f.line([[X(t), top], [X(t), bottom]], C['light'], .7)
        f.line([[x0, Y(t)], [x1, Y(t)]], C['light'], .7)
    f.line([[X(t), bottom], [X(t), bottom + 5]], C['edge'], .8)
    f.line([[x0 - 5, Y(t)], [x0, Y(t)]], C['edge'], .8)
    f.text(str(t), X(t) - 23, bottom + 7, 46, 26, 19, C['muted'], 'center')
    f.text(str(t), x0 - 45, Y(t) - 12, 35, 24, 19, C['muted'], 'right')
f.text('y (m)', x0 - 65, 32, 70, 24, 20, C['muted'])
f.text('x (m)', x0, 486, x1 - x0, 24, 20, C['muted'], 'center')
cmap = colormaps['viridis']
# Rectangle widths and heights use the same metric scale as the position axes.
for point, size in zip(positions, sizes):
    x, y, z = point
    color = to_hex(cmap(z / 4.5))
    f.rect(X(x - size[0] / 2), Y(y + size[1] / 2),
           size[0] / 42 * 400, size[1] / 42 * 400, color, '#4A4A4A', .5)
# Discrete native rectangles keep the continuous height legend editable in WPS.
cbx, cbw, steps = 1070, 12, 128
for i in range(steps):
    f.rect(cbx, top + i * 400 / steps, cbw, 400 / steps + .02,
           to_hex(cmap(1 - (i + .5) / steps)), width=0)
f.line([[cbx, top], [cbx + cbw, top], [cbx + cbw, bottom], [cbx, bottom], [cbx, top]], C['edge'], .7)
for z in [0, 1, 2, 3, 4]:
    y = bottom - z / 4.5 * 400
    f.line([[cbx + cbw, y], [cbx + cbw + 4, y]], C['edge'], .7)
    f.text(str(z), cbx + cbw + 7, y - 11, 25, 22, 18, C['muted'])
f.text('z₀ (m)', 1042, 32, 70, 24, 18, C['muted'], 'center')
f.data = [dict(kind='initial_moving_obstacle_footprint', index=i,
               position=point.tolist(), dimensions=size.tolist(),
               initial_z_m=float(point[2]), fill=to_hex(cmap(point[2] / 4.5)))
          for i, (point, size) in enumerate(zip(positions, sizes))]
f.save()
audit_path = ROOT / 'raw/flight_case/isaac_replay_audit.json'
audit = json.loads(audit_path.read_text())
assert audit['canonical_scene']['sha256'] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
trace = dict(
    artifact_id='fig:environment-setup',
    source_data=dict(dataset_id='frozen_density_D80_seed7', files={
        'visuals/data/d80_snapshot.json': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'figures/canonical_d80_scene_cropped.png': hashlib.sha256((ROOT / 'figures/canonical_d80_scene_cropped.png').read_bytes()).hexdigest(),
        str(reference_svg.relative_to(ROOT)): hashlib.sha256(reference_svg.read_bytes()).hexdigest(),
    }, rendering_audit=str(audit_path.relative_to(ROOT)), rendering_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest()),
    transformation=dict(script='visuals/environment_setup/build_environment_setup.py', hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), operations=[
        'Preserve the original manuscript obstacle-environment image without content editing.',
        'Verify all 80 centers, rectangle dimensions and height colors against the original vector distribution figure; draw exact metric footprints and color by initial z using viridis over 0–4.5 m.',
    ]),
    caption_claim='The original obstacle environment is shown alongside the spatial arrangement, footprints and initial heights of the frozen density-test D80 obstacles.',
    supported_manuscript_claims=[dict(claim='The density-test environment includes static and moving obstacles.', locator='sections/evaluation.tex')],
    limitations=[
        'Density-test D80 snapshot, distinct from the canonical primary evaluation snapshot.',
        'The scene image provides the environment context; the right panel specifies the frozen obstacle initialization.',
        'The right panel shows planar footprints and initial center heights; it does not encode vertical obstacle extents or a probability density estimate.',
    ],
)
(HERE / 'figure_trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n')
print('Environment figure: original scene, 80 verified metric footprints and initial-height colors.')

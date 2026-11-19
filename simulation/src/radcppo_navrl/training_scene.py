"""Sealed, first-construction training scenes and read-only geometry audits.

Manifest validation uses only the standard library. Isaac, NumPy and Torch are
loaded lazily for capture/binding. Geometry signatures cover USD mesh geometry,
world transforms and collision attributes under /World/ground only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

TRAINING_SCENE_PROTOCOL = 'navrl_native_training_scenes_v1'
SELECTION_RULE = 'first_native_construction_per_seed_no_policy_no_rollout'
CANONICAL_PROTOCOL = 'canonical_navrl_dynamic_obstacles_v2'
GEOMETRY_PROTOCOL = 'usd_ground_mesh_geometry_v1'
SEED_SEMANTICS = 'capture_config_seed_only_terrain_generator_resets_numpy_and_torch_to_0'


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _sealed_payload(path, expected_sha256):
    path = Path(path).resolve()
    if not _valid_sha256(expected_sha256) or _file_sha256(path) != expected_sha256:
        raise ValueError('Training scene file SHA256 differs: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def _float32_matrix(value, count, name, positive=False):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(name + ' count differs')
    flat = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(name + ' must have shape [N,3]')
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(name + ' contains nonfinite/non-numeric values')
            if positive and item <= 0:
                raise ValueError(name + ' contains nonpositive sizes')
            flat.append(float(item))
    return hashlib.sha256(struct.pack('<' + 'f' * len(flat), *flat)).hexdigest()


def validate_static_geometry_signature(value):
    if not isinstance(value, dict) or value.get('protocol') != GEOMETRY_PROTOCOL or value.get('root_path') != '/World/ground':
        raise ValueError('Static geometry signature protocol/root differs')
    meshes = value.get('meshes')
    if not isinstance(meshes, list) or not meshes:
        raise ValueError('Static geometry has no USD meshes')
    paths = []
    for mesh in meshes:
        path = mesh.get('prim_path', '')
        if not path.startswith('/World/ground/'):
            raise ValueError('Static mesh outside declared geometry scope')
        paths.append(path)
        for field, dtype, tail in [('vertices', 'float32_le', [3]), ('face_vertex_counts', 'int32_le', []),
                                   ('face_vertex_indices', 'int32_le', []), ('world_transform', 'float64_le', [4])]:
            record = mesh.get(field)
            if not isinstance(record, dict) or record.get('dtype') != dtype or not _valid_sha256(record.get('sha256')):
                raise ValueError('Static mesh field hash/dtype differs: ' + field)
            shape = record.get('shape')
            if not isinstance(shape, list) or not shape or shape[1:] != tail or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError('Static mesh field shape differs: ' + field)
            if field == 'world_transform' and shape != [4, 4]:
                raise ValueError('Static mesh world transform must be 4x4')
        if type(mesh.get('collision_enabled')) is not bool:
            raise ValueError('Static mesh collision-enabled attribute is missing')
        if not isinstance(mesh.get('collision_approximation'), str):
            raise ValueError('Static mesh collision approximation is missing')
        if not isinstance(mesh.get('orientation'), str) or not isinstance(mesh.get('subdivision_scheme'), str):
            raise ValueError('Static mesh topology attributes are missing')
    if paths != sorted(set(paths)) or value.get('mesh_count') != len(meshes):
        raise ValueError('Static mesh ordering/count differs')
    unhashed = {key: item for key, item in value.items() if key != 'sha256'}
    if value.get('sha256') != _json_sha256(unhashed):
        raise ValueError('Static geometry aggregate SHA256 differs')
    return value


def load_training_scene(path, expected_sha, expected_seed, expected_count=80):
    """Validate one existing canonical-format snapshot plus training provenance."""
    payload = _sealed_payload(path, expected_sha)
    if payload.get('protocol') != CANONICAL_PROTOCOL or type(payload.get('creation_seed')) is not int or payload['creation_seed'] != int(expected_seed):
        raise ValueError('Training snapshot protocol/config seed differs')
    if payload.get('creation_seed_semantics') != SEED_SEMANTICS or payload.get('upstream_terrain_seed') != 0:
        raise ValueError('Training capture seed semantics are not explicit')
    if type(payload.get('dynamic_obstacle_count')) is not int or payload['dynamic_obstacle_count'] != int(expected_count):
        raise ValueError('Training snapshot dynamic obstacle count differs')
    for field, positive in [('origin', False), ('size', True)]:
        actual = _float32_matrix(payload.get('dynamic_obstacle_' + field), expected_count, field, positive)
        if payload.get('dynamic_obstacle_' + field + '_sha256') != actual:
            raise ValueError('Training snapshot dynamic ' + field + ' SHA256 differs')
    extent = payload.get('map_range')
    if not isinstance(extent, list) or len(extent) != 3 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in extent):
        raise ValueError('Training map range is invalid')
    capture = payload.get('training_scene_capture')
    if (not isinstance(capture, dict) or capture.get('selection_rule') != SELECTION_RULE
            or capture.get('capture_config_seed') != int(expected_seed)
            or capture.get('num_envs') != 64 or capture.get('configured_static_obstacles') != 350
            or capture.get('policy_created') is not False or capture.get('measured_env_transitions') != 0):
        raise ValueError('Training snapshot is not the prescribed first native capture')
    validate_static_geometry_signature(payload.get('static_geometry'))
    return payload


def validate_training_scene_manifest(path, expectedsha, required_seeds=None):
    """Pure-stdlib seal/coverage/payload audit; returns the original manifest."""
    manifest = _sealed_payload(path, expectedsha)
    if (manifest.get('protocol') != TRAINING_SCENE_PROTOCOL or manifest.get('schema_version') != 1
            or manifest.get('selection_rule') != SELECTION_RULE
            or manifest.get('num_envs_at_capture') != 64
            or manifest.get('static_obstacle_count') != 350 or manifest.get('dynamic_obstacle_count') != 80):
        raise ValueError('Training scene manifest protocol/prescribed dimensions differ')
    entries = manifest.get('scenes')
    if not isinstance(entries, dict) or not entries:
        raise ValueError('Training scene manifest has no scenes')
    keys = []
    for key, entry in entries.items():
        if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
            raise ValueError('Training scene seed key is not canonical')
        seed = int(key)
        keys.append(seed)
        if not isinstance(entry, dict) or type(entry.get('seed')) is not int or entry['seed'] != seed:
            raise ValueError('Training scene entry seed differs')
        snapshot = Path(entry.get('snapshot', ''))
        if not snapshot.is_absolute() or not snapshot.is_file():
            raise ValueError('Training snapshot must be an existing absolute path')
        load_training_scene(snapshot, entry.get('sha256'), seed)
    if required_seeds is not None and not set(int(s) for s in required_seeds) <= set(keys):
        raise ValueError('Training manifest is missing required seeds')
    return manifest


def static_mesh_geometry_signature():
    """Read actual composed USD points/topology/world transforms; no stepping."""
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath('/World/ground')
    if not root or not root.IsValid():
        raise RuntimeError('Native USD terrain root /World/ground is absent')
    meshes = []
    def array_record(value, dtype):
        a = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        if a.dtype.kind == 'f' and not np.isfinite(a).all():
            raise RuntimeError('Nonfinite native terrain geometry')
        return {'dtype': {'<f4': 'float32_le', '<f8': 'float64_le', '<i4': 'int32_le'}[np.dtype(dtype).str],
                'shape': list(a.shape), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        enabled = prim.GetAttribute('physics:collisionEnabled').Get()
        approximation = prim.GetAttribute('physics:approximation').Get()
        meshes.append({'prim_path': str(prim.GetPath()),
            'vertices': array_record(mesh.GetPointsAttr().Get(), '<f4'),
            'face_vertex_counts': array_record(mesh.GetFaceVertexCountsAttr().Get(), '<i4'),
            'face_vertex_indices': array_record(mesh.GetFaceVertexIndicesAttr().Get(), '<i4'),
            'world_transform': array_record(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), '<f8'),
            'collision_enabled': bool(enabled), 'collision_approximation': '' if approximation is None else str(approximation),
            'orientation': str(mesh.GetOrientationAttr().Get()),
            'subdivision_scheme': str(mesh.GetSubdivisionSchemeAttr().Get())})
    signature = {'protocol': GEOMETRY_PROTOCOL, 'root_path': '/World/ground',
                 'mesh_count': len(meshes), 'meshes': sorted(meshes, key=lambda m: m['prim_path'])}
    signature['sha256'] = _json_sha256(signature)
    return validate_static_geometry_signature(signature)


def training_scene_signature(env):
    base = getattr(env, 'base_env', env)
    return {'configured_static_obstacles': int(base.cfg.env.num_obstacles),
            'actual_dynamic_obstacles': int(base.dyn_obs_state.shape[0]),
            'dynamic_origin_sha256': hashlib.sha256(base.dyn_obs_origin.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            'dynamic_size_sha256': hashlib.sha256(base.dyn_obs_size.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            'static_geometry_sha256': static_mesh_geometry_signature()['sha256'],
            'map_range': [float(x) for x in base.map_range]}


def assert_training_scene_signature(env, expected):
    current = training_scene_signature(env)
    if current != expected:
        raise RuntimeError('Training scene geometry/origin/size changed during execution')
    return current


def bind_training_scene(env, manifest_path, expected_manifest_sha256, seed):
    """Bind a sealed dynamic scene before policy/collector creation; no sim tick."""
    manifest = validate_training_scene_manifest(manifest_path, expected_manifest_sha256, [seed])
    entry = manifest['scenes'][str(int(seed))]
    payload = load_training_scene(entry['snapshot'], entry['sha256'], seed)
    base = getattr(env, 'base_env', env)
    if int(base.cfg.env.num_obstacles) != 350 or int(base.num_envs) != 64:
        raise RuntimeError('Training scene binding requires the prescribed N64/static350 environment')
    if [float(x) for x in base.map_range] != [float(x) for x in payload['map_range']]:
        raise RuntimeError('Training map range differs from sealed capture')
    geometry = static_mesh_geometry_signature()
    if geometry != payload['static_geometry']:
        raise RuntimeError('Constructed static terrain differs from sealed capture')
    from evaluate_v19_canonical import bind_navrl_canonical_dynamic_scene, reset_navrl_dynamic_obstacles
    before_step = int(base.sim.current_time_step_index)
    binding = bind_navrl_canonical_dynamic_scene(env, entry['snapshot'], entry['sha256'])
    dynamic_reset = reset_navrl_dynamic_obstacles(env)
    if int(base.sim.current_time_step_index) != before_step or int(base.dyn_obs_step_count) != 0:
        raise RuntimeError('Training scene binding advanced the physical/task clock')
    signature = training_scene_signature(env)
    if (signature['dynamic_origin_sha256'] != payload['dynamic_obstacle_origin_sha256']
            or signature['dynamic_size_sha256'] != payload['dynamic_obstacle_size_sha256']):
        raise RuntimeError('Training scene readback differs from sealed snapshot')
    record = {'protocol': TRAINING_SCENE_PROTOCOL, 'manifest': str(Path(manifest_path).resolve()),
              'manifest_sha256': expected_manifest_sha256, 'snapshot': entry['snapshot'],
              'snapshot_sha256': entry['sha256'], 'seed': int(seed),
              'creation_seed_semantics': SEED_SEMANTICS,
              'initial_signature': signature, 'canonical_binding': binding,
              'dynamic_reset': dynamic_reset, **signature}
    base._navrl_training_scene_binding = record
    return record

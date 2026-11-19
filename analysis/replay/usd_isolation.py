"""Read-only USD evidence for the fixed NavRL clone collision configuration.

The legacy PhysX scene inverts collision-group filtering: listed groups are
allowed to contact. This module never changes a USD attribute or physics state.
"""
from __future__ import annotations
import hashlib
import json

SCHEMA = 'qualified_anchor_usd_isolation_v1'


def snapshot_sha256(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _under(path, root):
    return path == root or path.startswith(root + '/')


def capture_isolation(stage, physics_scene_path, env_paths, dynamic_paths=()):
    """Capture resolved attributes/collections using actual USD schema identity."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    env_paths = [str(p) for p in env_paths]
    dynamic_paths = [str(p) for p in dynamic_paths]
    scene = stage.GetPrimAtPath(str(physics_scene_path))
    if not scene or not scene.HasAPI(PhysxSchema.PhysxSceneAPI):
        raise ValueError('Actual PhysX scene API is missing')
    for path in env_paths + dynamic_paths:
        if not stage.GetPrimAtPath(path):
            raise ValueError('Actual environment/dynamic root is missing: ' + path)
    colliders = []
    bodies = []
    # Default Traverse omits instance proxy children. The installed USD exposes
    # this predicate; no incomplete traversal fallback is permitted.
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    for prim in prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            bodies.append({'path': str(prim.GetPath()),
                           'enabled': api.GetRigidBodyEnabledAttr().Get(),
                           'kinematic': api.GetKinematicEnabledAttr().Get()})
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        colliders.append({'path': str(prim.GetPath()),
                          'collision_enabled': UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                          'rigid_body': str(owner.GetPath()) if owner else None})
    groups = []
    for prim in prims:
        # GetTypeName() == 'CollisionGroup' is WRONG for the installed USD.
        if not prim.IsA(UsdPhysics.CollisionGroup):
            continue
        collection = Usd.CollectionAPI(prim, 'colliders')
        query = collection.ComputeMembershipQuery()
        include_root = collection.GetIncludeRootAttr()
        inverse = prim.GetAttribute('physics:invertFilteredGroups')
        groups.append({'path': str(prim.GetPath()), 'type_name': str(prim.GetTypeName()),
                       'includes': [str(p) for p in collection.GetIncludesRel().GetTargets()],
                       'excludes': [str(p) for p in collection.GetExcludesRel().GetTargets()],
                       'expansion_rule': collection.GetExpansionRuleAttr().Get(),
                       'include_root': include_root.Get(),
                       'include_root_authored': include_root.HasAuthoredValueOpinion(),
                       'filtered_groups': [str(p) for p in prim.GetRelationship('physics:filteredGroups').GetTargets()],
                       'per_group_invert': inverse.Get() if inverse else None,
                       'per_group_invert_authored': bool(inverse and inverse.HasAuthoredValueOpinion()),
                       'resolved_collider_members': sorted(c['path'] for c in colliders
                           if query.IsPathIncluded(stage.GetPrimAtPath(c['path']).GetPath()))})
    snapshot = {'schema': SCHEMA, 'extraction': 'UsdPhysics.CollisionGroup.IsA_resolved_collection_query',
                'instance_proxy_traversal': True,
                'physics_scene_path': str(physics_scene_path),
                'scene_invert_collision_group_filter': PhysxSchema.PhysxSceneAPI(scene).GetInvertCollisionGroupFilterAttr().Get(),
                'env_paths': env_paths, 'dynamic_paths': dynamic_paths,
                'expected_cloner_global_paths': [],
                'collision_groups': sorted(groups, key=lambda g: g['path']),
                'colliders': sorted(colliders, key=lambda c: c['path']),
                'rigid_bodies': sorted(bodies, key=lambda b: b['path'])}
    # Fail before recording a purported proof if extraction is incomplete.
    validate_isolation(snapshot, expected_num_envs=len(env_paths), expected_dynamic_count=len(dynamic_paths))
    return snapshot


def validate_isolation(snapshot, expected_num_envs=64, expected_dynamic_count=80):
    """Recompute isolation proof. Unknown/shared mobile contacts fail closed."""
    if not isinstance(snapshot, dict) or snapshot.get('schema') != SCHEMA or snapshot.get('extraction') != 'UsdPhysics.CollisionGroup.IsA_resolved_collection_query':
        raise ValueError('Unknown USD isolation schema/extraction')
    if snapshot.get('scene_invert_collision_group_filter') is not True:
        raise ValueError('Legacy inverted collision filtering is not enabled')
    if snapshot.get('instance_proxy_traversal') is not True:
        raise ValueError('Collider inventory must include instance proxies')
    if snapshot.get('expected_cloner_global_paths') != []:
        raise ValueError('Frozen NavRL cloner global_paths must be empty')
    if not isinstance(snapshot.get('physics_scene_path'), str) or not snapshot['physics_scene_path'].startswith('/'):
        raise ValueError('Actual physics scene path is missing')
    envs = snapshot.get('env_paths'); dynamic = snapshot.get('dynamic_paths')
    if not isinstance(envs, list) or len(envs) != expected_num_envs or len(set(envs)) != len(envs) or envs != ['/World/envs/env_' + str(i) for i in range(expected_num_envs)]:
        raise ValueError('Actual environment roots/order are incomplete')
    if not isinstance(dynamic, list) or dynamic != ['/World/Origin' + str(i) for i in range(expected_dynamic_count)]:
        raise ValueError('Actual dynamic obstacle roots/order are incomplete')
    groups = snapshot.get('collision_groups'); colliders = snapshot.get('colliders'); bodies = snapshot.get('rigid_bodies')
    if not all(isinstance(x, list) for x in [groups, colliders, bodies]) or len(groups) != expected_num_envs:
        raise ValueError('Missing/unknown collision groups or collider/body inventory')
    def index(rows, label):
        if any(not isinstance(r, dict) or not isinstance(r.get('path'), str) or not r['path'].startswith('/') for r in rows):
            raise ValueError('Malformed ' + label + ' inventory')
        result = {r['path']: r for r in rows}
        if len(result) != len(rows):raise ValueError('Duplicate ' + label + ' path')
        return result
    cs = index(colliders, 'collider'); bs = index(bodies, 'body'); gs = index(groups, 'group')
    for body in bodies:
        if type(body.get('enabled')) is not bool or type(body.get('kinematic')) is not bool:
            raise ValueError('Unknown rigid-body flags')
    for c in colliders:
        if type(c.get('collision_enabled')) is not bool or 'rigid_body' not in c:
            raise ValueError('Unknown collider enabled/owner state')
        owner = c['rigid_body']
        if owner is not None and (owner not in bs or not _under(c['path'], owner)):
            raise ValueError('Unknown collider rigid-body owner')
        if c['collision_enabled'] and owner is not None and bs[owner]['enabled'] and not any(_under(c['path'], e) for e in envs):
            raise ValueError('Shared external mobile collider can influence environments')
    membership = {};enabled_counts = {};zero_contact = []
    for i, env in enumerate(envs):
        path = '/World/collisions/group' + str(i)
        g = gs.get(path)
        if not g or g.get('type_name') != 'PhysicsCollisionGroup':
            raise ValueError('Actual PhysicsCollisionGroup schema/path is missing')
        if g.get('includes') != [env] or g.get('excludes') != [] or g.get('expansion_rule') != 'expandPrims':
            raise ValueError('Environment collider collection is incomplete/overbroad')
        # Installed USD CPU probe returns None for an unauthored includeRoot,
        # rather than False. Preserve that raw value and its authored flag;
        # the resolved query below independently proves exact membership.
        if 'include_root' not in g or type(g.get('include_root_authored')) is not bool or not (g['include_root'] is False or (g['include_root'] is None and g['include_root_authored'] is False)):
            raise ValueError('Collection root inclusion is enabled/unknown')
        if g.get('filtered_groups') != [path]:
            raise ValueError('Inverted groups must allow self only; cross-env/unknown contact')
        if 'per_group_invert' not in g or 'per_group_invert_authored' not in g or not (g['per_group_invert'] is None or g['per_group_invert'] is False) or type(g['per_group_invert_authored']) is not bool or (g['per_group_invert'] is None and g['per_group_invert_authored']):
            raise ValueError('Unknown/per-group inverted filter override')
        expected = sorted(c for c in cs if _under(c, env))
        if g.get('resolved_collider_members') != expected:
            raise ValueError('Resolved collection does not cover every own-env collider exclusively')
        if not any(_under(b['path'],env) and b['enabled'] and not b['kinematic'] for b in bodies):
            raise ValueError('Environment has no actual own active drone rigid body')
        for c in expected:
            owner = cs[c]['rigid_body']
            if cs[c]['collision_enabled'] and (owner is None or not _under(owner, env) or not bs[owner]['enabled'] or bs[owner]['kinematic']):
                raise ValueError('Enabled drone collider has no own active non-kinematic rigid body')
        membership[env] = expected
        enabled_counts[env] = sum(cs[c]['collision_enabled'] for c in expected)
        if not enabled_counts[env]:zero_contact.append(env)
    for root in dynamic:
        rows = [c for c in colliders if _under(c['path'], root)]
        if not rows or any(c['collision_enabled'] for c in rows):
            raise ValueError('Dynamic obstacle actual collision flag is not disabled')
    return {'passed': True, 'schema': SCHEMA, 'snapshot_sha256': snapshot_sha256(snapshot),
            'environment_count': len(envs), 'dynamic_obstacle_root_count': len(dynamic),
            'collision_group_count': len(groups), 'collider_count': len(colliders),
            'environment_collider_membership': membership,
            'enabled_environment_collider_counts': enabled_counts,
            'zero_contact_environments': zero_contact,
            'global_cloner_group_present': False,
            'unknown_shared_mobile_colliders': [],
            'inverted_filter_semantics': 'filtered_groups_are_allowed_contacts',
            'physical_attributes_modified': False}

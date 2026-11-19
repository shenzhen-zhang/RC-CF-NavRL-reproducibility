"""Prospective qualified-anchor PC-ARP replay using the frozen native lifecycle.

No simulator snapshots are approximated by pose-only copying. Each candidate
starts from a fresh environment reset and the same saved velocity-command
prefix. Full state is saved; the declared qualification domain and all exact global controls are checked, and
repeated candidate trajectories must pass a frozen numerical/event tolerance.
Unexposed PhysX warm-start state is not claimed to be serialized.
"""
from __future__ import annotations
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _main():
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from replay_checks import (check_replay, check_repeated_outcome, array_hash,
        qualification_from_events, check_prefix_qualification, validate_outcome_schema,
        ReplayCheckError)
    from qualified_state_registry import (PROTOCOL, FIELD_REGISTRY, NUM_ENVS,
        REGISTRY_SHA256, QUALIFICATION_RULE, PREFIX_EVENTS)
    from usd_isolation import capture_isolation, validate_isolation
    from evaluate_v19_canonical import (bind_navrl_canonical_dynamic_scene,
                                        prepare_navrl_evaluation_rollout, seed_process,
                                        navrl_articulation_state_signature)
    from isaac_app import create_simulation_app, close_simulation_app
    from radcppo_navrl import ReachabilityTeacher, ReachabilityTeacherConfig
    from radcppo_navrl import observation_safety_cost, observation_min_clearance
    from radcppo_navrl.checkpointing import filter_shape_compatible_state_dict, materialize_shared_actor_head_aliases

    @hydra.main(config_path='/root/rc_cf_revision_20261020/fresh_project_scenefix1/methods/radcppo_navrl/cfg', config_name='train', version_base=None)
    def run(cfg):
        spec = OmegaConf.to_container(cfg.replay_study, resolve=True)
        if int(cfg.env.num_envs) != NUM_ENVS or int(cfg.env_dyn.num_obstacles) != 80:
            raise ValueError('Qualified protocol requires the frozen N64/D80 cohort')
        extension_path = Path(spec['teacher_validation_extension_path']).resolve()
        extension_sha = str(spec['teacher_validation_extension_sha256'])
        if file_hash(extension_path) != extension_sha:
            raise ValueError('Qualified extension manifest SHA mismatch')
        extension = json.loads(extension_path.read_text(encoding='utf-8'))
        if extension.get('protocol') != 'TEACHER_VALIDATION_EXTENSION_QUALIFIED_ANCHOR_V1':
            raise ValueError('Qualified extension protocol mismatch')
        parent = extension['parent_plan']
        if parent['sha256'] != '8142cb94b8ccf9918795df73c60646a8610e6522e2fd70b1537e44293288dac9' or file_hash(parent['path']) != parent['sha256']:
            raise ValueError('Qualified extension must bind the unchanged frozen parent plan')
        required_sources = {'collect_teacher_replay.py','replay_checks.py','qualified_state_registry.py',
                            'usd_isolation.py','analyze_teacher_replay.py'}
        sources = extension['qualified_sources']
        if set(sources) != required_sources:
            raise ValueError('Qualified extension source coverage differs')
        for name, identity in sources.items():
            path = Path(identity['path']).resolve()
            if path.name != name or path.parent != Path(__file__).resolve().parent or file_hash(path) != identity['sha256']:
                raise ValueError('Qualified actual source SHA/path mismatch: ' + name)
        if Path(sources['collect_teacher_replay.py']['path']).resolve() != Path(__file__).resolve():
            raise ValueError('Qualified collector entry identity differs')
        extension_identity = {'path':str(extension_path),'sha256':extension_sha,
            'protocol':extension['protocol'],'parent_plan':parent,'qualified_sources':sources}
        if int(cfg.sim.substeps) != 1:
            raise ValueError('Replay protocol requires sim.substeps=1; horizon is measured in control steps')
        output = Path(spec['output']).resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output/'protocol_identity.json').write_text(json.dumps({
            'teacher_validation_protocol':PROTOCOL,'extension_identity':extension_identity,
            'registry_sha256':REGISTRY_SHA256,'qualification_rule':QUALIFICATION_RULE,
            'prospective_only':True},indent=2),encoding='utf-8')
        app = create_simulation_app(headless=True)
        success = False
        try:
            from env import NavigationEnv
            from ppo import PPO
            from radcppo_adapter import RADCPPO
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            from torchrl.envs.transforms import Compose, TransformedEnv
            from torchrl.envs.utils import ExplorationType, set_exploration_type
            from utils import vec_to_world

            seed_process(int(cfg.seed))
            base = NavigationEnv(cfg)
            controller = LeePositionController(9.81, base.drone.params).to(cfg.device)
            env = TransformedEnv(base, Compose(VelController(controller, yaw_control=False)))
            env.eval()
            scene = bind_navrl_canonical_dynamic_scene(env, str(cfg.canonical_dynamic_scene_snapshot),
                                                     str(cfg.canonical_dynamic_scene_snapshot_sha256))
            checkpoint = Path(cfg.checkpoint).resolve()
            checkpoint_sha = file_hash(checkpoint)
            if checkpoint_sha != spec['checkpoint_sha256']:
                raise ValueError('Branch source checkpoint SHA mismatch')
            native = spec['policy_kind'] == 'official_navrl'
            policy = (PPO if native else RADCPPO)(cfg.algo, env.observation_spec, env.action_spec, cfg.device)
            weights = torch.load(checkpoint, map_location=cfg.device)
            if native:
                report = policy.load_state_dict(weights, strict=True)
            else:
                weights, skipped = filter_shape_compatible_state_dict(policy, weights)
                weights, aliases = materialize_shared_actor_head_aliases(policy, weights)
                report = policy.load_state_dict(weights, strict=True)
                policy.sync_training_schedule()
                policy.set_rollout_mode('actor')
            method = cfg.algo.method
            teacher = ReachabilityTeacher(ReachabilityTeacherConfig(
                horizon_s=float(method.teacher.horizon_s), horizon_steps=int(method.teacher.horizon_steps),
                max_acceleration_mps2=float(method.teacher.max_acceleration_mps2),
                max_speed_mps=float(cfg.algo.actor.action_limit), static_point_budget=int(method.teacher.static_point_budget),
                counterfactual_count=int(method.counterfactual_count), rollout_model=str(method.teacher.rollout_model))).to(cfg.device)
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            isolation_args = (stage, str(base.sim.get_physics_context().prim_path),
                              list(base.envs_prim_paths))
            dynamic_paths = ['/World/Origin'+str(i) for i in range(80)]
            isolation = capture_isolation(*isolation_args, dynamic_paths=dynamic_paths)
            isolation_validation = validate_isolation(isolation, expected_num_envs=NUM_ENVS)
            (output/'usd_isolation.json').write_text(json.dumps({'snapshot':isolation,
                'validation':isolation_validation},indent=2),encoding='utf-8')
            physics_dt = float(base.dt)
            control_dt = physics_dt * int(cfg.sim.substeps)
            requested_horizon = float(method.teacher.horizon_s)
            horizon = int(round(requested_horizon / control_dt))
            if horizon < 1 or abs(horizon * control_dt-requested_horizon) > .5*control_dt+1e-9:
                raise ValueError('Teacher physical branch must use the nearest control step')
            anchors = sorted(set(int(t) for t in spec['anchor_steps']))
            if not anchors or anchors[0] < 1 or anchors[-1] + horizon >= int(cfg.env.max_episode_length):
                raise ValueError('Invalid branch anchor schedule')
            tolerance = float(spec.get('replay_tolerance', 1e-4))
            if horizon != 50 or int(method.counterfactual_count) != 12:
                raise ValueError('Qualified cohort requires 50 control steps and 12 teacher entries plus nominal')
            if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 1e-4:
                raise ValueError('Qualified protocol retains the original maximum physical tolerance')
            records, control_records, reset_records, qualification_records = [], [], [], []
            started = time.monotonic()

            def array(value):
                return value.detach().cpu().numpy().copy()

            def state_signature(td):
                fields = {key:array(value) for key,value in navrl_articulation_state_signature(base).items()}
                for name in ['progress_buf', 'target_pos', 'target_dir', 'height_range', 'prev_drone_vel_w',
                             'dyn_obs_state', 'dyn_obs_origin', 'dyn_obs_goal', 'dyn_obs_vel', 'dyn_obs_vel_norm']:
                    value = getattr(base, name, None)
                    if isinstance(value, torch.Tensor): fields['environment/' + name] = array(value)
                fields['environment/dyn_obs_step_count'] = np.asarray(base.dyn_obs_step_count, dtype=np.int64)
                for name, value in vars(base.drone).items():
                    if isinstance(value, torch.Tensor): fields['drone/attribute/' + name] = array(value)
                if hasattr(base.drone, 'named_buffers'):
                    for name, value in base.drone.named_buffers(): fields['drone/buffer/' + name] = array(value)
                for channel in ['state', 'lidar', 'direction', 'dynamic_obstacle']:
                    fields['observation/' + channel] = array(td['agents', 'observation', channel])
                fields['random/torch_cpu'] = array(torch.get_rng_state())
                for i, value in enumerate(torch.cuda.get_rng_state_all()): fields[f'random/cuda_{i}'] = array(value)
                npr = np.random.get_state()
                fields['random/numpy_keys'] = npr[1].copy()
                fields['random/numpy_position'] = np.asarray(npr[2:], dtype=np.float64)
                fields['random/python_sha'] = np.frombuffer(hashlib.sha256(repr(random.getstate()).encode()).digest(), dtype=np.uint8)
                return fields

            def prefix_event(td):
                n = NUM_ENVS
                return {key:array(value.bool().reshape(n)) for key,value in [
                    ('done',td['done']),('reached_goal',td['stats','reach_goal']),
                    ('terminated',td['terminated']),('truncated',td['truncated']),
                    ('collision',td['stats','collision'])]}

            def stack_events(events, steps):
                return {key:np.stack([event[key] for event in events[:steps]],axis=1)
                        if steps else np.empty((NUM_ENVS,0),dtype=bool) for key in PREFIX_EVENTS}

            def raw_state(raw, prefix, state):
                for key,value in state.items(): raw[prefix+key.replace('/','__')] = value

            def raw_outcome(raw, prefix, outcome):
                for key,value in outcome.items():
                    if key == 'events':
                        for event,event_value in value.items():raw[prefix+'event_'+event] = event_value
                    else: raw[prefix+key] = value

            def reset(seed):
                if not native: policy.set_rollout_mode('actor')
                td, diagnostics = prepare_navrl_evaluation_rollout(env, seed)
                reset_records.append({'evaluation_seed':int(seed),'reset_index':len(reset_records)+1,
                                      'diagnostics':diagnostics})
                return td

            def step_world(td, command):
                td['agents', 'action'] = command.clone()
                return env.step(td)['next'].clone()

            @torch.no_grad()
            def branch_outcome(td, world):
                position, obstacle, costs, clearances, distances = [], [], [], [], []
                events = {key:[] for key in ['collision','terminated','truncated','done','reached_goal','valid_before_transition']}
                n = int(cfg.env.num_envs)
                collision = torch.zeros(n, dtype=torch.bool, device=cfg.device)
                unsafe = torch.zeros_like(collision)
                first = torch.full((n,), -1, dtype=torch.long, device=cfg.device)
                alive = torch.ones_like(collision)
                initial_distance = (base.target_pos-base.drone.get_state(env_frame=False)[..., :3]).norm(dim=-1).squeeze(-1)
                for t in range(horizon):
                    td = step_world(td, world)
                    obs = td['agents', 'observation']
                    c = td['stats', 'collision'].bool().reshape(n)
                    u = td['terminated'].bool().reshape(n)
                    for key, value in [('collision',c),('terminated',u),
                        ('truncated',td['truncated'].bool().reshape(n)),('done',td['done'].bool().reshape(n)),
                        ('reached_goal',td['stats','reach_goal'].bool().reshape(n)),('valid_before_transition',alive)]:
                        events[key].append(array(value))
                    first = torch.where(alive & u, torch.full_like(first, t), first)
                    collision |= alive & c
                    unsafe |= alive & u
                    cost = observation_safety_cost(obs['lidar'], obs['dynamic_obstacle'], td['stats', 'collision']).reshape(n)
                    clearance = observation_min_clearance(obs['lidar'], obs['dynamic_obstacle']).reshape(n)
                    position.append(array(base.drone.get_state(env_frame=False)[..., :3].squeeze(1)))
                    distances.append(array((base.target_pos-base.drone.get_state(env_frame=False)[..., :3]).norm(dim=-1).squeeze(-1)))
                    obstacle.append(array(base.dyn_obs_state[:, :3]))
                    costs.append(array(cost))
                    clearances.append(array(clearance))
                    alive &= ~u
                distance_array = np.stack(distances, axis=1)
                end_step = np.where(array(first)>=0, array(first), horizon-1)
                progress = array(initial_distance)-distance_array[np.arange(n), end_step]
                return {'positions': np.stack(position, axis=1), 'dynamic_positions': np.stack(obstacle),
                        'costs': np.stack(costs, axis=1), 'observed_clearance': np.stack(clearances, axis=1),
                        'collision': array(collision), 'unsafe': array(unsafe), 'first_unsafe_step': array(first),
                        'progress_m': progress, 'events':{key:np.stack(value,axis=1) for key,value in events.items()}}

            for evaluation_seed in spec['evaluation_seeds']:
                evaluation_seed = int(evaluation_seed)
                td = reset(evaluation_seed)
                prefix, prefix_events = [], []
                bank = {}
                alive = torch.ones(int(cfg.env.num_envs), dtype=torch.bool, device=cfg.device)
                with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
                    for t in range(anchors[-1] + 1):
                        if t in anchors:
                            predicted = policy(td.clone())
                            obs = td['agents', 'observation']
                            nominal = (2 * predicted['agents', 'action_normalized'] - 1) * float(cfg.algo.actor.action_limit)
                            if not native: nominal = predicted['method_base_action_local']
                            before = state_signature(td)
                            teacher_output = teacher(nominal, obs['state'], obs['lidar'], obs['dynamic_obstacle'])
                            after = state_signature(td)
                            try:
                                empty = check_replay(before, after, tolerance=0., empty_query=True)
                            except ReplayCheckError as error:
                                failed_empty = {}
                                raw_state(failed_empty,'reference__',before)
                                raw_state(failed_empty,'current__',after)
                                np.savez_compressed(output/f'failed_empty_query_seed{evaluation_seed}_anchor{t}.npz',**failed_empty)
                                (output/'failed_control.json').write_text(json.dumps({'evaluation_seed':evaluation_seed,
                                    'anchor_step':t,'control':error.control},indent=2),encoding='utf-8')
                                raise
                            pool = teacher._candidate_commands(nominal.reshape(-1,3), obs['state'].reshape(-1,8))
                            trajectories = teacher._reachable_trajectories(pool, obs['state'].reshape(-1,8)[...,5:8])
                            all_clearance = teacher._minimum_clearance(trajectories, obs['lidar'].reshape(-1,36,4), obs['dynamic_obstacle'].reshape(-1,5,10))
                            indices = torch.linspace(0, pool.shape[1]-1, int(method.counterfactual_count)-1, device=pool.device).round().long()
                            predicted_clearance = torch.cat([all_clearance[:, indices], teacher_output.clearance.reshape(-1,1)], dim=-1)
                            candidates = teacher_output.candidate_actions.reshape(-1,int(method.counterfactual_count),3)
                            # The pool norm-caps even its first entry, whereas
                            # NavRL bounds velocity components. Always execute
                            # the original, uncapped nominal as a separate
                            # reference so selected-vs-nominal is well defined.
                            nominal_trajectory = teacher._reachable_trajectories(nominal.reshape(-1,1,3), obs['state'].reshape(-1,8)[...,5:8])
                            nominal_clearance = teacher._minimum_clearance(nominal_trajectory,obs['lidar'].reshape(-1,36,4),obs['dynamic_obstacle'].reshape(-1,5,10))
                            candidates = torch.cat([candidates,nominal.reshape(-1,1,3)],dim=1)
                            predicted_clearance = torch.cat([predicted_clearance,nominal_clearance],dim=-1)
                            predicted_risk = torch.sigmoid((float(teacher.config.preferred_clearance_m)-predicted_clearance)/float(teacher.config.risk_temperature_m))
                            world = [vec_to_world(candidates[:, k], obs['direction']) for k in range(candidates.shape[1])]
                            reference_events = stack_events(prefix_events,t)
                            fixed_valid, first_event = qualification_from_events(reference_events,t)
                            if not np.array_equal(fixed_valid,array(alive)):
                                raise ValueError('Independent reference qualification differs from original alive rule')
                            bank[t] = {'signature': before, 'empty_after_signature':after,
                                       'valid': fixed_valid, 'prefix_events':reference_events,
                                       'first_disqualifying_step':first_event, 'world': world,
                                       'candidates': array(candidates), 'predicted_clearance': array(predicted_clearance),
                                       'predicted_risk': array(predicted_risk),
                                       'selected_index': array(teacher_output.selected_index.reshape(-1)),
                                       'empty_control': empty}
                        if t == anchors[-1]: break
                        policy_td = policy(td.clone())
                        world = policy_td['agents', 'action'].clone()
                        prefix.append(world)
                        td = step_world(td, world)
                        prefix_events.append(prefix_event(td))
                        alive &= ~(td['done'].reshape(-1).bool() | td['stats','reach_goal'].reshape(-1).bool())

                for anchor_step in anchors:
                    anchor = bank[anchor_step]
                    count = len(anchor['world'])
                    # The final two repetitions test candidate-order carryover.
                    order = list(range(count)) + [count-1, count-2]
                    reference_outcomes = {}
                    raw = {'valid_anchor_mask':anchor['valid'],
                           'reference_first_disqualifying_step':anchor['first_disqualifying_step']}
                    raw_state(raw,'anchor_state__',anchor['signature'])
                    raw_state(raw,'empty_query_after__',anchor['empty_after_signature'])
                    for key,value in anchor['prefix_events'].items():raw['reference_prefix_event_'+key] = value
                    raw['candidate_commands_local'] = anchor['candidates']
                    raw['command_prefix_world'] = np.stack([array(x) for x in prefix[:anchor_step]])
                    raw['teacher_clearances'] = anchor['predicted_clearance']
                    raw['teacher_risks'] = anchor['predicted_risk']
                    raw_path = output/f'seed{evaluation_seed}_anchor{anchor_step}.npz'
                    qualification_record = {'evaluation_seed':evaluation_seed,'anchor_step':anchor_step,
                        'planned_envs':NUM_ENVS,'qualified_count':int(anchor['valid'].sum()),
                        'excluded_envs':np.flatnonzero(~anchor['valid']).tolist(),
                        'qualified_mask_sha256':array_hash(anchor['valid']),
                        'reference_prefix_event_hashes':{k:array_hash(v) for k,v in anchor['prefix_events'].items()},
                        'reference_first_event_sha256':array_hash(anchor['first_disqualifying_step']),
                        'empty_estimate':bool(not anchor['valid'].any()),
                        'qualification_fixed_before_candidates':True}
                    qualification_records.append(qualification_record)
                    (output/'qualification_summary.json').write_text(json.dumps(qualification_records,indent=2),encoding='utf-8')
                    np.savez_compressed(raw_path,**raw)
                    for order_position, candidate in enumerate(order):
                        td = reset(evaluation_seed)
                        replay_events = []
                        for command in prefix[:anchor_step]:
                            td = step_world(td, command)
                            replay_events.append(prefix_event(td))
                        replay_events = stack_events(replay_events,anchor_step)
                        _, replay_first = qualification_from_events(replay_events,anchor_step)
                        current_signature = state_signature(td)
                        raw_state(raw,f'replay_{order_position}_state__',current_signature)
                        for key,value in replay_events.items():raw[f'replay_{order_position}_prefix_event_'+key] = value
                        raw[f'replay_{order_position}_first_disqualifying_step'] = replay_first
                        np.savez_compressed(raw_path,**raw)
                        comparison_failures, control = {}, None
                        try:
                            qualification_control = check_prefix_qualification(anchor['prefix_events'],replay_events,anchor['valid'])
                        except ReplayCheckError as error:
                            comparison_failures['qualification'] = error.control
                        try:
                            control = check_replay(anchor['signature'],current_signature,
                                valid_mask=anchor['valid'],tolerance=tolerance)
                        except ReplayCheckError as error:
                            comparison_failures['state_replay'] = error.control
                        if comparison_failures:
                            (output/'failed_control.json').write_text(json.dumps({'evaluation_seed':evaluation_seed,
                                'anchor_step':anchor_step,'candidate':candidate,'order_position':order_position,
                                'prepare_index':len(reset_records),'failed_controls':comparison_failures,
                                'state_replay':comparison_failures.get('state_replay',control)},indent=2),encoding='utf-8')
                            raise ValueError('Prospective qualified replay/prefix control failed')
                        outcome = branch_outcome(td, anchor['world'][candidate])
                        validate_outcome_schema(outcome)
                        raw_outcome(raw,('repetition_'+str(order_position)+'_' if candidate in reference_outcomes else 'candidate_'+str(candidate)+'_'),outcome)
                        np.savez_compressed(raw_path,**raw)
                        if reference_outcomes and not np.array_equal(
                            reference_outcomes[0]['dynamic_positions'], outcome['dynamic_positions']):
                            raise ValueError('Different candidate changed the exogenous obstacle trajectory')
                        control_record = {'evaluation_seed': evaluation_seed, 'anchor_step': anchor_step,
                                          'candidate': candidate, 'order_position': order_position,
                                          'state_replay': control, 'empty_control': anchor['empty_control'],
                                          'qualification':qualification_control,
                                          'common_future_obstacle_trajectory': True}
                        if candidate in reference_outcomes:
                            try:
                                control_record['outcome_repetition'] = check_repeated_outcome(reference_outcomes[candidate],
                                    outcome,valid_mask=anchor['valid'],tolerance=tolerance)
                            except ReplayCheckError as error:
                                (output/'failed_control.json').write_text(json.dumps({'evaluation_seed':evaluation_seed,
                                    'anchor_step':anchor_step,'candidate':candidate,'order_position':order_position,
                                    'prepare_index':len(reset_records),'control':error.control},indent=2),encoding='utf-8')
                                raise
                        else:
                            reference_outcomes[candidate] = outcome
                            raw[f'candidate_{candidate}_positions'] = outcome['positions']
                            raw[f'candidate_{candidate}_dynamic_positions'] = outcome['dynamic_positions']
                            raw[f'candidate_{candidate}_costs'] = outcome['costs']
                            raw[f'candidate_{candidate}_observed_clearance'] = outcome['observed_clearance']
                            for key,value in outcome['events'].items():
                                raw[f'candidate_{candidate}_event_{key}'] = value
                            for env_index, valid in enumerate(anchor['valid']):
                                if not valid: continue
                                first_unsafe = int(outcome['first_unsafe_step'][env_index])
                                stop = first_unsafe+1 if first_unsafe>=0 else horizon
                                records.append({'evaluation_seed': evaluation_seed, 'anchor_step': anchor_step,
                                    'env_index': env_index, 'candidate_index': candidate,
                                    'candidate_local_x': float(anchor['candidates'][env_index,candidate,0]),
                                    'candidate_local_y': float(anchor['candidates'][env_index,candidate,1]),
                                    'candidate_local_z': float(anchor['candidates'][env_index,candidate,2]),
                                    'teacher_predicted_clearance_m': float(anchor['predicted_clearance'][env_index,candidate]),
                                    'teacher_clearance_is_infinite': bool(not np.isfinite(anchor['predicted_clearance'][env_index,candidate])),
                                    'candidate_role': 'nominal' if candidate==count-1 else ('teacher_selected' if candidate==count-2 else ('speed_capped_nominal_pool' if candidate==0 else 'fixed_lattice')),
                                    'teacher_predicted_risk': float(anchor['predicted_risk'][env_index,candidate]),
                                    'teacher_predicted_feasible': bool(anchor['predicted_clearance'][env_index,candidate]>=0),
                                    'realized_collision': bool(outcome['collision'][env_index]),
                                    'realized_unsafe': bool(outcome['unsafe'][env_index]),
                                    'realized_first_unsafe_step': first_unsafe,
                                    'realized_observation_min_clearance_m': float(outcome['observed_clearance'][env_index,:stop].min()),
                                    'realized_mean_step_cost': float(outcome['costs'][env_index,:stop].mean()),
                                    'realized_progress_m': float(outcome['progress_m'][env_index]),
                                    'horizon_steps': horizon, 'horizon_s': horizon*control_dt,
                                    'requested_horizon_s': requested_horizon})
                        control_records.append(control_record)
                        (output/'replay_controls.json').write_text(json.dumps(control_records,indent=2),encoding='utf-8')
                    raw['candidate_commands_local'] = anchor['candidates']
                    raw['valid_anchor_mask'] = anchor['valid']
                    raw['command_prefix_world'] = np.stack([array(x) for x in prefix[:anchor_step]])
                    raw['teacher_clearances'] = anchor['predicted_clearance']
                    raw['teacher_risks'] = anchor['predicted_risk']
                    for field, value in anchor['signature'].items():
                        raw['anchor_state__'+field.replace('/','__')] = value
                    np.savez_compressed(output/f'seed{evaluation_seed}_anchor{anchor_step}.npz', **raw)
                    (output/'replay_controls.json').write_text(json.dumps(control_records, indent=2), encoding='utf-8')
                    print(f'branch seed={evaluation_seed} anchor={anchor_step} valid={int(anchor["valid"].sum())} controls=pass', flush=True)
            if not records: raise RuntimeError('All qualification domains empty: no estimable teacher outcomes; this is not zero risk')
            with (output/'candidate_outcomes.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            source_paths = {Path(identity['path']) for identity in extension_identity['qualified_sources'].values()} | {Path(__file__),Path(__file__).with_name('replay_checks.py'),Path(inspect.getfile(create_simulation_app)),Path(inspect.getfile(type(teacher))),
                            Path(inspect.getfile(type(base))),Path(inspect.getfile(type(base.drone))),
                            Path(inspect.getfile(type(controller))),Path(inspect.getfile(type(policy))),
                            Path(inspect.getfile(prepare_navrl_evaluation_rollout)),
                            Path(inspect.getfile(navrl_articulation_state_signature)),
                            Path(inspect.getfile(check_replay)),
                            Path(inspect.getfile(observation_safety_cost)),
                            Path(inspect.getfile(observation_min_clearance))}
            # Native bootstrap wraps create_simulation_app. Keep both its actual
            # wrapper file and the original isaac_app module in this manifest.
            for module_name in ['env','utils','ppo','radcppo_adapter','isaac_app','evaluate_v19_canonical']:
                module = sys.modules.get(module_name)
                if module is not None and getattr(module,'__file__',None): source_paths.add(Path(module.__file__))
            phase = str(spec.get('phase','pilot'))
            if phase not in {'pilot','formal'}: raise ValueError('Unknown replay study phase')
            final_isolation = capture_isolation(*isolation_args,dynamic_paths=dynamic_paths)
            final_isolation_validation = validate_isolation(final_isolation,expected_num_envs=NUM_ENVS)
            if final_isolation != isolation or final_isolation_validation != isolation_validation:
                raise ValueError('Actual USD collision isolation configuration changed during replay')
            metadata = {'status': 'QUALIFIED_PHYSICAL_REPLAY_FORMAL_COMPLETE' if phase=='formal' else 'QUALIFIED_PHYSICAL_REPLAY_PILOT_COMPLETE',
                        'teacher_validation_protocol':PROTOCOL,'extension_identity':extension_identity,
                        'qualified_replay':{'protocol':PROTOCOL,'registry_sha256':REGISTRY_SHA256,
                            'qualification_rule':QUALIFICATION_RULE,'num_envs':NUM_ENVS,
                            'qualification_by_anchor':qualification_records,
                            'usd_isolation':{'snapshot':isolation,'validation':isolation_validation},
                            'usd_isolation_at_completion':{'snapshot':final_isolation,'validation':final_isolation_validation},
                            'usd_isolation_unchanged_at_completion':True,
                            'excluded_envs_are_saved_and_audited_for_full_schema_and_finiteness':True,
                            'no_post_unsafe_time_mask_for_repetition':True,
                            'old_all_environment_failures_remain_rejected':True},
                        'raw_npz_sha256':{path.name:file_hash(path) for path in sorted(output.glob('seed*_anchor*.npz'))},
                        'checkpoint': str(checkpoint),
                        'checkpoint_sha256': checkpoint_sha, 'scene': scene, 'spec': spec,
                        'source_sha256': {str(path.resolve()):file_hash(path) for path in source_paths if path.is_file()},
                        'rows': len(records), 'unique_anchor_states': len(records)//(int(method.counterfactual_count)+1),
                        'reference_action': 'original component-bounded nominal, additionally executed after the 12 norm-capped teacher pool entries',
                        'anchor_policy': 'released native actor' if native else 'current adapted actor mean',
                        'teacher_nominal_input': 'released native actor mean' if native else 'frozen base-task actor mean, as used by distillation/counterfactual training labels',
                        'local_outcome_estimand': 'fixed world velocity to the control step nearest the requested teacher horizon; masks stop at first unsafe, goal is not absorbing',
                        'actual_timing': {'configured_physics_dt_s':float(cfg.sim.dt),'physics_dt_s':physics_dt,
                            'control_dt_s':control_dt,'requested_teacher_horizon_s':requested_horizon,
                            'realized_horizon_s':horizon*control_dt,'horizon_steps':horizon,
                            'rounding':'nearest_control_step'},
                        'elapsed_seconds': time.monotonic()-started, 'control_runs': len(control_records),
                        'restoration': 'canonical_three_commit_full_reset_and_complete_command_prefix_replay',
                        'articulation_signature_version': 2,
                        'initialization_diagnostics': reset_records,
                        'scope': 'prospectively fixed reference-prefix qualified environments for motion-dependent state/repeated trajectories; all RNG, fixed parameters and exogenous obstacles exact unmasked; all full tensors saved and finite; unexposed PhysX state not serialized',
                        'clearance_reference': 'realized NavRL observation clearance, not an exact terrain SDF',
                        'collision_reference': 'NavRL LiDAR/dynamic geometric collision event after physical execution, not exact PhysX contact truth',
                        'execution_environment': {'torch_version':torch.__version__, 'torch_cuda_runtime':torch.version.cuda,
                            'gpu_name':torch.cuda.get_device_name(cfg.device),
                            **{key:os.environ.get(key) for key in ('NAVRL_HEADLESS_NO_VIEWPORT','NAVRL_ISAAC_EXPERIENCE','NAVRL_OFFICIAL_ROOT')}},
                        'config': OmegaConf.to_container(cfg, resolve=True)}
            (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            success = True
        except BaseException as error:
            (output/'failure.json').write_text(json.dumps({'status':'FAILED_NO_FORMAL_BRANCH_RESULTS','teacher_validation_protocol':PROTOCOL,'extension_identity':extension_identity,
                'error':str(error), 'traceback':traceback.format_exc()}, indent=2), encoding='utf-8')
            raise
        finally:
            close_simulation_app(app, success=success)
    run()

if __name__ == '__main__': _main()

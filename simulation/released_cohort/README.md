# Released-checkpoint cohort

This cohort adapts the public NavRL initializer with nine variants and ten training seeds per variant. The [study plan](../../analysis/supplementary/study_plan.json) records the settings and archived commands. The [training scene manifest](training_scenes_scenefix1/manifest.json) includes the original digests and the redistributed digests after provenance paths were made portable. The [evaluation jobs](evaluation_jobs.json) preserve the checkpoint order and complete overrides for the 341 observed jobs.

## Prepare Isaac Sim

Use the pinned [NavRL setup](../README.md) in a disposable copy of its checkout. Apply the three headless patches with `patch -p1` from the NavRL `isaac-training` directory. Apply `runtime_env_patch.diff` to `training/scripts/env.py` using `patch training/scripts/env.py < /absolute/path/to/runtime_env_patch.diff`. The study plan records the pristine and patched SHA-256 checksums. Enable `NAVRL_HEADLESS_NO_VIEWPORT=1` for the headless patches.

Copy the `simulation` directory to a working directory and copy the contents of [overlay/](overlay/) onto that copy. The [overlay manifest](overlay-files.json) records the byte-for-byte archived versions of the six replaced files. Point `NAVRL_OFFICIAL_ROOT` at the patched `isaac-training` directory and add the working copy's `src/` to `PYTHONPATH`. Run the Hydra programs from the working copy using the Isaac Sim Python launcher.

Download the [public initializer](../../metadata/navrl-checkpoint.json) into your checkpoint folder as `navrl_checkpoint.pt`. Its SHA-256 is `51fa3dbdc6ba89626b5dad3a4638deb53d40aa6f0caa9b289657da4a8e0b60c3`. Extract the separate `released-cohort-checkpoints.zip` into the same folder to evaluate the authors' saved policies.

## Prepare input manifests

Run the commands below from the working copy. Replace `/absolute/path/to/checkpoints` with the checkpoint folder.

```bash
python3 released_cohort/prepare_inputs.py training training-scenes.local.json
python3 released_cohort/prepare_inputs.py evaluation eval_seed1183109_block00 \
  /absolute/path/to/checkpoints evaluation.local.json
```

For training, run `methods/radcppo_navrl/train.py` with the job's `argv` from the study plan. Replace the recorded input and output paths with your own paths. Set `training_scene_manifest` and `training_scene_manifest_sha256` to the relocated manifest and the digest printed by the first command. Set `training_scene_snapshot` and `training_scene_snapshot_sha256` to that training seed's absolute scene path and digest from the relocated manifest.

For the example evaluation job, use the following Hydra arguments with the prepared manifest and a fresh output directory.

```bash
"$ISAAC_PYTHON" methods/radcppo_navrl/evaluate_v19_checkpoint_suite.py \
  experiment=reviewer_F11_full checkpoint_suite_manifest="$PWD/evaluation.local.json" \
  +checkpoint_suite_order_scheme=permutation_cycle process_replicate=0 \
  "evaluation_seeds=[1183109]" seed=7 env.num_envs=64 env.num_obstacles=350 \
  env_dyn.num_obstacles=80 env.max_episode_length=2200 \
  canonical_dynamic_scene_snapshot="$PWD/configs/canonical_navrl_dynamic_scene_seed7_80obs_v1.json" \
  canonical_dynamic_scene_snapshot_sha256=737d7708d47dc3c3ac47881fad137cb61cc800d97be8b524b04e4d1f74944c0b \
  output_dir="$PWD/eval-example" revision_cost_trace_dir="$PWD/eval-example-cost" \
  hydra.run.dir="$PWD/eval-example-hydra"
```

The training scene inputs and sample evaluation manifest were validated on CPU. A fresh Isaac Sim training or rollout was outside that verification.

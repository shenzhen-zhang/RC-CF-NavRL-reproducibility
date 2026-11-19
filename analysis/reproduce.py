"""Recompute and check the reported results using the packaged CPU inputs."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis/supplementary'))
import analysis_core as released_core
import analyze_fresh_paired as released_paired


def read_json(path):
    return json.loads(Path(path).read_bytes())


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(script, *args, log=None):
    command = [sys.executable, script, *map(str, args)]
    if log is None:
        subprocess.run(command, check=True)
    else:
        with log.open('w') as stream:
            subprocess.run(command, check=True, stdout=stream)


def ood_arguments():
    for seed in (17, 29, 43):
        files = Path(f'raw/original/unseen_layouts/scene_{seed}')
        for group in ('group_012', 'group_34'):
            yield '--method'
            yield f'{seed}={files / group / "evaluation/episodes.csv"}'
        yield '--baseline'
        yield f'{seed}={files / "PPO拉格朗日基线/evaluation/episodes.csv"}'


def check_released(output):
    protocol = Path('analysis/supplementary')
    plan = read_json(protocol / 'study_plan.json')
    missing = read_json(protocol / 'missing-cases.json')
    shards = sorted(Path('raw/released_checkpoint/navigation').glob('*/episodes.csv'))
    panels, inputs = released_core.load_available(shards, plan, missing, released_paired)
    check(len(inputs) == 341 and sum(v['rows'] for v in inputs.values()) == 104_512,
          'Released-checkpoint input coverage differs')
    primary = released_core.primary_rows(panels, plan, released_paired)
    sensitivity, _ = released_core.sensitivity_rows(panels, plan, missing, released_paired)
    scope_plan = {**plan, 'evaluation_seeds': plan['evaluation_seeds'][:17]}
    common = {method: panel[:, :17] for method, panel in panels.items()}
    contrasts, absolute = released_paired.analyze(common, scope_plan)
    frozen = read_json('evidence/scope17_paired_effects_disclosed.json')['original_analyzer_result']
    for actual, expected in zip(contrasts, frozen['contrasts'], strict=True):
        # SciPy versions differ slightly when computing the Student t quantile.
        field = 'conditional_seed_t_ci95_sensitivity'
        check({k: v for k, v in actual.items() if k != field} ==
              {k: v for k, v in expected.items() if k != field},
              'Released common-17 contrasts differ')
        np.testing.assert_allclose(actual[field], expected[field], rtol=0, atol=1e-11)
    check(absolute == frozen['absolute'], 'Released common-17 rates differ')
    with Path('evidence/primary_complete18.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    fields = ('difference', 'ci95_low', 'ci95_high')
    for actual, row in zip(primary, rows, strict=True):
        check(all(actual[k] == row[k] for k in ('contrast', 'metric')),
              'Released primary outcome identity differs')
        np.testing.assert_allclose([actual[k] for k in fields], [float(row[k]) for k in fields],
                                   rtol=0, atol=1e-12)
    limit_file = next(Path('processed/supplementary').rglob('bounded_missing18_sensitivity.json'))
    check(sensitivity == read_json(limit_file)['rows'], 'Missing-case sensitivity differs')
    data = {'primary_complete18': primary, 'scope17_contrasts': contrasts,
            'scope17_absolute': absolute, 'sensitivity': sensitivity}
    (output / 'released-navigation.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print('Released cohort matches the reported 18-seed primary and common-17 analyses.', flush=True)


def check_replay(output):
    run('analysis/replay/analyze_teacher_replay.py', '--root', 'raw/released_checkpoint/replay',
        '--output', output / 'replay', '--bootstrap', 20_000)
    report = read_json(output / 'replay/teacher_validation.json')
    frozen_file = next(Path('processed/supplementary').rglob('source_teacher_validation.json'))
    frozen = read_json(frozen_file)
    check(len(report['source_files']) == 20, 'Replay job count differs')
    check(report['groups'] == frozen['groups'], 'Replay estimates or confidence intervals differ')
    print('Teacher replay estimates and 95% intervals match both anchor families.', flush=True)


def check_trajectory_and_runtime():
    case = read_json('visuals/data/flight_case.json')
    for row in case['methods']:
        source = Path('raw/flight_case', f'{row["method"]}_seed{case["evaluation_seed"]}.npz')
        check(hashlib.sha256(source.read_bytes()).hexdigest() == row['source_npz_sha256'],
              f'{source} checksum differs')
        with np.load(source, allow_pickle=False) as raw:
            end = row['terminal_state_index']
            for actual, field in [
                (raw['positions'][:end + 1], 'positions'),
                (raw['target_position'].reshape(-1), 'target_position'),
                (raw['dynamic_obstacle_positions'][end], 'terminal_obstacle_positions'),
                (raw['dynamic_obstacle_sizes'], 'obstacle_sizes'),
            ]:
                np.testing.assert_array_equal(actual, row[field])
    runtime = read_json('evidence/runtime_complexity_summary.json')
    repeats = [read_json(f'raw/runtime/repeat_{i}_runtime.json')['batches']['1'] for i in range(3)]
    for key in ('navrl_actor', 'v19_actor', 'v19_pcarp', 'v19_official_vo'):
        recorded = [r[key]['wall_mean_ms'] for r in repeats]
        target = runtime['aggregates']['1'][key]['wall_mean_ms']
        np.testing.assert_allclose([statistics.mean(recorded), statistics.stdev(recorded)],
                                   [target['mean'], target['sample_std']], rtol=0, atol=1e-12)
    print('Recorded flight geometry and batch-one runtime aggregates match.', flush=True)


def render():
    scripts = (
        'visuals/environment_setup/build_environment_setup.py',
        'build/reorganize_evidence_figures.py',
        'visuals/flight_case/build_flight_case.py',
        'visuals/teacher_validation/build_effect_matrix.py',
        'build/build_current_tables.py',
    )
    for script in scripts:
        run(script)
    figures = sorted(Path('visuals').glob('*/output/*.pdf'))
    check(len(figures) == 7 and all(p.read_bytes().startswith(b'%PDF-') for p in figures),
          'Expected seven regenerated evidence figures')
    tables = {'1': '03_runtime', 'A1': '04_configuration', 'A2': '05_components',
              'A3': '01_teacher', 'A4': '06_deployment'}
    for number, stem in tables.items():
        generated = Path('visuals/tables', f'{stem}.tex').read_bytes()
        check(generated == Path('manuscript/tables', f'table-{number}.tex').read_bytes(),
              f'Table {number} differs from the revision')
    print('Seven evidence figures rendered and all five tables match the revision.', flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('recomputed/run'))
    parser.add_argument('--quick', action='store_true',
                        help='Run primary, released-cohort, trajectory, runtime and figure/table checks')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)

    run('analysis/original/analyze_v19_simplified_vs_ppolag.py',
        '--episodes', 'raw/original/primary_paired_episodes.csv', '--output', output / 'primary.json',
        log=output / 'primary.log')
    check(read_json(output / 'primary.json') == read_json('evidence/canonical_paired_effects.json'),
          'Original primary statistics differ from the revision')
    print('Original primary results match all 5,760 paired episodes.', flush=True)
    check_released(output)
    check_trajectory_and_runtime()
    render()

    if not args.quick:
        run('analysis/original/analyze_v19_simplified_ood.py', *ood_arguments(),
            '--output', output / 'ood.json', log=output / 'ood.log')
        actual = read_json(output / 'ood.json')
        frozen = read_json('visuals/data/ood_paired_effects.json')
        check(actual['metrics'] == frozen['metrics'], 'Unseen-layout estimates differ')
        run('analysis/original/analyze_v19_paper_aligned_recheck.py',
            '--result-root', 'raw/original/density_noise', '--output-dir', output / 'density-noise')
        for name in ('paper_aligned_summary.csv', 'paper_aligned_paired_effects.csv'):
            check((output / 'density-noise' / name).read_bytes() == Path('evidence', name).read_bytes(),
                  f'Density or noise statistics differ in {name}')
        check_replay(output)
    print(f'Checks passed. Recomputed statistics are in {output}', flush=True)


if __name__ == '__main__':
    main()

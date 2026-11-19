"""Check the finite-episode safety cost results against the archived NPZ traces."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from zipfile import Path as ArchivePath, ZipFile

import numpy as np

from reproduce import ROOT, check, read_json, released_core
from supplementary import budget_traces


def check_statistics(actual, expected):
    if isinstance(expected, dict):
        check(actual.keys() == expected.keys(), 'Cost report fields differ')
        for key in expected:
            check_statistics(actual[key], expected[key])
    elif isinstance(expected, list):
        for value, reference in zip(actual, expected, strict=True):
            check_statistics(value, reference)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    else:
        check(actual == expected, f'Cost report value differs: {actual!r} vs {expected!r}')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('trace_archive', type=Path, help='released-cohort-step-cost-traces.zip from the data archive')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file for the recomputed estimates')
    args = parser.parse_args()

    supplementary = ROOT / 'analysis/supplementary'
    plan = read_json(supplementary / 'study_plan.json')
    missing = read_json(supplementary / 'missing-cases.json')
    with (ROOT / 'metadata/released-checkpoints.csv').open(newline='') as stream:
        checkpoint_hashes = {row['checkpoint']: row['sha256'] for row in csv.DictReader(stream)}

    with ZipFile(args.trace_archive) as archive:
        files = sorted(name for name in archive.namelist() if name.endswith('.npz'))
        check(len(files) == len(set(files)) == 1633, 'The complete archive must contain 1,633 distinct trace files')
        # Each member is one checkpoint and evaluation seed. The paper's cost summary uses these three variants.
        variants = ('F11_full_s', 'F01_no_gate_s', 'F10_no_Lcf_s')
        selected = [name for name in files if Path(name).stem.startswith(variants)]
        check(len(selected) == 535, 'Expected 535 recorded full/no-gate/no-counterfactual blocks')
        paths = (ArchivePath(archive, name) for name in selected)
        summaries, bounds = released_core.budget_rows(paths, plan, missing, budget_traces,
                                                     weight_hashes=checkpoint_hashes)

    check_statistics(summaries, read_json(ROOT / 'evidence/f11_budget_complete18.json')['results'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'full_policy': summaries, 'component_missing_case_bounds': bounds}, stream,
                  indent=2, allow_nan=False)
        stream.write('\n')
    print('All four reported F11 cost estimands match the reference.')


if __name__ == '__main__':
    main()

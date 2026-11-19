"""Relocate recorded scene and checkpoint manifests to a local Isaac Sim setup."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('training', help='Write a manifest for all training scenes')
    training.add_argument('output', type=Path)
    evaluation = commands.add_parser('evaluation', help='Write one recorded, ordered evaluation suite')
    evaluation.add_argument('job')
    evaluation.add_argument('checkpoint_root', type=Path)
    evaluation.add_argument('output', type=Path)
    args = parser.parse_args()

    if args.command == 'training':
        path = HERE / 'training_scenes_scenefix1/manifest.json'
        manifest = json.loads(path.read_bytes())
        for entry in manifest['scenes'].values():
            snapshot = path.parent / entry['snapshot']
            if digest(snapshot) != entry['sha256']:
                raise ValueError(f'Scene checksum differs: {snapshot}')
            entry['snapshot'] = str(snapshot)
        write_json(args.output, manifest)
        print(f'{digest(args.output)}  {args.output}')
        return

    jobs = json.loads((HERE / 'evaluation_jobs.json').read_bytes())
    job = next((entry for entry in jobs if entry['job'] == args.job), None)
    if job is None:
        parser.error(f'Unknown recorded evaluation job: {args.job}')
    for entry in job['entries']:
        checkpoint = args.checkpoint_root.resolve() / entry['checkpoint']
        if digest(checkpoint) != entry['checkpoint_sha256']:
            raise ValueError(f'Checkpoint checksum differs: {checkpoint}')
        entry['checkpoint'] = str(checkpoint)
    write_json(args.output, job['entries'])
    print(f'evaluation_seeds=[{job["evaluation_seed"]}]')


if __name__ == '__main__':
    main()

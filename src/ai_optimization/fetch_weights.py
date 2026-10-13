"""Restore fixed public checkpoints and verify their byte hashes.

Downloads occur only through this explicit command, never during cached inference.
Provider terms apply to the downloaded models.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
TABPFN = {
    'repository': 'Prior-Labs/TabPFN-v2-reg',
    'revision': '4972a65a1b30806315c6f92499959ffbfc69a673',
    'file': 'tabpfn-v2-regressor.ckpt',
    'bytes': 44390977,
    'sha256': '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736',
}


def specs(kind):
    if kind == 'tabpfn':
        return [dict(TABPFN, url=f"https://huggingface.co/{TABPFN['repository']}/resolve/{TABPFN['revision']}/{TABPFN['file']}")]
    manifest = ROOT / 'models/ai_optimization/design/checkpoint/source_manifest.json'
    return json.loads(manifest.read_text(encoding='utf-8'))['files']


def check(path, item):
    if path.stat().st_size != item['bytes']:
        raise ValueError(f"Size mismatch: {item['file']}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError(f"SHA256 mismatch: {item['file']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['hyenadna', 'tabpfn'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if not args.verify_only:
        if directory.exists():
            parser.error('Use a new directory; existing checkpoint directories are preserved.')
        directory.mkdir(parents=True)
    for item in specs(args.model):
        destination = directory / item['file']
        if not args.verify_only:
            with urllib.request.urlopen(item['url'], timeout=120) as source:
                with destination.open('xb') as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
        check(destination, item)
        print(json.dumps({'file': item['file'], 'sha256': item['sha256'], 'verified': True}))
    if args.model == 'hyenadna' and not args.verify_only:
        original = ROOT / 'models/ai_optimization/design/checkpoint/source_manifest.json'
        (directory / 'source_manifest.json').write_bytes(original.read_bytes())


if __name__ == '__main__':
    main()

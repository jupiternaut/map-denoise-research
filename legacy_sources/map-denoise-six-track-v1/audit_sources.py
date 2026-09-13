"""Freeze/read-only audit of research dependencies; no restoration or deletion."""
import argparse
import hashlib
import json
from pathlib import Path

OLD = Path('/home/grf/Documents/Codex/2026-09-10/map-denoise-v0')
PAPER = Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-paper-v1')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def snapshot():
    paths = [p for p in OLD.rglob('*') if p.is_file() and
             p.suffix in {'.py', '.md', '.cpp', '.json', '.tla', '.cfg'} and
             '__pycache__' not in p.parts]
    paths += [PAPER / 'main.tex', PAPER / 'output/pdf/main.pdf']
    return {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(paths) if p.exists()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'check'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        data = snapshot()
        with args.manifest.open('x') as f:
            json.dump(data, f, indent=2)
        print(json.dumps({'frozen_files': len(data), 'manifest': str(args.manifest)}))
    else:
        data = json.loads(args.manifest.read_text())
        mismatches = [name for name, item in data.items()
                      if not Path(name).is_file() or digest(Path(name)) != item['sha256']]
        print(json.dumps({'checked_files': len(data), 'mismatches': mismatches}))
        raise SystemExit(bool(mismatches))

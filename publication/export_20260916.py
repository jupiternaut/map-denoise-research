"""Curated, non-destructive export from the original liekkas workspaces.

This is a publication utility, not an experiment runner. --apply copies only
allowlisted artifacts, refuses different existing destinations, and records hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket

ROOT = Path(__file__).resolve().parents[1]
DOC = Path('/home/grf/Documents/Codex')
MAPPINGS = [
    (DOC / '2026-09-15' / name, Path('research_snapshots/2026-09-15') / name)
    for name in (
        'map-denoise-v25-surface-regeneration', 'e0',
        'e0-diagnostics-20260915T072044Z', 'CPR2_E0D_FORMAL_REPORT_20260915',
        'INITIAL_CONSTRUCTION.md', 'CONSTRUCTION_REVISION_1.md',
        'HISTORY_CHALLENGE.md', 'UNIFIED_GEOMETRY_META_RESEARCH_HANDOFF.md',
        'START_FRESH_FBALE5.md',
    )
] + [
    (DOC / '2026-09-16' / name, Path('research_snapshots/2026-09-16') / name)
    for name in ('e0-e-selective-revision-20260915T160016Z',
                 'e0-f-incumbent-support-20260916T060448Z',
                 'pnp-oracle-lab-20260916T112018Z')
] + [
    (Path('/home/grf/.hermes/attachments/outputs'), Path('meta_research/hermes/outputs')),
    (Path('/home/grf/Documents/meta-research-open-world-challenge'), Path('meta_research/open_world')),
] + [
    (Path('/home/grf/.hermes/attachments') / name, Path('meta_research/hermes') / name)
    for name in ('TASK.md', 'TASK.tla', 'AUDIT.md', 'AGENTS.md', 'AGENTS-2.md',
                 'experiment.py', 'FORMULATION.md', 'METHOD.md', 'RESULTS.json')
] + [
    (Path('/srv/slam-research/grf/map-denoise/runs') / name, Path('evidence/progress_20260916') / name)
    for name in ('field-budget-v24-uvg3jss3', 'loss-alignment-v23-9g6_i29x',
                 'real-support-bridge-v1-9de7jep3')
]
ALLOW = {'.md', '.py', '.json', '.jsonl', '.csv', '.tsv', '.txt', '.log',
         '.tla', '.cfg', '.sql', '.sh', '.toml', '.yaml', '.yml', '.png', '.svg'}
SKIP_DIRS = {'__pycache__', '.git', '.venv', 'venv', '.pytest_cache',
             'reproduction', 'qa', 'node_modules', 'states', 'downloads',
             'overlays', 'smoke'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    rows, excluded = [], []
    for source, dest in MAPPINGS:
        if not source.exists():
            raise FileNotFoundError(source)
        files = sorted(source.rglob('*')) if source.is_dir() else [source]
        for src in files:
            if not src.is_file():
                continue
            rel = src.relative_to(source) if source.is_dir() else Path(src.name)
            reason = None
            if src.is_symlink():
                reason = 'symlink: do not follow'
            elif set(rel.parts) & SKIP_DIRS:
                reason = 'cache, duplicate replay, raw download, or rendering QA'
            elif src.suffix.lower() not in ALLOW:
                reason = 'not allowlisted: binary arrays, point clouds, archives, third-party binaries'
            elif src.name.startswith('.env') or any(s in src.name.lower() for s in ('credentials', 'private_key')):
                reason = 'sensitive filename'
            elif src.stat().st_size > 15 * 1024 * 1024:
                reason = 'large file: retain locally'
            if reason:
                excluded.append({'source': str(src), 'reason': reason, 'bytes': src.stat().st_size})
                continue
            target = dest / rel if source.is_dir() else dest
            if target.name in {'AGENTS.md', 'AGENT.md'}:
                target = target.with_name('SOURCE_' + target.name)
            rows.append({'source': str(src), 'path': target.as_posix(),
                         'bytes': src.stat().st_size, 'sha256': digest(src)})
    return rows, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('This source-export utility is only for liekkas.')
    rows, excluded = collect()
    if len({r['path'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate destinations')
    if args.apply:
        for row in rows:
            target = ROOT / row['path']
            if target.exists() and digest(target) != row['sha256']:
                raise RuntimeError(f'Refusing to overwrite {target}')
        for row in rows:
            target = ROOT / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(row['source'], target)
            assert digest(target) == digest(Path(row['source'])) == row['sha256']
        manifest = {'date': '2026-09-16', 'host': 'liekkas',
                    'scope': 'curated publication, not a complete disk backup',
                    'historical_agent_files': 'renamed SOURCE_AGENTS/AGENT: evidence, not active instructions',
                    'files': rows, 'excluded': excluded}
        (ROOT / 'publication/SNAPSHOT_20260916.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'applied': args.apply, 'files': len(rows),
                      'bytes': sum(r['bytes'] for r in rows), 'excluded': len(excluded),
                      'largest': sorted(rows, key=lambda r: r['bytes'], reverse=True)[:8]}, indent=2))


if __name__ == '__main__':
    main()

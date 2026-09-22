"""Maintainer-only export from pinned local Git objects; not a Windows setup command."""
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
COMMIT = '2cf73a80f5e41a5d126ebdc8321c0cd6c61f24f9'
PREFIX = 'research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/'
EXTRA = {'tests/test_runtime.py', 'scene_adapter.py', 'EXPERIMENT_REPORT.md',
         'EXPERIMENT_PROTOCOL.md', 'EXECUTION_NOTE.md', 'EXECUTION_CHECKPOINT.md',
         'TRAINING_LOCK.json', 'REPRODUCIBILITY.md', 'EXECUTION_AUDIT.json'}

def put_identical_or_new(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite different content: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    assert socket.gethostname() == 'liekkas'
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', COMMIT]).decode().strip() == COMMIT
    paths = subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-rz', '--name-only', COMMIT, '--', PREFIX]).decode().split('\0')
    provenance = []
    for path in filter(None, paths):
        rel = path.removeprefix(PREFIX)
        if not (rel.startswith('package/') or rel in EXTRA):
            continue
        data = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{path}'])
        dst = ROOT / 'reference' / rel
        put_identical_or_new(dst, data)
        provenance.append({'path': str(dst.relative_to(ROOT)), 'git_path': path,
                           'sha256': digest(data), 'bytes': len(data)})
    origin = {'repository': 'https://github.com/jupiternaut/map-denoise-research',
              'commit': COMMIT, 'export': 'exact Git blob bytes; no source rewriting',
              'files': provenance}
    put_identical_or_new(ROOT/'SOURCE_MANIFEST.json', (json.dumps(origin, ensure_ascii=False, indent=2)+'\n').encode())
    excluded = {'BUNDLE_MANIFEST.json', 'PACKAGING_VALIDATION.json'}
    rows = []
    patterns = [rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{40,}',
                rb'https?://[^\s/@:]+:[^\s/@]+@']
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.name in excluded:
            continue
        if p.is_symlink() or '__pycache__' in p.parts:
            raise RuntimeError('Unexpected link/cache: '+str(p))
        b = p.read_bytes()
        if any(re.search(pattern, b) for pattern in patterns):
            raise RuntimeError('Credential-pattern review required: '+str(p.relative_to(ROOT)))
        rows.append({'path': str(p.relative_to(ROOT)), 'bytes': len(b), 'sha256': digest(b)})
    manifest = {'manifest_version': 1, 'commit': COMMIT,
                'excluded_self_and_packaging_report': sorted(excluded), 'files': rows}
    put_identical_or_new(ROOT/'BUNDLE_MANIFEST.json', (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps({'reference_files': len(provenance), 'manifest_files': len(rows),
                      'bytes': sum(r['bytes'] for r in rows), 'secrets_pattern_scan': 'no matches'}))

if __name__ == '__main__':
    main()

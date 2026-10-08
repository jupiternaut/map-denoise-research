"""One-time byte-preserving archive; synthetic arrays explicitly included.

This exporter is tied to the verified source host. The companion verifier is
portable and does not depend on that host. Historical sources are never edited.
"""
import hashlib
import json
import re
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/srv/slam-research/grf/map-denoise')
SOURCES = [
    BASE / 'runs/surface-owned-support-20261008T022918Z',
    BASE / 'runs/footprint-support-20261008T025757Z',
    BASE / 'runs/mixed-pixel-20261008T041249Z',
    BASE / 'plans/mixed-pixel-study-20261008T035954Z',
]
MANIFEST = REPO / 'publication/MIXED_PIXEL_20261008_MANIFEST.json'
TEXT = {'.md', '.py', '.json', '.jsonl', '.csv', '.txt', '.sha256', '.log'}
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})'),
    'api_secret': re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}'),
    'credential_url': re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),
    'secret_assignment': re.compile(r'(?i)(?:api_key|access_token|auth_token|password)\s*[=:]\s*[\"\x27][A-Za-z0-9_/-]{24,}[\"\x27]'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname() == 'liekkas'
    assert REPO == Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
    assert not MANIFEST.exists(), 'Already exported; verify without overwriting'
    planned, excluded, hits = [], [], []
    for source in SOURCES:
        destination = REPO / 'research_snapshots/2026-10-08' / source.name
        assert source.is_dir() and not destination.exists(), str(source)
        for path in sorted(source.rglob('*')):
            if not path.is_file() and not path.is_symlink():
                continue
            rel = path.relative_to(source)
            reason = None
            if path.is_symlink():
                reason = 'symlink not followed'
            elif '.aris' in rel.parts:
                reason = 'private reviewer prompts and traces; formal audit retained'
            elif '__pycache__' in rel.parts or '.pytest_cache' in rel.parts or path.suffix == '.pyc':
                reason = 'runtime cache'
            elif rel.as_posix() == 'audit/response.txt':
                reason = 'raw reviewer response; public structured audit retained'
            elif path.suffix == '.npz':
                allowed = (
                    (source.name.startswith('surface-owned-') and rel.parts[0] == 'mechanism')
                    or (source.name.startswith('footprint-') and rel.parts[0] in {'observed', 'oracle_inputs', 'curves'})
                    or (source.name.startswith('mixed-pixel-2026') and rel.parts[0] in {'data', 'ordinary_stage', 'oracle_stage', 'diagnostics'})
                )
                assert allowed, 'Unreviewed array location: ' + str(path)
            elif path.suffix not in TEXT | {'.png'}:
                reason = 'file type outside declared publication subset'
            if reason:
                excluded.append(dict(source=str(path), bytes=path.lstat().st_size, reason=reason))
                continue
            assert path.stat().st_size < 90_000_000, 'Review large file: ' + str(path)
            if path.suffix in TEXT:
                content = path.read_text()
                for kind, pattern in PATTERNS.items():
                    for match in pattern.finditer(content):
                        hits.append(dict(source=str(path), kind=kind, line=content[:match.start()].count('\n') + 1))
            published = rel.with_name('SOURCE_AGENTS.md') if rel.name == 'AGENTS.md' else rel
            target = destination / published
            planned.append((path, target, digest(path), path.stat().st_size))
    if hits:
        print(json.dumps(dict(secret_pattern_locations=hits), indent=2))
        raise SystemExit('Review detections before publishing (values not printed)')
    files = []
    for source, target, sha, size in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copy2(source, target)
        assert digest(source) == digest(target) == sha
        files.append(dict(path=target.relative_to(REPO).as_posix(), source=str(source),
                          bytes=size, sha256=sha,
                          transformation='filename only: archived instructions' if source.name == 'AGENTS.md' else 'none'))
    payload = dict(created_at=datetime.now(timezone.utc).isoformat(), source_host='liekkas',
        target='https://github.com/jupiternaut/map-denoise-research', branch='main',
        archive_roots=['research_snapshots/2026-10-08/' + x.name for x in SOURCES],
        files=files, excluded=excluded, credential_pattern_detections=0,
        scope='Three frozen research runs and their mixed-pixel plan, including synthetic inputs, curves and pixel diagnostics; not a raw real-data backup.',
        source_files_modified=False, new_research_executed=False,
        portability='Historical absolute paths retained as evidence. Companion verifier resolves source paths through this manifest; original runners still require path adaptation.')
    with MANIFEST.open('x') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(files=len(files), bytes=sum(x['bytes'] for x in files),
                         arrays=sum(x['path'].endswith('.npz') for x in files),
                         excluded=len(excluded)), indent=2))


if __name__ == '__main__':
    main()

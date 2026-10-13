"""Export the frozen Oct 10--13 LOVELACE chain without modifying sources.

Large textual evidence is gzip-compressed losslessly; scientific source bytes,
failed attempts and historical seals are preserved. This is a host-specific
exporter, not a portable scientific runner. See the companion verifier.
"""
import gzip
import hashlib
import json
import re
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/srv/slam-research/grf/map-denoise/runs')
NAMES = (
    'lovelace-challenge-20261010T205013Z',
    'lovelace-hybrid-20261010T214217Z',
    'lovelace-nuisance-20261011T044500Z',
    'lovelace-shared-appearance-20261011T052300Z',
    'lovelace-shared-ablation-20261011T055200Z',
    'lovelace-view-bias-20261011T092300Z',
    'lovelace-observation-bridge-20261011T232502Z',
    'lovelace-structured-view-bias-20261012T022500Z',
    'lovelace-structured-bias-confirmation-20261012T035916Z',
    'lovelace-baseline-intervention-20261012T045547Z',
)
ARCHIVE = 'research_snapshots/2026-10-13'
MANIFEST = REPO/'publication/LOVELACE_20261013_MANIFEST.json'
TEXT = {'.py', '.md', '.json', '.log', '.txt', '.csv'}
PATTERNS = {
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})'),
    'api_secret': re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}'),
    'credential_url': re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
    'secret_assignment': re.compile(rb'(?i)(?:api_key|access_token|auth_token|password)\s*[=:]\s*[\"\x27][A-Za-z0-9_/-]{24,}[\"\x27]'),
}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert socket.gethostname() == 'liekkas'
    assert REPO == Path('/srv/slam-research/grf/map-denoise/publications/github-20261013')
    assert not MANIFEST.exists() and not (REPO/ARCHIVE).exists()
    planned, excluded, hits = [], [], []
    for name in NAMES:
        root = SOURCE/name
        assert root.is_dir() and not root.is_symlink()
        for path in sorted(root.rglob('*')):
            if not path.is_file() and not path.is_symlink():
                continue
            rel = path.relative_to(SOURCE)
            reason = None
            if path.is_symlink():
                raise RuntimeError('Unreviewed symlink: '+str(path))
            if '.aris' in rel.parts:
                reason = 'private agent review traces and local review gates'
            elif any(p in rel.parts for p in ('__pycache__', '.pytest_cache')) or path.suffix == '.pyc':
                reason = 'runtime cache'
            if reason:
                excluded.append(dict(source_relative=rel.as_posix(), reason=reason,
                                     bytes=path.stat().st_size, sha256=digest(path)))
                continue
            assert path.suffix in TEXT | {'.npz'}, 'Unreviewed type: '+str(path)
            # These arrays are the small synthetic test/confirmation fixtures.
            if path.suffix == '.npz':
                assert name in NAMES[:2] and rel.parts[1:3] == ('outputs', 'confirmation-v1') and path.name.startswith('forecasts-')
            raw = path.read_bytes()
            if path.suffix in TEXT:
                for kind, pattern in PATTERNS.items():
                    for match in pattern.finditer(raw):
                        hits.append(dict(path=rel.as_posix(), kind=kind,
                                         line=raw[:match.start()].count(b'\n')+1))
            published = rel.with_name('SOURCE_AGENTS.md') if rel.name == 'AGENTS.md' else rel
            encoding = 'gzip' if path.suffix in {'.json', '.csv', '.log'} and len(raw)>1048576 else 'identity'
            if encoding == 'gzip':
                published = Path(str(published)+'.gz')
            planned.append(dict(source=str(path), source_relative=rel.as_posix(),
                                path=(Path(ARCHIVE)/published).as_posix(),
                                source_sha256=hashlib.sha256(raw).hexdigest(),
                                source_bytes=len(raw), encoding=encoding,
                                renamed_instructions=rel.name == 'AGENTS.md'))
    if hits:
        print(json.dumps({'credential_pattern_locations':hits}, indent=2))
        raise SystemExit('Review detections before publication; no values printed')
    assert len({p['path'] for p in planned}) == len(planned)
    # All sources have been inventoried and scanned before any copy is written.
    for record in planned:
        src, dst = Path(record['source']), REPO/record['path']
        assert digest(src) == record['source_sha256'], 'Source changed during export'
        dst.parent.mkdir(parents=True, exist_ok=True)
        if record['encoding'] == 'gzip':
            with dst.open('xb') as out, gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as compressed, src.open('rb') as inp:
                shutil.copyfileobj(inp, compressed)
            with gzip.open(dst, 'rb') as inp:
                assert hashlib.file_digest(inp, 'sha256').hexdigest() == record['source_sha256']
        else:
            assert not dst.exists()
            shutil.copy2(src, dst)
            assert digest(dst) == record['source_sha256']
        record['sha256'], record['bytes'] = digest(dst), dst.stat().st_size
        assert record['bytes'] < 90_000_000, 'Review large published file'
    for record in planned:
        assert digest(Path(record['source'])) == record['source_sha256']
    payload = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_host='liekkas', source_root=str(SOURCE),
        target='https://github.com/jupiternaut/map-denoise-research', branch='main',
        base_commit='bd8d05aad65e2c55cd9cd288beae4ef949606984', archive_root=ARCHIVE,
        source_roots=list(NAMES), files=planned, excluded=excluded,
        credential_pattern_detections=0, source_files_modified=False,
        new_research_executed=False,
        scope='Ten frozen run roots, including nested A1-A10, synthetic inputs/truth, code, protocols, failed attempts, results and public proof records. Not real-data or environment backup.',
        portability='Original paths and seals preserved. Private .aris excluded. Original runners are not portable unchanged; use companion publication verifier for read-only hash and A10 arithmetic checks.')
    with MANIFEST.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(files=len(planned), source_bytes=sum(p['source_bytes'] for p in planned),
                         published_bytes=sum(p['bytes'] for p in planned), excluded=len(excluded),
                         compressed=sum(p['encoding']=='gzip' for p in planned)), indent=2))


if __name__ == '__main__':
    main()

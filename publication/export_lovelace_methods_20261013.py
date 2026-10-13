"""One-shot, host-specific export of frozen A11--A14; never edits source runs.

The previous publication supplies credential patterns and the file hash helper.
Only the archive copies are renamed/compressed. Use the separate portable
verifier to inspect a downloaded repository without this host's source paths.
"""
import gzip
import hashlib
import json
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from export_lovelace_20261013 import PATTERNS, digest

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/srv/slam-research/grf/map-denoise/runs')
NAMES = (
    'lovelace-evidence-accumulation-20261013T051931Z',
    'lovelace-representation-search-20261013T080031Z',
    'lovelace-certified-boundary-20261013T093552Z',
    'lovelace-joint-boundary-20261013T123336Z',
)
ARCHIVE = 'research_snapshots/2026-10-13-method-search'
MANIFEST = REPO/'publication/LOVELACE_METHODS_20261013_MANIFEST.json'
TEXT = {'.py', '.md', '.json', '.log', '.txt', '.csv'}


def main():
    assert socket.gethostname() == 'liekkas'
    assert REPO == Path('/srv/slam-research/grf/map-denoise/publications/github-20261013')
    remote = subprocess.check_output(['git', 'remote', 'get-url', '--push', 'origin'], cwd=REPO, text=True).strip()
    assert remote == 'https://github.com/jupiternaut/map-denoise-research.git'
    assert not MANIFEST.exists() and not (REPO/ARCHIVE).exists(), 'Never overwrite a prior export'
    planned, excluded, hits = [], [], []
    for name in NAMES:
        root = SOURCE/name
        assert root.is_dir() and not root.is_symlink()
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise RuntimeError('Unreviewed symlink: '+str(path))
            if not path.is_file():
                continue
            rel = path.relative_to(SOURCE)
            reason = None
            if '.aris' in rel.parts:
                reason = 'private agent review traces and local review gates'
            elif any(p in rel.parts for p in ('__pycache__', '.pytest_cache', '.cache')) or path.suffix == '.pyc':
                reason = 'runtime cache'
            if reason:
                excluded.append(dict(source_relative=rel.as_posix(), reason=reason,
                                     bytes=path.stat().st_size, sha256=digest(path)))
                continue
            assert path.suffix in TEXT, 'Unreviewed type: '+str(path)
            raw = path.read_bytes()
            raw.decode('utf-8')
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
        raise SystemExit('Review detections before publication; no secret values printed')
    assert len({p['path'] for p in planned}) == len(planned)
    # Inventory and scan the entire allowlist before making any publication copy.
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
        assert record['bytes'] < 90_000_000, 'Review oversized publication'
    for record in planned:
        assert digest(Path(record['source'])) == record['source_sha256']
    payload = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_host='liekkas', source_root=str(SOURCE),
        target=remote.removesuffix('.git'), branch='main',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        archive_root=ARCHIVE, source_roots=list(NAMES), files=planned, excluded=excluded,
        credential_pattern_detections=0, source_files_modified=False,
        new_research_executed=False,
        scope='Four frozen A11-A14 roots: scientific code, synthetic inputs and evaluation truth, protocols, saved predictions, public certificates, failures and corrections. No real data/environment backup or private review traces.',
        portability='Original paths and historical seals preserved. Historical AGENTS renamed SOURCE_AGENTS. Large text losslessly gzipped. Host-specific original runners are not portable unchanged. Companion verifier reads publication bytes only; it checks archive integrity and A14 arithmetic, not all original proof kernels.',
        historical_gaps=['A12 REPORT and CHECKPOINT absent; original tracker stale; formal records and EXPERIMENT_AUDIT preserved without retrospectively rewriting them.',
                         'A14 pre-run SOURCE_FREEZE omitted 11 ancillary dynamic generation-verification dependencies. Original seal preserved; this later archive manifest is not a retroactive pre-run seal.'])
    with MANIFEST.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(files=len(planned), source_bytes=sum(p['source_bytes'] for p in planned),
                         published_bytes=sum(p['bytes'] for p in planned), excluded=len(excluded),
                         compressed=sum(p['encoding']=='gzip' for p in planned)), indent=2))


if __name__ == '__main__':
    main()

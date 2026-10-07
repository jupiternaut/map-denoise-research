"""One-time, byte-preserving publication export from the two frozen local runs."""
import hashlib
import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/srv/slam-research/grf/map-denoise/runs')
RUNS = ('track-discrimination-20261007T160716Z', 'selector-attribution-20261007T180539Z')
ALLOWED = {'.md', '.py', '.json', '.jsonl', '.csv', '.txt', '.sha256', '.png'}
MANIFEST = REPO / 'publication/SELECTOR_ATTRIBUTION_20261008_MANIFEST.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname() == 'liekkas'
    assert REPO == Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
    assert not MANIFEST.exists(), 'Publication already exported; verify instead of overwriting'
    files, excluded = [], []
    for run in RUNS:
        source = SOURCE / run
        destination = REPO / 'research_snapshots/2026-10-07' / run
        assert source.is_dir() and not destination.exists()
        for path in sorted(source.rglob('*')):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = path.relative_to(source)
            reason = None
            if path.is_symlink(): reason = 'symlink: not followed'
            elif '.aris' in relative.parts: reason = 'private reviewer prompts, responses and traces'
            elif '__pycache__' in relative.parts or path.suffix == '.pyc': reason = 'bytecode cache'
            elif path.suffix.lower() not in ALLOWED: reason = 'array/cache/raw binary not in text+research-image publication subset'
            if reason:
                excluded.append(dict(source=str(path), bytes=path.stat().st_size, reason=reason))
                continue
            published_relative = relative.with_name('SOURCE_AGENTS.md') if relative.name == 'AGENTS.md' else relative
            target = destination / published_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists()
            before = digest(path)
            shutil.copy2(path, target)
            assert digest(target) == before == digest(path)
            files.append(dict(path=str(target.relative_to(REPO)), source=str(path), bytes=path.stat().st_size,
                              sha256=before, transformation='filename only: archived task instructions' if relative.name=='AGENTS.md' else 'none'))
    data = dict(created_at=datetime.now(timezone.utc).isoformat(), source_host='liekkas',
        target='https://github.com/jupiternaut/map-denoise-research', branch='main',
        archive_roots=['research_snapshots/2026-10-07/'+r for r in RUNS],
        files=files, excluded=excluded,
        scope='Frozen source and public research evidence subset; private traces, arrays and environments excluded. Not a complete dataset backup.',
        source_files_modified=False, new_research_executed=False)
    with MANIFEST.open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(files=len(files), bytes=sum(f['bytes'] for f in files), excluded=len(excluded)),indent=2))


if __name__ == '__main__': main()

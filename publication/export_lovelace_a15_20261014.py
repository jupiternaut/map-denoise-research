"""Export A15 original evidence; copies alone are renamed/compressed."""
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
NAMES = ('lovelace-proof-reuse-20261014T074405Z',)
ARCHIVE = 'research_snapshots/2026-10-14-proof-reuse'
MANIFEST = REPO/'publication/LOVELACE_A15_20261014_MANIFEST.json'
TEXT = {'.py', '.md', '.json', '.log', '.txt', '.csv'}

def main():
    assert socket.gethostname() == 'liekkas'
    assert REPO == Path('/srv/slam-research/grf/map-denoise/publications/github-20261013')
    remote = subprocess.check_output(['git','remote','get-url','--push','origin'],cwd=REPO,text=True).strip()
    assert remote == 'https://github.com/jupiternaut/map-denoise-research.git'
    assert not MANIFEST.exists() and not (REPO/ARCHIVE).exists()
    root = SOURCE/NAMES[0]
    assert root.is_dir() and not root.is_symlink()
    # Read-only check of both experiment freezes and the final source inventory.
    seals = {}
    for name in ('FREEZE_tolerance.json','FREEZE_switches.json','FREEZE_supplement.json'):
        data=json.loads((root/name).read_text())
        assert all(digest(Path(p)) == h for p,h in data['files'].items()), name
        seals[name]=digest(root/name)
    original=json.loads((root/'ARTIFACTS.json').read_text())
    for rel, record in original['files'].items():
        assert digest(root/rel)==record['sha256'] and (root/rel).stat().st_size==record['bytes'],rel
    planned,excluded,hits=[],[],[]
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise RuntimeError('unreviewed symlink: '+str(path))
        if not path.is_file():continue
        rel=path.relative_to(SOURCE)
        reason=None
        if '.aris' in rel.parts:reason='private agent review traces and metadata'
        elif any(x in rel.parts for x in ('__pycache__','.pytest_cache','.cache')) or path.suffix=='.pyc':reason='runtime cache'
        if reason:
            excluded.append(dict(source_relative=rel.as_posix(),reason=reason,bytes=path.stat().st_size,sha256=digest(path)))
            continue
        assert path.suffix in TEXT,'unreviewed type: '+str(path)
        raw=path.read_bytes();raw.decode('utf-8')
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(raw):
                hits.append(dict(path=rel.as_posix(),kind=kind,line=raw[:match.start()].count(b'\n')+1))
        published=rel.with_name('SOURCE_AGENTS.md') if rel.name=='AGENTS.md' else rel
        encoding='gzip' if path.suffix in {'.json','.csv','.log'} and len(raw)>1048576 else 'identity'
        if encoding=='gzip':published=Path(str(published)+'.gz')
        planned.append(dict(source=str(path),source_relative=rel.as_posix(),path=(Path(ARCHIVE)/published).as_posix(),
            source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),encoding=encoding,
            renamed_instructions=rel.name=='AGENTS.md'))
    if hits:
        print(json.dumps({'credential_pattern_locations':hits},indent=2))
        raise SystemExit('Review credential detections; secret values omitted')
    assert len({p['path'] for p in planned})==len(planned)
    for record in planned:
        src,dst=Path(record['source']),REPO/record['path']
        assert digest(src)==record['source_sha256']
        dst.parent.mkdir(parents=True,exist_ok=True)
        if record['encoding']=='gzip':
            with dst.open('xb') as output,gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as archive,src.open('rb') as input_file:
                shutil.copyfileobj(input_file,archive)
            with gzip.open(dst,'rb') as decoded:assert hashlib.file_digest(decoded,'sha256').hexdigest()==record['source_sha256']
        else:
            assert not dst.exists();shutil.copy2(src,dst)
        record.update(sha256=digest(dst),bytes=dst.stat().st_size)
        assert record['bytes']<90_000_000
    assert all(digest(Path(p['source']))==p['source_sha256'] for p in planned)
    payload=dict(created_at=datetime.now(timezone.utc).isoformat(),source_host='liekkas',source_root=str(SOURCE),
        target=remote.removesuffix('.git'),branch='main',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        archive_root=ARCHIVE,source_roots=list(NAMES),files=planned,excluded=excluded,
        credential_pattern_detections=0,source_files_modified=False,new_research_executed=False,source_seals=seals,
        previous_evidence_manifest='publication/LOVELACE_METHODS_20261013_MANIFEST.json',
        previous_evidence_manifest_sha256=digest(REPO/'publication/LOVELACE_METHODS_20261013_MANIFEST.json'),
        scope='A15 complete original run: controlled A14 replay, code, protocols, raw jobs/results/receipts, exact boundary witnesses, reports, public audit and final inventory. Private review traces excluded.',
        portability='Source absolute paths and historical seals retained. SOURCE_AGENTS is archival data. Two large JSON records losslessly gzipped. Portable verifier reads publication bytes only; original scientific reruns still require path/environment adaptation.',
        historical_gaps=['Boundary diagnostic implementation first appears in supplemental seal after diagnostic execution. Matching runtime hash and independent exact witness checks retained; do not claim prospective implementation freeze.',
            'Cost ledger is partial child/development CPU; verification, parent orchestration and audit costs have separate or incomplete accounting.',
            '18 exposed simulation worlds, not new confirmation; three repeats measure timing only. Same-family semantic audit remains provisional.'])
    with MANIFEST.open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(files=len(planned),source_bytes=sum(p['source_bytes'] for p in planned),
        published_bytes=sum(p['bytes'] for p in planned),excluded=len(excluded),compressed=sum(p['encoding']=='gzip' for p in planned)),indent=2))

if __name__=='__main__':main()

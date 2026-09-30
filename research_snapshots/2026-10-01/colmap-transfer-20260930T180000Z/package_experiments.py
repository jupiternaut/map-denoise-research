"""Create explicit public experimental subsets without changing source runs."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
RUNS=[ROOT.parent/'camera-pairing-replay-20260930T162130Z',
      ROOT.parent/'colmap-fixed-pixels-20260930T172250Z',ROOT]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    records=[];exclusions=[]
    for source in RUNS:
        dest=REPO/'research_snapshots/2026-10-01'/source.name
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            r=p.relative_to(source)
            if r.name in ('EXPERIMENT_PUBLICATION_EXPECTED.json','GIT_PUBLICATION_VERIFICATION.json'):
                exclusions.append(dict(source=str(p),reason='external publication verification record; avoid self-referential snapshot'));continue
            if any(x in r.parts for x in ['.aris','__pycache__']) or r.parts[0] in ('inputs','workspaces','inference') or (r.parts[0]=='references' and r.name!='MANIFEST.json'):
                exclusions.append(dict(source=str(p),reason='internal traces/cache or source data/dense workspace; see source manifests'));continue
            if r.parts[0] in ('predictions','initialization') and p.suffix not in ('.npz','.json','.md','.csv'):
                exclusions.append(dict(source=str(p),reason='outside explicit prediction/initialization publication types'));continue
            if p.suffix not in ('.md','.json','.csv','.py','.txt','.png','.pdf','.npz','.log') or p.stat().st_size>20*1024**2:
                exclusions.append(dict(source=str(p),reason='not lightweight evidence type or >20MiB; no silent truncation'));continue
            target=dest/r;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
            records.append(dict(path=str(target.relative_to(REPO)),source=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    # Bind every external executable import needed by the transfer protocol as a source artifact.
    lock=json.loads((ROOT/'RUN_LOCK.json').read_text())
    for filename,expected in lock['dependencies'].items():
        p=Path(filename);assert sha(p)==expected
        target=REPO/'research_snapshots/2026-10-01'/ROOT.name/'dependencies'/p.parent.name/p.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        records.append(dict(path=str(target.relative_to(REPO)),source=str(p),bytes=p.stat().st_size,sha256=expected))
    for r in records:assert sha(REPO/r['path'])==r['sha256']
    manifest=dict(scope='Full selected code, protocols, tables, lightweight fixed predictions and figures; NOT all dataset/dense workspace bytes',
      records=records,exclusions=exclusions,source_runs=[str(p) for p in RUNS],
      retained_absolute_paths='Historical exact-host scripts retain original paths; dependency copies aid inspection, not automatic portability',
      old_seal_semantics='Original source seals intentionally include omitted dense inputs; public completeness is defined only by this manifest')
    target=REPO/'publication/TRANSFER_20261001_MANIFEST.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    expected={'files':[{'path':r['path'],'sha256':r['sha256']} for r in records]+[{'path':str(target.relative_to(REPO)),'sha256':sha(target)}]}
    (ROOT/'EXPERIMENT_PUBLICATION_EXPECTED.json').write_text(json.dumps(expected,indent=2)+'\n')
    print('PACKAGED',len(records),'files',sum(r['bytes'] for r in records),'bytes',len(exclusions),'explicit exclusions')
if __name__=='__main__':main()

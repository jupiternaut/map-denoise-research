"""Index completed A11 artifacts without modifying any scientific output."""
from datetime import datetime,timezone
from a11_io_contract import *
def write_exclusive(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
    target();require(load(OUT/'STAGE_STATUS.json')['status']=='COMPLETE','formal incomplete')
    require(load(OUT/'verification/VERIFY.json')['status']=='PASS','proof incomplete')
    require(sources()==load(OUT/'SOURCE_LOCK.json')['files'] and inputs()==load(OUT/'INPUT_PRELOCK.json')['files'],'frozen bytes drift')
    require(all((ROOT/name).is_file() for name in ('REPORT.md','CHECKPOINT.md','EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json')),'handoff incomplete')
    excluded={'MANIFEST.md','ARTIFACTS.json','FINAL_SEAL.json'}
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '.aris' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.name not in excluded)
    lines=['# Research Output Manifest','',
        '同一A11工作区的公开科学工件；`.aris`原始审计轨迹保持本地私有，不计入公开清单。',
        '', '| Timestamp (UTC) | Skill | File | Stage | Description |',
        '|---|---|---|---|---|']
    for p in files:
        stamp=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()
        rel=p.relative_to(ROOT).as_posix()
        skill='/experiment-audit' if ('AUDIT' in rel or 'verification/' in rel) else '/goal-first-reporting' if p.name in ('REPORT.md','CHECKPOINT.md') else '/experiment-plan'
        lines.append(f'| {stamp} | {skill} | {rel} | implementation | A11保留工件 |')
    with (ROOT/'MANIFEST.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    files.append(ROOT/'MANIFEST.md')
    index={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    write_exclusive(ROOT/'ARTIFACTS.json',dict(stage='A11',files=dict(sorted(index.items()))))
    write_exclusive(ROOT/'FINAL_SEAL.json',dict(stage='A11',host='liekkas',logical_root=str(LOGICAL_ROOT),physical_root=str(ROOT),
        status='COMPLETE',verification='PASS',artifact_index_sha256=sha(ROOT/'ARTIFACTS.json'),indexed_files=len(index),
        source_lock_sha256=sha(OUT/'SOURCE_LOCK.json'),report_sha256=sha(ROOT/'REPORT.md'),
        semantic_review=load(ROOT/'EXPERIMENT_AUDIT.json')['verdict'],private_raw_traces_excluded=True))
    print(json.dumps(load(ROOT/'FINAL_SEAL.json')),flush=True)
if __name__=='__main__':main()

"""Archive completed A10 local artifacts after reporting and arithmetic review."""
from datetime import datetime
import json
from pathlib import Path
from io_contract import *
def main():
    target();validate_old()
    require(load(OUT/'STAGE_STATUS.json')['status']=='COMPLETE','formal incomplete')
    require(load(OUT/'verification/VERIFY.json')['status']=='PASS','verification gate')
    require(load(ROOT/'.aris/ARITHMETIC_GATE.json')['status']=='PASS','arithmetic gate')
    require(sources()==load(OUT/'SOURCE_LOCK.json')['files'] and inputs()==load(OUT/'INPUT_PRELOCK.json')['files'],'frozen bytes')
    require(not (ROOT/'ARTIFACTS.json').exists() and not (ROOT/'FINAL_SEAL.json').exists(),'seal already exists')
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    require(not any(p.suffix=='.pending' for p in paths),'pending output remains')
    stamp=datetime.now().astimezone().isoformat()
    manifest='# Research Output Manifest\n\n| Timestamp | Skill | File | Stage | Description |\n|-----------|-------|------|-------|-------------|\n'
    for p in paths:
        rel=str(p.relative_to(ROOT));stage='review' if rel.startswith('.aris') else 'implementation'
        desc='私有审查记录，不发布' if rel.startswith('.aris') else ('正式运行产物' if rel.startswith('outputs/') else 'A10源码、计划或分析')
        manifest+=f'| {stamp} | experiment-bridge / analyze-results | {rel} | {stage} | {desc} |\n'
    manifest+=f'| {stamp} | experiment-bridge | MANIFEST.md | implementation | 本索引 |\n'
    manifest+=f'| {stamp} | experiment-bridge | ARTIFACTS.json | implementation | 归档摘要索引 |\n'
    manifest+=f'| {stamp} | experiment-bridge | FINAL_SEAL.json | implementation | 最终本地封存 |\n'
    (ROOT/'MANIFEST.md').write_text(manifest)
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    (ROOT/'ARTIFACTS.json').write_text(json.dumps({'stage':'A10','files':files},indent=2,ensure_ascii=False)+'\n')
    seal={'stage':'A10','host':'liekkas','root':str(ROOT),'formal_status':'COMPLETE','verification':'PASS','arithmetic_review':'PASS',
        'artifact_index_sha256':sha(ROOT/'ARTIFACTS.json'),'indexed_files':len(files),'total_files_including_self':len(files)+2,
        'source_lock_sha256':sha(OUT/'SOURCE_LOCK.json'),'report_sha256':sha(ROOT/'REPORT.md'),
        'analysis_sha256':sha(ROOT/'ANALYSIS.json'),'verified_locked_sources':len(sources()),'verified_prelocked_inputs':len(inputs()),
        'assurance':'same-family provisional; exact certificates under fixed synthetic model','sealed_at':stamp}
    (ROOT/'FINAL_SEAL.json').write_text(json.dumps(seal,indent=2,ensure_ascii=False)+'\n')
    require(all(sha(ROOT/p)==h for p,h in files.items()),'artifact readback');print(json.dumps(seal,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

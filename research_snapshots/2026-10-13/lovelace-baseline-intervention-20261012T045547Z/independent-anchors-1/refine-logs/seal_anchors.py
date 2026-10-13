"""Inventory new A8 artifacts only, after formal PASS and final report."""
from datetime import datetime,timezone,timedelta
import hashlib,json,socket
from pathlib import Path
R=Path(__file__).resolve().parent.parent;O=R/'outputs/anchor-v1'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
    if socket.gethostname()!='liekkas' or R.name!='independent-anchors-1':raise RuntimeError('wrongtarget')
    if load(O/'STAGE_STATUS.json')['status']!='COMPLETE' or load(O/'verification/VERIFY.json')['status']!='PASS':raise RuntimeError('formal stage not accepted')
    for lock in ('SOURCE_LOCK.json','INPUTS.json'):
        for path,digest in load(O/lock)['files'].items():
            if sha(Path(path))!=digest:raise RuntimeError('frozen bytes drift '+path)
    for oldname in ('point-feasibility-1','common-pose-feasibility-1'):
        old=R.parent/oldname
        for relative,digest in load(old/'ARTIFACTS.json')['files'].items():
            if sha(old/relative)!=digest:raise RuntimeError('old archive changed '+str(old/relative))
    for p in ('REPORT.md','CHECKPOINT.md','ANALYSIS.json','COMMANDS.md'):
        if not (R/p).is_file():raise RuntimeError('missing final '+p)
    expected={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file()}
    if any(name in expected for name in ('ARTIFACTS.json','FINAL_SEAL.json','MANIFEST.md')):raise RuntimeError('seal already exists')
    expected.update(('ARTIFACTS.json','FINAL_SEAL.json','MANIFEST.md'))
    lines=['# A8 工件清单','','每个本目录文件一行。private与.aris为科研盲法/审查私有材料；不是已发布GitHub包。旧绝对路径依赖由SOURCE_LOCK/INPUTS绑定，未复制为新文件。','',
        '| 文件 | 类型 | 状态 |','|---|---|---|']
    for name in sorted(expected):
        kind='私有盲法/审查' if '/private/' in '/'+name or name.startswith('.aris/') else '源码/测试' if name.endswith('.py') else '实验数据/结果' if name.startswith('data/') or name.startswith('outputs/') else '报告/归档'
        lines.append('| '+name+' | '+kind+' | COMPLETE；审查same-family provisional |')
    with (R/'MANIFEST.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    files={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file()}
    if set(files)!=expected-{'ARTIFACTS.json','FINAL_SEAL.json'}:raise RuntimeError('inventory enumeration mismatch')
    dump(R/'ARTIFACTS.json',{'stage':'A8','host':'liekkas','root':str(R),'count':len(files),'files':files,
        'excluded_self_files':['ARTIFACTS.json','FINAL_SEAL.json'],'total_files_including_self':len(files)+2})
    dump(R/'FINAL_SEAL.json',{'stage':'A8','host':'liekkas','root':str(R),'formal_status':'COMPLETE','verification':'PASS',
        'artifact_index_sha256':sha(R/'ARTIFACTS.json'),'indexed_files':len(files),'total_files_including_self':len(files)+2,
        'source_lock_sha256':sha(O/'SOURCE_LOCK.json'),'report_sha256':sha(R/'REPORT.md'),'analysis_sha256':sha(R/'ANALYSIS.json'),
        'old_stage_status_preserved':'A7 INCOMPLETE','sealed_at':datetime.now(timezone(timedelta(hours=8))).isoformat()})
    if {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file()}!=expected:raise RuntimeError('complete file enumeration drift')
    print(json.dumps({'indexed':len(files),'total':len(expected),'index_sha256':sha(R/'ARTIFACTS.json'),'old_stages_untouched':True}))
if __name__=='__main__':main()

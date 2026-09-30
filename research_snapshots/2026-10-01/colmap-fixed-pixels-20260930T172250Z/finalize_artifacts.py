"""Verify sealed inputs and enumerate delivery artifacts, never overwrite old runs."""
from pathlib import Path
import hashlib,json,datetime,shutil
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for sealfile,base in [('PREDICTIONS_SEALED.json',ROOT),('RUN_LOCK.json',ROOT)]:
    d=json.loads((ROOT/sealfile).read_text())
    for p,h in d['files'].items():assert sha(base/p)==h,p
old=ROOT.parent/'camera-pairing-replay-20260930T162130Z'
for p,h in json.loads((old/'PREDICTIONS_SEALED.json').read_text())['files'].items():assert sha(old/p)==h,p
base=ROOT.parent/'upstream-photo-holdout-20260930T113213Z'
deps={str(base/n):sha(base/n) for n in ['evaluate.py','audit_sources.py']}
for p in [ROOT.parent.parent/'datasets/published-outputs-v2-reference/MANIFEST.json',
          ROOT.parent.parent/'datasets/reconstruction-v22-scan37/DOWNLOAD_RETRY_1789282524663424354.json']:
    deps[str(p)]=sha(p)
with (ROOT/'EVALUATION_DEPENDENCIES_POSTRUN.json').open('x') as f:
    json.dump(dict(files=deps,timing='post-run provenance completion, not a pre-evaluation seal'),f,indent=2)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
fresh=json.loads((ROOT/'review/FRESH_INTEGRITY.json').read_text())
audit={**fresh,'audit_skill':'experiment-audit','reviewer_model':'gpt-6-astra',
    'reviewer_reasoning':'ultra','agent_id':'/root/colmap_fresh_integrity',
    'trace_path':'.aris/traces/experiment-audit/20261001_run01/',
    'audit_source':'review/FRESH_INTEGRITY.json',
    'final_report_followup':'review/FRESH_INTEGRITY_ADDENDUM.json'}
with (ROOT/'EXPERIMENT_AUDIT.json').open('x') as f:json.dump(audit,f,indent=2)
shutil.copy2(ROOT/'review/FRESH_INTEGRITY.md',ROOT/'EXPERIMENT_AUDIT.md')
trace=ROOT/'.aris/traces/experiment-audit/20261001_run01'
for target,src in [('001-fresh.response.md','FRESH_INTEGRITY.md'),
                   ('002-fresh.response.md','FRESH_INTEGRITY_ADDENDUM.md'),
                   ('003-existing.response.md','DESIGN_REVIEW_PRE_GT.md'),
                   ('004-existing.response.md','CODE_REVIEW_PRE_GT.md'),
                   ('005-existing.response.json','INDEPENDENT_NUMERIC_CHECK.json')]:
    shutil.copy2(ROOT/'review'/src,trace/target)
for i,who in [('001-fresh','/root/colmap_fresh_integrity'),('002-fresh','/root/colmap_fresh_integrity'),('003-existing','/root/upstream_run_integrity_reviewer'),
              ('004-existing','/root/upstream_run_integrity_reviewer'),('005-existing','/root/upstream_run_integrity_reviewer')]:
    with (trace/(i+'.meta.json')).open('x') as f:
        json.dump(dict(agent_id=who,review_independence='same-family',acceptance_status='provisional',
            status='ok',saved_at_utc=stamp,response_is_agent_authored_artifact=True),f,indent=2)
(ROOT/'.aris/meta').mkdir(exist_ok=True)
with (ROOT/'.aris/meta/events.jsonl').open('a') as f:
    f.write(json.dumps(dict(event='review_trace',skill='experiment-audit',trace_path=str(trace.relative_to(ROOT)),status='ok'))+'\n')
tracker=f'''# Experiment Tracker — {stamp} UTC

| ID | 内容 | 状态 | 依据 |
|---|---|---|---|
| M0 | 同相机/同灰图/固定像素与独立环境 | DONE | RUN_LOCK + review |
| M1 | 官方MVS15次调用、独立workspace复跑 | DONE | job_records + RUN.log |
| M2 | 118项封存、同512/497口径评分 | DONE | PREDICTIONS_SEALED + evaluation |
| M3 | 独立核验、同点作图、报告 | DONE | review + figures + REPORT |

旧两场景开发回放，非独立迁移。部署默认未改。无后台MVS任务。
'''
version=ROOT/'refine-logs'/f'EXPERIMENT_TRACKER_{stamp}.md';version.write_text(tracker)
shutil.copy2(version,ROOT/'refine-logs/EXPERIMENT_TRACKER.md')
manifest=ROOT/'MANIFEST.md'
existing=manifest.read_text()
with manifest.open('a') as f:
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '.aris' in p.parts or p.name=='MANIFEST.md' or '__pycache__' in p.parts:continue
        rel=str(p.relative_to(ROOT))
        if f'| {rel} |' not in existing:f.write(f'| {stamp} | experiment-plan / experiment-audit | {rel} | implementation | 本轮生成工件 |\n')
files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and '.aris' not in p.parts and '__pycache__' not in p.parts}
with (ROOT/'DELIVERY_MANIFEST.json').open('x') as f:json.dump(dict(files=files,old_prediction_seal_unchanged=True),f,indent=2)
print('DELIVERY',len(files),'files; old/new seals verified')

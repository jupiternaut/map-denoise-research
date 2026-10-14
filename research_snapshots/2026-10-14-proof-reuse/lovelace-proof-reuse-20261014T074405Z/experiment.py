"""Predeclared A14 replay: immutable stages, cold serial jobs, exact verification."""
import argparse
from collections import defaultdict,Counter
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
LOGICAL=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-proof-reuse-20261014T074405Z')
A14=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z')
PY='/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'
STAGES={'tolerance':(('original','pure'),1),'switches':(('pure','off','hot','inherit','both'),3)}
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(d,f,indent=2,allow_nan=False)
def env():
    e=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): e[k]='1'
    return e
def target():
    assert socket.gethostname()=='liekkas' and ROOT==LOGICAL.resolve()
def check(stage):
    target(); f=load(ROOT/f'FREEZE_{stage}.json')
    assert all(sha(p)==h for p,h in f['files'].items()),'frozen source/data changed'
    return f
def freeze(stage):
    target(); paths={ROOT/'experiment.py',ROOT/'worker.py',ROOT/'pure_search.py',ROOT/'PROTOCOL.md'}
    for arm in STAGES[stage][0]:
        result=subprocess.run([PY,'-B',str(ROOT/'worker.py'),'--deps',arm],cwd=ROOT,env=env(),capture_output=True,text=True,check=True)
        paths.update(map(Path,load_text(result.stdout)))
    paths.update(A14.glob('*.py'))
    paths.update((A14/'data/confirm'/p for p in ('public.json','tasks.json')))
    historical=load(A14/'SOURCE_FREEZE.json')['files']
    assert all(sha(p)==h for p,h in historical.items())
    paths.update(map(Path,historical))
    save(ROOT/f'FREEZE_{stage}.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
        'files':{str(p.resolve()):sha(p) for p in sorted(paths)},'arms_repeats':STAGES[stage],
        'scope':'A14 exposed replay; no new confirmation','historical_freeze_sha':sha(A14/'SOURCE_FREEZE.json')})
def load_text(t): return json.loads(t)
def predict(stage):
    freeze_record=check(stage); arms,reps=STAGES[stage]; folder=ROOT/'outputs'/stage
    grouped=defaultdict(list)
    for t in load(A14/'data/confirm/tasks.json'):
        grouped[t['observation_id']].append({k:t[k] for k in ('task_id','incumbent')})
    jobs=[]
    for rep in range(reps):
        for i,p in enumerate(load(A14/'data/confirm/public.json')['observations']):
            shift=(rep+i)%len(arms);order=arms[shift:]+arms[:shift]
            for arm in order:
                jobs.append(dict(arm=arm,replicate=rep,budget_seconds=6.,observation_id=p['observation_id'],
                                 observation=p['observation'],tasks=grouped[p['observation_id']]))
    save(folder/'MANIFEST.json',jobs); receipts=[]; started=time.perf_counter()
    totalcpu=0.
    for i,job in enumerate(jobs):
        if totalcpu+18>1200: raise RuntimeError('batch CPU guard')
        ip=folder/f'jobs/{i:03}.job.json';op=folder/f'jobs/{i:03}.result.json';save(ip,job)
        t=time.perf_counter();c=resource.getrusage(resource.RUSAGE_CHILDREN)
        r=subprocess.run([PY,'-B',str(ROOT/'worker.py'),str(ip),str(op)],cwd=ROOT,env=env(),capture_output=True,text=True,timeout=18.)
        elapsed=time.perf_counter()-t;c1=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=c1.ru_utime+c1.ru_stime-c.ru_utime-c.ru_stime;totalcpu+=cpu
        receipt={k:job[k] for k in ('arm','replicate','observation_id','budget_seconds')}
        receipt.update(index=i,result=str(op.relative_to(ROOT)),process_seconds=elapsed,process_cpu_seconds=cpu,
                       returncode=r.returncode,stderr=r.stderr,stdout=r.stdout,sha256=sha(op) if op.exists() else None)
        save(folder/f'jobs/{i:03}.receipt.json',receipt);receipts.append(receipt)
        if r.returncode or not op.exists(): raise RuntimeError(f'job {i} failed; preserved receipt')
        out=load(op)
        assert all(p in freeze_record['files'] for p in out['dependencies']), 'unfrozen dynamic dependency'
        print(json.dumps(dict(index=i,arm=job['arm'],seconds=round(elapsed,3),reason=out['state']['reason'] if out['state'] else out['prepare_failure'])),flush=True)
    check(stage);save(folder/'READY.json',{'receipts':receipts,'calls':len(receipts),'wall_seconds':time.perf_counter()-started,
        'child_cpu_seconds':totalcpu,'truth_loaded':False,'freeze_sha':sha(ROOT/f'FREEZE_{stage}.json')})
def verify(stage):
    check(stage);sys.path.append(str(A14))
    spec=importlib.util.spec_from_file_location('a15_verify_core',A14/'experiment.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    obs={p['observation_id']:p['observation'] for p in load(A14/'data/confirm/public.json')['observations']}
    folder=ROOT/'outputs'/stage;records=[];t=time.perf_counter();cpu=time.process_time();memo={}
    for r in load(folder/'READY.json')['receipts']:
        assert sha(ROOT/r['result'])==r['sha256'];o=load(ROOT/r['result']);s=o['state']
        assert s is not None and o['prepare_failure'] is None
        proofstate={k:s[k] for k in ('cells','exclusions','lower','upper')}
        key=(o['observation_id'],hashlib.sha256(json.dumps(proofstate,sort_keys=True).encode()).hexdigest())
        if key not in memo:memo[key]=m.verify_one(obs[o['observation_id']],s)
        v=memo[key]
        decode=all(p['failure'] is None and p['selected']==min(s['upper'],max(s['lower'],p['incumbent'])) for p in o['predictions'])
        records.append({'result':r['result'],'valid':v['valid'] and decode,'proof_check':v,'decoder_valid':decode})
    save(folder/'VERIFICATION.json',{'all_valid':all(x['valid'] for x in records),'records':records,
        'unique_proof_states':len(memo),'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,'truth_loaded':False})
    assert all(x['valid'] for x in records)
    print(json.dumps({'verified':len(records),'unique':len(memo),'seconds':time.perf_counter()-t}),flush=True)
def score(stage):
    check(stage);folder=ROOT/'outputs'/stage;assert load(folder/'VERIFICATION.json')['all_valid']
    truth={r['observation_id']:r for r in load(A14/'data/confirm/private/truth.json')}
    receipts=load(folder/'READY.json')['receipts'];rows=[];states={};costs=defaultdict(list)
    for r in receipts:
        assert sha(ROOT/r['result'])==r['sha256'];o=load(ROOT/r['result']);s=o['state'];tr=truth[o['observation_id']]
        z=tr['truth']['numerator']/tr['truth']['denominator'];key=(o['arm'],o['replicate'],o['observation_id']);states[key]=s
        costs[o['arm']].append(r)
        for p in o['predictions']:
            initial=(p['incumbent']-z)**2; final=(p['selected']-z)**2
            rows.append(dict(arm=o['arm'],replicate=o['replicate'],observation_id=o['observation_id'],task_id=p['task_id'],
                truth=z,incumbent=p['incumbent'],selected=p['selected'],initial_se=initial,final_se=final,
                correct_input=abs(p['incumbent']-z)<=1e-9,covered=s['lower']<=z<=s['upper']))
    with (folder/'METRICS.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for arm,cc in costs.items():
        rr=[r for r in rows if r['arm']==arm and r['replicate']==0];shift=[r for r in rr if not r['correct_input']]
        ss=[s for (a,rep,oid),s in states.items() if a==arm]
        summary[arm]={'n_inputs_first_repeat':len(rr),'shifted_mse':sum(r['final_se'] for r in shift)/len(shift),
            'improved':sum(r['final_se']<r['initial_se']-1e-9 for r in shift),
            'same':sum(abs(r['final_se']-r['initial_se'])<=1e-9 for r in shift),
            'harmed':sum(r['final_se']>r['initial_se']+1e-9 for r in shift),
            'correct_moved':sum(r['correct_input'] and r['final_se']>1e-9 for r in rr),
            'all_covered':all(r['covered'] for r in rr),'cold_mean':sum(c['process_seconds'] for c in cc)/len(cc),
            'cpu_mean':sum(c['process_cpu_seconds'] for c in cc)/len(cc),
            'reasons':dict(Counter(s['reason'] for s in ss)),
            'repeat_output_consistent':all(s['lower']==states[(arm,0,oid)]['lower'] and s['upper']==states[(arm,0,oid)]['upper']
                                           for (a,rep,oid),s in states.items() if a==arm)}
    old=list(csv.DictReader((A14/'outputs/formal-v1/METRICS.csv').open()))
    oldmap={(r['arm'],float(r['budget_seconds']),r['task_id']):r for r in old}
    groups=defaultdict(list)
    new={r['task_id']:r for r in rows if r['arm']=='pure' and r['replicate']==0 and not r['correct_input']}
    for tid,r in new.items():
        j=float(oldmap['joint',6.,tid]['final_se']);f=float(oldmap['fine',6.,tid]['final_se']);l=float(oldmap['legacy',24.,tid]['final_se'])
        a,b=j>f+1e-9,j>l+1e-9;g='both' if a and b else 'fine_only' if a else 'legacy_only' if b else 'neither'
        groups[g].append({'task_id':tid,'observation_id':r['observation_id'],'old_joint_se':j,'fine_se':f,'legacy24_se':l,
                         'new_se':r['final_se'],'fine_loss_removed':r['final_se']<=f+1e-9,
                         'legacy_loss_removed':r['final_se']<=l+1e-9,'reason':states['pure',0,r['observation_id']]['reason']})
    grouped={g:{'n':len(rr),'improved_over_joint':sum(r['new_se']<r['old_joint_se']-1e-9 for r in rr),
        'remaining_fine_losses':sum(not r['fine_loss_removed'] for r in rr),
        'remaining_legacy_losses':sum(not r['legacy_loss_removed'] for r in rr),'records':rr} for g,rr in groups.items()}
    equivalence={}
    if 'original' in costs:
        equivalence['historical_original_exact_selected']=all(r['selected']==float(oldmap['joint',6.,r['task_id']]['selected']) for r in rows if r['arm']=='original')
    if 'off' in costs:
        baseline={(r['replicate'],r['task_id']):r for r in rows if r['arm']=='pure'}
        for arm in ('off','hot','inherit','both'):
            aa=[r for r in rows if r['arm']==arm]
            equivalence[arm]={'exact_selected_equal':all(r['selected']==baseline[r['replicate'],r['task_id']]['selected'] for r in aa),
                             'max_abs_selected_delta':max(abs(r['selected']-baseline[r['replicate'],r['task_id']]['selected']) for r in aa)}
    save(folder/'SUMMARY.json',{'arms':summary,'historical_groups':grouped,'equivalence':equivalence,
                              'scope':'18 exposed A14 worlds, not new confirmation','truth_sha':sha(A14/'data/confirm/private/truth.json')})
    print(json.dumps({'arms':summary,'groups':{g:{k:v for k,v in s.items() if k!='records'} for g,s in grouped.items()},'equivalence':equivalence}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','predict','verify','score'));p.add_argument('stage',choices=tuple(STAGES))
    a=p.parse_args();globals()[a.action](a.stage)

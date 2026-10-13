"""A14 bounded CPU experiment, reusing frozen A13 I/O and cold worker queue."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parent
LOGICAL=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z')
A13=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
spec=importlib.util.spec_from_file_location('a14_queue_core',A13/'run_experiment.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
core.ROOT=ROOT;core.LOGICAL=LOGICAL
core.SEEDS=(20261013141,20261013142,20261013143)
ARMS=(('base',6.),('fine',6.),('joint',6.),('legacy',6.),('legacy',24.))
EXPECTED_CALLS=18*len(ARMS)
load,save,sha=core.load,core.save,core.sha

def freeze():
    core.target()
    previous=load(A13/'SOURCE_FREEZE.json')
    assert all(sha(p)==h for p,h in previous['files'].items())
    paths=[Path(p) for p in previous['files']]+list(A13.glob('*.py'))+list(ROOT.glob('*.py'))
    paths += [ROOT/'refine-logs/EXPERIMENT_PLAN.md',ROOT/'CONFIRMATION_LOCK.md',ROOT/'JOINT_DERIVATION.md']
    save(ROOT/'SOURCE_FREEZE.json',{'utc':datetime.now(timezone.utc).isoformat(),'host':'liekkas',
        'files':{str(p.resolve()):sha(p) for p in paths},'arms':ARMS,'seeds':core.SEEDS,
        'ancestor_freeze_sha256':sha(A13/'SOURCE_FREEZE.json')})

def generate():core.generate()

def smoke():
    core.target()
    d=load(core.OLD/'data/dev/dev.json')
    p=d['observations'][0]
    ts=[{'task_id':t['task_id'],'incumbent':t['incumbent']} for t in d['tasks'] if t['observation_id']==p['observation_id']]
    for i,(arm,budget) in enumerate(ARMS[:3]):
        core.run_job(dict(arm=arm,budget_seconds=budget,observation_id=p['observation_id'],
            observation=p['observation'],tasks=ts),ROOT/'outputs/smoke-v1',i)

def predict():
    core.target();core.check_freeze()
    folder=ROOT/'outputs/formal-v1';grouped=defaultdict(list)
    for t in load(ROOT/'data/confirm/tasks.json'):
        grouped[t['observation_id']].append({'task_id':t['task_id'],'incumbent':t['incumbent']})
    jobs=[]
    for i,p in enumerate(load(ROOT/'data/confirm/public.json')['observations']):
        order=ARMS[i%len(ARMS):]+ARMS[:i%len(ARMS)]
        for arm,budget in order:
            jobs.append(dict(arm=arm,budget_seconds=budget,observation_id=p['observation_id'],
                observation=p['observation'],tasks=grouped[p['observation_id']]))
    save(folder/'MANIFEST.json',jobs);start=time.perf_counter();receipts=[]
    initial=resource.getrusage(resource.RUSAGE_CHILDREN)
    for i,job in enumerate(jobs):
        used=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=used.ru_utime+used.ru_stime-initial.ru_utime-initial.ru_stime
        if cpu+job['budget_seconds']+12>570 or time.perf_counter()-start>1200:
            save(folder/'INCOMPLETE.json',{'completed':len(receipts),'reason':'batch budget','cpu':cpu});return
        save(folder/f'state/{i:03}-running.json',{'index':i,'status':'running'})
        r=core.run_job(job,folder/'jobs',i);receipts.append(r)
        save(folder/f'state/{i:03}-terminal.json',{'index':i,'status':'completed' if r['returncode']==0 else 'failed'})
    core.check_freeze()
    save(folder/'PREDICTIONS_READY.json',{'status':'COMPLETE','calls':len(receipts),
        'prediction_count':sum(len(load(ROOT/r['result'])['predictions']) for r in receipts),
        'seconds':time.perf_counter()-start,'receipts':receipts,
        'source_freeze_sha256':sha(ROOT/'SOURCE_FREEZE.json'),'truth_not_loaded_by_predict':True})

def verify_one(observation,state):
    import joint_verify as j
    original=j.old.verify_certificate
    def dispatch(obs,entry):
        cert=entry.get('certificate',entry)
        if cert.get('schema')!=j.SCHEMA:return original(obs,entry)
        r=j.verify_certificate(obs,entry)
        return {**r,'valid':r['verified'],'reason':r.get('error','four exact tiles cover camera domain')}
    j.old.verify_certificate=dispatch
    try:return j.old.verify_state(observation,state)
    finally:j.old.verify_certificate=original

def verify():
    core.target();core.check_freeze();folder=ROOT/'outputs/formal-v1'
    ready=load(folder/'PREDICTIONS_READY.json')
    assert ready['calls']==EXPECTED_CALLS and ready['prediction_count']==EXPECTED_CALLS*7
    obs={x['observation_id']:x['observation'] for x in load(ROOT/'data/confirm/public.json')['observations']}
    start=time.perf_counter();cpu=time.process_time();checks=[]
    for r in ready['receipts']:
        assert sha(ROOT/r['result'])==r['sha256']
        if r['arm']=='legacy':continue
        out=load(ROOT/r['result']);s=out['state']
        check=verify_one(obs[out['observation_id']],s) if s else {'valid':out['prepare_failure'] is not None,'reason':'explicit worker fallback'}
        check['decoder_valid']=all(p['selected']==(p['incumbent'] if p['failure'] is not None or s is None else min(s['upper'],max(s['lower'],p['incumbent']))) for p in out['predictions'])
        check['valid']=check['valid'] and check['decoder_valid']
        checks.append({'arm':out['arm'],'observation_id':out['observation_id'],**check})
    assert len(checks)==54
    result={'all_valid':all(c['valid'] for c in checks),'records':checks,
        'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'truth_loaded':False}
    save(folder/'INDEPENDENT_VERIFICATION.json',result)
    print(json.dumps({'verified_states':len(checks),'all_valid':result['all_valid']}),flush=True)

def score():
    core.target();core.check_freeze();folder=ROOT/'outputs/formal-v1'
    ready=load(folder/'PREDICTIONS_READY.json');assert ready['status']=='COMPLETE'
    assert ready['calls']==EXPECTED_CALLS and ready['prediction_count']==EXPECTED_CALLS*7
    assert load(folder/'INDEPENDENT_VERIFICATION.json')['all_valid']
    truth={x['observation_id']:x for x in load(ROOT/'data/confirm/private/truth.json')}
    rows=[];costs=[];coverage=defaultdict(list)
    for receipt in ready['receipts']:
        assert sha(ROOT/receipt['result'])==receipt['sha256']
        out=load(ROOT/receipt['result']);f=truth[out['observation_id']]
        z=f['truth']['numerator']/f['truth']['denominator'];s=out['state'];arm=out['arm'];budget=out['budget_seconds']
        hull=None if s is None else (s.get('hull') if arm=='legacy' else [s['lower'],s['upper']])
        coverage[(arm,budget)].append(hull is not None and hull[0]<=z<=hull[1])
        costs.append({k:receipt[k] for k in ('arm','budget_seconds','observation_id','process_seconds','process_cpu_seconds','returncode')}|out['cost'])
        for p in out['predictions']:
            a=abs(p['incumbent']-z);b=abs(p['selected']-z)
            rows.append(dict(arm=arm,budget_seconds=budget,observation_id=out['observation_id'],
                task_id=p['task_id'],truth=z,incumbent=p['incumbent'],selected=p['selected'],initial_ae=a,
                final_ae=b,initial_se=a*a,final_se=b*b,correct_input=a<=1e-9,failure=p['failure'] is not None,
                branch=f['rear_branch'],stratum=f['stratum'],negative_control=f['negative_control'],seed_block=f['seed_block']))
    assert len({(x['arm'],x['budget_seconds'],x['task_id']) for x in rows})==EXPECTED_CALLS*7
    summary={}
    for arm,budget in ARMS:
        rr=[r for r in rows if r['arm']==arm and r['budget_seconds']==budget]
        cc=[c for c in costs if c['arm']==arm and c['budget_seconds']==budget]
        summary[f'{arm}@{budget:g}']={'all':core.aggregate(rr),
            'shifted':core.aggregate([r for r in rr if not r['correct_input']]),
            'correct':core.aggregate([r for r in rr if r['correct_input']]),
            'negative':core.aggregate([r for r in rr if r['negative_control']]),
            'by_branch':{b:core.aggregate([r for r in rr if r['branch']==b]) for b in ('none','partial','full')},
            'covered_worlds':sum(coverage[(arm,budget)]),'worlds':18,
            'mean_cold_seconds':sum(c['process_seconds'] for c in cc)/len(cc),
            'cpu_seconds':sum(c['process_cpu_seconds'] for c in cc),
            'overruns':sum(c.get('overrun_gt_0_25',0) for c in cc)}
    with (folder/'METRICS.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(folder/'SUMMARY.json',summary);save(folder/'COSTS.json',costs)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('smoke','freeze','generate','predict','verify','score'))
    globals()[p.parse_args().action]()

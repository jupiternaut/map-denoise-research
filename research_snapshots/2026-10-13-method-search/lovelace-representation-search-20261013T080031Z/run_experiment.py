"""Single bounded A12 batch; prediction first, private truth only in score()."""
import argparse
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
LOGICAL=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
PY='/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'
ARMS=('fixed','library','free')
BUDGETS=(6.,24.)

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False)
def target():
    if socket.gethostname()!='liekkas' or ROOT!=LOGICAL.resolve():
        raise RuntimeError('wrong target identity')
def env():
    e=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        e[key]='1'
    e['PYTHONDONTWRITEBYTECODE']='1'
    return e

def freeze():
    target()
    paths=[ROOT/x for x in ('data_builder.py','worker.py','run_experiment.py','task_bundle/TASK.md',
           'task_bundle/forward.py','task_bundle/dev.json','library/methods.py',
           'refine-logs/EXPERIMENT_PLAN.md')]
    paths.extend(p for a in ARMS for p in (ROOT/'arms'/a).glob('*.py'))
    for a in ARMS:
        if not (ROOT/'arms'/a/'method.py').is_file(): raise RuntimeError('missing '+a)
    result={'utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),
            'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}}
    save(ROOT/'SOURCE_FREEZE.json',result)
    print(json.dumps({'frozen_files':len(result['files'])}),flush=True)

def check_freeze():
    r=load(ROOT/'SOURCE_FREEZE.json')
    for p,h in r['files'].items():
        if sha(ROOT/p)!=h: raise RuntimeError('frozen source changed: '+p)
    return r

def run_job(job,folder,index):
    folder.mkdir(parents=True,exist_ok=True)
    infile=folder/f'{index:03d}.job.json'; outfile=folder/f'{index:03d}.result.json'
    save(infile,job)
    t=time.perf_counter(); child0=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        process=subprocess.run([PY,'-B',str(ROOT/'worker.py'),str(infile),str(outfile)],
                               cwd=ROOT,env=env(),capture_output=True,text=True,
                               timeout=job['budget_seconds']+12.)
        code=process.returncode; stdout=process.stdout; stderr=process.stderr
    except subprocess.TimeoutExpired as error:
        code=-9; stdout=str(error.stdout or ''); stderr='parent process timeout'
    wall=time.perf_counter()-t
    child1=resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu=child1.ru_utime+child1.ru_stime-child0.ru_utime-child0.ru_stime
    if code==0 and outfile.exists():
        result=load(outfile)
    else:
        result={'arm':job['arm'],'observation_id':job['observation_id'],
          'budget_seconds':job['budget_seconds'],'ablation':job.get('ablation',False),
          'state':None,'prepare_failure':{'type':'WorkerFailure','message':stderr[-2000:]},
          'predictions':[{'task_id':t['task_id'],'incumbent':t['incumbent'],'selected':t['incumbent'],
                         'failure':{'type':'WorkerFailure'}} for t in job['tasks']],
          'cost':{}}
        if not outfile.exists(): save(outfile,result)
    try:
        finish=json.loads(stdout.strip().splitlines()[-1]) if code==0 else None
    except (ValueError,IndexError):
        finish=None
    receipt={'index':index,'job':str(infile.relative_to(ROOT)),
             'result':str(outfile.relative_to(ROOT)),'result_sha256':sha(outfile),
             'process_seconds':wall,'process_cpu_seconds':child_cpu,'finish':finish,
             'returncode':code,'stdout':stdout,'stderr':stderr,
             'arm':job['arm'],'budget_seconds':job['budget_seconds'],
             'observation_id':job['observation_id'],'ablation':job.get('ablation',False)}
    save(folder/f'{index:03d}.receipt.json',receipt)
    print(json.dumps({'job':index,'arm':job['arm'],'budget':job['budget_seconds'],
                     'seconds':round(wall,3),'returncode':code}),flush=True)
    return receipt

def predict():
    target(); check_freeze()
    folder=ROOT/'outputs/formal-v1'
    if folder.exists(): raise FileExistsError('formal output exists')
    folder.mkdir(parents=True)
    start=time.perf_counter(); deadline=start+2400.
    public=load(ROOT/'data/holdout/public.json')['observations']
    tasks=load(ROOT/'data/holdout/private/tasks.json') # incumbents, not truth
    grouped={p['observation_id']:[] for p in public}
    for t in tasks:
        grouped[t['observation_id']].append({'task_id':t['task_id'],'incumbent':t['incumbent']})
    receipts=[]
    for bi,budget in enumerate(BUDGETS):
        for oi,p in enumerate(public):
            shift=(oi+bi)%len(ARMS); order=ARMS[shift:]+ARMS[:shift]
            for arm in order:
                if time.perf_counter()+budget+12>deadline:
                    save(folder/'INCOMPLETE.json',{'reason':'batch deadline','completed':len(receipts)})
                    return
                job={'arm':arm,'budget_seconds':budget,'observation_id':p['observation_id'],
                     'observation':p['observation'],'tasks':grouped[p['observation_id']]}
                receipts.append(run_job(job,folder/'jobs',len(receipts)))
    check_freeze()
    save(folder/'PREDICTIONS_READY.json',{'status':'COMPLETE','calls':len(receipts),
         'prediction_count':sum(len(load(ROOT/r['result'])['predictions']) for r in receipts),
         'seconds':time.perf_counter()-start,'receipts':receipts,
         'truth_not_loaded_by_predict':True,'source_freeze_sha256':sha(ROOT/'SOURCE_FREEZE.json')})

def aggregate(rows):
    n=len(rows)
    if not n: return {'n':0}
    mse=sum(r['final_se'] for r in rows)/n; initial=sum(r['initial_se'] for r in rows)/n
    return {'n':n,'worlds':len({r['observation_id'] for r in rows}),
            'mae':sum(r['final_ae'] for r in rows)/n,'mse':mse,
            'identity_mae':sum(r['initial_ae'] for r in rows)/n,'identity_mse':initial,
            'mse_improvement_percent':None if initial==0 else 100*(1-mse/initial),
            'improved':sum(r['initial_ae']-r['final_ae']>1e-9 for r in rows),
            'harmed':sum(r['final_ae']-r['initial_ae']>1e-9 for r in rows),
            'same':sum(abs(r['final_ae']-r['initial_ae'])<=1e-9 for r in rows),
            'max_error':max(r['final_ae'] for r in rows),'failures':sum(r['failure'] for r in rows)}

def supplement():
    target(); check_freeze()
    folder=ROOT/'outputs/supplement-v1'
    if folder.exists(): raise FileExistsError('supplement output exists')
    ready=load(ROOT/'outputs/formal-v1/PREDICTIONS_READY.json')
    if ready['status']!='COMPLETE': raise RuntimeError('main predictions incomplete')
    public=load(ROOT/'data/holdout/public.json')['observations']
    tasks=load(ROOT/'data/holdout/private/tasks.json')
    # Prespecified first-block contrast cases: none/partial/full. No truth read.
    receipts=[]
    for index in (0,2,4):
        p=public[index]
        selected=[{'task_id':t['task_id'],'incumbent':t['incumbent']} for t in tasks
                  if t['observation_id']==p['observation_id']]
        job={'arm':'free','budget_seconds':24.,'ablation':True,
             'observation_id':p['observation_id'],'observation':p['observation'],'tasks':selected}
        receipts.append(run_job(job,folder/'jobs',len(receipts)))
    save(folder/'PREDICTIONS_READY.json',{'status':'COMPLETE','receipts':receipts,
         'selected_public_indices':[0,2,4],'truth_not_loaded':True})

def score():
    target(); check_freeze(); folder=ROOT/'outputs/formal-v1'
    ready=load(folder/'PREDICTIONS_READY.json')
    if ready['status']!='COMPLETE' or ready['calls']!=108 or ready['prediction_count']!=756:
        raise RuntimeError('all three arms/two budgets must precede truth scoring')
    for r in ready['receipts']:
        if sha(ROOT/r['result'])!=r['result_sha256']: raise RuntimeError('prediction changed')
    # First read of held-out truth in this scoring path.
    facts={x['observation_id']:x for x in load(ROOT/'data/holdout/private/truth.json')}
    rows=[]; costs=[]
    for r in ready['receipts']:
        result=load(ROOT/r['result']); f=facts[r['observation_id']]
        truth=f['truth']['numerator']/f['truth']['denominator']
        costs.append({**{k:r[k] for k in ('arm','budget_seconds','observation_id','process_seconds','process_cpu_seconds','returncode','finish')},
                      **result['cost']})
        for p in result['predictions']:
            initial=abs(p['incumbent']-truth); final=abs(p['selected']-truth)
            rows.append({'arm':r['arm'],'budget_seconds':r['budget_seconds'],
              'observation_id':r['observation_id'],'task_id':p['task_id'],'truth':truth,
              'incumbent':p['incumbent'],'selected':p['selected'],
              'initial_ae':initial,'final_ae':final,'initial_se':initial**2,'final_se':final**2,
              'correct_input':initial<=1e-9,'failure':p['failure'] is not None,
              'branch':f['rear_branch'],'stratum':f['stratum'],'negative_control':f['negative_control'],
              'seed_block':f['seed_block']})
    if len({(r['arm'],r['budget_seconds'],r['task_id']) for r in rows})!=756:
        raise RuntimeError('duplicates/missing predictions')
    summary={}
    for arm in ARMS:
        for budget in BUDGETS:
            take=[r for r in rows if r['arm']==arm and r['budget_seconds']==budget]
            cs=[c for c in costs if c['arm']==arm and c['budget_seconds']==budget]
            summary[f'{arm}@{budget:g}']={'all':aggregate(take),
              'shifted':aggregate([r for r in take if not r['correct_input']]),
              'correct':aggregate([r for r in take if r['correct_input']]),
              'negative':aggregate([r for r in take if r['negative_control']]),
              'by_branch':{b:aggregate([r for r in take if r['branch']==b]) for b in ('none','partial','full')},
              'by_stratum':{s:aggregate([r for r in take if r['stratum']==s]) for s in ('contrast','lowcontrast')},
              'by_seed':{str(b):aggregate([r for r in take if r['seed_block']==b]) for b in range(3)},
              'cost':{'process_seconds_total':sum(c['process_seconds'] for c in cs),
                      'process_cpu_seconds_total':sum(c['process_cpu_seconds'] for c in cs),
                      'prepare_seconds_total':sum(c.get('prepare_seconds',0) for c in cs),
                      'prepare_cpu_seconds_total':sum(c['prepare_cpu_seconds'] for c in cs if 'prepare_cpu_seconds' in c),
                      'missing_prepare_cpu_records':sum('prepare_cpu_seconds' not in c for c in cs),
                      'overruns_gt_0_25':sum(c.get('overrun_gt_0_25',False) for c in cs),
                      'process_failures':sum(c['returncode']!=0 for c in cs),
                      'max_rss_kib':max(c.get('max_rss_kib',0) for c in cs)}}
    with (folder/'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(folder/'COSTS.json',costs);save(folder/'SUMMARY.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('freeze','predict','supplement','score'));a=ap.parse_args()
    {'freeze':freeze,'predict':predict,'supplement':supplement,'score':score}[a.command]()

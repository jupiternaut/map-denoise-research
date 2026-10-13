"""A13 bounded local CPU queue. Freeze -> generate -> predict -> score."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
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
LOGICAL=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
OLD=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
PY='/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'
ARMS=('numeric','certified','legacy'); BUDGETS=(6.,24.)
SEEDS=(20261013131,20261013132,20261013133)

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(data,f,indent=2,allow_nan=False)
def target():
    assert socket.gethostname()=='liekkas' and ROOT==LOGICAL.resolve(), 'wrong exact target'
def env():
    e=dict(os.environ)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): e[k]='1'
    e['PYTHONDONTWRITEBYTECODE']='1'
    return e
def load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m)
    return m

def freeze():
    target()
    # Record the executed, read-only dependency chain as well as this adapter.
    g=load_source('a13_freeze_generator',OLD/'data_builder.py');g.dependencies()
    import search_method as m
    m.warm_load('certified')
    paths=list(ROOT.glob('*.py'))+[ROOT/'refine-logs/EXPERIMENT_PLAN.md']
    paths += [OLD/p for p in ('data_builder.py','arms/free/method.py','library/methods.py',
                              'task_bundle/forward.py','task_bundle/TASK.md')]
    # Generator and legacy both import a module named baseline_geometry but
    # intentionally target different frozen files. Snapshot legacy in its own
    # cold process, just as formal worker execution does; never reuse a cached
    # generator module or weaken its existing exact-path check.
    probe='import json,sys; import search_method as m; m.warm_load("legacy"); print(json.dumps([x.__file__ for x in list(sys.modules.values()) if getattr(x,"__file__",None) and "/map-denoise/runs/" in x.__file__ and x.__file__.endswith(".py")]))'
    result=subprocess.run([PY,'-B','-c',probe],cwd=ROOT,env=env(),capture_output=True,text=True,check=True,timeout=30.)
    paths += [Path(p) for p in json.loads(result.stdout)]
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and '/map-denoise/runs/' in name and name.endswith('.py'):
            paths.append(Path(name))
    save(ROOT/'SOURCE_FREEZE.json',{'utc':datetime.now(timezone.utc).isoformat(),
         'host':socket.gethostname(), 'files':{str(p.resolve()):sha(p) for p in paths}})
def check_freeze():
    f=load(ROOT/'SOURCE_FREEZE.json')
    assert all(sha(p)==h for p,h in f['files'].items()),'frozen source changed'
    return f

def generate():
    target(); check_freeze()
    g=load_source('a13_original_generator',OLD/'data_builder.py')
    start=time.perf_counter(); public,truth,tasks=g.build_population('holdout',seeds=SEEDS)
    assert len(public)==len(truth)==18 and len(tasks)==126
    folder=ROOT/'data/confirm'
    save(folder/'public.json',{'observations':public})
    save(folder/'tasks.json',tasks); save(folder/'private/truth.json',truth)
    save(folder/'GENERATION.json',{'seeds':SEEDS,'worlds':18,'tasks':126,
         'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,
         'source_freeze_sha256':sha(ROOT/'SOURCE_FREEZE.json'),
         'exact_forward_verified':all(x['independent_exact_verified'] for x in truth),
         'observation_transport':'float(exact rational) then JSON binary64 round-trip; certificate uses closed rounding preimages',
         'float_forward_max_abs_error':max(x['float_forward_max_abs_error'] for x in truth),
         'original_generator':str(OLD/'data_builder.py')})
    print(json.dumps({'generated_worlds':18,'tasks':126}),flush=True)

def run_job(job,folder,index):
    folder.mkdir(parents=True,exist_ok=True)
    ip=folder/f'{index:03}.job.json'; op=folder/f'{index:03}.result.json'
    save(ip,job); t=time.perf_counter(); c=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        r=subprocess.run([PY,'-B',str(ROOT/'worker.py'),str(ip),str(op)],cwd=ROOT,
                         env=env(),capture_output=True,text=True,timeout=job['budget_seconds']+12.)
        code,out,err=r.returncode,r.stdout,r.stderr
    except subprocess.TimeoutExpired:
        code,out,err=-9,'','parent timeout'
    c1=resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed=time.perf_counter()-t; cpu=c1.ru_utime+c1.ru_stime-c.ru_utime-c.ru_stime
    if code!=0 or not op.exists():
        if op.exists(): raise RuntimeError('failed process left result; inspect before continuing')
        save(op,{'arm':job['arm'],'budget_seconds':job['budget_seconds'],
          'observation_id':job['observation_id'],'state':None,'cost':{},
          'prepare_failure':{'type':'WorkerFailure','message':err[-2000:]},
          'predictions':[dict(t,selected=t['incumbent'],failure={'type':'WorkerFailure'}) for t in job['tasks']]})
    receipt={'index':index,'arm':job['arm'],'budget_seconds':job['budget_seconds'],
        'observation_id':job['observation_id'],'job':str(ip.relative_to(ROOT)),
        'result':str(op.relative_to(ROOT)),'sha256':sha(op),'process_seconds':elapsed,
        'process_cpu_seconds':cpu,'returncode':code,'stdout':out,'stderr':err}
    save(folder/f'{index:03}.receipt.json',receipt)
    print(json.dumps({'job':index,'arm':job['arm'],'budget':job['budget_seconds'],
                      'seconds':round(elapsed,3),'returncode':code}),flush=True)
    return receipt

def smoke():
    d=load(OLD/'task_bundle/dev.json'); p=d['observations'][0]
    ts=[{'task_id':t['task_id'],'incumbent':t['incumbent']} for t in d['tasks'] if t['observation_id']==p['observation_id']]
    for arm in ('numeric','certified'):
        run_job({'arm':arm,'budget_seconds':6.,'observation_id':p['observation_id'],
                 'observation':p['observation'],'tasks':ts},ROOT/'outputs/smoke-v1',0 if arm=='numeric' else 1)

def predict():
    target(); check_freeze(); data=ROOT/'data/confirm'; folder=ROOT/'outputs/formal-v1'
    public=load(data/'public.json')['observations']; tasks=load(data/'tasks.json')
    grouped=defaultdict(list)
    for t in tasks: grouped[t['observation_id']].append({'task_id':t['task_id'],'incumbent':t['incumbent']})
    jobs=[]
    for bi,budget in enumerate(BUDGETS):
        for oi,p in enumerate(public):
            shift=(oi+bi)%3
            for arm in ARMS[shift:]+ARMS[:shift]:
                jobs.append({'arm':arm,'budget_seconds':budget,'observation_id':p['observation_id'],
                             'observation':p['observation'],'tasks':grouped[p['observation_id']]})
    save(folder/'MANIFEST.json',jobs)
    start=time.perf_counter(); receipts=[]
    for i,job in enumerate(jobs):
        if time.perf_counter()-start+job['budget_seconds']+12>2400:
            save(folder/'INCOMPLETE.json',{'completed':len(receipts),'reason':'batch deadline'}); return
        save(folder/f'state/{i:03}-running.json',{'index':i,'status':'running'})
        r=run_job(job,folder/'jobs',i); receipts.append(r)
        save(folder/f'state/{i:03}-terminal.json',{'index':i,'status':'completed' if r['returncode']==0 else 'failed'})
    check_freeze()
    save(folder/'PREDICTIONS_READY.json',{'status':'COMPLETE','calls':len(receipts),
         'prediction_count':sum(len(load(ROOT/r['result'])['predictions']) for r in receipts),
         'seconds':time.perf_counter()-start,'receipts':receipts,
         'source_freeze_sha256':sha(ROOT/'SOURCE_FREEZE.json'),'truth_not_loaded_by_predict':True})

def diagnose():
    """Same frozen dev proposals; proof never changes the proposal trajectory."""
    target();check_freeze()
    import search_method as m
    m.warm_load('numeric');m.warm_load('certified')
    d=load(OLD/'data/dev/public.json'); records=[]
    for p in d['observations']:
        t=time.perf_counter()
        state=m.prepare(p['observation'],6.,'numeric')
        ctx=m.CERT.ExactContext(p['observation']); proofs=[]
        deadline=time.perf_counter()+24.
        for e in state['exclusions']:
            if time.perf_counter()>=deadline:
                checked={'verified':False,'certificate':None,'reason':'diagnostic_deadline'}
            else:
                checked=ctx.check_cell(e['branch'],e['lower'],e['upper'],deadline)
            proofs.append({**e,**checked})
        retained=[(c['lower'],c['upper']) for c in state['cells']]
        retained += [(c['lower'],c['upper']) for c in proofs if not c['verified']]
        adjusted=[min(x[0] for x in retained),max(x[1] for x in retained)] if retained else [540.,660.]
        records.append({'observation_id':p['observation_id'],'numeric_state':state,
           'proofs':proofs,'verified':sum(c['verified'] for c in proofs),
           'attempted':len(proofs),'verified_replay_hull':adjusted,
           'seconds':time.perf_counter()-t})
        print(json.dumps({'diagnostic_world':p['observation_id'],'verified':records[-1]['verified'],
                           'attempted':len(proofs),'seconds':records[-1]['seconds']}),flush=True)
    save(ROOT/'outputs/diagnostic-v1/RESULTS.json',{'scope':'six exposed dev worlds, same proposals',
                                                  'records':records})

def aggregate(rows):
    if not rows:return {'n':0}
    n=len(rows); a=sum(r['initial_se'] for r in rows)/n; b=sum(r['final_se'] for r in rows)/n
    return {'n':n,'worlds':len({r['observation_id'] for r in rows}),
      'mse':b,'mae':sum(r['final_ae'] for r in rows)/n,'identity_mse':a,
      'mse_improvement_percent':100*(1-b/a) if a else None,
      'improved':sum(r['initial_ae']-r['final_ae']>1e-9 for r in rows),
      'same':sum(abs(r['initial_ae']-r['final_ae'])<=1e-9 for r in rows),
      'harmed':sum(r['final_ae']-r['initial_ae']>1e-9 for r in rows),
      'max_error':max(r['final_ae'] for r in rows),'failures':sum(r['failure'] for r in rows)}

def verify():
    """Independent standalone verifier replays saved states, no labels loaded."""
    target();check_freeze()
    from verify_certificates import verify_state
    folder=ROOT/'outputs/formal-v1'
    ready=load(folder/'PREDICTIONS_READY.json')
    assert ready['calls']==108 and ready['prediction_count']==756
    observations={x['observation_id']:x['observation'] for x in load(ROOT/'data/confirm/public.json')['observations']}
    start=time.perf_counter();cpu=time.process_time(); records=[]
    for receipt in ready['receipts']:
        if receipt['arm']!='certified': continue
        assert sha(ROOT/receipt['result'])==receipt['sha256']
        out=load(ROOT/receipt['result']); t=time.perf_counter()
        if out['state'] is None:
            check={'valid':all(p['selected']==p['incumbent'] for p in out['predictions']),
                   'reason':'identity worker fallback, no interval certificate', 'checked_exclusions':0}
        else:
            check=verify_state(observations[out['observation_id']],out['state'])
        decoder_valid=all(p['selected']==(p['incumbent'] if p['failure'] is not None or out['state'] is None
            else min(out['state']['upper'],max(out['state']['lower'],p['incumbent']))) for p in out['predictions'])
        check['decoder_valid']=decoder_valid
        check['valid']=check['valid'] and decoder_valid
        records.append({'budget_seconds':out['budget_seconds'],'observation_id':out['observation_id'],
                        'seconds':time.perf_counter()-t,**check})
    assert len(records)==36
    dev=load(OLD/'data/dev/public.json')
    observations={x['observation_id']:x['observation'] for x in dev['observations']}
    diagnostics=[]
    for r in load(ROOT/'outputs/diagnostic-v1/RESULTS.json')['records']:
        state=dict(r['numeric_state'])
        state['cells']=state['cells']+[{'branch':x['branch'],'lower':x['lower'],'upper':x['upper']}
                                      for x in r['proofs'] if not x['verified']]
        state['exclusions']=[x for x in r['proofs'] if x['verified']]
        state['lower'],state['upper']=r['verified_replay_hull']
        diagnostics.append({'observation_id':r['observation_id'],**verify_state(observations[r['observation_id']],state)})
    result={'all_valid':all(x['valid'] for x in records+diagnostics),'formal_states':records,
            'diagnostic_states':diagnostics,'wall_seconds':time.perf_counter()-start,
            'cpu_seconds':time.process_time()-cpu,'truth_loaded':False}
    save(folder/'INDEPENDENT_VERIFICATION.json',result)
    print(json.dumps(result),flush=True)

def score():
    target();check_freeze(); folder=ROOT/'outputs/formal-v1'
    r=load(folder/'PREDICTIONS_READY.json')
    assert r['status']=='COMPLETE' and r['calls']==108 and r['prediction_count']==756
    assert load(folder/'INDEPENDENT_VERIFICATION.json')['all_valid'], 'independent certificate checks failed; do not certify results'
    truth={f['observation_id']:f for f in load(ROOT/'data/confirm/private/truth.json')}
    rows=[]; costs=[]; coverage=defaultdict(list)
    for receipt in r['receipts']:
        assert sha(ROOT/receipt['result'])==receipt['sha256']
        out=load(ROOT/receipt['result']); f=truth[out['observation_id']]
        z=f['truth']['numerator']/f['truth']['denominator']; s=out['state']
        label=f"{out['arm']}@{out['budget_seconds']:g}"
        hull=None if s is None else (s.get('hull') if out['arm']=='legacy' else [s['lower'],s['upper']])
        coverage[label].append(hull is not None and hull[0]<=z<=hull[1])
        costs.append({k:receipt[k] for k in ('arm','budget_seconds','observation_id','process_seconds','process_cpu_seconds','returncode')} | out['cost'])
        for p in out['predictions']:
            a=abs(p['incumbent']-z); b=abs(p['selected']-z)
            rows.append(dict(arm=out['arm'],budget_seconds=out['budget_seconds'],
              observation_id=out['observation_id'],task_id=p['task_id'],truth=z,
              incumbent=p['incumbent'],selected=p['selected'],initial_ae=a,final_ae=b,
              initial_se=a*a,final_se=b*b,correct_input=a<=1e-9,failure=p['failure'] is not None,
              branch=f['rear_branch'],stratum=f['stratum'],negative_control=f['negative_control'],seed_block=f['seed_block']))
    assert len({(x['arm'],x['budget_seconds'],x['task_id']) for x in rows})==756
    summary={}
    for arm in ARMS:
        for budget in BUDGETS:
            rr=[x for x in rows if x['arm']==arm and x['budget_seconds']==budget]
            cc=[x for x in costs if x['arm']==arm and x['budget_seconds']==budget]
            label=f'{arm}@{budget:g}'
            summary[label]={'all':aggregate(rr),'shifted':aggregate([x for x in rr if not x['correct_input']]),
              'correct':aggregate([x for x in rr if x['correct_input']]),'negative':aggregate([x for x in rr if x['negative_control']]),
              'by_branch':{b:aggregate([x for x in rr if x['branch']==b]) for b in ('none','partial','full')},
              'by_stratum':{b:aggregate([x for x in rr if x['stratum']==b]) for b in ('contrast','lowcontrast')},
              'covered_worlds':sum(coverage[label]),'worlds':18,
              'cost':{key:sum(x.get(key,0) for x in cc) for key in ('process_seconds','process_cpu_seconds','prepare_seconds','prepare_cpu_seconds','overrun_gt_0_25')}}
    with (folder/'METRICS.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(folder/'SUMMARY.json',summary);save(folder/'COSTS.json',costs)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('freeze','generate','smoke','predict','score','diagnose','verify'))
    args=ap.parse_args();globals()[args.action]()

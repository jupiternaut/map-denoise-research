"""Post-review switch harness: global receipts, timeout accounting, state gate.

The frozen tolerance runner is retained unchanged. This wrapper repairs its
limited global accounting and adds a pre-truth full-state equivalence gate.
"""
from collections import defaultdict
import json
from pathlib import Path
import resource
import subprocess
import time
import experiment as e

ROOT=e.ROOT
def spent():
    # READY duplicates are deliberately not counted. Only unique job receipts
    # plus independent diagnostic outputs and declared development CPU count.
    jobs=sum(e.load(p)['process_cpu_seconds'] for p in (ROOT/'outputs').glob('*/jobs/*.receipt.json'))
    diagnostic=sum(e.load(p).get('process_cpu_seconds',0.) for p in (ROOT/'outputs').glob('boundary-*.json'))
    return {'job_cpu':jobs,'diagnostic_child_cpu':diagnostic,'declared_development_cpu':1.402,
            'total_child_and_development_cpu':jobs+diagnostic+1.402}

def freeze():
    e.freeze('switches')
    files=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md']
    e.save(ROOT/'FREEZE_supplement.json',{'files':{str(p.resolve()):e.sha(p) for p in files},
       'global_guard':1200.,'counter_note':'legacy kernel LP statuses count completed returns, not exception attempts'})
def check():
    f=e.check('switches');s=e.load(ROOT/'FREEZE_supplement.json')
    assert all(e.sha(p)==h for p,h in s['files'].items())
    return f

def predict():
    f=check();arms,reps=e.STAGES['switches'];folder=ROOT/'outputs/switches'
    grouped=defaultdict(list)
    for t in e.load(e.A14/'data/confirm/tasks.json'):
        grouped[t['observation_id']].append({k:t[k] for k in ('task_id','incumbent')})
    jobs=[]
    for rep in range(reps):
        for i,p in enumerate(e.load(e.A14/'data/confirm/public.json')['observations']):
            shift=(rep+i)%len(arms)
            for arm in arms[shift:]+arms[:shift]:
                jobs.append(dict(arm=arm,replicate=rep,budget_seconds=6.,observation_id=p['observation_id'],
                                 observation=p['observation'],tasks=grouped[p['observation_id']]))
    e.save(folder/'MANIFEST.json',jobs);start=time.perf_counter();receipts=[];ledger_before=spent()
    for i,job in enumerate(jobs):
        ledger=spent()
        if ledger['total_child_and_development_cpu']+18>1200:
            e.save(folder/'INCOMPLETE.json',{'reason':'global CPU guard','ledger':ledger,'completed':len(receipts)})
            return
        ip=folder/f'jobs/{i:03}.job.json';op=folder/f'jobs/{i:03}.result.json';e.save(ip,job)
        t=time.perf_counter();c=resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            result=subprocess.run([e.PY,'-B',str(ROOT/'worker.py'),str(ip),str(op)],cwd=ROOT,env=e.env(),
                                  capture_output=True,text=True,timeout=18.)
            code,out,err=result.returncode,result.stdout,result.stderr
        except subprocess.TimeoutExpired as exc:
            code=-9;out=str(exc.stdout or '');err='parent process timeout: '+str(exc.stderr or '')
        finally:
            elapsed=time.perf_counter()-t;c1=resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu=c1.ru_utime+c1.ru_stime-c.ru_utime-c.ru_stime
        receipt={k:job[k] for k in ('arm','replicate','observation_id','budget_seconds')}
        receipt.update(index=i,result=str(op.relative_to(ROOT)),process_seconds=elapsed,process_cpu_seconds=cpu,
                       returncode=code,stderr=err,stdout=out,sha256=e.sha(op) if op.exists() else None)
        e.save(folder/f'jobs/{i:03}.receipt.json',receipt);receipts.append(receipt)
        if code or not op.exists():raise RuntimeError(f'failed job {i}; all costs preserved')
        o=e.load(op);assert all(p in f['files'] for p in o['dependencies'])
        if i%15==0:print(json.dumps({'job':i,'arm':job['arm'],'seconds':elapsed,'cpu_so_far':spent()}),flush=True)
    check();e.save(folder/'READY.json',{'receipts':receipts,'calls':len(receipts),'wall_seconds':time.perf_counter()-start,
        'child_cpu_seconds':sum(r['process_cpu_seconds'] for r in receipts),'ledger_before':ledger_before,'ledger_after':spent(),
        'truth_loaded':False,'freeze_sha':e.sha(ROOT/'FREEZE_switches.json')})

def state_key(s):
    cells=sorted((c['branch'],c['lower'],c['upper'],c['checked'],c['terminal']) for c in s['cells'])
    excludes=sorted((c['branch'],c['lower'],c['upper'],c['mode']) for c in s['exclusions'])
    return {'lower':s['lower'],'upper':s['upper'],'cells':cells,'reason':s['reason'],'exclusion_cells':excludes}

def state_audit():
    check();folder=ROOT/'outputs/switches';outputs={}
    for r in e.load(folder/'READY.json')['receipts']:
        assert e.sha(ROOT/r['result'])==r['sha256'];o=e.load(ROOT/r['result'])
        assert o['state'] is not None and o['prepare_failure'] is None
        outputs[o['arm'],o['replicate'],o['observation_id']]=o
    records=[]
    for (arm,rep,oid),o in outputs.items():
        if arm=='pure':continue
        a,b=state_key(o['state']),state_key(outputs['pure',rep,oid]['state'])
        records.append({'arm':arm,'replicate':rep,'observation_id':oid,
                        'fields_equal':{k:a[k]==b[k] for k in a},'all_equal':a==b})
    e.save(folder/'STATE_EQUIVALENCE.json',{'records':records,'all_equal':all(r['all_equal'] for r in records),'truth_loaded':False})
    assert all(r['all_equal'] for r in records if r['arm']=='off'),'all-off bridge not structurally equivalent'
    print(json.dumps({'states_compared':len(records),'all_equal':all(r['all_equal'] for r in records)}))

def profile():
    check();folder=ROOT/'outputs/switches';group=defaultdict(list)
    for r in e.load(folder/'READY.json')['receipts']:
        o=e.load(ROOT/r['result']);group[o['arm']].append((o,r))
    result={}
    for arm,items in group.items():
        times=defaultdict(float);counts=defaultdict(int);repeat_cold=defaultdict(list)
        for o,r in items:
            s=o['state'];repeat_cold[o['replicate']].append(r['process_seconds'])
            for k,v in s['component_seconds'].items():times[k]+=v
            for k,v in s.get('inner_counts',{}).items():counts[k]+=v
            for k,v in s.get('tile_reuse',{}).get('counts',{}).items():counts['reuse_'+k]+=v
            for event in s.get('tile_order_trace',[]):
                counts['actual_first_nonzero_tile']+=bool(event['order'] and event['order'][0]!=0)
            for a in s['certificate_attempts']:
                child=a.get('tile_results',[]) if a['mode']=='tiles' else [a]
                for x in child:
                    if x.get('error'): counts['cert_errors_or_deadlines']+=1
        result[arm]={'n_calls':len(items),'mean_components_seconds':{k:v/len(items) for k,v in times.items()},
                     'completed_lp_and_reuse_counts':dict(counts),
                     'cold_mean_by_repeat':{k:sum(v)/len(v) for k,v in repeat_cold.items()},
                     'cold_mean':sum(r['process_seconds'] for o,r in items)/len(items),
                     'cpu_mean':sum(r['process_cpu_seconds'] for o,r in items)/len(items)}
    pairs=[]
    keyed={(o['arm'],o['replicate'],o['observation_id']):r for items in group.values() for o,r in items}
    for rep in range(3):
        for p in e.load(e.A14/'data/confirm/public.json')['observations']:
            oid=p['observation_id'];r={a:keyed[a,rep,oid] for a in group}
            pairs.append({'replicate':rep,'observation_id':oid,'hot_minus_off':r['hot']['process_seconds']-r['off']['process_seconds'],
                'inherit_minus_off':r['inherit']['process_seconds']-r['off']['process_seconds'],
                'interaction':r['both']['process_seconds']-r['hot']['process_seconds']-r['inherit']['process_seconds']+r['off']['process_seconds']})
    e.save(folder/'PROFILE.json',{'arms':result,'paired_costs':pairs,'ledger':spent(),
        'note':'nested times must not be summed; counts are completed LP returns, exception attempts are not inferred'})
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()

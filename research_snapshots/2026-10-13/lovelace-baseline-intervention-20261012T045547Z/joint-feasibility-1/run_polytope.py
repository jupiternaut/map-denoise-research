"""A5 public sealed-leaf filtering; all actions precede private evaluation."""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction as Q
import argparse, csv, hashlib, importlib.util, json, os, signal, socket, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1')
PYTHON=Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
OLD=ROOT.parent/'appearance-variation-1/outputs/variation-v1'
DATA=ROOT.parent/'appearance-variation-1/data'
sys.path.insert(0,str(ROOT))
import polytope_filter as contractor
sys.path.insert(0,str(ROOT.parent/'appearance-variation-1'))
import variation_solver as original
spec=importlib.util.spec_from_file_location('a5_frozen_helpers',ROOT.parent/'appearance-variation-1/run_variation.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
exact,encode,load=helpers.exact,helpers.encode,helpers.load
select,make_row,aggregate=helpers.select,helpers.make_row,helpers.aggregate
METHODS=('calibrated','joint')
ARMS=('box','polytope')
LIMIT=3600
BUDGET_SCOPE='A5 full sealed-leaf main plus independent verification; no reset'
class blocked_alarm:
    def __enter__(self):self.previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    def __exit__(self,*_):signal.pthread_sigmask(signal.SIG_SETMASK,self.previous)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value,replace=False):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve()==ROOT:
        raise ValueError('write outside exact phase')
    if replace and path.name!='RUN_STATUS.json':raise ValueError('only status replaces')
    path.parent.mkdir(parents=True,exist_ok=True)
    with blocked_alarm():
        target=path.with_name(path.name+'.pending') if replace else path
        with target.open('x',encoding='utf-8') as f:
            json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        if replace:target.replace(path)
def csv_save(path,rows):
    with blocked_alarm(),Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def sources():
    result=load(OLD/'SOURCE_LOCK.json')['files'].copy()
    for p,h in result.items():
        if sha(p)!=h:raise RuntimeError('old source changed: '+p)
    result.update({str(p.resolve()):sha(p) for p in sorted({ROOT/'AGENTS.md',ROOT/'PROTOCOL.md',ROOT/'REGISTRATION.json',*ROOT.glob('*.py')})})
    return dict(sorted(result.items()))
def inputs():
    paths=list(load(OLD/'INPUTS.json')['files'])
    paths += [str(OLD/r) for r in ('SOURCE_LOCK.json','INPUTS.json','RUN_STATUS.json','STAGE_STATUS.json','verification/VERIFY.json','outers.json','decisions.json')]
    return {str(Path(p).resolve()):sha(p) for p in sorted(set(paths))}
def population():
    ps,ts=load(DATA/'observations.json'),load(DATA/'tasks.json')
    expected=[f'p{i:03d}b80e{v}' for v in range(2) for i in range(1,19)]
    if len(ps)!=36 or [p['observation_id'] for p in ps]!=expected:raise RuntimeError('public identity changed')
    if len(ts)!=324 or [t['task_id'] for t in ts]!=[o+f's{s}' for o in expected for s in range(9)]:
        raise RuntimeError('task identity changed')
    return ps,ts
def leaf_rows(public,arm,branch,node,method):
    ts=tuple(tuple(exact(x) for x in pair) for pair in arm['calibration_intervals'])
    areas=original.geometry.geometry_bounds(branch,exact(public['radius']),exact(node['lo']),exact(node['hi']),exact(public['baseline']))
    rows=original.necessary_rows(original.appearance_engine._image(public['observation'],public['clipped']),
        areas,method,clipped=public['clipped'],drift_bound=Q(1,10),calibration_intervals=ts)
    pairs=node['darkness_bounds']+(node['t_bounds'] if method=='joint' else [])
    bounds=tuple(tuple(exact(x) for x in pair) for pair in pairs)
    return rows,bounds
def filter_arm(public,old_arm,method,deadline):
    tick=time.perf_counter();arm=deepcopy(old_arm);records={};counts=Counter()
    old_count=0;single=0;budget_leaves=0;lp_seconds=0.0
    original.clear_geometry_cache()
    for branch in original.BRANCHES:
        tree=arm['branches'][branch]
        node_by_id={n['id']:n for n in tree['trace']}
        records[branch]=[];retained=[]
        for ident in tree['retained_ids']:
            if time.time()>=deadline:raise TimeoutError('A5 shared deadline during leaves')
            n=node_by_id[ident]
            if n['status'] not in ('retained_tolerance','unresolved_budget'):raise RuntimeError('unregistered retained status')
            rows,bounds=leaf_rows(public,old_arm,branch,n,method)
            result=contractor.classify(rows,bounds,deadline=time.monotonic()+max(0,deadline-time.time()))
            if result['status'] not in ('CERTIFIED_INFEASIBLE','FEASIBLE_WITNESS','UNKNOWN'):
                raise RuntimeError('unknown classification')
            if result['status']=='CERTIFIED_INFEASIBLE':
                if not contractor.certificate_check(rows,bounds,result['certificate']):
                    raise RuntimeError('unverified exclusion certificate')
            elif result['status']=='FEASIBLE_WITNESS':
                if not contractor.witness_check(rows,bounds,result['witness']):
                    raise RuntimeError('unverified exact witness')
                retained.append(ident)
            else:retained.append(ident)
            records[branch].append({'node_id':ident,'lo':n['lo'],'hi':n['hi'],'old_status':n['status'],'classification':result})
            counts[result['status']]+=1;single+=int(result['single_row_contradiction'])
            lp_seconds+=result['lp_seconds'];budget_leaves+=int(n['status']=='unresolved_budget');old_count+=1
        tree['retained_ids']=retained
        tree['hull']=None if not retained else [encode(min(exact(node_by_id[i]['lo']) for i in retained)),encode(max(exact(node_by_id[i]['hi']) for i in retained))]
    arm['outer']=original.frozen.union_outer(arm['branches'],method=method,clipped=public['clipped'],
        beta=original.BETA,epsilon=original.EPSILON,rounds=original.ROUNDS,max_boxes=original.MAX_BOXES,tolerance=original.TOLERANCE)
    arm['polytope_records']=records
    arm['polytope_stats']={'original_leaves':old_count,'status_counts':dict(sorted(counts.items())),
        'single_row_contradictions':single,'budget_leaves_rebuilt':budget_leaves,
        'lp_seconds':lp_seconds,'filter_seconds':time.perf_counter()-tick}
    return arm
def action(task,public,arm,name):
    result=select(arm['outer']['hull'],task['incumbent'],'hybrid')
    for field in ('output','gain_lower'):
        if result[field] is not None:result[field]=encode(result[field])
    return {'task_id':task['task_id'],'unit_id':public['unit_id'],'method':arm['outer']['mode'],
        'appearance_model':name,'reused':name=='box',**result}
def check_actions(ds,tasks):
    expected={(t['task_id'],a,m) for t in tasks for a in ARMS for m in METHODS}
    keys=[(d['task_id'],d['appearance_model'],d['method']) for d in ds]
    if len(ds)!=1296 or len(set(keys))!=1296 or set(keys)!=expected:
        raise RuntimeError('incomplete1296 two-arm action matrix')
    if any(d['reused']!=(d['appearance_model']=='box') for d in ds):raise RuntimeError('reuse identity')
def check_keep(ks,tasks):
    ids={t['task_id']:t['incumbent'] for t in tasks}
    if len(ks)!=324 or len({d['task_id'] for d in ks})!=324 or {d['task_id'] for d in ks}!=set(ids):
        raise RuntimeError('incomplete324 KEEP')
    if any(d['method']!='keep' or d['status']!='KEEP' or d['output']!=ids[d['task_id']] for d in ks):
        raise RuntimeError('nonidentity KEEP')
def evaluate(output,ps,ts,arms,ds,ks):
    truth={x['observation_id']:x for x in load(DATA/'private/truth.json')}
    tm={t['task_id']:t for t in ts};pm={p['observation_id']:p for p in ps}
    rows=[];keep=[]
    for d in ds:
        t=tm[d['task_id']];p=pm[t['observation_id']]
        row=make_row(t,p,truth[p['observation_id']],d,arms[p['observation_id']][d['appearance_model']][d['method']],d['appearance_model'])
        row['variation_index']=p['variation_index'];rows.append(row)
    for d in ks:
        t=tm[d['task_id']];p=pm[t['observation_id']]
        row=make_row(t,p,truth[p['observation_id']],d,{'outer':{'hull':None}},'keep')
        row['variation_index']=p['variation_index'];keep.append(row)
    cov=[]
    for p in ps:
        for a in ARMS:
            for method in METHODS:
                c=helpers.coverage(p,truth[p['observation_id']],method,arms[p['observation_id']][a][method],'bounded')
                c['appearance_model']=a;cov.append(c)
    for c in cov:
        if not all(c[k] is True for k in ('covered','generating_branch_covered','latent_material_covered','joint_witness_covered')):
            raise RuntimeError('lost actual simultaneous generating witness')
        if c['method']=='joint' and c['latent_bias_covered'] is not True:raise RuntimeError('lost bias')
    if any(r['status']!='KEEP' for r in rows if exact(r['initial_ae_mm'])==0):raise RuntimeError('moved correct input')
    for rel,values in (('results',rows),('keep/results',keep)):
        save(output/(rel+'.json'),values);csv_save(output/(rel+'.csv'),values)
    save(output/'coverage.json',cov)
    save(output/'summary.json',{'methods':aggregate(rows),'keep':aggregate(keep)})
def run(output):
    output=Path(output).resolve()
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or Path(sys.executable)!=PYTHON or output.parent!=ROOT/'outputs':
        raise RuntimeError('wrong target or runtime')
    budget=load(output/'BUDGET.json');start=budget['start_epoch'];deadline=budget['deadline_epoch']
    source_hashes=load(output/'SOURCE_LOCK.json')['files']
    if sources()!=source_hashes:raise RuntimeError('source changed')
    receipt=load(OLD/'verification/VERIFY.json')
    if receipt['status']!='PASS' or receipt['output']!=str(OLD):raise RuntimeError('old receipt wrong target')
    input_hashes=inputs();save(output/'INPUTS.json',{'files':input_hashes})
    save(output/'REGISTRATION.json',load(ROOT/'REGISTRATION.json'))
    ps,ts=population();old=load(OLD/'outers.json')
    old_ds={(d['task_id'],d['method']):d for d in load(OLD/'decisions.json') if d['appearance_model']=='bounded'}
    save(output/'observations.json',ps);save(output/'tasks.json',ts)
    taskmap=defaultdict(list)
    for t in ts:taskmap[t['observation_id']].append(t)
    arms={};ds=[];ks=[];timing=[];calls=0
    def timed_out(*_):raise TimeoutError('A5 shared deadline')
    prev=signal.signal(signal.SIGALRM,timed_out)
    if deadline<=time.time():raise TimeoutError('no main budget')
    signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    try:
        for i,p in enumerate(ps):
            oid=p['observation_id'];arms[oid]={'box':{},'polytope':{}}
            for t in taskmap[oid]:
                ks.append({'task_id':t['task_id'],'unit_id':p['unit_id'],'method':'keep','appearance_model':'keep',
                    'reused':True,'output':t['incumbent'],'gain_lower':encode(0),'status':'KEEP','reason':'identity','branch':'keep'})
            for method in METHODS:
                b=old[oid]['bounded'][method];arms[oid]['box'][method]=b
                new=filter_arm(p,b,method,deadline);arms[oid]['polytope'][method]=new;calls+=1
                timing.append({'observation_id':oid,'variation_index':p['variation_index'],'method':method,**new['polytope_stats']})
                for name,arm in (('box',b),('polytope',new)):
                    for t in taskmap[oid]:
                        d=action(t,p,arm,name)
                        if name=='box':
                            for key in ('status','output','gain_lower','reason','branch'):
                                if d[key]!=old_ds[(t['task_id'],method)][key]:raise RuntimeError('baseline action changed')
                        ds.append(d)
            save(output/f'checkpoint/{oid}.json',{'observation_id':oid,'arms':arms[oid]})
            save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':i+1,'cases_complete':calls,
                'decisions_complete':len(ds),'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
            print(json.dumps({'stage':'A5','observations_complete':i+1,'cases_complete':calls,'elapsed_seconds':round(time.time()-start,1)}),flush=True)
        save(output/'outers.json',arms);save(output/'decisions.json',ds);save(output/'keep/decisions.json',ks);save(output/'TIMING.json',timing)
        ds=load(output/'decisions.json');ks=load(output/'keep/decisions.json');arms=load(output/'outers.json')
        check_actions(ds,ts);check_keep(ks,ts)
        if calls!=72 or len(arms)!=36:raise RuntimeError('incomplete72 cases')
        if sum(c['original_leaves'] for c in timing)!=5103:raise RuntimeError('changed leaf population')
        evaluate(output,ps,ts,arms,ds,ks)
        if sources()!=source_hashes or inputs()!=input_hashes:raise RuntimeError('frozen bytes changed')
        save(output/'RUN_STATUS.json',{'status':'COMPLETE','observations_complete':36,'cases_complete':72,'decisions_complete':1296,
            'leaf_checks':5103,'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prev)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/polytope-v1');run(p.parse_args().output)


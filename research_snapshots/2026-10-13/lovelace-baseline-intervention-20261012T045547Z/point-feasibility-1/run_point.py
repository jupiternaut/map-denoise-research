"""A6 exact public incumbent-depth diagnostics, never a new geometry policy."""
from collections import Counter, defaultdict
from fractions import Fraction as Q
import argparse, csv, hashlib, json, os, signal, socket, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1')
PYTHON=Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
PRIOR=ROOT.parent/'joint-feasibility-1/outputs/polytope-v1'
DATA=ROOT.parent/'appearance-variation-1/data'
sys.path.insert(0,str(ROOT))
import point_solver as solver
exact,encode=solver.exact,solver.encode
METHODS=('calibrated','joint')
BRANCHES=('none','partial','full')
LIMIT=3600
BUDGET_SCOPE='A6 exact incumbent-point diagnostic main plus independent verification; no reset'
class blocked_alarm:
    def __enter__(self):self.previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    def __exit__(self,*_):signal.pthread_sigmask(signal.SIG_SETMASK,self.previous)
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value,replace=False):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve()==ROOT:raise ValueError('write outside exact phase')
    if replace and path.name!='RUN_STATUS.json':raise ValueError('only run status replaces')
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
    result=load(PRIOR/'SOURCE_LOCK.json')['files'].copy()
    for p,h in result.items():
        if sha(p)!=h:raise RuntimeError('old source changed: '+p)
    result.update({str(p.resolve()):sha(p) for p in sorted({ROOT/'AGENTS.md',ROOT/'PROTOCOL.md',ROOT/'REGISTRATION.json',*ROOT.glob('*.py')})})
    return dict(sorted(result.items()))
def inputs():
    paths=list(load(PRIOR/'INPUTS.json')['files'])
    paths += [str(PRIOR/r) for r in ('SOURCE_LOCK.json','INPUTS.json','RUN_STATUS.json','STAGE_STATUS.json','verification/VERIFY.json','outers.json','decisions.json')]
    return {str(Path(p).resolve()):sha(p) for p in sorted(set(paths))}
def population():
    ps,ts=load(DATA/'observations.json'),load(DATA/'tasks.json')
    expected=[f'p{i:03d}b80e{v}' for v in range(2) for i in range(1,19)]
    if len(ps)!=36 or [p['observation_id'] for p in ps]!=expected:raise RuntimeError('changed public36')
    if len(ts)!=324 or [t['task_id'] for t in ts]!=[o+f's{s}' for o in expected for s in range(9)]:
        raise RuntimeError('changed public324')
    return ps,ts
def containing(arm,depth):
    z=exact(depth);result=[]
    for branch in BRANCHES:
        tree=arm['branches'][branch];nodes={n['id']:n for n in tree['trace']}
        result.extend((branch,nodes[i]) for i in tree['retained_ids']
                      if exact(nodes[i]['lo'])<=z<=exact(nodes[i]['hi']))
    return result
def check_classification(rows,bounds,result):
    if result['status']=='CERTIFIED_INFEASIBLE':
        if not solver.contractor.certificate_check(rows,bounds,result['certificate']):raise RuntimeError('unverified exact exclusion')
    elif result['status']=='FEASIBLE_WITNESS':
        if not solver.contractor.witness_check(rows,bounds,result['witness']):raise RuntimeError('unverified exact witness')
    elif result['status']!='UNKNOWN':raise RuntimeError('unregistered classification')
def classify(public,arm,method,branch,z,node,deadline):
    if time.time()>=deadline:raise TimeoutError('shared point deadline')
    rows,bounds=solver.build_problem(public,arm,method,branch,z,node)
    result=solver.contractor.classify(rows,bounds,deadline=time.monotonic()+max(0,deadline-time.time()))
    check_classification(rows,bounds,result)
    return result
def diagnose(task,public,old_arm,prior_arm,method,a4_action,a5_action,deadline):
    tick=time.perf_counter();z=exact(task['incumbent']);local=[];global_records=[]
    prior_index={(b,c['node_id']):c['classification'] for b,cs in prior_arm['polytope_records'].items() for c in cs}
    matched=containing(old_arm,z)
    for branch,node in matched:
        result=classify(public,old_arm,method,branch,z,node,deadline)
        local.append({'branch':branch,'node_id':node['id'],'lo':node['lo'],'hi':node['hi'],
                      'box_classification':prior_index[branch,node['id']],'point_classification':result})
    for branch in BRANCHES:
        result=classify(public,old_arm,method,branch,z,None,deadline)
        global_records.append({'branch':branch,'point_classification':result})
        if result['status']=='FEASIBLE_WITNESS':
            same_branch=[n for b,n in matched if b==branch]
            if not same_branch:raise RuntimeError('global point witness outside frozen branch support')
            for node in same_branch:
                rows,bounds=solver.build_problem(public,old_arm,method,branch,z,node)
                if not solver.contractor.witness_check(rows,bounds,result['witness']):
                    raise RuntimeError('global witness contradicts same-leaf bounds')
        elif result['status']=='CERTIFIED_INFEASIBLE':
            if any(c['branch']==branch and c['point_classification']['status']=='FEASIBLE_WITNESS' for c in local):
                raise RuntimeError('local point witness contradicts global point proof')
    hull=old_arm['outer']['hull']
    return {'task_id':task['task_id'],'observation_id':public['observation_id'],'unit_id':public['unit_id'],
        'variation_index':public['variation_index'],'method':method,'incumbent':task['incumbent'],
        'source_hull_contains':hull is not None and exact(hull[0])<=z<=exact(hull[1]),
        'source_union_contains':bool(matched),'local':local,'global':global_records,
        'box_status':solver.summarize([c['box_classification']['status'] for c in local]),
        'local_status':solver.summarize([c['point_classification']['status'] for c in local]),
        'global_status':solver.summarize([c['point_classification']['status'] for c in global_records]),
        'a4_action':{k:a4_action[k] for k in ('status','output')},
        'a5_action':{k:a5_action[k] for k in ('status','output')},
        'diagnostic_seconds':time.perf_counter()-tick,
        'lp_seconds':sum(c['point_classification']['lp_seconds'] for c in local+global_records)}
def check_records(records,tasks):
    expected={(t['task_id'],m) for t in tasks for m in METHODS}
    keys=[(d['task_id'],d['method']) for d in records]
    if len(records)!=648 or len(set(keys))!=648 or set(keys)!=expected:raise RuntimeError('incomplete unique648 point matrix')
    globals_count=sum(len(d['global']) for d in records)
    locals_count=Counter({m:sum(len(d['local']) for d in records if d['method']==m) for m in METHODS})
    if globals_count!=1944 or locals_count!={'calibrated':494,'joint':486}:raise RuntimeError('changed global1944/local980')
    for d in records:
        if [x['branch'] for x in d['global']]!=list(BRANCHES):raise RuntimeError('missing branch or early exit')
        keys=[(x['branch'],x['node_id']) for x in d['local']]
        if len(keys)!=len(set(keys)):raise RuntimeError('duplicated containing leaf')
def summary_rows(rows):
    groups=defaultdict(list)
    for r in rows:
        common=(r['unit_id'],r['method'])
        groups[common+('all','')].append(r)
        shifted=Q(r['initial_ae_mm'])!=0
        groups[common+('all_shifted' if shifted else 'correct','')].append(r)
        if shifted:
            groups[common+('negative_shifted' if r['case']=='equal_full_negative' else 'informative_shifted','')].append(r)
            groups[common+('case_shifted',r['case'])].append(r)
            if r['case']!='equal_full_negative':groups[common+('depth_informative',r['depth_mm'])].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        out.append(dict(zip(('unit_id','method','group','value'),key),n=len(rs),
            **{label:dict(sorted(Counter(r[label] for r in rs).items())) for label in ('box_status','local_status','global_status')},
            new_point_excluded=sum(r['box_status']!='CERTIFIED_INFEASIBLE' and r['box_status']!='SOURCE_EXCLUDED'
                                   and r['local_status']=='CERTIFIED_INFEASIBLE' for r in rs),
            a5_moves=sum(r['a5_status']=='MOVE' for r in rs)))
    return out
def evaluation_row(record,task,public,fact):
    z,truth=exact(task['incumbent']),exact(fact['truth'])
    return {'task_id':task['task_id'],'observation_id':public['observation_id'],'pair_id':public['pair_id'],
        'unit_id':public['unit_id'],'variation_index':public['variation_index'],'method':record['method'],
        'case':fact['case'],'depth_mm':str(truth),'incumbent_mm':str(z),'initial_ae_mm':str(abs(z-truth)),
        'source_hull_contains':record['source_hull_contains'],'source_union_contains':record['source_union_contains'],
        'box_status':record['box_status'],'local_status':record['local_status'],'global_status':record['global_status'],
        'local_checks':len(record['local']),'a4_status':record['a4_action']['status'],
        'a4_output_mm':str(exact(record['a4_action']['output'])),'a5_status':record['a5_action']['status'],
        'a5_output_mm':str(exact(record['a5_action']['output']))}
def actual_coverage(public,fact,arm,method,records):
    depth=exact(fact['truth']);branch=fact['rear_branch']
    darkness=[1-exact(c) for c in fact['view_colors']]
    bias=[Q(1,100)-exact(b) for b in fact['actual_view_bias']]
    witness=[encode(x) for x in darkness+(bias if method=='joint' else [])]
    rows,bounds=solver.build_problem(public,arm,method,branch,depth)
    if not solver.contractor.witness_check(rows,bounds,witness):raise RuntimeError('lost global true point witness')
    matched=[n for b,n in containing(arm,depth) if b==branch]
    if not matched:raise RuntimeError('lost actual generating branch')
    for node in matched:
        rows,bounds=solver.build_problem(public,arm,method,branch,depth,node)
        if not solver.contractor.witness_check(rows,bounds,witness):raise RuntimeError('lost same-leaf true point witness')
    correct=[d for d in records if d['method']==method and d['incumbent']==encode(depth)]
    if len(correct)!=1 or correct[0]['local_status']=='CERTIFIED_INFEASIBLE' or correct[0]['global_status']=='CERTIFIED_INFEASIBLE':
        raise RuntimeError('wrong correct-input diagnosis')
    return {'observation_id':public['observation_id'],'method':method,'generating_branch':branch,
        'global_actual_point_witness':True,'all_containing_actual_leaf_witnesses':True,
        'actual_leaf_ids':[n['id'] for n in matched]}
def evaluate(output,ps,ts,old,records):
    facts={f['observation_id']:f for f in load(DATA/'private/truth.json')}
    pm={p['observation_id']:p for p in ps};tm={t['task_id']:t for t in ts}
    rows=[evaluation_row(d,tm[d['task_id']],pm[d['observation_id']],facts[d['observation_id']]) for d in records]
    coverage=[]
    for p in ps:
        for method in METHODS:
            coverage.append(actual_coverage(p,facts[p['observation_id']],old[p['observation_id']]['box'][method],method,
                                           [d for d in records if d['observation_id']==p['observation_id']]))
    save(output/'results.json',rows);csv_save(output/'results.csv',rows)
    save(output/'coverage.json',coverage);save(output/'summary.json',summary_rows(rows))
def run(output):
    output=Path(output).resolve()
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or Path(sys.executable)!=PYTHON or output.parent!=ROOT/'outputs':
        raise RuntimeError('wrong target or runtime')
    budget=load(output/'BUDGET.json');start=budget['start_epoch'];deadline=budget['deadline_epoch']
    locked=load(output/'SOURCE_LOCK.json')['files']
    if sources()!=locked:raise RuntimeError('source drift')
    receipt=load(PRIOR/'verification/VERIFY.json');stage=load(PRIOR/'STAGE_STATUS.json')
    if receipt['status']!='PASS' or receipt['output']!=str(PRIOR) or stage['status']!='COMPLETE':
        raise RuntimeError('old exact target not COMPLETE/PASS')
    input_hashes=inputs();save(output/'INPUTS.json',{'files':input_hashes})
    save(output/'REGISTRATION.json',load(ROOT/'REGISTRATION.json'))
    ps,ts=population();old=load(PRIOR/'outers.json')
    a5_actions={(d['task_id'],d['method']):d for d in load(PRIOR/'decisions.json') if d['appearance_model']=='polytope'}
    a4_actions={(d['task_id'],d['method']):d for d in load(PRIOR/'decisions.json') if d['appearance_model']=='box'}
    save(output/'observations.json',ps);save(output/'tasks.json',ts)
    taskmap=defaultdict(list)
    for t in ts:taskmap[t['observation_id']].append(t)
    records=[]
    def timed_out(*_):raise TimeoutError('A6 shared deadline')
    prev=signal.signal(signal.SIGALRM,timed_out)
    if deadline<=time.time():raise TimeoutError('no main budget')
    signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    try:
        for i,p in enumerate(ps):
            oid=p['observation_id'];done=[]
            solver.original.clear_geometry_cache()
            for method in METHODS:
                arm=old[oid]['box'][method];prior_arm=old[oid]['polytope'][method]
                for t in taskmap[oid]:
                    done.append(diagnose(t,p,arm,prior_arm,method,a4_actions[t['task_id'],method],a5_actions[t['task_id'],method],deadline))
            records.extend(done)
            save(output/f'checkpoint/{oid}.json',{'observation_id':oid,'records':done})
            save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':i+1,'diagnoses_complete':len(records),
                'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
            print(json.dumps({'stage':'A6','observations_complete':i+1,'diagnoses_complete':len(records),
                              'elapsed_seconds':round(time.time()-start,1)}),flush=True)
        save(output/'diagnoses.json',records)
        records=load(output/'diagnoses.json')
        check_records(records,ts)
        evaluate(output,ps,ts,old,records)
        if sources()!=locked or inputs()!=input_hashes:raise RuntimeError('frozen bytes changed')
        save(output/'RUN_STATUS.json',{'status':'COMPLETE','observations_complete':36,'diagnoses_complete':648,
             'local_checks':980,'global_checks':1944,'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prev)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/point-v1');run(p.parse_args().output)


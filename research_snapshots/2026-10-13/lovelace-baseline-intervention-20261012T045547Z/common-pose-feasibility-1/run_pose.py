"""A7 public common-pose completion; never geometry selection."""
import argparse,csv,hashlib,json,os,signal,socket,sys,time
from collections import Counter,defaultdict
from fractions import Fraction as Q
from pathlib import Path
sys.dont_write_bytecode=True
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_n]='1'
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1')
PARENT=ROOT.parent
A6=PARENT/'point-feasibility-1'
PRIOR=A6/'outputs/point-v1'
A5=PARENT/'joint-feasibility-1/outputs/polytope-v1'
DATA=PARENT/'appearance-variation-1/data'
PYTHON=Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(A6))
import pose_solver as solver
import pose_lift as lift
import point_solver as old
exact,encode=old.exact,old.encode
BRANCHES=old.BRANCHES
LIMIT=3600
BUDGET_SCOPE='A7 joint fixed-depth common-pose main plus independent verification; no reset'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class blocked_alarm:
    def __enter__(self):self.previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    def __exit__(self,*_):signal.pthread_sigmask(signal.SIG_SETMASK,self.previous)
def save(path,value,replace=False):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve()==ROOT:raise ValueError('wrong write root')
    if replace and path.name!='RUN_STATUS.json':raise ValueError('replace only status')
    path.parent.mkdir(parents=True,exist_ok=True)
    with blocked_alarm():
        target=path.with_name(path.name+'.pending') if replace else path
        with target.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        if replace:target.replace(path)
def sources():
    result=load(PRIOR/'SOURCE_LOCK.json')['files'].copy()
    for p,h in result.items():
        if sha(p)!=h:raise RuntimeError('frozen source drift '+p)
    for p in sorted({ROOT/'AGENTS.md',ROOT/'PROTOCOL.md',ROOT/'REGISTRATION.json',*ROOT.glob('*.py')}):result[str(p)]=sha(p)
    return dict(sorted(result.items()))
def inputs():
    inv=load(A6/'ARTIFACTS.json')
    for rel,h in inv['files'].items():
        if sha(A6/rel)!=h:raise RuntimeError('prior indexed artifact drift '+rel)
    paths=[A6/'ARTIFACTS.json',A6/'FINAL_SEAL.json',PRIOR/'diagnoses.json',PRIOR/'observations.json',PRIOR/'tasks.json',
           PRIOR/'STAGE_STATUS.json',PRIOR/'verification/VERIFY.json',A5/'outers.json',DATA/'private/truth.json']
    return {str(p):sha(p) for p in paths}
def aggregate(statuses):
    vals=tuple(statuses)
    if len(vals)!=3 or any(s not in ('FORWARD_WITNESS','UNKNOWN','INHERITED_INFEASIBLE') for s in vals):raise ValueError('3 branches required')
    if 'FORWARD_WITNESS' in vals:return 'FORWARD_WITNESS'
    if all(s=='INHERITED_INFEASIBLE' for s in vals):return 'INHERITED_INFEASIBLE'
    return 'UNKNOWN'
def make_seed_witness(public,depth,branch,seed,offsets):
    ds=tuple(exact(v) for v in seed[:9]);ts=tuple(exact(v) for v in seed[9:])
    areas=old.geometry._point_areas(branch,exact(depth),tuple(exact(v) for v in offsets),Q(80))
    noise=[]
    for i,obs in enumerate(public['observation']):
        pixel=i//3;view=pixel//48
        pred=1-sum(a*d for a,d in zip(areas[pixel],ds[3*view:3*view+3]))+Q(1,100)-ts[view]
        y=exact(obs)
        noise.append(encode(-Q(1,100) if y==0 else Q(1,100) if y==1 else y-pred))
    return {'offsets':offsets,'darkness':seed[:9],'t':seed[9:],'noise':noise}
def branch_record(public,arm,task,source,deadline):
    if time.time()>=deadline:raise TimeoutError('A7 shared deadline before branch')
    branch=source['branch'];c=source['point_classification'];z=task['incumbent']
    rows,bounds=old.build_problem(public,arm,'joint',branch,z)
    base={'branch':branch,'source_status':c['status']}
    if c['status']=='CERTIFIED_INFEASIBLE':
        if not old.certificate_check(rows,bounds,c['certificate']):raise RuntimeError('bad inherited certificate')
        return {**base,'status':'INHERITED_INFEASIBLE','seed':None,'search':None,'witness':None}
    if c['status']!='FEASIBLE_WITNESS' or not old.witness_check(rows,bounds,c['witness']):raise RuntimeError('needs exact prior joint branch classification')
    seed=lift.lift_seed(public,arm,z,branch,c['witness'])
    seed_w=None
    if seed['status']=='SEED_LIFTED':
        seed_w=make_seed_witness(public,z,branch,c['witness'],seed['offsets'])
        if not solver.check_forward_witness(public,arm,z,branch,seed_w):raise RuntimeError('invalid lifted forward seed')
    search=solver.complete_branch(public,arm,z,branch,deadline=time.monotonic()+max(0,deadline-time.time()))
    if time.time()>=deadline:raise TimeoutError('A7 shared deadline after search')
    if search['status']=='FORWARD_WITNESS':
        if not solver.check_forward_witness(public,arm,z,branch,search['witness']):raise RuntimeError('invalid searched witness')
        chosen=search['witness']
    elif search['status']=='UNKNOWN':
        if search['witness'] is not None:raise RuntimeError('UNKNOWN has witness')
        chosen=seed_w
    else:raise RuntimeError('new search exclusion forbidden')
    return {**base,'status':'FORWARD_WITNESS' if chosen else 'UNKNOWN','seed':seed,'search':search,'witness':chosen}
def check_records(records,tasks):
    if len(records)!=324 or [r['task_id'] for r in records]!=[t['task_id'] for t in tasks]:raise RuntimeError('public324 matrix')
    if sum(len(r['branches']) for r in records)!=972:raise RuntimeError('972 branches')
    for r in records:
        if [c['branch'] for c in r['branches']]!=list(BRANCHES):raise RuntimeError('branch completeness')
        if r['status']!=aggregate(c['status'] for c in r['branches']):raise RuntimeError('wrong union')
def summary_rows(rows):
    groups=defaultdict(list)
    for r in rows:
        u=r['unit_id'];groups[u,'all',''].append(r)
        shifted=Q(r['initial_ae_mm'])!=0
        groups[u,'shifted' if shifted else 'correct',''].append(r)
        if shifted:
            groups[u,'negative_shifted' if r['case']=='equal_full_negative' else 'informative_shifted',''].append(r)
            groups[u,'case_shifted',r['case']].append(r)
            if r['case']!='equal_full_negative':groups[u,'depth_informative',r['truth_mm']].append(r)
    out=[]
    for (u,g,v),rs in sorted(groups.items()):
        out.append({'unit_id':u,'group':g,'value':v,'n':len(rs),
          **{k:dict(sorted(Counter(r[k] for r in rs).items())) for k in ('a6_status','status','nominal_status','seed_status')},
          'additional_verified_over_nominal':sum(r['status']=='FORWARD_WITNESS' and r['nominal_status']!='FORWARD_WITNESS' for r in rs)})
    return out
def evaluate(output,ps,tasks,records):
    # First private JSON parse is AFTER saved complete records gate.
    facts={f['observation_id']:f for f in load(DATA/'private/truth.json')}
    pm={p['observation_id']:p for p in ps};tm={t['task_id']:t for t in tasks}
    rows=[]
    for r in records:
        p=pm[r['observation_id']];f=facts[r['observation_id']];t=tm[r['task_id']]
        ns=['FORWARD_WITNESS' if c['search'] and c['search']['center_status']=='FORWARD_WITNESS' else
            'INHERITED_INFEASIBLE' if c['status']=='INHERITED_INFEASIBLE' else 'UNKNOWN' for c in r['branches']]
        ss=['FORWARD_WITNESS' if c['seed'] and c['seed']['status']=='SEED_LIFTED' else
            'INHERITED_INFEASIBLE' if c['status']=='INHERITED_INFEASIBLE' else 'UNKNOWN' for c in r['branches']]
        rows.append({'task_id':r['task_id'],'observation_id':r['observation_id'],'unit_id':p['unit_id'],
            'variation_index':p['variation_index'],'case':f['case'],'truth_mm':str(exact(f['truth'])),
            'incumbent_mm':str(exact(t['incumbent'])),'initial_ae_mm':str(abs(exact(t['incumbent'])-exact(f['truth']))),
            'a6_status':r['a6_status'],'status':r['status'],'nominal_status':aggregate(ns),'seed_status':aggregate(ss)})
    save(output/'results.json',rows)
    with blocked_alarm(),(output/'results.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(output/'summary.json',summary_rows(rows))
def run(output):
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or Path(sys.executable)!=PYTHON or output!=ROOT/'outputs/pose-v1':raise RuntimeError('wrong target/runtime')
    budget=load(output/'BUDGET.json');start,deadline=budget['start_epoch'],budget['deadline_epoch']
    locked=load(output/'SOURCE_LOCK.json')['files']
    if sources()!=locked:raise RuntimeError('source drift')
    if load(PRIOR/'STAGE_STATUS.json')['status']!='COMPLETE' or load(PRIOR/'verification/VERIFY.json')['status']!='PASS':raise RuntimeError('prior notready')
    hashes=inputs();save(output/'INPUTS.json',{'files':hashes});save(output/'REGISTRATION.json',load(ROOT/'REGISTRATION.json'))
    ps,ts=load(PRIOR/'observations.json'),load(PRIOR/'tasks.json')
    expected=[f'p{i:03d}b80e{v}' for v in range(2) for i in range(1,19)]
    if [p['observation_id'] for p in ps]!=expected or [t['task_id'] for t in ts]!=[o+f's{s}' for o in expected for s in range(9)]:raise RuntimeError('population changed')
    prior={r['task_id']:r for r in load(PRIOR/'diagnoses.json') if r['method']=='joint'}
    old_arms=load(A5/'outers.json')
    save(output/'observations.json',ps);save(output/'tasks.json',ts)
    grouped=defaultdict(list)
    for t in ts:grouped[t['observation_id']].append(t)
    records=[]
    def timeout(*_):raise TimeoutError('A7 shared deadline')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    try:
        for i,p in enumerate(ps):
            oid=p['observation_id'];arm=old_arms[oid]['box']['joint'];done=[]
            old.geometry.clear_geometry_cache()
            for task in grouped[oid]:
                src=prior[task['task_id']]
                branches=[branch_record(p,arm,task,c,deadline) for c in src['global']]
                done.append({'task_id':task['task_id'],'observation_id':oid,'incumbent':task['incumbent'],
                    'a6_status':src['global_status'],'status':aggregate(c['status'] for c in branches),'branches':branches})
            records.extend(done);save(output/f'checkpoint/{oid}.json',{'observation_id':oid,'records':done})
            save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':i+1,'diagnoses_complete':len(records),
                    'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
            print(json.dumps({'stage':'A7','observations_complete':i+1,'diagnoses_complete':len(records),'seconds':round(time.time()-start,2)}),flush=True)
        save(output/'diagnoses.json',records);records=load(output/'diagnoses.json');check_records(records,ts)
        evaluate(output,ps,ts,records)
        if sources()!=locked or inputs()!=hashes:raise RuntimeError('frozen bytes drift')
        save(output/'RUN_STATUS.json',{'status':'COMPLETE','observations_complete':36,'diagnoses_complete':324,
                'branch_records':972,'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/pose-v1');run(parser.parse_args().output.resolve())

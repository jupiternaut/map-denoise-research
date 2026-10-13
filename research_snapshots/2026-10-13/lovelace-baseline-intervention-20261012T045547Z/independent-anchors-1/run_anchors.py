"""A8 public-only 2x2 inference. No private JSON is parsed here."""
import argparse, hashlib, json, os, signal, socket, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/independent-anchors-1')
P=ROOT.parent; A6=P/'point-feasibility-1'; A7=P/'common-pose-feasibility-1'
PRIOR=A6/'outputs/point-v1';OLDOUT=A7/'outputs/pose-v1'
PYTHON=Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
sys.path.insert(0,str(ROOT))
import anchor_solver as solver
exact,enc=solver.exact,solver.encode
SCIENCE_NAMES=('AGENTS.md','PROTOCOL.md','REGISTRATION.json','prepare_anchors.py','anchor_solver.py',
    'run_anchors.py','execute_anchors.py','verify_anchors.py','test_anchor_solver.py',
    'test_verify_anchors.py','test_anchor_pipeline.py')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x,replace=False):
    p=Path(p)
    if not p.resolve().is_relative_to(ROOT/'outputs'):raise ValueError('wrong output target')
    if replace and p.name!='RUN_STATUS.json':raise ValueError('only status replaces')
    p.parent.mkdir(parents=True,exist_ok=True)
    oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    try:
        target=p.with_name(p.name+'.pending') if replace else p
        with target.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        if replace:target.replace(p)
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
def sources():
    files=load(PRIOR/'SOURCE_LOCK.json')['files'].copy()
    for p in (A7/'pose_solver.py',A7/'pose_lift.py',*[ROOT/n for n in SCIENCE_NAMES]):files[str(p)]=sha(p)
    return dict(sorted(files.items()))
def input_paths():
    return (A6/'ARTIFACTS.json',A6/'FINAL_SEAL.json',PRIOR/'diagnoses.json',PRIOR/'observations.json',
        PRIOR/'tasks.json',PRIOR/'STAGE_STATUS.json',PRIOR/'verification/VERIFY.json',
        P/'joint-feasibility-1/outputs/polytope-v1/outers.json',
        A7/'ARTIFACTS.json',A7/'FINAL_SEAL.json',OLDOUT/'diagnoses.json',OLDOUT/'STAGE_STATUS.json',
        A7/'diagnostic-audit/SAVED_RECORD_AUDIT.json',P/'appearance-variation-1/data/private/truth.json',
        ROOT/'data/anchors.json',ROOT/'data/private/measurements.json',ROOT/'data/GENERATION.json')
def inputs():return {str(p):sha(p) for p in input_paths()}
def validate_old_seals():
    for root in (A6,A7):
        inv=load(root/'ARTIFACTS.json')
        for rel,h in inv['files'].items():
            if sha(root/rel)!=h:raise RuntimeError('old sealed artifact drift '+str(root/rel))
    for lock in (PRIOR/'INPUTS.json',OLDOUT/'INPUTS.json'):
        for path,h in load(lock)['files'].items():
            if sha(path)!=h:raise RuntimeError('old frozen external input drift '+path)
    if load(PRIOR/'STAGE_STATUS.json')['status']!='COMPLETE':raise RuntimeError('A6 not complete')
    if load(PRIOR/'verification/VERIFY.json')['status']!='PASS':raise RuntimeError('A6 not verified')
    audit=load(A7/'diagnostic-audit/SAVED_RECORD_AUDIT.json')
    if audit['status']!='SAVED_RECORD_NUMERICAL_PASS' or audit['formal_acceptance'] is not False:
        raise RuntimeError('A7 diagnostic provenance mismatch')
    if load(OLDOUT/'STAGE_STATUS.json')['status']!='INCOMPLETE':raise RuntimeError('A7 status must remain incomplete')
def aggregate(values):
    values=tuple(values)
    if len(values)!=3 or any(s not in ('FORWARD_WITNESS','UNKNOWN','INHERITED_INFEASIBLE','CERTIFIED_INFEASIBLE') for s in values):
        raise ValueError('three legal statuses required')
    if 'FORWARD_WITNESS' in values:return 'FORWARD_WITNESS'
    if all(s in ('INHERITED_INFEASIBLE','CERTIFIED_INFEASIBLE') for s in values):return 'CERTIFIED_INFEASIBLE'
    return 'UNKNOWN'
def branch_record(public,oldarm,anchor,name,task,old6,old7,main_deadline):
    branch=old6['branch'];c=old6['point_classification'];z=task['incumbent']
    base={'branch':branch,'source_status':c['status'],'inherited_classification':None,
          'outer_classification':None,'saved_witness_status':'NOT_AVAILABLE','saved_extended_witness':None,
          'search':None,'witness':None,'termination':None}
    if c['status']=='CERTIFIED_INFEASIBLE':
        rows,bounds=solver.point.build_problem(public,oldarm,'joint',branch,z)
        if not solver.point.certificate_check(rows,bounds,c['certificate']):raise RuntimeError('bad old certificate')
        return {**base,'status':'INHERITED_INFEASIBLE','inherited_classification':c,'termination':'old_global_certificate'}
    if c['status']!='FEASIBLE_WITNESS':raise RuntimeError('unregistered prior status')
    if old7['status']=='FORWARD_WITNESS':
        if not solver.pose.check_forward_witness(public,oldarm,z,branch,old7['witness']):raise RuntimeError('bad saved photo witness')
        w=solver.revalidate_old_witness(public,oldarm,anchor,name,z,branch,old7['witness'])
        base.update(saved_witness_status='ACCEPTED' if w else 'REJECTED',saved_extended_witness=w)
        if w:return {**base,'status':'FORWARD_WITNESS','witness':w,'termination':'saved_exact_extension'}
    if time.time()>=main_deadline:
        return {**base,'status':'UNKNOWN','termination':'UNKNOWN_BUDGET'}
    rows,bounds=solver.build_problem(public,oldarm,anchor,name,z,branch)
    mono_end=time.monotonic()+max(0,main_deadline-time.time())
    classified=solver.classify(rows,bounds,deadline=mono_end)
    base['outer_classification']=classified
    if classified['status']=='CERTIFIED_INFEASIBLE':
        if not solver.certificate_check(rows,bounds,classified['certificate']):raise RuntimeError('bad new global certificate')
        return {**base,'status':'CERTIFIED_INFEASIBLE','termination':'new_global_certificate'}
    search=solver.complete_branch(public,oldarm,anchor,name,z,branch,deadline=mono_end)
    base['search']=search
    if search['status']=='FORWARD_WITNESS':
        w=search['witness']
        if not solver.check_forward_witness(public,oldarm,anchor,name,z,branch,w):raise RuntimeError('bad full witness')
        return {**base,'status':'FORWARD_WITNESS','witness':w,'termination':search['termination']}
    if search['status']!='UNKNOWN':raise RuntimeError('fixedpose/search exclusion forbidden')
    return {**base,'status':'UNKNOWN','termination':search['termination']}
def main(output):
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or Path(sys.executable)!=PYTHON or output!=ROOT/'outputs/anchor-v1':
        raise RuntimeError('wrong execution target')
    budget=load(output/'BUDGET.json');start=budget['start_epoch'];deadline=budget['deadline_epoch'];main_end=budget['main_deadline_epoch']
    frozen=load(output/'SOURCE_LOCK.json')['files']
    if sources()!=frozen or any(sha(p)!=h for p,h in frozen.items()):raise RuntimeError('source lock drift')
    validate_old_seals();input_hashes=inputs();save(output/'INPUTS.json',{'files':input_hashes})
    publics,tasks=load(PRIOR/'observations.json'),load(PRIOR/'tasks.json')
    anchors={a['observation_id']:a for a in load(ROOT/'data/anchors.json')}
    old6={r['task_id']:r for r in load(PRIOR/'diagnoses.json') if r['method']=='joint'}
    old7={r['task_id']:r for r in load(OLDOUT/'diagnoses.json')}
    oldarms=load(P/'joint-feasibility-1/outputs/polytope-v1/outers.json')
    expected=[f'p{i:03d}b80e{v}' for v in range(2) for i in range(1,19)]
    if [p['observation_id'] for p in publics]!=expected or [t['task_id'] for t in tasks]!=[o+f's{s}' for o in expected for s in range(9)]:
        raise RuntimeError('fixed population changed')
    save(output/'observations.json',publics);save(output/'tasks.json',tasks)
    save(output/'REGISTRATION.json',load(ROOT/'REGISTRATION.json'))
    def alarm(*_):raise TimeoutError('A8 shared whole deadline')
    oldalarm=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    records=[]
    try:
        for index,pub in enumerate(publics):
            oid=pub['observation_id'];oldarm=oldarms[oid]['box']['joint'];done=[]
            solver.point.clear_geometry_cache();solver.clear_geometry_cache()
            for task in tasks[9*index:9*(index+1)]:
                s6=old6[task['task_id']];s7=old7[task['task_id']]
                if [b['branch'] for b in s6['global']]!=list(solver.BRANCHES):raise RuntimeError('old branch identity')
                for name in solver.ARM_NAMES:
                    bs=[branch_record(pub,oldarm,anchors[oid],name,task,b6,b7,main_end)
                        for b6,b7 in zip(s6['global'],s7['branches'])]
                    done.append({'arm':name,'task_id':task['task_id'],'observation_id':oid,'incumbent':task['incumbent'],
                        'a6_status':s6['global_status'],'a7_status':s7['status'],
                        'status':aggregate(b['status'] for b in bs),'branches':bs})
            records.extend(done);save(output/f'checkpoint/{oid}.json',{'observation_id':oid,'records':done})
            save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':index+1,'records_complete':len(records),
                'seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
            print(json.dumps({'stage':'A8-main','observations_complete':index+1,'records':len(records),'seconds':round(time.time()-start,2)}),flush=True)
        if len(records)!=1296 or sum(len(r['branches']) for r in records)!=3888:raise RuntimeError('full matrix missing')
        save(output/'diagnoses.json',records)
        reread=load(output/'diagnoses.json')
        if reread!=records:raise RuntimeError('record readback mismatch')
        if sources()!=frozen or inputs()!=input_hashes:raise RuntimeError('frozen bytes changed')
        save(output/'RUN_STATUS.json',{'status':'COMPLETE','records_complete':1296,'branch_records':3888,
            'observations_complete':36,'seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldalarm)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/anchor-v1')
    main(ap.parse_args().output.resolve())

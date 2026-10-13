"""Fixed A4-V public matrix; every action persists before private evaluation."""
from collections import defaultdict
from fractions import Fraction as Q
import argparse, csv, hashlib, importlib.util, json, os, signal, socket, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1')
PARENT=ROOT.parent
DATA=ROOT/'data'
OLD=PARENT/'appearance-1/outputs/appearance-v1'
FULL=PARENT/'recovery-1/outputs/baseline-v1'
METHODS=('calibrated','joint')
ARMS=('strict_shared','independent','bounded')
DRIFT={'strict_shared':Q(0),'independent':Q(1),'bounded':Q(1,10)}
LIMIT=3600
BUDGET_SCOPE='A4-V prepare plus public inference plus independent verification; no reset'
sys.path.insert(0,str(ROOT))
import variation_solver as solver
if Path(solver.__file__).resolve()!=ROOT/'variation_solver.py':
    raise RuntimeError('substitute solver')
sys.path.insert(0,str(PARENT/'ablation-1'))
spec=importlib.util.spec_from_file_location('variation_frozen_helpers',PARENT/'ablation-1/run_ablation.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
exact,encode,load=helpers.exact,helpers.encode,helpers.load
select,make_row,aggregate=helpers.select,helpers.make_row,helpers.aggregate
class blocked_alarm:
    def __enter__(self):
        self.previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    def __exit__(self,*_):
        signal.pthread_sigmask(signal.SIG_SETMASK,self.previous)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value,replace=False):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve()==ROOT:
        raise ValueError('write outside exact phase')
    if replace and path.name!='RUN_STATUS.json':
        raise ValueError('only run status can replace')
    path.parent.mkdir(parents=True,exist_ok=True)
    with blocked_alarm():
        target=path.with_name(path.name+'.pending') if replace else path
        with target.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
        if replace:target.replace(path)
def sources():
    result=load(OLD/'SOURCE_LOCK.json')['files'].copy()
    for path,digest in result.items():
        if sha(path)!=digest:raise RuntimeError('old frozen source changed: '+path)
    result.update({str(p.resolve()):sha(p) for p in sorted(
        {ROOT/'AGENTS.md',ROOT/'PROTOCOL.md',ROOT/'REGISTRATION.json',*ROOT.glob('*.py')})})
    return dict(sorted(result.items()))
def inputs():
    paths=list(load(OLD/'INPUTS.json')['files'])
    paths += [str(OLD/rel) for rel in ('SOURCE_LOCK.json','RUN_STATUS.json','verification/VERIFY.json','outers.json','decisions.json')]
    paths += [str(DATA/rel) for rel in ('observations.json','tasks.json','private/truth.json','GENERATION.json')]
    return {str(Path(p).resolve()):sha(p) for p in sorted(paths)}
def population():
    observations,tasks=load(DATA/'observations.json'),load(DATA/'tasks.json')
    if len(observations)!=36 or len(tasks)!=324:
        raise RuntimeError('incomplete public population')
    expected=[f'p{i:03d}b80e{v}' for v in range(2) for i in range(1,19)]
    if [p['observation_id'] for p in observations]!=expected:
        raise RuntimeError('changed public identity')
    if [t['task_id'] for t in tasks]!=[oid+f's{s}' for oid in expected for s in range(9)]:
        raise RuntimeError('changed task identity')
    return observations,tasks
def action(task,public,arm,model,reused):
    result=select(arm['outer']['hull'],task['incumbent'],'hybrid')
    for field in ('output','gain_lower'):
        if result[field] is not None:result[field]=encode(result[field])
    return {'task_id':task['task_id'],'unit_id':public['unit_id'],'method':arm['outer']['mode'],
            'appearance_model':model,'reused':reused,**result}
def check_actions(records,tasks):
    expected={(t['task_id'],model,method) for t in tasks for model in ARMS for method in METHODS}
    keys=[(r['task_id'],r['appearance_model'],r['method']) for r in records]
    if len(keys)!=1944 or len(set(keys))!=1944 or set(keys)!=expected:
        raise RuntimeError('incomplete persisted1944 matrix')
    if sum(r['reused'] is False for r in records)!=1296 or sum(r['reused'] is True for r in records)!=648:
        raise RuntimeError('changed action reuse counts')
def check_keep(records,tasks):
    expected={t['task_id']:t['incumbent'] for t in tasks}
    if len(records)!=324 or len({r['task_id'] for r in records})!=324 or {r['task_id'] for r in records}!=set(expected):
        raise RuntimeError('incomplete unique324 KEEP matrix')
    if any(r['method']!='keep' or r['status']!='KEEP' or r['output']!=expected[r['task_id']] for r in records):
        raise RuntimeError('nonidentity KEEP matrix')
def coverage(public,fact,method,arm,model):
    truth,branch=exact(fact['truth']),fact['rear_branch']
    hull=arm['outer']['hull']
    true_nodes={}
    for name,value in arm['branches'].items():
        true_nodes[name]=[n for n in value['trace'] if n['id'] in value['retained_ids'] and exact(n['lo'])<=truth<=exact(n['hi'])]
    leaves=true_nodes[branch]
    material_required=model!='strict_shared' or public['variation_index']==0
    dark=[1-exact(c) for c in (fact['colors'] if model=='strict_shared' else fact['view_colors'])]
    bias=[Q(1,100)-exact(b) for b in fact['actual_view_bias']]
    def contained(node,key,values):
        return len(node[key])==len(values) and all(exact(lo)<=v<=exact(hi) for (lo,hi),v in zip(node[key],values))
    material=any(contained(n,'darkness_bounds',dark) for n in leaves) if material_required else None
    bias_ok=any(contained(n,'t_bounds',bias) for n in leaves) if method=='joint' else None
    witness=any(contained(n,'darkness_bounds',dark) and (method!='joint' or contained(n,'t_bounds',bias)) for n in leaves) if material_required else None
    return {'observation_id':public['observation_id'],'base_observation_id':public['base_observation_id'],
        'variation_index':public['variation_index'],'unit_id':public['unit_id'],'method':method,
        'appearance_model':model,'covered':any(true_nodes.values()),'generating_branch':branch,
        'generating_branch_covered':bool(leaves),'incompatible':hull is None,
        'latent_material_covered':material,'latent_bias_covered':bias_ok,'joint_witness_covered':witness,
        'material_witness_required':material_required,'hull_width_mm':None if hull is None else str(exact(hull[1])-exact(hull[0])),
        'branch_coverage':{name:bool(nodes) for name,nodes in true_nodes.items()}}
def csv_save(path,rows):
    with blocked_alarm(),Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def evaluate(output,observations,tasks,outers,decisions,keep):
    truths={f['observation_id']:f for f in load(DATA/'private/truth.json')}
    tm={t['task_id']:t for t in tasks};pm={p['observation_id']:p for p in observations}
    rows=[]
    for d in decisions:
        t=tm[d['task_id']];p=pm[t['observation_id']]
        rows.append(make_row(t,p,truths[p['observation_id']],d,outers[p['observation_id']][d['appearance_model']][d['method']],d['appearance_model']))
    keeps=[]
    for d in keep:
        t=tm[d['task_id']];p=pm[t['observation_id']]
        keeps.append(make_row(t,p,truths[p['observation_id']],d,{'outer':{'hull':None}},'keep'))
    cov=[coverage(p,truths[p['observation_id']],method,outers[p['observation_id']][model][method],model)
         for p in observations for model in ARMS for method in METHODS]
    for rel,values in (('results',rows),('keep/results',keeps)):
        save(output/(rel+'.json'),values);csv_save(output/(rel+'.csv'),values)
    save(output/'coverage.json',cov)
    save(output/'summary.json',{'methods':aggregate(rows),'keep':aggregate(keeps)})
    save(output/'intervention_diagnostic.json',[{'observation_id':p['observation_id'],
        'base_observation_id':p['base_observation_id'],'eta':p['eta'],
        'actual_material_spread':truths[p['observation_id']]['actual_material_spread'],
        'image_unchanged':truths[p['observation_id']]['image_unchanged']} for p in observations])
def run(output):
    output=Path(output).resolve()
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or output.parent!=ROOT/'outputs':
        raise RuntimeError('wrong target')
    budget=load(output/'BUDGET.json');start=budget['start_epoch'];deadline=budget['deadline_epoch']
    source_hashes=load(output/'SOURCE_LOCK.json')['files']
    if sources()!=source_hashes:raise RuntimeError('sources changed before inference')
    input_hashes=inputs();save(output/'INPUTS.json',{'files':input_hashes})
    save(output/'REGISTRATION.json',load(ROOT/'REGISTRATION.json'))
    def reached(*_):raise TimeoutError('shared A4-V deadline')
    previous=signal.signal(signal.SIGALRM,reached)
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('no inference budget')
    signal.setitimer(signal.ITIMER_REAL,remaining)
    calls,reuses,decisions,outers,timing,keep=0,0,[],{},[],[]
    try:
        observations,tasks=population()
        save(output/'observations.json',observations);save(output/'tasks.json',tasks)
        old_full,old_independent=load(FULL/'outers.json'),load(OLD/'outers.json')
        full_decisions={(d['task_id'],d['method']):d for d in load(FULL/'decisions.json')}
        independent_decisions={(d['task_id'],d['method']):d for d in load(OLD/'decisions.json')}
        taskmap=defaultdict(list)
        for t in tasks:taskmap[t['observation_id']].append(t)
        for index,p in enumerate(observations):
            oid,base=p['observation_id'],p['base_observation_id'];outers[oid]={}
            for t in taskmap[oid]:
                keep.append({'task_id':t['task_id'],'unit_id':p['unit_id'],'method':'keep','appearance_model':'keep',
                    'reused':True,'output':t['incumbent'],'gain_lower':encode(0),'status':'KEEP','reason':'identity','branch':'keep'})
            for model in ARMS:
                outers[oid][model]={}
                for method in METHODS:
                    reused=p['variation_index']==0 and model!='bounded'
                    if reused:
                        source=old_full if model=='strict_shared' else old_independent
                        arm={**source[base][method],'appearance_model':model,'drift_bound':encode(DRIFT[model])};reuses+=1
                    else:
                        solver.clear_geometry_cache();tick=time.perf_counter()
                        arm=solver.solve_public(p['observation'],p['radius'],p['baseline'],method,DRIFT[model],clipped=p['clipped'])
                        seconds=time.perf_counter()-tick;calls+=1
                        timing.append({'observation_id':oid,'appearance_model':model,'method':method,
                            'cold_seconds':seconds,'evaluated_boxes':arm['outer']['evaluated_boxes']})
                    if arm['calibration']!=old_full[base][method]['calibration']:
                        raise RuntimeError('material intervention changed guaranteed-background calibration')
                    outers[oid][model][method]=arm
                    for t in taskmap[oid]:
                        record=action(t,p,arm,model,reused)
                        if reused:
                            old=(full_decisions if model=='strict_shared' else independent_decisions)[(t['base_task_id'],method)]
                            for field in ('status','output','gain_lower','reason','branch'):
                                if record[field]!=old[field]:raise RuntimeError('zero-variation action identity failed')
                        decisions.append(record)
            save(output/f'checkpoint/{oid}.json',{'observation_id':oid,'arms':outers[oid]})
            save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':index+1,
                'decisions_complete':len(decisions),'new_solver_calls':calls,'reused_solver_calls':reuses,
                'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
            print(json.dumps({'stage':'A4-V','observations_complete':index+1,'new_solver_calls':calls,
                'elapsed_seconds':round(time.time()-start,1)}),flush=True)
        save(output/'outers.json',outers);save(output/'decisions.json',decisions)
        save(output/'keep/decisions.json',keep);save(output/'TIMING.json',timing)
        decisions=load(output/'decisions.json');outers=load(output/'outers.json');keep=load(output/'keep/decisions.json')
        check_actions(decisions,tasks)
        check_keep(keep,tasks)
        if calls!=144 or reuses!=72 or len(outers)!=36 or len(keep)!=324:
            raise RuntimeError('incomplete solve/keep matrix')
        evaluate(output,observations,tasks,outers,decisions,keep)
        if sources()!=source_hashes or inputs()!=input_hashes:
            raise RuntimeError('frozen bytes changed')
        save(output/'RUN_STATUS.json',{'status':'COMPLETE','observations_complete':36,
            'decisions_complete':1944,'new_solver_calls':144,'reused_solver_calls':72,
            'elapsed_seconds':time.time()-start,'deadline_epoch':deadline},replace=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    return str(output)

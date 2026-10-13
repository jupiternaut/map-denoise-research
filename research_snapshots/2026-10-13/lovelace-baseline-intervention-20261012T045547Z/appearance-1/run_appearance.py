"""A4: public inference with per-view material, persisted actions, then evaluation."""
from collections import defaultdict
from fractions import Fraction as Q
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True
for thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[thread_variable] = '1'
ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1')
PARENT = ROOT.parent
DATA = PARENT/'data'
FULL = PARENT/'recovery-1/outputs/baseline-v1'
PREVIOUS = PARENT/'ablation-1/outputs/ablation-v1'
METHODS = ('calibrated','joint')
LIMIT = 3600
BUDGET_SCOPE = 'A4 per-view appearance main plus independent verification; no reset'
sys.path.insert(0,str(ROOT))
import appearance_solver as solver
if Path(solver.__file__).resolve() != ROOT/'appearance_solver.py':
    raise RuntimeError('substitute solver import')
sys.path.insert(0,str(PARENT/'ablation-1'))
_spec = importlib.util.spec_from_file_location('a4_frozen_helpers',PARENT/'ablation-1/run_ablation.py')
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
exact, encode, load = helpers.exact, helpers.encode, helpers.load
select, make_row, aggregate = helpers.select, helpers.make_row, helpers.aggregate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class blocked_alarm:
    def __enter__(self):
        self.previous = signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    def __exit__(self,*_):
        signal.pthread_sigmask(signal.SIG_SETMASK,self.previous)


def save(path,value,replace=False):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve() == ROOT:
        raise ValueError('write outside exact phase root')
    if replace and path.name != 'RUN_STATUS.json':
        raise ValueError('only RUN_STATUS may be replaced')
    path.parent.mkdir(parents=True,exist_ok=True)
    with blocked_alarm():
        target = path.with_name(path.name+'.pending') if replace else path
        with target.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
        if replace:
            target.replace(path)


def sources():
    result = load(PREVIOUS/'SOURCE_LOCK.json')['files'].copy()
    for path,digest in result.items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: '+path)
    result.update({str(path.resolve()):sha(path) for path in sorted(
        {ROOT/'AGENTS.md',ROOT/'PROTOCOL.md',ROOT/'REGISTRATION.json',*ROOT.glob('*.py')})})
    return dict(sorted(result.items()))


def inputs():
    paths = [DATA/relative for relative in ('observations.json','tasks.json','private/truth.json','GENERATION.json')]
    paths += [FULL/relative for relative in ('outers.json','decisions.json','RUN_STATUS.json','verification/VERIFY.json','SOURCE_LOCK.json')]
    paths += [PREVIOUS/relative for relative in ('SOURCE_LOCK.json','RUN_STATUS.json','verification/VERIFY.json')]
    return {str(path.resolve()):sha(path) for path in sorted(paths)}


def population():
    observations,tasks = helpers.population()
    observations = [public for public in observations if public['unit_id']=='baseline_80']
    ids = {public['observation_id'] for public in observations}
    tasks = [task for task in tasks if task['observation_id'] in ids]
    if len(observations)!=18 or len(tasks)!=162:
        raise RuntimeError('incomplete publicly defined B80 population')
    return observations,tasks


def action(task,public,arm):
    result = select(arm['outer']['hull'],task['incumbent'],'hybrid')
    for field in ('output','gain_lower'):
        if result[field] is not None:
            result[field] = encode(result[field])
    return {'task_id':task['task_id'],'unit_id':public['unit_id'],'method':arm['outer']['mode'],
            'appearance_scope':'per_view','reused':False,**result}


def check_actions(records,tasks):
    expected = {(task['task_id'],method) for task in tasks for method in METHODS}
    keys = [(record['task_id'],record['method']) for record in records]
    if (len(keys)!=len(expected) or len(set(keys))!=len(expected) or set(keys)!=expected
        or any(record['reused'] is not False or record['appearance_scope']!='per_view' for record in records)):
        raise RuntimeError('incomplete or duplicated saved A4 action matrix')


def coverage(public,private,method,arm):
    truth,branch,hull = exact(private['truth']),private['rear_branch'],arm['outer']['hull']
    retained = {name:[node for node in value['trace'] if node['id'] in value['retained_ids']]
                for name,value in arm['branches'].items()}
    true_nodes = {name:[node for node in nodes if exact(node['lo'])<=truth<=exact(node['hi'])]
                  for name,nodes in retained.items()}
    leaves = true_nodes[branch]
    darkness = [1-exact(color) for color in private['colors']]*3
    actual_t = [Q(1,100)-exact(bias) for bias in private['actual_view_bias']]
    def latent(key,values):
        return bool(leaves) and all(
            len(node[key])==len(values) and all(exact(pair[0])<=value<=exact(pair[1])
                                               for pair,value in zip(node[key],values))
            for node in leaves)
    return {'appearance_scope':'per_view','observation_id':public['observation_id'],
            'unit_id':public['unit_id'],'method':method,'case':private['case'],'seed':private['seed'],
            'covered':any(true_nodes.values()),'incompatible':hull is None,
            'hull_width_mm':None if hull is None else str(exact(hull[1])-exact(hull[0])),
            'retained_branches':arm['outer']['retained_branches'],
            'branch_coverage':{name:bool(nodes) for name,nodes in true_nodes.items()},
            'generating_branch':branch,'generating_branch_covered':bool(leaves),
            'true_depth_leaf_ids':{name:[node['id'] for node in nodes] for name,nodes in true_nodes.items()},
            'latent_material_covered':latent('darkness_bounds',darkness),
            'latent_bias_covered':latent('t_bounds',actual_t) if method=='joint' else None}


def csv_save(path,rows):
    with blocked_alarm(),Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output,observations,tasks,new_outers,full_outers,decisions):
    # The caller first persists and reads back all 324 public actions, then checks completeness.
    truths = {fact['observation_id']:fact for fact in load(DATA/'private/truth.json')}
    tm = {task['task_id']:task for task in tasks}
    om = {public['observation_id']:public for public in observations}
    rows = []
    for decision in decisions:
        task = tm[decision['task_id']]
        public = om[task['observation_id']]
        rows.append(make_row(task,public,truths[public['observation_id']],decision,
                             new_outers[public['observation_id']][decision['method']],'per_view'))
    reference_decisions = [d for d in load(FULL/'decisions.json') if d['task_id'] in tm]
    if len(reference_decisions)!=486:
        raise RuntimeError('incomplete frozen B80 reference actions')
    references = []
    for decision in reference_decisions:
        task = tm[decision['task_id']]
        public = om[task['observation_id']]
        references.append(make_row(task,public,truths[public['observation_id']],decision,
                                   full_outers[public['observation_id']][decision['method']],
                                   'keep' if decision['method']=='keep' else 'global_shared'))
    cov = [coverage(public,truths[public['observation_id']],method,new_outers[public['observation_id']][method])
           for public in observations for method in METHODS]
    for relative,values in (('results',rows),('reference/results',references)):
        save(output/(relative+'.json'),values)
        csv_save(output/(relative+'.csv'),values)
    save(output/'coverage.json',cov)
    save(output/'summary.json',{'new':aggregate(rows),'reference':aggregate(references)})


def run(output,ownership=None):
    output = Path(output).resolve()
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or not output.is_relative_to(ROOT/'outputs') or output.exists():
        raise RuntimeError('wrong host/root/output or existing run')
    output.mkdir(parents=True)
    if ownership is not None:
        ownership['created']=True
    start = time.time()
    budget = {'start_epoch':start,'deadline_epoch':start+LIMIT,'limit_seconds':LIMIT,'scope':BUDGET_SCOPE}
    save(output/'BUDGET.json',budget)
    source_hashes,input_hashes = sources(),inputs()
    save(output/'SOURCE_LOCK.json',{'source_root':str(ROOT),'files':source_hashes})
    save(output/'INPUTS.json',{'files':input_hashes})
    save(output/'REGISTRATION.json',load(ROOT/'REGISTRATION.json'))
    def reached(_signum,_frame):
        raise TimeoutError('absolute A4 deadline reached')
    previous = signal.signal(signal.SIGALRM,reached)
    remaining = budget['deadline_epoch']-time.time()
    if remaining<=0:
        raise TimeoutError('A4 exhausted before inference')
    signal.setitimer(signal.ITIMER_REAL,remaining)
    calls,decisions,outers,timing = 0,[],{},[]
    try:
        observations,tasks = population()
        save(output/'observations.json',observations)
        save(output/'tasks.json',tasks)
        full_outers = load(FULL/'outers.json')
        task_map = defaultdict(list)
        for task in tasks:
            task_map[task['observation_id']].append(task)
        for index,public in enumerate(observations):
            oid = public['observation_id']
            outers[oid] = {}
            for method in METHODS:
                solver.clear_geometry_cache()
                tick = time.perf_counter()
                arm = solver.solve_public(public['observation'],public['radius'],public['baseline'],method,
                                          appearance_scope='per_view',clipped=public['clipped'])
                seconds = time.perf_counter()-tick
                calls += 1
                if arm['calibration']!=full_outers[oid][method]['calibration']:
                    raise RuntimeError('A4 changed full-image calibration')
                outers[oid][method] = arm
                timing.append({'observation_id':oid,'appearance_scope':'per_view','method':method,
                               'cold_seconds':seconds,'evaluated_boxes':arm['outer']['evaluated_boxes']})
                decisions.extend(action(task,public,arm) for task in task_map[oid])
            save(output/f'checkpoint/{oid}.json',{'observation_id':oid,'arms':outers[oid]})
            save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':index+1,
                 'decisions_complete':len(decisions),'new_solver_calls':calls,'elapsed_seconds':time.time()-start,
                 'deadline_epoch':budget['deadline_epoch']},replace=True)
            print(json.dumps({'stage':'A4','observations_complete':index+1,'new_solver_calls':calls,
                              'elapsed_seconds':round(time.time()-start,1)}),flush=True)
        save(output/'outers.json',outers)
        save(output/'decisions.json',decisions)
        save(output/'TIMING.json',timing)
        decisions = load(output/'decisions.json')
        outers = load(output/'outers.json')
        check_actions(decisions,tasks)
        if calls!=36 or len(outers)!=18:
            raise RuntimeError('incomplete A4 solve matrix')
        evaluate(output,observations,tasks,outers,full_outers,decisions)
        if sources()!=source_hashes or inputs()!=input_hashes:
            raise RuntimeError('frozen byte graph changed')
        save(output/'RUN_STATUS.json',{'status':'COMPLETE','observations_complete':18,'decisions_complete':324,
             'new_solver_calls':36,'elapsed_seconds':time.time()-start,'deadline_epoch':budget['deadline_epoch']},replace=True)
    except BaseException as error:
        save(output/'INCOMPLETE.json',{'error':type(error).__name__,'message':str(error),'calls':calls,
                                    'decisions':len(decisions),'elapsed_seconds':time.time()-start})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    return str(output)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/appearance-v1')
    print(run(parser.parse_args().output))

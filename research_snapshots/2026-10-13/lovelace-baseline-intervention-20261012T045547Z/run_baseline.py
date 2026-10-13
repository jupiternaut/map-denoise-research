"""Public inference then sealed paired evaluation; one shared absolute deadline."""
import os
import sys
sys.dont_write_bytecode = True
for _baseline_thread in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_baseline_thread] = '1'
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import socket
import time
import baseline_solver as solver

ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z')
CONF = ROOT.parent / 'lovelace-structured-bias-confirmation-20261012T035916Z'
FROZEN = ROOT.parent / 'lovelace-structured-view-bias-20261012T022500Z'
DATA = ROOT / 'data'
spec = importlib.util.spec_from_file_location('baseline_frozen_helpers', CONF / 'run_confirmation.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
helpers, old, encode, exact, load = previous.helpers, previous.old, previous.encode, previous.exact, previous.load
METHODS, UNITS, LIMIT = ('keep','calibrated','joint'), ('baseline_70','baseline_80'), 1800
SAMPLING_NOTE = 'one paired noise/pose realization per template/depth; controlled synthetic imaging intervention, not real transfer'
TIMING_SCOPE = previous.TIMING_SCOPE
PROGRESS = {'decisions.json','RUN_STATUS.json'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value, replace=False):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve() == ROOT:
        raise ValueError('only exact new root outputs allowed')
    if replace and path.name not in PROGRESS:
        raise ValueError('only progress may be replaced')
    with old.finish_record_before_alarm():
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.with_name(path.name+'.pending') if replace else path
        with target.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        if replace:
            target.replace(path)

def csv_save(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)

def source_files():
    spec = importlib.util.spec_from_file_location('baseline_source_inventory', FROZEN / 'verify_structured.py')
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    result = dict(inventory.scientific_files())
    frozen_lock = load(FROZEN / 'outputs/structured-v1/SOURCE_LOCK.json')['files']
    for path, digest in frozen_lock.items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: '+path)
    for path in sorted({ROOT/'AGENTS.md',ROOT/'PROTOCOL.md',ROOT/'REGISTRATION.json',*ROOT.glob('*.py')}):
        result[str(path.resolve())] = sha(path)
    # CONF helpers are direct dependencies, although the older graph predates them.
    result[str((CONF/'run_confirmation.py').resolve())] = sha(CONF/'run_confirmation.py')
    return dict(sorted(result.items()))

def input_files():
    return {str(path.resolve()):sha(path) for path in sorted(DATA/part for part in
        ('observations.json','tasks.json','private/truth.json','GENERATION.json'))}

def population():
    observations, tasks = load(DATA/'observations.json'), load(DATA/'tasks.json')
    if (len(observations),len(tasks)) != (36,324):
        raise RuntimeError('wrong public population')
    expected = [(f'p{i:03d}b{b}', f'p{i:03d}', b) for i in range(1,19) for b in ('70','80')]
    for item,(oid,pid,b) in zip(observations,expected):
        if (set(item) != {'observation_id','pair_id','unit_id','baseline','clipped','radius','observation'} or
            item['observation_id'] != oid or item['pair_id'] != pid or item['baseline'] != b or
            item['unit_id'] != 'baseline_'+b or item['clipped'] is not True or item['radius'] != '1/5' or
            len(item['observation']) != 432):
            raise RuntimeError('invalid public image identity/schema')
    expected_tasks = [(o,state) for o in observations for state in range(9)]
    for task,(item,state) in zip(tasks,expected_tasks):
        if (set(task) != {'task_id','base_task_id','observation_id','incumbent'} or
            task['task_id'] != f'{item["observation_id"]}s{state}' or
            task['base_task_id'] != f'{item["pair_id"]}s{state}' or task['observation_id'] != item['observation_id']):
            raise RuntimeError('invalid public task identity/schema')
    return observations,tasks

def complete_saved_actions(output, decisions):
    if (len(decisions) != 972 or len({(r['task_id'],r['method']) for r in decisions}) != 972 or
        load(output/'decisions.json') != decisions or Counter((r['method'],r['reused']) for r in decisions) !=
        {('keep',False):324,('calibrated',False):324,('joint',False):324}):
        raise RuntimeError('972 unique saved/read-back fresh actions required before private evaluation')

def private_truths():
    return load(DATA/'private/truth.json')

def execute_new(item, method):
    solver.clear_geometry_cache()
    return solver.solve_public(item['observation'],item['radius'],item['baseline'],method,item['clipped'])

def aggregate_rows(rows, truths):
    result = []
    facts = {r['observation_id']:str(exact(r['truth'])) for r in truths}
    for unit in UNITS:
        values = [r for r in rows if r['unit_id'] == unit]
        result.extend({**r,'unit_id':unit} for r in helpers.aggregate(values))
        for depth in sorted(set(facts.values()),key=Q):
            for r in helpers.aggregate([r for r in values if facts[r['observation_id']] == depth]):
                if r['grouping'] == 'overall':
                    result.append({**r,'grouping':'by_depth','condition':depth,'unit_id':unit})
    return result

def method_pairs(rows):
    result = []
    for unit in UNITS:
        values = [r for r in rows if r['unit_id'] == unit]
        result.extend({**r,'unit_id':unit} for r in previous.paired_rows(values))
    return result

def baseline_pairs(rows):
    matrix = {(r['base_task_id'],r['unit_id'],r['method']):r for r in rows}
    result = []
    for bid in sorted({r['base_task_id'] for r in rows}):
        for method in METHODS:
            a,b = matrix[bid,UNITS[0],method],matrix[bid,UNITS[1],method]
            x,y = exact(a['ae_mm']),exact(b['ae_mm'])
            result.append({'base_task_id':bid,'method':method,'case':a['case'],'seed':a['seed'],
                'state_group':a['state_group'],'left_unit_id':UNITS[0],'right_unit_id':UNITS[1],
                'left_ae_mm':str(x),'right_ae_mm':str(y),'mae_gain_mm':str(x-y),
                'mse_gain_mm2':str(x*x-y*y),'better':y<x,'same':y==x,'worse':y>x})
    return result

def paired_baseline_claims(rows,pairs,facts):
    matrix = {(r['base_task_id'],r['unit_id'],r['method']):r for r in rows}
    def count(values):
        result = {'total':len(values),'better':sum(p['better'] for p in values),
            'same':sum(p['same'] for p in values),'worse':sum(p['worse'] for p in values),
            'new_repairs':0,'lost_repairs':0}
        for p in values:
            initial = exact(matrix[p['base_task_id'],UNITS[0],p['method']]['initial_ae_mm'])
            x,y = exact(p['left_ae_mm']),exact(p['right_ae_mm'])
            result['new_repairs'] += int(x>=initial and y<initial)
            result['lost_repairs'] += int(x<initial and y>=initial)
        return result
    result = count(pairs)
    result['by_method'] = {m:count([p for p in pairs if p['method']==m]) for m in METHODS}
    depths = {r['base_task_id']:exact(facts[r['observation_id']]['truth']) for r in rows}
    deepest = max(depths.values())
    deep = [p for p in pairs if depths[p['base_task_id']]==deepest and p['case']!='equal_full_negative' and p['state_group']=='small']
    result['deep_informative'] = {m:count([p for p in deep if p['method']==m]) for m in METHODS}
    return result

def evaluate(output, observations, tasks, truths, decisions, outers, timing, selectors, started):
    facts = {r['observation_id']:r for r in truths}
    states = {r['task_id']:r for r in tasks}
    rows = []
    for action in decisions:
        task = states[action['task_id']]
        oid,method = task['observation_id'],action['method']
        private,outer = facts[oid],outers[oid][method]['outer']
        a,b,t = exact(task['incumbent']),exact(action['output']),exact(private['truth'])
        initial,error = abs(a-t),abs(b-t)
        hull = None if outer['hull'] is None else tuple(map(exact,outer['hull']))
        rows.append({'task_id':task['task_id'],'base_task_id':task['base_task_id'],'observation_id':oid,
            'unit_id':private['unit_id'],'pair_id':private['pair_id'],'baseline':private['baseline'],
            'method':method,'reused':False,'case':private['case'],'seed':private['seed'],
            'state_group':'correct' if initial==0 else 'small','prior_position':'inside_prior' if 540<=a<=660 else 'outside_prior',
            'incumbent':str(a),'output':str(b),'status':action['status'],'initial_ae_mm':str(initial),
            'ae_mm':str(error),'se_mm2':str(error*error),'actual_gain_mm2':str(initial*initial-error*error),
            'covered':helpers.covered(outer,t),'hull_width_mm':None if hull is None else str(hull[1]-hull[0]),
            'hull_lo_mm':None if hull is None else str(hull[0]),'hull_hi_mm':None if hull is None else str(hull[1]),
            'retained_branch_count':len(outer['retained_branches']),'retained_interval_count':len(outer['retained_intervals']),
            'evaluated_boxes':outer['evaluated_boxes'],'unresolved_budget_leaves':outer['unresolved_budget_leaves'],'auxiliary_samples_used':0})
    coverage = []
    for unit in UNITS:
        subset = [o for o in observations if o['unit_id']==unit]
        coverage.extend({**r,'unit_id':unit} for r in previous.coverage_rows(subset,facts,outers))
    mp,bp = method_pairs(rows),baseline_pairs(rows)
    for name, values, csvname in [('results',rows,'results.csv'),('coverage',coverage,'COVERAGE.csv'),
        ('PAIRED_METHOD',mp,'PAIRED_METHOD.csv'),('PAIRED_BASELINE',bp,'PAIRED_BASELINE.csv')]:
        save(output/(name+'.json'),values)
        csv_save(output/csvname,values)
    safety,paired = {},{}
    for unit in UNITS:
        values = [r for r in rows if r['unit_id']==unit]
        cov = [r for r in coverage if r['unit_id']==unit]
        safety[unit] = previous.safety_claims(values,cov)
        unit_outers = {o['observation_id']:outers[o['observation_id']] for o in observations if o['unit_id']==unit}
        paired[unit] = previous.paired_claims(values,[p for p in mp if p['unit_id']==unit],unit_outers)
    return {'population':{'observations':36,'tasks':324,'decisions':972,'new_solver_calls':72},
        'rows':aggregate_rows(rows,truths),'paired_method':paired,'paired_baseline':paired_baseline_claims(rows,bp,facts),
        'safety':safety,'costs':previous.cost_rows(timing,outers,selectors),
        'sampling_note':SAMPLING_NOTE,'timing_scope':TIMING_SCOPE,'elapsed_seconds':time.perf_counter()-started}

def _run(output,started,budget):
    sources,inputs = source_files(),input_files()
    observations,tasks = population()
    output.mkdir(parents=True)
    save(output/'SOURCE_LOCK.json',{'source_root':str(ROOT),'files':sources,'utc':datetime.now(timezone.utc).isoformat()})
    save(output/'INPUTS.json',{'source_root':str(DATA),'files':inputs})
    save(output/'BUDGET.json',budget)
    save(output/'REGISTRATION.json',load(ROOT/'REGISTRATION.json'))
    save(output/'observations.json',observations)
    save(output/'tasks.json',tasks)
    by_oid = defaultdict(list)
    for task in tasks:
        by_oid[task['observation_id']].append(task)
    decisions,outers,timing,selectors = [],{},[],[]
    for item in observations:
        oid,unit = item['observation_id'],item['unit_id']
        arms = {'keep':helpers.identity_arm()}
        for method in METHODS[1:]:
            arms[method] = execute_new(item,method)
            entry = arms[method]
            timing.append({'observation_id':oid,'unit_id':unit,'method':method,'branch_seconds':entry['branch_seconds'],
                'outer_seconds':entry['outer_seconds'],'preparation_seconds':entry['preparation_seconds'],'reused':False})
        seconds = {}
        for method in METHODS:
            tick = time.perf_counter()
            for task in by_oid[oid]:
                decisions.append({'task_id':task['task_id'],'unit_id':unit,'method':method,'reused':False,
                    **old.select_action(arms[method]['outer'],exact(task['incumbent']),keep=method=='keep')})
            seconds[method] = time.perf_counter()-tick
        outers[oid] = arms
        selectors.append({'observation_id':oid,'seconds_by_method':seconds})
        save(output/('checkpoint/'+oid+'.json'),{'observation_id':oid,'arms':arms})
        save(output/'decisions.json',decisions,replace=True)
        save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':len(outers),
            'decisions_complete':len(decisions),'elapsed_seconds':time.perf_counter()-started},replace=True)
        print(json.dumps({'observations_complete':len(outers),'decisions_complete':len(decisions),
            'seconds':round(time.perf_counter()-started,3)}),flush=True)
    save(output/'outers.json',outers)
    save(output/'timing.json',timing)
    save(output/'selector_timing.json',selectors)
    complete_saved_actions(output,decisions)
    truths = private_truths()
    summary = evaluate(output,observations,tasks,truths,decisions,outers,timing,selectors,started)
    save(output/'summary.json',summary)
    csv_save(output/'SUMMARY.csv',summary['rows'])
    if source_files()!=sources or input_files()!=inputs:
        raise RuntimeError('locked bytes changed during run')
    save(output/'RUN_STATUS.json',{'status':'COMPLETE','observations_complete':36,'decisions_complete':972,
        'new_solver_calls':72,'elapsed_seconds':time.perf_counter()-started,'deadline_epoch':budget['deadline_epoch']},replace=True)
    print(json.dumps({'stage':'complete','elapsed_seconds':summary['elapsed_seconds']}),flush=True)

def expired(_signal,_frame):
    raise TimeoutError('single shared main/verification window exhausted')

def run(output):
    if socket.gethostname()!='liekkas' or ROOT.resolve()!=EXPECTED:
        raise RuntimeError('requires exact registered host/root')
    output = Path(output).resolve()
    if output.exists() or output==ROOT or not output.is_relative_to(ROOT):
        raise ValueError('fresh output inside exact root required')
    start,tick = time.time(),time.perf_counter()
    budget = {'start_epoch':start,'deadline_epoch':start+LIMIT,'limit_seconds':LIMIT,
        'scope':'one main run plus complete verification; do not reset'}
    previous_handler = signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,LIMIT)
    try:
        _run(output,tick,budget)
    except BaseException as error:
        if output.exists():
            save(output/'RUN_STATUS.json',{'status':'INCOMPLETE','reason':str(error),
                'elapsed_seconds':time.perf_counter()-tick,'deadline_epoch':budget['deadline_epoch']},replace=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'outputs/baseline-v1'))
    run(parser.parse_args().output)

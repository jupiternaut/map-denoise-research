"""One bounded same-data replay, preserving all prior scientific artifacts."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z'
DATA = PRIOR / 'outputs/confirmation-v1'
NUISANCE = ROOT.parent / 'lovelace-nuisance-20261011T044500Z'
sys.path.insert(0, str(NUISANCE))
import run_nuisance as old
sys.path.insert(0, str(ROOT))
from grouped_solver import solve_arm, clear_geometry_cache

METHODS = ('keep', 'channel511', 'global_gray511', 'pixel_gray511', 'view_gray511', 'global_rgb511')
MODES = dict(zip(METHODS[1:], ('channel', 'global_gray', 'pixel_gray', 'view_gray', 'global_rgb')))
MAX_BOXES, ROUNDS, TIMEOUT = 511, 8, 5400
EPSILON, TOLERANCE = Q(1, 100), Q(15, 32)
INPUT_RELATIVE = ('observations.json', 'tasks.json', 'evaluation/truth.json', 'outers.json',
                  'decisions.json', 'evaluation/SUMMARY.json', 'SOURCE_LOCK.json', 'verification/VERIFY.json')
decode, encode = old.decode, old.encode

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def input_lock():
    return {str((DATA / rel).resolve()): sha(DATA / rel) for rel in INPUT_RELATIVE}

def science_lock():
    files = dict(load(DATA / 'SOURCE_LOCK.json')['files'])
    for path, expected in files.items():
        if sha(path) != expected:
            raise RuntimeError('historical frozen scientific source changed: '+path)
    for path in sorted(ROOT.glob('*.py')):
        files[str(path.resolve())] = sha(path)
    files[str((ROOT / 'PROTOCOL.md').resolve())] = sha(ROOT / 'PROTOCOL.md')
    return files

def configuration():
    return {'methods': METHODS, 'method_configs': {m: {'mode': MODES.get(m, 'identity'),
            'max_boxes': MAX_BOXES, 'rounds': ROUNDS} for m in METHODS},
            'epsilon': str(EPSILON), 'tolerance_mm': str(TOLERANCE),
            'max_boxes_per_branch': MAX_BOXES, 'expected_observations': 54,
            'expected_tasks': 486, 'expected_decisions': 2916,
            'allowed_branches': old.BRANCHES, 'cpu_threads': 1, 'gpu_count': 0,
            'wall_time_limit_seconds': TIMEOUT, 'source_root': str(DATA),
            'sampling_note': 'same 54 synthetic configurations replayed; not a new confirmation or independent scenes',
            'cache_policy': 'clear all geometry/model caches before each arm/observation; reuse within its three branches',
            'old_255_results': 'frozen prior artifacts, not rerun or overwritten'}

def identity_arm():
    outer = old.prior_outer()
    outer['max_boxes_per_branch'] = MAX_BOXES
    return {'branches': {}, 'outer': outer, 'branch_seconds': {}, 'outer_seconds': 0.0}

def execute_arm(observation, radius, method):
    clear_geometry_cache()
    if method == 'keep':
        return identity_arm()
    return solve_arm(tuple(map(decode, observation)), Q(radius), MODES[method],
                     max_boxes=MAX_BOXES, rounds=ROUNDS, tolerance=TOLERANCE, epsilon=EPSILON)

def csv_save(path, rows, json_columns=()):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if k in json_columns and isinstance(v, (dict,list)) else v
                         for k,v in row.items()} for row in rows)

def aggregate(rows):
    result = old.aggregates(rows)
    for seed in sorted({r['seed'] for r in rows}):
        subset = [r for r in rows if r['seed'] == seed]
        result += [{**r, 'grouping': 'by_seed', 'condition': str(seed)}
                   for r in old.aggregates(subset) if r['grouping'] == 'overall']
        for case in sorted({r['case'] for r in subset}):
            result += [{**r, 'grouping': 'by_case_seed', 'condition': [case, seed]}
                       for r in old.aggregates([r for r in subset if r['case'] == case]) if r['grouping'] == 'overall']
    return result

def covered(outer, truth):
    return any(decode(lo) <= truth <= decode(hi) for lo,hi in outer['retained_intervals'])

def evaluate(output, observations, tasks, truths, decisions, outers, timing, started):
    tm = {t['task_id']: t for t in tasks}
    pm = {p['observation_id']: p for p in truths}
    om = {o['observation_id']: o['arms'] for o in outers}
    rows = []
    for decision in decisions:
        task = tm[decision['task_id']]
        oid, method = task['observation_id'], decision['method']
        private, outer = pm[oid], om[oid][method]['outer']
        truth, incumbent, prediction = decode(private['truth']), decode(task['incumbent']), decode(decision['output'])
        error, initial = abs(prediction-truth), abs(incumbent-truth)
        width = None if outer['hull'] is None else str(decode(outer['hull'][1])-decode(outer['hull'][0]))
        rows.append({'task_id': task['task_id'], 'observation_id': oid, 'case': private['case'], 'seed': private['seed'],
                     'method': method, 'state_group': 'correct' if initial == 0 else 'small',
                     'prior_position': 'inside_prior' if 540 <= incumbent <= 660 else 'outside_prior',
                     'incumbent': str(incumbent), 'output': str(prediction), 'status': decision['status'],
                     'initial_ae_mm': str(initial), 'ae_mm': str(error), 'se_mm2': str(error*error),
                     'actual_gain_mm2': str(initial*initial-error*error), 'covered': covered(outer,truth),
                     'hull_width_mm': width, 'retained_branch_count': len(outer['retained_branches']),
                     'evaluated_boxes': outer['evaluated_boxes'], 'unresolved_budget_leaves': outer['unresolved_budget_leaves'],
                     'auxiliary_samples_used': 0})
    save(output/'evaluation/ROWS.json',rows)
    csv_save(output/'evaluation/ROWS.csv',rows)
    coverage = []
    for record in outers:
        oid, arms = record['observation_id'], record['arms']
        private = pm[oid]
        truth = decode(private['truth'])
        for method,entry in arms.items():
            outer = entry['outer']
            coverage.append({'observation_id':oid,'case':private['case'],'seed':private['seed'],'method':method,
                             'covered':covered(outer,truth),'incompatible':outer['hull'] is None,
                             'hull_width_mm':None if outer['hull'] is None else str(decode(outer['hull'][1])-decode(outer['hull'][0])),
                             'retained_branches':outer['retained_branches'],
                             'branch_coverage':{b:any(decode(node['lo']) <= truth <= decode(node['hi'])
                                                       for node in branch['trace'] if node['id'] in branch['retained_ids'])
                                                for b,branch in entry['branches'].items()}})
    save(output/'evaluation/COVERAGE.json',coverage)
    csv_save(output/'evaluation/COVERAGE.csv',coverage,('retained_branches','branch_coverage'))
    costs = {}
    for method in METHODS:
        a = sum(t['outer_seconds_by_method'][method] for t in timing)
        b = sum(t['selector_seconds_by_method'][method] for t in timing)
        costs[method] = {'outer_seconds':a,'selector_seconds':b,'total_seconds':a+b,
                         'evaluated_boxes':sum(o['arms'][method]['outer']['evaluated_boxes'] for o in outers),
                         'unresolved_budget_leaves':sum(o['arms'][method]['outer']['unresolved_budget_leaves'] for o in outers)}
    summary = {'observation_configurations':len(observations),'task_count':len(tasks),'decision_count':len(decisions),
               'rows':aggregate(rows),'per_method_timing':costs,'sampling_note':configuration()['sampling_note'],
               'timing_scope':configuration()['cache_policy'],'elapsed_seconds':time.perf_counter()-started}
    save(output/'evaluation/SUMMARY.json',summary)
    csv_save(output/'evaluation/SUMMARY.csv',summary['rows'],('condition',))
    csv_save(output/'evaluation/TIMING.csv',[{'method':m,**v} for m,v in costs.items()])
    return summary

def validate_output(output):
    path = Path(output).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('output must be inside exact new ablation root')
    if path.exists():
        raise FileExistsError('refuse overwriting an existing replay')
    return path

def run(output):
    output = validate_output(output)
    inputs, sources = input_lock(), science_lock()
    if load(DATA/'verification/VERIFY.json')['status'] != 'PASS':
        raise RuntimeError('frozen prior confirmation is not verified')
    observations, tasks, truths = (load(DATA/name) for name in ('observations.json','tasks.json','evaluation/truth.json'))
    if (len(observations),len(tasks),len(truths)) != (54,486,54):
        raise RuntimeError('wrong fixed population')
    output.mkdir(parents=True)
    save(output/'INPUTS.json',{'source_root':str(DATA),'files':inputs})
    save(output/'SOURCE_LOCK.json',{'files':sources,'source_root':str(ROOT),'lock_scope':'scientific sources and frozen protocol',
                                  'utc':datetime.now(timezone.utc).isoformat()})
    save(output/'PROTOCOL.json',configuration())
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['observation_id']].append(task)
    decisions,outers,timing = [],[],[]
    started = time.perf_counter()
    for observation in observations:
        oid, arms, action_times = observation['observation_id'],{},{}
        completed = []
        for method in METHODS:
            entry = execute_arm(observation['observation'],observation['radius'],method)
            action_start = time.perf_counter()
            for task in grouped[oid]:
                decision = old.select_action(entry['outer'],decode(task['incumbent']),keep=method=='keep')
                completed.append({'task_id':task['task_id'],'method':method,**decision})
            action_times[method] = time.perf_counter()-action_start
            arms[method] = entry
        with old.finish_record_before_alarm():
            record = {'observation_id':oid,'arms':arms}
            save(output/f'checkpoint/{oid}.json',record)
            outers.append(record)
            decisions.extend(completed)
            timing.append({'observation_id':oid,'outer_seconds_by_method':{m:arms[m]['outer_seconds'] for m in METHODS},
                           'selector_seconds_by_method':action_times})
            save(output/'decisions.json',decisions)
            save(output/'TIMING_PARTIAL.json',{'observations':timing})
            save(output/'RUN_STATUS.json',{'status':'RUNNING','observations_complete':len(outers),
                                          'decisions_complete':len(decisions),'elapsed_seconds':time.perf_counter()-started})
        print(json.dumps({'observations_complete':len(outers),'decisions_complete':len(decisions),
                          'elapsed_seconds':round(time.perf_counter()-started,2)}),flush=True)
    save(output/'outers.json',outers)
    if load(output/'decisions.json') != decisions or len(decisions)!=2916:
        raise RuntimeError('all actions must be recorded before truth evaluation')
    save(output/'TIMING.json',{'observations':timing,'cold_cache_note':configuration()['cache_policy']})
    summary = evaluate(output,observations,tasks,truths,decisions,outers,timing,started)
    if input_lock()!=inputs or science_lock()!=sources:
        raise RuntimeError('frozen input/source changed during replay')
    save(output/'RUN_STATUS.json',{'status':'COMPLETE','observations_complete':54,'decisions_complete':2916,
                                  'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'stage':'complete','elapsed_seconds':summary['elapsed_seconds'],
                      'main':[r for r in summary['rows'] if r['grouping']=='overall' and r['state_group']=='small']}),flush=True)
    return summary

def smoke():
    public,_ = old.generate_observation('smoke_only','partial',(Q(1,4),Q(1,2),Q(3,4)),Q(600),99123,
                                      radius=Q(1,5),prefix='shared-ablation-smoke-v1')
    for method in METHODS:
        tick=time.perf_counter()
        entry=execute_arm(public['observation'],public['radius'],method)
        assert covered(entry['outer'],Q(600))
        assert entry['outer']['unresolved_budget_leaves']==0
        for initial in (Q(600),Q(2401,4),Q(597)):
            action=old.select_action(entry['outer'],initial,keep=method=='keep')
            assert abs(decode(action['output'])-600)<=abs(initial-600)
        print(json.dumps({'smoke':method,'seconds':time.perf_counter()-tick,
                          'hull':entry['outer']['hull'],'boxes':entry['outer']['evaluated_boxes']}),flush=True)

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ablation-v1')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv)
    if args.smoke:
        smoke()
        return 0
    output=validate_output(args.output)
    previous=signal.signal(signal.SIGALRM,old.deadline_reached)
    signal.alarm(TIMEOUT)
    try:
        run(output)
    except (Exception,KeyboardInterrupt) as error:
        if output.exists():
            state=load(output/'RUN_STATUS.json') if (output/'RUN_STATUS.json').exists() else {}
            save(output/'RUN_STATUS.json',{**state,'status':'INCOMPLETE','reason':str(error)})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
    return 0

if __name__=='__main__':
    raise SystemExit(main())

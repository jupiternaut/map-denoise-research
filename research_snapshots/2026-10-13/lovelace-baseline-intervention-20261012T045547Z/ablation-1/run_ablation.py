"""Public inference, complete persisted action gates, then paired evaluation."""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as Q
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True
for thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_variable] = '1'
import ablation_solver as solver

ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1')
DATA = ROOT.parent / 'data'
FULL = ROOT.parent / 'recovery-1/outputs/baseline-v1'
METHODS, MASKS, SELECTORS = ('calibrated', 'joint'), ('drop_F', 'matched_drop'), ('hybrid', 'midpoint', 'projection')
LIMIT = 3600


def exact(value):
    if isinstance(value, dict):
        if set(value) != {'numerator', 'denominator'} or type(value['numerator']) is not int or type(value['denominator']) is not int or value['denominator'] <= 0:
            raise ValueError('invalid exact rational')
        return Q(value['numerator'], value['denominator'])
    if isinstance(value, float) or isinstance(value, bool):
        raise ValueError('inexact rational input')
    return Q(value)


def encode(value):
    q = exact(value)
    return {'numerator': q.numerator, 'denominator': q.denominator}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, replace=False):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT) or path.resolve() == ROOT:
        raise ValueError('write outside exact phase root')
    if replace and path.name != 'RUN_STATUS.json':
        raise ValueError('only RUN_STATUS may be replaced')
    path.parent.mkdir(parents=True, exist_ok=True)
    with blocked_alarm():
        target = path.with_name(path.name + '.pending') if replace else path
        with target.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        if replace:
            target.replace(path)


class blocked_alarm:
    def __enter__(self):
        self.previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    def __exit__(self, *_):
        signal.pthread_sigmask(signal.SIG_SETMASK, self.previous)


def sources():
    result = load(FULL / 'SOURCE_LOCK.json')['files'].copy()
    for path, digest in result.items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: ' + path)
    result.update({str(path.resolve()): sha(path) for path in sorted(
        {ROOT / 'AGENTS.md', ROOT / 'PROTOCOL.md', ROOT / 'REGISTRATION.json', *ROOT.glob('*.py')})})
    return dict(sorted(result.items()))


def inputs():
    paths = [DATA / relative for relative in ('observations.json', 'tasks.json', 'private/truth.json', 'GENERATION.json')]
    paths += [FULL / relative for relative in ('outers.json', 'decisions.json', 'RUN_STATUS.json', 'verification/VERIFY.json', 'SOURCE_LOCK.json')]
    return {str(path.resolve()): sha(path) for path in sorted(paths)}


def population():
    observations, tasks = load(DATA / 'observations.json'), load(DATA / 'tasks.json')
    if len(observations) != 36 or len(tasks) != 324:
        raise RuntimeError('changed registered population')
    for public, (index, b) in zip(observations, ((i, b) for i in range(1, 19) for b in ('70', '80'))):
        oid = f'p{index:03d}b{b}'
        if public['observation_id'] != oid or public['pair_id'] != f'p{index:03d}' or public['unit_id'] != 'baseline_' + b or public['baseline'] != b or public['radius'] != '1/5' or public['clipped'] is not True or len(public['observation']) != 432:
            raise RuntimeError('changed public identity or encoded boundary')
    expected = [(public, s) for public in observations for s in range(9)]
    for task, (public, state) in zip(tasks, expected):
        if task['task_id'] != public['observation_id'] + f's{state}' or task['base_task_id'] != public['pair_id'] + f's{state}' or task['observation_id'] != public['observation_id']:
            raise RuntimeError('changed task identity')
    return observations, tasks


def endpoint(hull, incumbent, proposal):
    a = exact(incumbent)
    if hull is None:
        return {'status': 'INCOMPATIBLE', 'output': a, 'gain_lower': None, 'reason': 'empty_certified_outer_set', 'branch': 'incompatible'}
    left, right = map(exact, hull)
    b = (left + right) / 2 if proposal == 'midpoint' else min(max(a, left), right)
    gain = min((a-left)**2-(b-left)**2, (a-right)**2-(b-right)**2)
    move = a != b and gain > 0
    return {'status': 'MOVE' if move else 'KEEP', 'output': b if move else a, 'gain_lower': gain if move else Q(0),
            'reason': 'strict_gain_over_continuous_outer_hull' if move else 'no_strict_safe_gain_for_proposal',
            'branch': proposal if move else 'keep'}


def select(hull, incumbent, selector='hybrid'):
    if selector != 'hybrid':
        if selector not in ('midpoint', 'projection'):
            raise ValueError('unregistered selector')
        return endpoint(hull, incumbent, selector)
    result = endpoint(hull, incumbent, 'midpoint')
    return result if result['status'] == 'MOVE' else endpoint(hull, incumbent, 'projection')


def action(task, public, arm, condition, selector, replay=False):
    result = select(arm['outer']['hull'], task['incumbent'], selector)
    for field in ('output', 'gain_lower'):
        if result[field] is not None:
            result[field] = encode(result[field])
    return {'task_id': task['task_id'], 'unit_id': public['unit_id'], 'method': arm['outer']['mode'],
            'selector' if replay else 'mask_name': condition, 'reused': replay, **result}


def check_actions(records, tasks, conditions, replay):
    field = 'selector' if replay else 'mask_name'
    expected = {(task['task_id'], method, condition) for task in tasks for method in METHODS for condition in conditions}
    keys = [(r['task_id'], r['method'], r[field]) for r in records]
    if len(keys) != len(expected) or len(set(keys)) != len(expected) or set(keys) != expected or any(r['reused'] != replay for r in records):
        raise RuntimeError('incomplete or duplicated persisted action matrix')


def make_row(task, public, private, decision, arm, condition):
    truth, a, b = exact(private['truth']), exact(task['incumbent']), exact(decision['output'])
    hull = arm['outer']['hull']
    return {'task_id': task['task_id'], 'base_task_id': task['base_task_id'], 'observation_id': task['observation_id'],
            'pair_id': public['pair_id'], 'unit_id': public['unit_id'], 'baseline': public['baseline'],
            'method': decision['method'], 'condition': condition, 'case': private['case'], 'depth_mm': str(truth),
            'initial_ae_mm': str(abs(a-truth)), 'ae_mm': str(abs(b-truth)), 'se_mm2': str((b-truth)**2),
            'actual_gain_mm2': str((a-truth)**2-(b-truth)**2), 'output': str(b), 'status': decision['status'],
            'covered': hull is not None and exact(hull[0]) <= truth <= exact(hull[1]),
            'hull_width_mm': None if hull is None else str(exact(hull[1])-exact(hull[0]))}


def metric_rows(decisions, observations, tasks, truths, arm_lookup, condition_field):
    tm = {task['task_id']: task for task in tasks}
    om = {public['observation_id']: public for public in observations}
    rows = []
    for decision in decisions:
        task = tm[decision['task_id']]
        public = om[task['observation_id']]
        arm = arm_lookup(public['observation_id'], decision)
        rows.append(make_row(task, public, truths[public['observation_id']], decision, arm, decision[condition_field]))
    return rows


def coverage(public, private, method, arm, mask_name):
    truth, branch, hull = exact(private['truth']), private['rear_branch'], arm['outer']['hull']
    retained = {name: [node for node in value['trace'] if node['id'] in value['retained_ids']] for name, value in arm['branches'].items()}
    true_nodes = {name: [n for n in nodes if exact(n['lo']) <= truth <= exact(n['hi'])] for name, nodes in retained.items()}
    true_leaves = true_nodes[branch]
    darkness = [1-exact(color) for color in private['colors']]
    actual_t = [Q(1,100)-exact(bias) for bias in private['actual_view_bias']]
    latent = lambda key, values: bool(true_leaves) and all(all(exact(pair[0]) <= value <= exact(pair[1]) for pair, value in zip(node[key], values)) for node in true_leaves)
    return {'mask_name': mask_name, 'observation_id': public['observation_id'], 'unit_id': public['unit_id'], 'method': method,
            'case': private['case'], 'seed': private['seed'], 'covered': any(true_nodes.values()), 'incompatible': hull is None,
            'hull_width_mm': None if hull is None else str(exact(hull[1])-exact(hull[0])), 'retained_branches': arm['outer']['retained_branches'],
            'branch_coverage': {name: bool(nodes) for name, nodes in true_nodes.items()}, 'generating_branch': branch,
            'generating_branch_covered': bool(true_leaves), 'true_depth_leaf_ids': {name: [n['id'] for n in nodes] for name, nodes in true_nodes.items()},
            'latent_material_covered': latent('darkness_bounds', darkness), 'latent_bias_covered': latent('t_bounds', actual_t) if method == 'joint' else None}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        common = (row['unit_id'], row['method'], row['condition'])
        shifted = exact(row['initial_ae_mm']) != 0
        groups[common + ('all_shifted' if shifted else 'correct', '')].append(row)
        if shifted:
            groups[common + ('negative_shifted' if row['case'] == 'equal_full_negative' else 'informative_shifted', '')].append(row)
            groups[common + ('case_shifted', row['case'])].append(row)
            if row['case'] != 'equal_full_negative':
                groups[common + ('depth_informative', row['depth_mm'])].append(row)
    result = []
    for key, selected in sorted(groups.items()):
        n = len(selected)
        initial = [exact(r['initial_ae_mm']) for r in selected]
        ae = [exact(r['ae_mm']) for r in selected]
        result.append(dict(zip(('unit_id', 'method', 'condition', 'group', 'value'), key), n=n,
            mae_mm=str(sum(ae,Q(0))/n), mse_mm2=str(sum((v*v for v in ae),Q(0))/n),
            keep_mse_mm2=str(sum((v*v for v in initial),Q(0))/n),
            improved=sum(b<a for a,b in zip(initial,ae)), same=sum(b==a for a,b in zip(initial,ae)), harmed=sum(b>a for a,b in zip(initial,ae))))
    return result


def csv_save(path, rows):
    with blocked_alarm(), Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output, observations, tasks, new_outers, full_outers, new_decisions, selector_decisions):
    # Called only after BOTH complete matrices are persisted and read back.
    truths = {fact['observation_id']: fact for fact in load(DATA / 'private/truth.json')}
    new_rows = metric_rows(new_decisions, observations, tasks, truths,
        lambda oid, d: new_outers[oid][d['mask_name']][d['method']], 'mask_name')
    selector_rows = metric_rows(selector_decisions, observations, tasks, truths,
        lambda oid, d: full_outers[oid][d['method']], 'selector')
    references = [{**decision, 'condition': 'full'} for decision in load(FULL / 'decisions.json')]
    reference_rows = metric_rows(references, observations, tasks, truths,
        lambda oid, d: full_outers[oid][d['method']], 'condition')
    cov = [coverage(public, truths[public['observation_id']], method,
        new_outers[public['observation_id']][mask][method], mask) for public in observations for mask in MASKS for method in METHODS]
    for path, rows in (('results',new_rows), ('selectors/results',selector_rows), ('reference/results',reference_rows)):
        save(output / (path+'.json'), rows)
        csv_save(output / (path+'.csv'), rows)
    save(output / 'coverage.json', cov)
    save(output / 'summary.json', {'masked': aggregate(new_rows), 'selectors': aggregate(selector_rows), 'reference': aggregate(reference_rows)})


def run(output):
    output = Path(output).resolve()
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED or not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise RuntimeError('wrong host/root/output or existing run')
    output.mkdir(parents=True)
    start = time.time()
    budget = {'start_epoch': start, 'deadline_epoch': start+LIMIT, 'limit_seconds': LIMIT,
              'scope': 'A7 and A1+A5 main plus independent verification; no reset'}
    save(output/'BUDGET.json', budget)
    source_hashes, input_hashes = sources(), inputs()
    save(output/'SOURCE_LOCK.json', {'source_root': str(ROOT), 'files':source_hashes})
    save(output/'INPUTS.json', {'files':input_hashes})
    save(output/'REGISTRATION.json', load(ROOT/'REGISTRATION.json'))
    def reached(_signum, _frame):
        raise TimeoutError('absolute phase deadline reached')
    previous = signal.signal(signal.SIGALRM, reached)
    remaining = budget['deadline_epoch'] - time.time()
    if remaining <= 0:
        raise TimeoutError('phase window exhausted before inference')
    signal.setitimer(signal.ITIMER_REAL, remaining)
    calls, decisions, selector_decisions, outers, timing = 0, [], [], {}, []
    try:
        observations, tasks = population()
        save(output/'observations.json', observations)
        save(output/'tasks.json', tasks)
        full_outers = load(FULL/'outers.json')
        task_map = defaultdict(list)
        for task in tasks:
            task_map[task['observation_id']].append(task)
        for public in observations:
            oid = public['observation_id']
            for method in METHODS:
                for selector in SELECTORS:
                    selector_decisions.extend(action(task,public,full_outers[oid][method],selector,selector,True) for task in task_map[oid])
        save(output/'selectors/decisions.json', selector_decisions)
        selector_decisions = load(output/'selectors/decisions.json')
        check_actions(selector_decisions,tasks,SELECTORS,True)
        for index, public in enumerate(observations):
            oid = public['observation_id']
            outers[oid] = {}
            for mask in MASKS:
                outers[oid][mask] = {}
                for method in METHODS:
                    solver.clear_geometry_cache()
                    tick = time.perf_counter()
                    arm = solver.solve_public(public['observation'],public['radius'],public['baseline'],method,mask,clipped=public['clipped'])
                    seconds = time.perf_counter()-tick
                    calls += 1
                    if arm['calibration'] != full_outers[oid][method]['calibration']:
                        raise RuntimeError('mask changed full-image calibration')
                    outers[oid][mask][method] = arm
                    timing.append({'observation_id':oid,'mask_name':mask,'method':method,'cold_seconds':seconds,'evaluated_boxes':arm['outer']['evaluated_boxes']})
                    decisions.extend(action(task,public,arm,mask,'hybrid') for task in task_map[oid])
            save(output/f'checkpoint/{oid}.json', {'observation_id':oid,'arms':outers[oid]})
            save(output/'RUN_STATUS.json', {'status':'RUNNING','observations_complete':index+1,'decisions_complete':len(decisions),
                'selector_decisions_complete':len(selector_decisions),'new_solver_calls':calls,'elapsed_seconds':time.time()-start,'deadline_epoch':budget['deadline_epoch']},replace=True)
            print(json.dumps({'stage':'A1_A5','observations_complete':index+1,'new_solver_calls':calls,'elapsed_seconds':round(time.time()-start,1)}),flush=True)
        save(output/'outers.json',outers)
        save(output/'decisions.json',decisions)
        save(output/'TIMING.json',timing)
        decisions = load(output/'decisions.json')
        outers = load(output/'outers.json')
        check_actions(decisions,tasks,MASKS,False)
        if calls != 144 or len(outers) != 36:
            raise RuntimeError('incomplete solve matrix')
        evaluate(output,observations,tasks,outers,full_outers,decisions,selector_decisions)
        if sources()!=source_hashes or inputs()!=input_hashes:
            raise RuntimeError('frozen byte graph changed')
        save(output/'RUN_STATUS.json', {'status':'COMPLETE','observations_complete':36,'decisions_complete':1296,
            'selector_decisions_complete':1944,'new_solver_calls':144,'elapsed_seconds':time.time()-start,'deadline_epoch':budget['deadline_epoch']},replace=True)
    except BaseException as error:
        save(output/'INCOMPLETE.json', {'error':type(error).__name__,'message':str(error),'calls':calls,'decisions':len(decisions),'elapsed_seconds':time.time()-start})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    return str(output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ablation-v1')
    print(run(parser.parse_args().output))

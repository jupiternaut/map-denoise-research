#!/usr/bin/env python3
"""Portable, read-only A15 archive and saved-arithmetic audit (stdlib only).

Only publication utilities are imported. Archived research programs, solvers,
and absolute provenance paths are never executed or opened. Saved proof audit
receipts are reconciled, not a new proof of every geometric certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ARCHIVE_ROOT = 'research_snapshots/2026-10-14-proof-reuse'
MANIFEST_PATH = 'publication/LOVELACE_A15_20261014_MANIFEST.json'
A15 = 'lovelace-proof-reuse-20261014T074405Z'
STAGES = {'tolerance': (('original', 'pure'), 1),
          'switches': (('pure', 'off', 'hot', 'inherit', 'both'), 3)}
EPSILON = 1e-9


def isolated(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Never modify the globals of an ordinarily imported older verifier.
_seal = isolated('_a15_private_seal', 'verify_lovelace_20261013.py')
_seal.ARCHIVE_ROOT, _seal.MANIFEST_PATH = ARCHIVE_ROOT, MANIFEST_PATH
_history = isolated('_a15_private_history', 'verify_lovelace_methods_20261013.py')
AuditError, require, rational = _seal.AuditError, _seal.require, _seal.rational
A14, FORMAL = _history.A14, _history.FORMAL


class Snapshot(_seal.Snapshot):
    def source_bytes(self, relative):
        _seal.canonical_relative(relative)
        require(relative in self.sources, 'missing archived record: ' + relative)
        return self.read_bytes(self.sources[relative])

    def source_json(self, relative):
        return json.loads(self.source_bytes(relative))


class Comparison:
    def __init__(self):
        self.leaves = 0
        self.max_numeric_difference = 0.0

    def equal(self, expected, actual, label):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(expected) == set(actual), label + ': keys differ')
            for key, value in expected.items():
                self.equal(value, actual[key], label + '.' + key)
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(expected) == len(actual), label + ': length differs')
            for i, (left, right) in enumerate(zip(expected, actual)):
                self.equal(left, right, f'{label}[{i}]')
        elif type(expected) is float:
            finite(actual, label)
            difference = abs(expected - actual)
            self.max_numeric_difference = max(self.max_numeric_difference, difference)
            require(math.isfinite(expected) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12),
                    label + ': numeric mismatch')
            self.leaves += 1
        else:
            require(type(expected) is type(actual) and expected == actual, label + ': value/type mismatch')
            self.leaves += 1


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + ': finite number required')
    return value


def nonnegative(value, label):
    require(finite(value, label) >= 0, label + ': negative value')
    return value


def exact_text(value):
    value = Fraction(value)
    return f'{value.numerator}/{value.denominator}'


def unique(records, keys, label):
    require(isinstance(records, list), label + ': array required')
    result = {}
    for record in records:
        require(isinstance(record, dict), label + ': object required')
        key = tuple(record[k] for k in keys)
        require(key not in result, label + ': duplicate identity')
        result[key] = record
    return result


def metrics(raw, expected):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
    fields = list(expected[0])
    require(reader.fieldnames == fields, 'METRICS.csv: columns differ')
    saved = list(reader)
    require(len(saved) == len(expected), 'METRICS.csv: row count differs')
    converted = []
    for row, template in zip(saved, expected):
        require(set(row) == set(fields) and all(v is not None for v in row.values()), 'METRICS.csv: malformed row')
        parsed = {}
        for key, value in template.items():
            if type(value) is bool:
                require(row[key] in ('True', 'False'), 'METRICS.csv: invalid boolean')
                parsed[key] = row[key] == 'True'
            else:
                parsed[key] = type(value)(row[key])
        converted.append(parsed)
    return converted


def state_key(state):
    return {'lower': state['lower'], 'upper': state['upper'],
            'cells': sorted((c['branch'], c['lower'], c['upper'], c['checked'], c['terminal'])
                            for c in state['cells']),
            'reason': state['reason'],
            'exclusion_cells': sorted((c['branch'], c['lower'], c['upper'], c['mode'])
                                      for c in state['exclusions'])}


def validate_state(state):
    require(isinstance(state, dict), 'successful call requires a state')
    lower, upper = finite(state['lower'], 'state lower'), finite(state['upper'], 'state upper')
    require(540 <= lower <= upper <= 660, 'state hull outside depth domain')
    require(state['reason'] == 'boundary_resolution', 'unexpected stop reason/budget cutoff')
    require(isinstance(state['cells'], list) and state['cells'], 'nonempty retained cells required')
    for cell in state['cells'] + state['exclusions']:
        require(cell['branch'] in ('none', 'partial', 'full'), 'invalid cell branch')
        lo, hi = finite(cell['lower'], 'cell lower'), finite(cell['upper'], 'cell upper')
        require(540 <= lo <= hi <= 660, 'invalid cell interval')
    require(lower == min(c['lower'] for c in state['cells']) and
            upper == max(c['upper'] for c in state['cells']), 'hull differs from retained-cell envelope')
    for cell in state['cells']:
        require(type(cell['checked']) is bool and type(cell['terminal']) is bool, 'invalid retained cell flags')
    for cell in state['exclusions']:
        require(cell['verified'] is True and cell['mode'] in ('whole', 'tiles'), 'unverified excluded cell')
    # Recheck the one-dimensional partition, without re-deriving geometry.
    for branch in ('none', 'partial', 'full'):
        intervals = sorted((c['lower'], c['upper']) for c in state['cells'] + state['exclusions']
                           if c['branch'] == branch)
        require(intervals and intervals[0][0] == 540 and intervals[-1][1] == 660,
                'saved partition does not cover depth domain')
        require(all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])),
                'saved partition has gap or overlap')
    for name, value in state['component_seconds'].items():
        nonnegative(value, 'component ' + name)


def summary_for(rows, outputs, receipts, oldmap):
    costs, states = defaultdict(list), {}
    for output, receipt in zip(outputs, receipts):
        costs[output['arm']].append(receipt)
        states[output['arm'], output['replicate'], output['observation_id']] = output['state']
    arms = {}
    for arm, cc in costs.items():
        rr = [r for r in rows if r['arm'] == arm and r['replicate'] == 0]
        shift = [r for r in rr if not r['correct_input']]
        ss = [s for (a, rep, oid), s in states.items() if a == arm]
        require(len(rr) == 126 and len(shift) == 108, 'arm requires 126 inputs, 108 shifted')
        arms[arm] = dict(n_inputs_first_repeat=len(rr), shifted_mse=sum(r['final_se'] for r in shift)/len(shift),
                         improved=sum(r['final_se'] < r['initial_se']-EPSILON for r in shift),
                         same=sum(abs(r['final_se']-r['initial_se']) <= EPSILON for r in shift),
                         harmed=sum(r['final_se'] > r['initial_se']+EPSILON for r in shift),
                         correct_moved=sum(r['correct_input'] and r['final_se'] > EPSILON for r in rr),
                         all_covered=all(r['covered'] for r in rr),
                         cold_mean=sum(c['process_seconds'] for c in cc)/len(cc),
                         cpu_mean=sum(c['process_cpu_seconds'] for c in cc)/len(cc),
                         reasons=dict(Counter(s['reason'] for s in ss)),
                         repeat_output_consistent=all(s['lower'] == states[arm, 0, oid]['lower'] and
                                                     s['upper'] == states[arm, 0, oid]['upper']
                                                     for (a, rep, oid), s in states.items() if a == arm))
    groups = defaultdict(list)
    for row in rows:
        if row['arm'] != 'pure' or row['replicate'] != 0 or row['correct_input']:
            continue
        tid, oid = row['task_id'], row['observation_id']
        joint, fine, legacy = (float(oldmap[a, b, tid]['final_se'])
                               for a, b in (('joint', 6.), ('fine', 6.), ('legacy', 24.)))
        a, b = joint > fine+EPSILON, joint > legacy+EPSILON
        group = 'both' if a and b else 'fine_only' if a else 'legacy_only' if b else 'neither'
        groups[group].append(dict(task_id=tid, observation_id=oid, old_joint_se=joint, fine_se=fine,
                                 legacy24_se=legacy, new_se=row['final_se'],
                                 fine_loss_removed=row['final_se'] <= fine+EPSILON,
                                 legacy_loss_removed=row['final_se'] <= legacy+EPSILON,
                                 reason=states['pure', 0, oid]['reason']))
    grouped = {g: dict(n=len(rr), improved_over_joint=sum(r['new_se'] < r['old_joint_se']-EPSILON for r in rr),
                       remaining_fine_losses=sum(not r['fine_loss_removed'] for r in rr),
                       remaining_legacy_losses=sum(not r['legacy_loss_removed'] for r in rr), records=rr)
               for g, rr in groups.items()}
    require({g: v['n'] for g, v in grouped.items()} ==
            {'both': 26, 'fine_only': 12, 'legacy_only': 26, 'neither': 44}, 'historical loss groups differ')
    equivalence = {}
    if 'original' in costs:
        equivalence['historical_original_exact_selected'] = all(
            r['selected'] == float(oldmap['joint', 6., r['task_id']]['selected'])
            for r in rows if r['arm'] == 'original')
        require(equivalence['historical_original_exact_selected'], 'original replay differs from A14')
    else:
        baseline = {(r['replicate'], r['task_id']): r for r in rows if r['arm'] == 'pure'}
        for arm in ('off', 'hot', 'inherit', 'both'):
            aa = [r for r in rows if r['arm'] == arm]
            deltas = [abs(r['selected']-baseline[r['replicate'], r['task_id']]['selected']) for r in aa]
            equivalence[arm] = dict(exact_selected_equal=all(x == 0 for x in deltas), max_abs_selected_delta=max(deltas))
            require(equivalence[arm]['exact_selected_equal'], 'switch output differs from pure')
    return arms, grouped, equivalence


def audit_stage(snapshot, historical, stage, observations, tasks, truth, oldmap, check):
    load = lambda name: snapshot.source_json(A15 + '/' + name)
    prefix = 'outputs/' + stage + '/'
    arms, repeats = STAGES[stage]
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['observation_id']].append({k: task[k] for k in ('task_id', 'incumbent')})
    expected_jobs = []
    for rep in range(repeats):
        for i, public in enumerate(observations):
            shift = (rep+i) % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                expected_jobs.append(dict(arm=arm, replicate=rep, budget_seconds=6.,
                                          observation_id=public['observation_id'], observation=public['observation'],
                                          tasks=grouped[public['observation_id']]))
    jobs, ready = load(prefix + 'MANIFEST.json'), load(prefix + 'READY.json')
    check.equal(expected_jobs, jobs, stage + '.MANIFEST.json')
    count = len(expected_jobs)
    require(ready['calls'] == count and len(ready['receipts']) == count and ready['truth_loaded'] is False,
            stage + ': READY call count/truth gate differs')
    freeze_raw = snapshot.source_bytes(A15 + '/FREEZE_' + stage + '.json')
    require(ready['freeze_sha'] == hashlib.sha256(freeze_raw).hexdigest(), stage + ': freeze digest differs')
    receipts = ready['receipts']
    require(len(unique(receipts, ('arm', 'replicate', 'observation_id'), 'receipts')) == count,
            stage + ': receipt matrix differs')
    verification = load(prefix + 'VERIFICATION.json')
    require(verification['all_valid'] is True and verification['truth_loaded'] is False and
            len(verification['records']) == count, stage + ': saved verification receipt count/gate differs')
    rows, outputs, exact_covered, proof_keys = [], [], 0, set()
    for i, (job, receipt, audit) in enumerate(zip(jobs, receipts, verification['records'])):
        stem = prefix + f'jobs/{i:03}'
        check.equal(job, load(stem + '.job.json'), stage + f'.job {i}')
        check.equal(receipt, load(stem + '.receipt.json'), stage + f'.receipt {i}')
        require(receipt['index'] == i and receipt['result'] == stem + '.result.json', 'receipt result/index differs')
        raw = snapshot.source_bytes(A15 + '/' + receipt['result'])
        require(hashlib.sha256(raw).hexdigest() == receipt['sha256'], 'receipt result digest mismatch')
        output = json.loads(raw)
        for key in ('arm', 'replicate', 'observation_id', 'budget_seconds'):
            check.equal(job[key], receipt[key], stage + f'.receipt {i}.{key}')
            check.equal(job[key], output[key], stage + f'.result {i}.{key}')
        require(output['schema'] == 'a15-cold-v1' and output['prepare_failure'] is None and
                receipt['returncode'] == 0 and type(receipt['returncode']) is int, 'failed cold invocation')
        for key in ('process_seconds', 'process_cpu_seconds'):
            nonnegative(receipt[key], 'receipt ' + key)
        for key, value in output['cost'].items():
            nonnegative(value, 'worker cost ' + key)
        state = output['state']
        validate_state(state)
        require(output['dependencies'] and set(output['dependencies']) <= set(json.loads(freeze_raw)['files']),
                'unfrozen dynamic dependency')
        if output['arm'] not in ('original', 'pure'):
            require(state['tolerance'] == .001 and state['hot'] == (output['arm'] in ('hot', 'both')) and
                    state['inherit'] == (output['arm'] in ('inherit', 'both')), 'switch flags differ')
        proof = audit['proof_check']
        require(audit['result'] == receipt['result'] and audit['valid'] is True and audit['decoder_valid'] is True and
                proof['valid'] is True and proof['partitioned'] is True and proof['conservative'] is True,
                'saved proof/decoder validation flag differs')
        check.equal(len(state['exclusions']), proof['checked_exclusions'], 'proof checked_exclusions')
        check.equal(len(state['cells']), proof['retained_cells'], 'proof retained_cells')
        check.equal([exact_text(state['lower']), exact_text(state['upper'])], proof['hull'], 'proof hull')
        proofstate = {k: state[k] for k in ('cells', 'exclusions', 'lower', 'upper')}
        proof_keys.add((output['observation_id'], hashlib.sha256(json.dumps(proofstate, sort_keys=True).encode()).hexdigest()))
        predictions = output['predictions']
        require(len(predictions) == 7, 'seven predictions per cold call required')
        check.equal(job['tasks'], [{k: p[k] for k in ('task_id', 'incumbent')} for p in predictions], 'prediction inputs')
        exact_truth = rational(truth[output['observation_id']]['truth'])
        z = float(exact_truth)
        covered = state['lower'] <= z <= state['upper']
        exact = Fraction(state['lower']) <= exact_truth <= Fraction(state['upper'])
        require(covered and exact, 'truth outside saved hull')
        exact_covered += exact
        for prediction in predictions:
            incumbent = finite(prediction['incumbent'], 'incumbent')
            selected = finite(prediction['selected'], 'selected')
            require(prediction['failure'] is None and
                    selected == min(state['upper'], max(state['lower'], incumbent)), 'prediction differs from hull projection')
            rows.append(dict(arm=output['arm'], replicate=output['replicate'], observation_id=output['observation_id'],
                             task_id=prediction['task_id'], truth=z, incumbent=incumbent, selected=selected,
                             initial_se=(incumbent-z)**2, final_se=(selected-z)**2,
                             correct_input=abs(incumbent-z) <= EPSILON, covered=covered))
        outputs.append(output)
    require(len(unique(rows, ('arm', 'replicate', 'task_id'), 'predictions')) == count*7, 'prediction matrix differs')
    check.equal(len(proof_keys), verification['unique_proof_states'], stage + '.unique_proof_states')
    check.equal(rows, metrics(snapshot.source_bytes(A15 + '/' + prefix + 'METRICS.csv'), rows), stage + '.METRICS.csv')
    arm_summary, groups, equivalence = summary_for(rows, outputs, receipts, oldmap)
    expected_summary = dict(arms=arm_summary, historical_groups=groups, equivalence=equivalence,
                            scope='18 exposed A14 worlds, not new confirmation',
                            truth_sha=hashlib.sha256(historical.source_bytes(A14 + '/data/confirm/private/truth.json')).hexdigest())
    check.equal(expected_summary, load(prefix + 'SUMMARY.json'), stage + '.SUMMARY.json')
    cpu = sum(r['process_cpu_seconds'] for r in receipts)
    check.equal(cpu, ready['child_cpu_seconds'], stage + '.READY.child_cpu_seconds')
    nonnegative(ready['wall_seconds'], stage + '.READY.wall_seconds')
    return dict(outputs=outputs, receipts=receipts, rows=rows, ready=ready, summaries=arm_summary,
                groups=groups, exact_covered=exact_covered,
                exclusions=sum(len(o['state']['exclusions']) for o in outputs))


def audit_equivalence(snapshot, switches, check):
    indexed = {(o['arm'], o['replicate'], o['observation_id']): o for o in switches['outputs']}
    records = []
    for output in switches['outputs']:
        arm, rep, oid = (output[k] for k in ('arm', 'replicate', 'observation_id'))
        if arm == 'pure':
            continue
        left, right = state_key(output['state']), state_key(indexed['pure', rep, oid]['state'])
        require(left == right, 'full state equivalence differs: ' + repr((arm, rep, oid)))
        records.append(dict(arm=arm, replicate=rep, observation_id=oid,
                            fields_equal={k: left[k] == right[k] for k in left}, all_equal=left == right))
    require(len(records) == 216, '216 full state comparisons required')
    check.equal(dict(records=records, all_equal=True, truth_loaded=False),
                snapshot.source_json(A15 + '/outputs/switches/STATE_EQUIVALENCE.json'), 'STATE_EQUIVALENCE.json')
    return len(records)


def profile_for(switches, observations, ledger):
    grouped = defaultdict(list)
    for output, receipt in zip(switches['outputs'], switches['receipts']):
        grouped[output['arm']].append((output, receipt))
    arms = {}
    for arm, items in grouped.items():
        times, counts, repeat_cold = defaultdict(float), defaultdict(int), defaultdict(list)
        for output, receipt in items:
            state = output['state']
            repeat_cold[str(output['replicate'])].append(receipt['process_seconds'])
            for key, value in state['component_seconds'].items():
                times[key] += value
            for key, value in state.get('inner_counts', {}).items():
                require(type(value) is int and value >= 0, 'invalid inner counter')
                counts[key] += value
            reuse = state.get('tile_reuse', {})
            for key, value in reuse.get('counts', {}).items():
                require(type(value) is int and value >= 0, 'invalid reuse counter')
                require(sum(t['counts'][key] for t in reuse['per_tile']) == value, 'reuse per-tile counter mismatch')
                counts['reuse_' + key] += value
            for event in state.get('tile_order_trace', []):
                require(event['order'] and len(set(event['order'])) == len(event['order']) and
                        set(event['order']) <= {0, 1, 2, 3} and len(event['order']) == len(event['statuses']),
                        'invalid tile order trace')
                counts['actual_first_nonzero_tile'] += bool(event['order'] and event['order'][0] != 0)
            for attempt in state['certificate_attempts']:
                children = attempt.get('tile_results', []) if attempt['mode'] == 'tiles' else [attempt]
                for child in children:
                    if child.get('error'):
                        counts['cert_errors_or_deadlines'] += 1
            if arm != 'pure':
                inner = state['inner_counts']
                attempts = state['certificate_attempts']
                require(inner['whole_float_lp'] == state['stats']['lp_calls'] and
                        inner['tile_float_lp'] == sum(len(e['statuses']) for e in state['tile_order_trace']),
                        'completed floating LP counter mismatch')
                require(inner['whole_certificate_calls'] == sum(a['mode'] == 'whole' for a in attempts) and
                        inner['tile_cover_certificate_calls'] == sum(a['mode'] == 'tiles' for a in attempts),
                        'certificate call counter mismatch')
        arms[arm] = dict(n_calls=len(items), mean_components_seconds={k: v/len(items) for k, v in times.items()},
                         completed_lp_and_reuse_counts=dict(counts),
                         cold_mean_by_repeat={k: sum(v)/len(v) for k, v in repeat_cold.items()},
                         cold_mean=sum(r['process_seconds'] for o, r in items)/len(items),
                         cpu_mean=sum(r['process_cpu_seconds'] for o, r in items)/len(items))
    keyed = {(o['arm'], o['replicate'], o['observation_id']): r
             for items in grouped.values() for o, r in items}
    pairs = []
    for rep in range(3):
        for public in observations:
            oid = public['observation_id']
            r = {arm: keyed[arm, rep, oid] for arm in grouped}
            pairs.append(dict(replicate=rep, observation_id=oid,
                              hot_minus_off=r['hot']['process_seconds']-r['off']['process_seconds'],
                              inherit_minus_off=r['inherit']['process_seconds']-r['off']['process_seconds'],
                              interaction=r['both']['process_seconds']-r['hot']['process_seconds']-
                              r['inherit']['process_seconds']+r['off']['process_seconds']))
    return dict(arms=arms, paired_costs=pairs, ledger=ledger,
                note='nested times must not be summed; counts are completed LP returns, exception attempts are not inferred')


def boundary_cells(state):
    result = {}
    for cell in state['cells']:
        sides = [side for side, yes in (('lower', cell['lower'] == state['lower']),
                                       ('upper', cell['upper'] == state['upper'])) if yes]
        if not sides:
            continue
        lo, hi = (exact_text(cell[k]) for k in ('lower', 'upper'))
        key = cell['branch'], lo, hi
        if key not in result:
            result[key] = dict(cell, lo_exact=lo, hi_exact=hi, sides=sides)
        else:
            result[key]['sides'] = sorted(set(result[key]['sides'] + sides))
    return list(result.values())


def audit_boundary(snapshot, historical, tolerance, check):
    diagnostic = snapshot.source_json(A15 + '/outputs/boundary-tolerance.json')
    expected_sources, targets = {}, {}
    for output, receipt in zip(tolerance['outputs'], tolerance['receipts']):
        path, state, oid = receipt['result'], output['state'], output['observation_id']
        expected_sources[path] = receipt['sha256']
        for cell in boundary_cells(state):
            key = oid, cell['branch'], cell['lo_exact'], cell['hi_exact']
            target = targets.setdefault(key, dict(observation_id=oid, cell=cell, sources=[]))
            target['sources'].append(dict(path=path, arm=output['arm'], budget_seconds=output['budget_seconds'],
                                          hull=[state['lower'], state['upper']], stop_reason=state['reason'],
                                          checked=cell['checked'], terminal=cell['terminal'], sides=cell['sides']))
    check.equal(expected_sources, diagnostic['input_result_sha256'], 'boundary input source mapping')
    require(diagnostic['schema'] == 'a15-boundary-four-tile-diagnostic-v1' and
            diagnostic['unique_target_cells'] == diagnostic['diagnosed_cells'] == len(targets) == 72 and
            len(diagnostic['records']) == 72 and diagnostic['all_targets_completed'] is True and
            diagnostic['skipped'] == [], 'boundary target counts/completion differ')
    require(diagnostic['script_sha256'] == hashlib.sha256(snapshot.source_bytes(A15 + '/boundary_diagnostic.py')).hexdigest(),
            'boundary diagnostic script digest differs')
    require(diagnostic['public_sha256'] == hashlib.sha256(historical.source_bytes(A14 + '/data/confirm/public.json')).hexdigest(),
            'boundary public digest differs')
    counts, statuses, process_cpu = Counter(), Counter(), 0.
    seen = set()
    for record in diagnostic['records']:
        cell, oid = record['cell'], record['observation_id']
        key = oid, cell['branch'], cell['lo_exact'], cell['hi_exact']
        require(key in targets and key not in seen, 'boundary missing/duplicate cell source identity')
        seen.add(key)
        check.equal(targets[key], {k: record[k] for k in ('observation_id', 'cell', 'sources')}, 'boundary source record')
        tiles = record['tiles']
        require([t['tile_id'] for t in tiles] == [0, 1, 2, 3], 'boundary requires exactly four tiles')
        for tile in tiles:
            status, validation = tile['status'], tile['independent_validation']
            require(status in ('exact_infeasible', 'exact_necessary_lp_feasible') and
                    validation['verified'] is True and validation['kind'] == status and tile['worker_exitcode'] == 0,
                    'boundary unknown/rejected exact object')
            if status == 'exact_infeasible':
                proof = tile['certificate']
                require(proof['branch'] == cell['branch'] and proof['lo'] == cell['lo_exact'] and
                        proof['hi'] == cell['hi_exact'] and proof['tile_id'] == tile['tile_id'] and
                        proof['observation_sha256'] == tile['observation_sha256'] and
                        proof['camera_box'] == tile['camera_box'] and rational(proof['margin']) > 0,
                        'boundary certificate identity/positive margin differs')
                check.equal(proof['margin'], validation['margin'], 'boundary recorded margin')
            else:
                require(tile['certificate'] is None and len(tile['witness']) == 15, 'boundary 15-coordinate witness required')
                for value in tile['witness']:
                    rational(value)
            require(tile['hard_cap_seconds'] == diagnostic['tile_hard_cap_seconds'] == 3., 'boundary cap differs')
            process_cpu += nonnegative(tile['process_cpu_seconds'], 'boundary process CPU')
            statuses[status] += 1
        classification = ('necessary_relaxation_retains_at_least_one_tile' if
                          any(t['status'] == 'exact_necessary_lp_feasible' for t in tiles)
                          else 'all_four_exactly_infeasible_offline')
        require(record['classification'] == classification, 'boundary cell classification differs')
        counts[classification] += 1
    require(seen == set(targets), 'boundary target set incomplete')
    check.equal(dict(counts), diagnostic['counts'], 'boundary counts')
    check.equal(dict(statuses), diagnostic['tile_counts'], 'boundary tile_counts')
    check.equal(process_cpu, diagnostic['process_cpu_seconds'], 'boundary process_cpu_seconds')
    require(dict(statuses) == {'exact_infeasible': 125, 'exact_necessary_lp_feasible': 163},
            'boundary expected 125 infeasible and 163 necessary-feasible tiles')
    return dict(cells=len(targets), tiles=sum(statuses.values()), tile_counts=dict(statuses),
                source_results=len(expected_sources), process_cpu_seconds=process_cpu,
                exact_geometry_reproved=False)


def numerical_audit(snapshot, historical=None):
    historical = historical or _history.Snapshot(snapshot.repository)
    # Historical comparison inputs must themselves have consistent arithmetic.
    _history.numerical_audit(historical)
    check = Comparison()
    tasks = historical.source_json(A14 + '/data/confirm/tasks.json')
    observations = historical.source_json(A14 + '/data/confirm/public.json')['observations']
    facts = historical.source_json(A14 + '/data/confirm/private/truth.json')
    truth = {key[0]: value for key, value in unique(facts, ('observation_id',), 'truth').items()}
    require(len(observations) == len(truth) == 18 and len(unique(tasks, ('task_id',), 'tasks')) == 126 and
            Counter(t['observation_id'] for t in tasks) == Counter({oid: 7 for oid in truth}),
            '18 worlds and 126 tasks required')
    require(set(truth) == {p['observation_id'] for p in observations}, 'public/truth identities differ')
    for task in tasks:
        require(task['incumbent'] == float(rational(task['incumbent_exact'])), 'incumbent exact value differs')
    oldrows = list(csv.DictReader(io.StringIO(historical.source_bytes(A14 + '/' + FORMAL + 'METRICS.csv').decode())))
    oldmap = {(r['arm'], float(r['budget_seconds']), r['task_id']): r for r in oldrows}
    stages = {stage: audit_stage(snapshot, historical, stage, observations, tasks, truth, oldmap, check)
              for stage in STAGES}
    comparisons = audit_equivalence(snapshot, stages['switches'], check)
    boundary = audit_boundary(snapshot, historical, stages['tolerance'], check)
    diagnostic_cpu = sum(snapshot.source_json(name)['process_cpu_seconds'] for name in snapshot.sources
                         if name.startswith(A15 + '/outputs/boundary-') and name.endswith('.json'))
    tolerance_cpu = sum(r['process_cpu_seconds'] for r in stages['tolerance']['receipts'])
    job_cpu = tolerance_cpu + sum(r['process_cpu_seconds'] for r in stages['switches']['receipts'])
    ledger = dict(job_cpu=job_cpu, diagnostic_child_cpu=diagnostic_cpu, declared_development_cpu=1.402,
                  total_child_and_development_cpu=job_cpu + diagnostic_cpu + 1.402)
    before = dict(ledger, job_cpu=tolerance_cpu,
                  total_child_and_development_cpu=tolerance_cpu + diagnostic_cpu + 1.402)
    check.equal(before, stages['switches']['ready']['ledger_before'], 'switch ledger_before')
    check.equal(ledger, stages['switches']['ready']['ledger_after'], 'switch ledger_after')
    require(ledger['total_child_and_development_cpu'] < 1200, 'partial CPU ledger exceeds guard')
    profile = profile_for(stages['switches'], observations, ledger)
    check.equal(profile, snapshot.source_json(A15 + '/outputs/switches/PROFILE.json'), 'PROFILE.json')
    return dict(worlds=18, tasks=126, cold_calls=306, task_outputs_recomputed=2142,
                per_stage={k: dict(calls=len(v['outputs']), predictions=len(v['rows']),
                                   exact_truth_covered_calls=v['exact_covered'], exclusions=v['exclusions'],
                                   arms=v['summaries'], historical_groups={g: {f: x for f, x in group.items() if f != 'records'}
                                                                         for g, group in v['groups'].items()})
                           for k, v in stages.items()},
                full_state_equivalences=comparisons, boundary=boundary,
                profile=profile, partial_cpu_ledger=ledger, compared_leaves=check.leaves,
                max_numeric_difference=check.max_numeric_difference,
                numeric_relative_tolerance=1e-12, numeric_absolute_tolerance=1e-12,
                classification_tolerance=EPSILON,
                limits=['Saved arithmetic, archive identity, and receipt reconciliation; no scientific solver rerun.',
                        'Saved partitions and verification receipts are checked; full exact geometry is not reproved.',
                        'Boundary witness/certificate identities and counts are checked; geometric constraints are not rebuilt.',
                        'Timing and counters are saved measurements, not remeasured performance.',
                        'CPU ledger is explicitly partial, not total research cost.',
                        '18 exposed A14 worlds are same-family replay, not new confirmation.',
                        'Manifest hashes establish internal consistency, not independent authenticity.',
                        'Free-form Markdown is not parsed or certified.'])


def verify(repository):
    snapshot = Snapshot(repository)
    require(snapshot.manifest.get('source_roots') == [A15], 'unexpected A15 source root')
    require({name.split('/')[0] for name in snapshot.sources} == {A15}, 'archive roots differ from A15')
    seal = snapshot.seal()
    historical = _history.Snapshot(repository)
    historical_seal = historical.seal()
    return dict(status='PASS', archive_root=ARCHIVE_ROOT, seal=seal, historical_seal=historical_seal,
                a15_arithmetic=numerical_audit(snapshot, historical), read_only=True,
                source_provenance_paths_opened=False, archived_research_modules_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = verify(args.repo)
    except (AuditError, _history.AuditError, OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, EOFError) as error:
        print(json.dumps(dict(status='FAIL', error=str(error)[:2000]), sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

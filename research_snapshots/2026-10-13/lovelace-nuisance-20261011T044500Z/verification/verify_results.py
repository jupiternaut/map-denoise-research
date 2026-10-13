"""Independent deterministic verification of the registered nuisance evidence.

All endpoint, trace, union, random-tape and scoring arithmetic is implemented
here rather than obtained from the runner/action. The ColoredModel image and
enclosure kernels are deliberately shared and their mathematical soundness is
an external contract, not an independently established result of this verifier.
"""
import os
import sys

sys.dont_write_bytecode = True
for _verification_thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_verification_thread_variable] = '1'

import argparse
from datetime import datetime, timezone
from fractions import Fraction as Q
import json
from pathlib import Path
import socket
import time

EXPERIMENT_ROOT = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-nuisance-20261011T044500Z')
sys.path.insert(0, str(EXPERIMENT_ROOT))
from verification.arithmetic import (DOMAIN, EPSILON, TOLERANCE, exact, encoded, indexed, pair,
                                     require, verify_trace, verify_union, verify_decision)
from verification.evidence import (ROOT, METHODS, CASES, TRUTHS, SEEDS, OFFSETS, BRANCHES,
                                   registration, regenerate, arm_contracts, contains, make_row,
                                   reconstruct_aggregates, same, load, verify_csv, finite_seconds,
                                   verify_source_lock, scientific_files, verify_strategy_boundary,
                                   branch_counts)
from nuisance_model import ColoredModel


def check_enclosures(outer, checked, observation, model, where):
    """Re-evaluate every evaluated box using the shared enclosure kernel."""
    calls, witnesses = 0, 0
    for node in outer['trace']:
        if node['status'] == 'unresolved_budget':
            continue
        low, high = checked['boxes'][node['id']]
        bounds = model.enclosure(low, high)
        require(len(bounds) == len(observation), f'{where}: enclosure dimension mismatch')
        require(all(isinstance(a, Q) and isinstance(b, Q) and 0 <= a <= b <= 1 for a, b in bounds),
                f'{where}: invalid shared-kernel interval')
        conflict = next(((i, a, b) for i, (a, b) in enumerate(bounds)
                         if a > observation[i] + EPSILON or b < observation[i] - EPSILON), None)
        if conflict is not None:
            channel, a, b = conflict
            require(node['status'] == 'excluded' and node['witness_channel'] == channel,
                    f'{where}: shared-kernel box exclusion/status differs')
            same(node['prediction_interval'], [encoded(a), encoded(b)], f'{where}: witness {node["id"]}')
            witnesses += 1
        else:
            require(node['status'] == ('retained_tolerance' if high - low <= TOLERANCE else 'split'),
                    f'{where}: nonconflicting box status differs')
        calls += 1
    require(calls == checked['evaluated'], f'{where}: evaluated-kernel call count mismatch')
    return calls, witnesses


def verify_timing(output, outers, summary):
    timing = load(output / 'TIMING.json')
    require(set(timing) == {'generation_seconds', 'observations', 'shared_cost_note'}, 'TIMING: invalid schema')
    generation = finite_seconds(timing['generation_seconds'], 'generation time')
    same(timing['shared_cost_note'], 'geometry cache is shared across models; each arm records its actually observed solve cost, including all allowed branches', 'TIMING scope')
    timings = indexed(timing['observations'], lambda t: t['observation_id'], 'TIMING')
    require(set(timings) == set(outers), 'TIMING: missing or extra observation')
    same(load(output / 'TIMING_PARTIAL.json'), timing['observations'], 'TIMING final checkpoint')
    for oid, record in timings.items():
        require(set(record) == {'observation_id', 'outer_seconds_by_method', 'selector_seconds_by_method'}, f'{oid}: timing schema')
        require(set(record['outer_seconds_by_method']) == set(METHODS) and
                set(record['selector_seconds_by_method']) == set(METHODS), f'{oid}: timing method set')
        for method in METHODS:
            measured = finite_seconds(record['outer_seconds_by_method'][method], f'{oid}/{method}: solve time')
            finite_seconds(record['selector_seconds_by_method'][method], f'{oid}/{method}: action time')
            require(measured == outers[oid]['arms'][method]['outer_seconds'], f'{oid}/{method}: inconsistent solve time')
    per_method = {}
    for method in METHODS:
        outer = sum(t['outer_seconds_by_method'][method] for t in timing['observations'])
        action = sum(t['selector_seconds_by_method'][method] for t in timing['observations'])
        per_method[method] = {'outer_seconds': outer, 'selector_seconds': action, 'total_seconds': outer + action,
                              'evaluated_boxes': sum(o['arms'][method]['outer']['evaluated_boxes'] for o in outers.values()),
                              'unresolved_budget_leaves': sum(o['arms'][method]['outer']['unresolved_budget_leaves'] for o in outers.values())}
    same(summary['per_method_timing'], per_method, 'SUMMARY per-method measured costs')
    require(summary['generation_seconds'] == generation, 'SUMMARY: inconsistent generation time')
    verify_csv(output / 'evaluation/TIMING.csv', [{'method': method, **values} for method, values in per_method.items()])
    return generation


def primary_claims(rows, unions, truths):
    result = {}
    for method in ('expanded_unanchored', 'calibrated_union'):
        values = [r for r in rows if r['method'] == method]
        coverage = sum(contains(unions[oid][method]['intervals'], exact(private['truth'])) for oid, private in truths.items())
        correct = [r for r in values if r['state_group'] == 'correct']
        harmed = [r['task_id'] for r in values if exact(r['ae_mm']) > exact(r['initial_ae_mm'])]
        require(coverage == 54 and len(correct) == 54 and
                all(r['status'] == 'KEEP' and exact(r['ae_mm']) == 0 for r in correct) and not harmed,
                f'C1: {method} violates declared coverage/conditional safety')
        negative = [r for r in values if r['case'] == 'equal_full_negative']
        require(len(negative) == 81 and all(r['status'] == 'KEEP' for r in negative),
                f'negative control: {method} moves despite depth-independent full-branch explanation')
        for oid, private in truths.items():
            if private['case'] == 'equal_full_negative':
                require(unions[oid][method]['intervals'] == [DOMAIN], f'negative control: {method} lost full depth domain')
        result[method] = {'covered_observations': coverage, 'correct_initial_kept': len(correct),
                          'harm': len(harmed), 'negative_control_small_kept': sum(r['state_group'] == 'small' for r in negative)}
    comparison = {}
    for method in ('expanded_unanchored', 'calibrated_union'):
        small = [r for r in rows if r['method'] == method and r['case'] != 'equal_full_negative' and r['state_group'] == 'small']
        require(len(small) == 360, f'C2: incorrect informative small-state denominator for {method}')
        comparison[method] = {'n': len(small), 'improved': sum(exact(r['ae_mm']) < exact(r['initial_ae_mm']) for r in small),
                              'mae_exact_mm': str(sum(exact(r['ae_mm']) for r in small) / len(small))}
    a, b = comparison['expanded_unanchored'], comparison['calibrated_union']
    c2 = b['improved'] > a['improved'] and Q(b['mae_exact_mm']) < Q(a['mae_exact_mm'])
    return {'C1': {'status': 'PASS', 'valid_contract_arms': result},
            'C2': {'status': 'SUPPORTED' if c2 else 'NEGATIVE', 'informative_small_states': comparison,
                   'negative_is_scientific_result_not_verification_failure': True}}


def verify(output):
    output = Path(output).resolve()
    require(socket.gethostname() == 'liekkas', 'target identity: expected host liekkas')
    require(ROOT.resolve() == EXPERIMENT_ROOT and output.is_relative_to(ROOT) and output != ROOT,
            'target identity: evidence must be inside the exact new experiment directory')
    start_sources = verify_source_lock(load(output / 'SOURCE_LOCK.json'))
    information_check = verify_strategy_boundary()
    same(load(output / 'PROTOCOL.json'), registration(), 'registered protocol snapshot')
    status = load(output / 'RUN_STATUS.json')
    require(set(status) == {'status', 'observations_complete', 'decisions_complete', 'elapsed_seconds'} and
            status['status'] == 'COMPLETE' and type(status['observations_complete']) is int and
            status['observations_complete'] == 54 and type(status['decisions_complete']) is int and
            status['decisions_complete'] == 2430, 'RUN_STATUS: completed registered population required')
    finite_seconds(status['elapsed_seconds'], 'complete elapsed time')
    require(status['elapsed_seconds'] <= 1800, 'RUN_STATUS: exceeded registered wall-time budget')
    observations_list = load(output / 'observations.json')
    truths_list = load(output / 'evaluation/truth.json')
    tasks_list = load(output / 'tasks.json')
    decisions = load(output / 'decisions.json')
    outers_list = load(output / 'outers.json')
    observations = indexed(observations_list, lambda o: o['observation_id'], 'observations')
    truths = indexed(truths_list, lambda t: t['observation_id'], 'truths')
    tasks = indexed(tasks_list, lambda t: t['task_id'], 'tasks')
    outers = indexed(outers_list, lambda o: o['observation_id'], 'outers')
    decision_map = indexed(decisions, lambda d: (d['task_id'], d['method']), 'decisions')
    require(len(observations) == len(truths) == len(outers) == 54 and len(tasks) == 486 and len(decision_map) == 2430,
            'population: expected 54 observations / 486 tasks / 2430 decisions')
    ids = [f'o{i:03d}' for i in range(54)]
    require(list(observations) == list(truths) == list(outers) == ids, 'population: changed observation identities/order')
    expected_tasks, expected_decisions = [], []
    i = 0
    for case, branch, colors in CASES:
        for truth in TRUTHS:
            for seed in SEEDS:
                oid = ids[i]
                public, private = regenerate(case, branch, colors, truth, seed, ColoredModel)
                same(observations[oid], {'observation_id': oid, **public}, f'{oid}: public image and sourced samples')
                same(truths[oid], {'observation_id': oid, **private}, f'{oid}: private tape and truth')
                for j, offset in enumerate(OFFSETS):
                    tid = f'{oid}-s{j}'
                    expected_tasks.append({'task_id': tid, 'observation_id': oid, 'incumbent': encoded(truth + offset)})
                    expected_decisions.extend((tid, method) for method in METHODS)
                i += 1
    same(tasks_list, expected_tasks, 'registered independent task incumbents')
    require(list(decision_map) == expected_decisions, 'decisions: missing, extra or reordered registered arm/task')
    unions, branch_checks = {}, {}
    kernel_calls, witness_count, partition_leaves = 0, 0, 0
    for oid in ids:
        entry, public = outers[oid], observations[oid]
        require(set(entry) == {'observation_id', 'arms'} and list(entry['arms']) == list(METHODS), f'{oid}: outer method schema/order')
        image = tuple(exact(v) for v in public['observation'])
        contracts = arm_contracts(public['calibration'])
        unions[oid], branch_checks[oid] = {}, {}
        for method in METHODS:
            arm = entry['arms'][method]
            where = f'{oid}/{method}'
            require(set(arm) == {'branches', 'outer', 'branch_seconds', 'outer_seconds'}, f'{where}: invalid arm schema')
            allowed, colors = contracts[method]
            require(list(arm['branches']) == list(allowed) and list(arm['branch_seconds']) == list(allowed),
                    f'{where}: altered whole-scene branch contract')
            seconds = finite_seconds(arm['outer_seconds'], where + ': total solve time')
            measured = sum(finite_seconds(value, where + ': branch time') for value in arm['branch_seconds'].values())
            require(seconds >= measured, f'{where}: union time below branch cost')
            if method == 'keep':
                require(seconds == 0, f'{where}: unexpected KEEP solver cost')
            checked_branches = {}
            for branch in allowed:
                outer = arm['branches'][branch]
                checked = verify_trace(outer, image, where + '/' + branch)
                model = ColoredModel(branch, camera_radius=Q(public['radius']), colors=colors)
                calls, witnesses = check_enclosures(outer, checked, image, model, where + '/' + branch)
                kernel_calls += calls
                witness_count += witnesses
                partition_leaves += sum(n['status'] != 'split' for n in outer['trace'])
                checked_branches[branch] = checked
            branch_checks[oid][method] = checked_branches
            unions[oid][method] = verify_union(arm['outer'], checked_branches, where, prior_only=method == 'keep')
    rows = []
    for decision in decisions:
        task = tasks[decision['task_id']]
        oid, method = task['observation_id'], decision['method']
        checked = unions[oid][method]
        verify_decision(decision, task['incumbent'], checked['hull'], f'{task["task_id"]}/{method}')
        rows.append(make_row(task, truths[oid], decision, checked))
    same(load(output / 'evaluation/ROWS.json'), rows, 'ROWS exact truth evaluation')
    verify_csv(output / 'evaluation/ROWS.csv', rows)
    coverage = []
    for oid in ids:
        private = truths[oid]
        truth = exact(private['truth'])
        for method in METHODS:
            checked, branches = unions[oid][method], branch_checks[oid][method]
            hull = checked['hull']
            coverage.append({'observation_id': oid, 'case': private['case'], 'method': method,
                             'covered': contains(checked['intervals'], truth), 'incompatible': hull is None,
                             'hull_width_mm': None if hull is None else str(hull[1] - hull[0]),
                             'retained_branches': checked['retained_branches'],
                             'branch_coverage': {name: contains(branch['intervals'], truth) for name, branch in branches.items()}})
    same(load(output / 'evaluation/COVERAGE.json'), coverage, 'COVERAGE exact whole-branch membership')
    verify_csv(output / 'evaluation/COVERAGE.csv', coverage, ('retained_branches', 'branch_coverage'))
    summary = load(output / 'evaluation/SUMMARY.json')
    fields = {'observation_configurations', 'template_count', 'task_count', 'decision_count', 'rows', 'sampling_structure',
              'sampling_note', 'hybrid_branch_counts', 'sample_counts', 'per_method_timing', 'generation_seconds',
              'elapsed_seconds', 'timing_scope'}
    require(set(summary) == fields, 'SUMMARY: invalid root schema')
    for field, value in {'observation_configurations': 54, 'template_count': 6, 'task_count': 486, 'decision_count': 2430}.items():
        same(summary[field], value, f'SUMMARY {field}')
    aggregates = reconstruct_aggregates(rows)
    require(len(aggregates) == 135, 'SUMMARY: incorrect overall/case/information × state × method denominator')
    same(summary['rows'], aggregates, 'SUMMARY independently reconstructed grouped metrics')
    verify_csv(output / 'evaluation/SUMMARY.csv', aggregates)
    same(summary['hybrid_branch_counts'], branch_counts(decisions), 'SUMMARY hybrid branches')
    same(summary['sampling_structure'], {'case_templates': 6, 'depth_settings': 3,
         'noise_replicates_per_case_depth': 3, 'shared_incumbents_per_observation': 9}, 'SUMMARY sampling structure')
    same(summary['sampling_note'], 'configurations are synthetic case/depth settings with repeated noise seeds; initial states and arms share observations; these counts are not independent real scenes', 'SUMMARY sampling note')
    same(summary['sample_counts'], {'image_configurations': 54, 'image_channels_per_observation': 432,
         'auxiliary_samples_per_observation': 3, 'auxiliary_samples_total': 162,
         'auxiliary_samples_used_by_method': {m: 162 if m.startswith('calibrated_') else 0 for m in METHODS},
         'shared_samples_note': 'both calibrated arms reuse the same samples; they are not independent extra observations'}, 'SUMMARY information accounting')
    same(summary['timing_scope'], 'observed shared-geometry-cache execution order; includes interval solves and hybrid actions, excludes a standalone cold-cache extrapolation', 'SUMMARY timing scope')
    finite_seconds(summary['elapsed_seconds'], 'SUMMARY elapsed')
    generation_seconds = verify_timing(output, outers, summary)
    require(generation_seconds <= summary['elapsed_seconds'] <= status['elapsed_seconds'], 'timing: inconsistent elapsed sequence')
    claims = primary_claims(rows, unions, truths)
    same(scientific_files(), start_sources, 'source inventory unchanged throughout verification')
    return {'status': 'PASS', 'host': socket.gethostname(), 'output': str(output),
            'observations': 54, 'tasks': 486, 'decisions': 2430, 'summary_rows': len(aggregates),
            'branch_outer_traces': 432, 'terminal_partition_leaves': partition_leaves,
            'shared_kernel_evaluated_boxes_recomputed': kernel_calls, 'all_exclusion_witnesses_recomputed': witness_count,
            'claims': claims, 'source_files_verified': len(start_sources), 'information_boundary': information_check,
            'checks': ['exact registration and complete population', 'private random tape, poses, RGB noise and three sourced samples',
                       'every branch partition/BFS budget/status/witness', 'whole-branch depth union and provenance',
                       'all frozen hybrid endpoint actions', 'exact ROWS/COVERAGE and independent grouped SUMMARY',
                       'ROWS/COVERAGE/SUMMARY/TIMING CSV-to-JSON equality', 'source-lock absolute inventory and unchanged hashes'],
            'limitations': ['Shared ColoredModel point/enclosure kernels are not an independent geometry/enclosure proof.',
                            'Recorded wall times are validated and summed, not reproduced; shared-cache timings are not cold-cache comparisons.',
                            'Conditional safety requires actual-truth coverage; wrong nominal/isolated exclusion and incompatibility are scientific results.',
                            'Channelwise enclosure discards shared-color cross-pixel correlations; wide unanchored outer/KEEP is not proof that original images contain no depth information.',
                            'Sourced appearance samples add information and may tighten computational relaxation; their necessity is not established.',
                            'Six templates and repeated depths/noise are not 54 independent real scenes; no real-MVS migration is tested.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify frozen nuisance evidence without running an experiment.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/confirmation-v1', help='completed evidence directory inside the exact experiment root')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or not output.is_dir():
        parser.error('output must be an existing evidence directory inside the exact experiment root')
    destination = output / 'verification/VERIFY.json'
    if destination.exists():
        print('REFUSE: verification/VERIFY.json already exists; evidence is never overwritten', file=sys.stderr)
        return 2
    started = time.perf_counter()
    try:
        result = verify(output)
    except Exception as error:
        result = {'status': 'FAIL', 'host': socket.gethostname(), 'output': str(output),
                  'error': f'{type(error).__name__}: {error}'}
    result['utc'] = datetime.now(timezone.utc).isoformat()
    result['verification_seconds'] = time.perf_counter() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except FileExistsError:
        print('REFUSE: concurrent verification created VERIFY.json', file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

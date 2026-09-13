"""Single-thread warm cost replay of already evaluated, unchanged V7 outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUNS = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
REFERENCE = RUNS/'reassociation-v7-development-vbwoojgf'
METHODS = (('original', 0), ('local_multistart', 6), ('reassociate', 6))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def process_memory():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        field = line.split(':', 1)[0]
        if field in ('VmRSS', 'VmHWM'):
            result[field+'_kib'] = int(line.split()[1])
    result['ru_maxrss_kib'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return result


def load_algorithm():
    path = PROJECT/'exploration_v7/algorithm/reassociation.py'
    spec = importlib.util.spec_from_file_location('v7_cost_replay_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats != 3:
        raise ValueError('frozen cost protocol uses exactly three warm repeats')
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('incorrect target host')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
        if os.environ.get(key) != '1':
            raise RuntimeError(key+' must be 1')
    started = time.perf_counter()
    memory_start = process_memory()
    manifest = json.loads((REFERENCE/'SOURCE_MANIFEST.json').read_text())
    before = {path: digest(path) for path in manifest}
    if before != manifest:
        raise RuntimeError('formal source snapshot no longer matches current source')
    inputs = json.loads((REFERENCE/'INPUT_MANIFEST.json').read_text())
    dest = Path(tempfile.mkdtemp(prefix='v7-warm-cost-', dir=RUNS))
    print(dest, flush=True)
    shutil.copyfile(__file__, dest/'benchmark_v7.py')
    save(dest/'CONFIG.json', dict(host=socket.gethostname(), formal_run=str(REFERENCE),
        repeats=3, methods=[list(m) for m in METHODS], inputs=len(inputs),
        timing='one new measured freeze per input; one unmeasured warm fit per arm; three measured post-upstream fits',
        order='cyclic method order each repeat to reduce fixed order bias',
        total_time_estimate='measured freeze + warm fit; composed, not separately timed end-to-end executions',
        memory_scope='one CPU process; /proc VmHWM and VmRSS plus resource ru_maxrss, not per-method peaks',
        no_new_geometry_experiment=True, no_new_parameters=True, no_gpu=True))
    save(dest/'SOURCE_BEFORE.json', before)
    model = load_algorithm()
    rows, freezes, warmchecks, memories = [], [], [], [dict(stage='entry_after_imports', **memory_start)]

    def check(case, method, output, info):
        reference_path = REFERENCE/'outputs'/(case+'__'+method+'.npz')
        with np.load(reference_path, allow_pickle=False) as data:
            expected = data['xyz_world'].copy()
        expected_info = json.loads(reference_path.with_suffix('.json').read_text())['info']
        same = bool(np.array_equal(output, expected))
        fingerprint = model.V5._fingerprint(xyz_world_m=output)
        fingerprint_equal = fingerprint == expected_info['output_sha256'] == info['output_sha256']
        if not same or not fingerprint_equal:
            raise AssertionError('cost replay geometry changed: '+case+' '+method)
        if info['search_work'] != expected_info['search_work']:
            raise AssertionError('cost replay search work changed')
        return dict(bitwise_equal=True, fingerprint_equal=True, output_array_sha256=fingerprint)

    for source in inputs:
        case = source['case']
        if digest(source['input']) != source['input_sha256']:
            raise RuntimeError('input modified')
        with np.load(source['input'], allow_pickle=False) as data:
            world, scans = data['xyz_world'].copy(), data['scan_id'].copy()
        tick = time.perf_counter()
        state = model.freeze(world, scans, 1.)
        freeze_seconds = time.perf_counter()-tick
        freezes.append(dict(case=case, freeze_seconds=freeze_seconds,
                            recorded_internal_freeze_seconds=state['freeze_seconds'],
                            frozen_common_sha256=state['frozen_common_sha256'], **process_memory()))
        for variant, budget in METHODS:
            output, info, _ = model.fit_frozen(state, variant=variant, budget=max(1, budget), sharing='independent')
            method = f'{variant}_b{budget}_independent'
            warmchecks.append(dict(case=case, method=method, **check(case, method, output, info)))
        for repeat in range(args.repeats):
            ordering = METHODS[repeat:] + METHODS[:repeat]
            for position, (variant, budget) in enumerate(ordering):
                tick = time.perf_counter()
                output, info, _ = model.fit_frozen(state, variant=variant, budget=max(1, budget), sharing='independent')
                elapsed = time.perf_counter()-tick
                method = f'{variant}_b{budget}_independent'
                verified = check(case, method, output, info)
                work = info['search_work']
                rows.append(dict(case=case, method=method, repeat=repeat, position=position,
                    measured_fit_seconds=elapsed, measured_freeze_seconds=freeze_seconds,
                    composed_freeze_plus_fit_seconds=freeze_seconds+elapsed,
                    search_seconds=info['search_seconds'], final_fit_seconds=info['final_fit_seconds'],
                    search_work_proxy=info['search_work_proxy'],
                    residual_pair_evaluations=work['residual_pair_evaluations'],
                    weighted_fit_row_visits=work['weighted_fit_row_visits'],
                    weighted_design_elements=work['weighted_design_elements'],
                    least_squares_calls=work['least_squares_calls'],
                    final_fit_row_visits=info['final_fit_row_visits'],
                    final_fit_design_elements=info['final_fit_design_elements'],
                    fit_rank=info['fit_rank'], fit_parameter_count=info['fit_parameter_count'],
                    compute_target=info['compute_target'], compute_target_met=info['compute_target_met'],
                    **verified, **process_memory()))
        memories.append(dict(stage=case, **process_memory()))
        print(case+' 9/9 timed, 3/3 warm outputs verified', flush=True)
    with (dest/'TIMINGS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save(dest/'FREEZE_TIMINGS.json', freezes)
    save(dest/'WARMUP_CHECKS.json', warmchecks)
    save(dest/'MEMORY_TIMELINE.json', memories)
    summaries = {}
    for variant, budget in METHODS:
        method = f'{variant}_b{budget}_independent'
        method_rows = [r for r in rows if r['method'] == method]
        case_medians = [float(np.median([r['measured_fit_seconds'] for r in method_rows if r['case'] == s['case']])) for s in inputs]
        summaries[method] = dict(timed_replays=len(method_rows),
            mean_case_median_fit_ms=float(np.mean(case_medians)*1000),
            median_case_median_fit_ms=float(np.median(case_medians)*1000),
            mean_case_median_composed_ms=float((np.mean(case_medians)+np.mean([f['freeze_seconds'] for f in freezes]))*1000),
            mean_search_work_proxy=float(np.mean([r['search_work_proxy'] for r in method_rows])),
            mean_residual_pair_evaluations=float(np.mean([r['residual_pair_evaluations'] for r in method_rows])),
            mean_weighted_fit_row_visits=float(np.mean([r['weighted_fit_row_visits'] for r in method_rows])),
            mean_final_design_elements=float(np.mean([r['final_fit_design_elements'] for r in method_rows])))
    local = summaries['local_multistart_b6_independent']
    reassoc = summaries['reassociate_b6_independent']
    after = {path: digest(path) for path in manifest}
    save(dest/'SOURCE_AFTER.json', after)
    if before != after:
        raise AssertionError('source changed during cost replay')
    memory_end = process_memory()
    summary = dict(host=socket.gethostname(), run_dir=str(dest), formal_run=str(REFERENCE),
        input_count=len(inputs), timed_output_checks=len(rows), warmup_output_checks=len(warmchecks),
        all_output_fingerprints_equal=True, source_unchanged=True,
        mean_freeze_ms=float(np.mean([f['freeze_seconds'] for f in freezes])*1000),
        median_freeze_ms=float(np.median([f['freeze_seconds'] for f in freezes])*1000),
        methods=summaries,
        reassociate_to_local_mean_fit_ratio=reassoc['mean_case_median_fit_ms']/local['mean_case_median_fit_ms'],
        reassociate_to_local_work_proxy_ratio=reassoc['mean_search_work_proxy']/local['mean_search_work_proxy'],
        memory_entry=memory_start, memory_end=memory_end,
        memory_interpretation='VmHWM is /proc current process high-water memory; ru_maxrss may retain inherited pre-exec high water and is reported separately; neither is a per-method incremental peak',
        elapsed_seconds=time.perf_counter()-started, no_new_geometric_claim=True)
    save(dest/'SUMMARY.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

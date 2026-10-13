"""Measurement-only constraints diagnosis at points inside the observed gap.

These point LPs diagnose relaxation strength; they are not a new hull method.
All old contradictory classifications retain exact certificate rechecks.
"""
from fractions import Fraction as Q
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

START_CPU = time.process_time()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
A12 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
A13 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
sys.path.insert(0, str(A13))
import search_method as method
import numpy as np
method.warm_load('certified')
method.warm_load('legacy')
old = method.LEGACY


def exact_witness_check(context, branch, depth, answer):
    if answer.status != 0:
        return False
    rows, rhs, bounds = context.cell_constraints(branch, depth, depth)
    x = tuple(Q.from_float(float(v)) for v in answer.x)
    return (all(lo <= val <= hi for val, (lo, hi) in zip(x, bounds)) and
            all(sum((val * x[j] for j, val in row.items()), Q(0)) <= b
                for row, b in zip(rows, rhs)))


def classify_old(public, oldarm, anchors, branch, depth, camera):
    rows, bounds = old.a10.build_problem(public, oldarm, anchors, depth, depth, branch, camera)
    start = time.perf_counter()
    answer = old.a10.classify(rows, bounds, deadline=time.monotonic() + 2.)
    status = answer['status']
    checked = (old.a10.certificate_check(rows, bounds, answer['certificate'])
               if status == 'CERTIFIED_INFEASIBLE' else
               old.a10.witness_check(rows, bounds, answer['witness'])
               if status == 'FEASIBLE_WITNESS' else None)
    return {'status': status, 'independent_exact_recheck': checked,
            'seconds': time.perf_counter() - start, 'classification': answer}


def main():
    if socket.gethostname().split('.')[0] != 'liekkas':
        raise RuntimeError('wrong host')
    resource.setrlimit(resource.RLIMIT_CPU, (35, 38))
    observations = {row['observation_id']: row['observation'] for row in
                    json.loads((A12 / 'data/dev/public.json').read_text())['observations']}
    prior = json.loads((ROOT / 'diagnostic_depth_gap.json').read_text())
    records = []
    for legacy in prior['records']:
        if legacy['arm'] != 'historical_accumulated' or 'not_run' in legacy:
            continue
        oid = legacy['observation_id']
        fast = next(r for r in prior['records'] if r['observation_id'] == oid and
                    r['arm'] == 'A13_certified' and r['depth_tolerance'] == .001)
        obs = observations[oid]
        public, oldarm, anchors = old.legacy_inputs(obs)
        _, cameras = old.a9.anchor.camera_intervals(anchors)
        context = method.CERT.ExactContext(obs)
        encoded = method.FREE._encode(obs, True)
        branch = next(c['branch'] for c in fast['state']['cells'])
        for side, f, h in zip(('lower', 'upper'), fast['hull'], legacy['hull']):
            if (f >= h if side == 'lower' else f <= h):
                continue
            z = Q.from_float((f + h) / 2.)
            new_areas = method.CERT.area_enclosures(branch, z, z, context.cameras)
            old_areas = old.a9.interval_area_bounds(branch, z, z, cameras)
            differences = []
            for v in range(3):
                for p in range(48):
                    for k in range(3):
                        flo, fhi = new_areas[v][p][0][k], new_areas[v][p][1][k]
                        hlo, hhi = old_areas[48*v+p][2*k:2*k+2]
                        improvement = (fhi-flo)-(hhi-hlo)
                        if improvement > Q(1, 100000000):
                            differences.append({'view': v, 'pixel': p, 'material': k,
                                                'A13': [float(flo), float(fhi)],
                                                'historical': [float(hlo), float(hhi)],
                                                'width_reduction': float(improvement)})
            floated = method.FREE._lp_for_cell(encoded, branch, float(z), float(z), .5)
            direct = classify_old(public, oldarm, anchors, branch, z, cameras)
            tiled = [classify_old(public, oldarm, anchors, branch, z, tile)
                     for tile in old.a10.camera_tiles(cameras)]
            record = {'observation_id': oid, 'branch': branch, 'side': side,
                      'depth': float(z), 'depth_exact': str(z),
                      'A13_float_status': int(floated.status),
                      'A13_exact_outer_witness_verified': exact_witness_check(context, branch, z, floated),
                      'historical_whole_camera': direct, 'historical_four_camera_tiles': tiled,
                      'area_discrepancies': sorted(differences, key=lambda x: -x['width_reduction'])}
            records.append(record)
            print(json.dumps({'observation_id': oid, 'side': side, 'depth': float(z),
                              'A13_status': int(floated.status),
                              'old_whole_status': direct['status'],
                              'old_tile_statuses': [r['status'] for r in tiled],
                              'max_area_width_reduction': differences[0] if differences else None}), flush=True)
    result = {'scope': 'public development only; probes selected from hull gaps without truth/task reads',
              'invocation': 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B /srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z/diagnostic_constraints.py',
              'records': records, 'total_cpu_seconds_including_imports': time.process_time()-START_CPU,
              'cpu_guard_seconds': [35,38]}
    with (ROOT / 'diagnostic_constraints.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({'total_cpu_seconds_including_imports': result['total_cpu_seconds_including_imports']}))


if __name__ == '__main__':
    main()

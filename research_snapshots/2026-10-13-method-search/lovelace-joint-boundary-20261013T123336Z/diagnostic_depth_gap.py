"""Bounded A14 development diagnostic; public observations only enter solvers."""
from copy import deepcopy
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time

START_CPU = time.process_time()
ROOT = Path(__file__).resolve().parent
A12 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
A13 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
PUBLIC = A12 / 'data/dev/public.json'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(A13))
import search_method as method
method.warm_load('certified')


def main():
    if socket.gethostname().split('.')[0] != 'liekkas':
        raise RuntimeError('wrong host')
    resource.setrlimit(resource.RLIMIT_CPU, (63, 65))
    data = json.loads(PUBLIC.read_text())
    if len(data['observations']) != 6:
        raise RuntimeError('wrong development set')
    records = []
    for row in data['observations']:
        public = {key: deepcopy(row['observation'][key]) for key in ('photo', 'camera', 'appearance')}
        for tolerance in (.025, .001):
            method.FREE.CELL_TOLERANCE = tolerance
            start, cpu = time.perf_counter(), time.process_time()
            state = method.prepare(public, 4., 'certified')
            record = {'observation_id': row['observation_id'], 'arm': 'A13_certified',
                      'depth_tolerance': tolerance, 'budget_seconds': 4.,
                      'hull': [state['lower'], state['upper']],
                      'width': state['upper'] - state['lower'],
                      'wall_seconds': time.perf_counter() - start,
                      'cpu_seconds': time.process_time() - cpu, 'state': state}
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != 'state'}), flush=True)
    method.warm_load('legacy')
    # Selected using public surviving branch/geometry only: one partial and one
    # informative full rear; no truths, incumbents, task groups or scores read.
    for index in (2, 4):
        row = data['observations'][index]
        remaining = 61. - (time.process_time() - START_CPU)
        if remaining < 24.5:
            records.append({'observation_id': row['observation_id'], 'arm': 'historical_accumulated',
                            'not_run': 'remaining total CPU budget < 24.5 seconds'})
            continue
        start, cpu = time.perf_counter(), time.process_time()
        public = {key: deepcopy(row['observation'][key]) for key in ('photo', 'camera', 'appearance')}
        state = method.prepare(public, 24., 'legacy')
        record = {'observation_id': row['observation_id'], 'arm': 'historical_accumulated',
                  'budget_seconds': 24., 'hull': state['hull'],
                  'width': None if state['hull'] is None else state['hull'][1] - state['hull'][0],
                  'wall_seconds': time.perf_counter() - start,
                  'cpu_seconds': time.process_time() - cpu, 'state': state}
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != 'state'}), flush=True)
    saved = json.loads((A12 / 'arms/library/DEV_OUTCOMES.json').read_text())
    history = [{key: row[key] for key in ('observation_id', 'method', 'budget_seconds',
                                          'raw_hull', 'cpu_seconds', 'wall_seconds')}
               for row in saved['trials'] if row['method'] == 'accumulated']
    result = {'scope': 'six A12 public development observations only; no truth or tasks loaded',
              'host': socket.gethostname(), 'logical_root': str(ROOT),
              'public_path': str(PUBLIC), 'public_sha256': hashlib.sha256(PUBLIC.read_bytes()).hexdigest(),
              'invocation': 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B /srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z/diagnostic_depth_gap.py',
              'records': records, 'historical_saved_development_6s': history,
              'total_cpu_seconds_including_imports': time.process_time() - START_CPU,
              'cpu_guard_seconds': [63, 65]}
    with (ROOT / 'diagnostic_depth_gap.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({'total_cpu_seconds_including_imports': result['total_cpu_seconds_including_imports']}))


if __name__ == '__main__':
    main()

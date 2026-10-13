"""Replay saved exact historical whole-camera witnesses in A13 coordinates."""
from fractions import Fraction as Q
import json
import os
from pathlib import Path
import sys
import time

START_CPU = time.process_time()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
A12 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
sys.path.insert(0, '/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
import cert_kernel

observations = {r['observation_id']: r['observation'] for r in
                json.loads((A12 / 'data/dev/public.json').read_text())['observations']}
records = []
for probe in json.loads((ROOT / 'diagnostic_constraints.json').read_text())['records']:
    context = cert_kernel.ExactContext(observations[probe['observation_id']])
    depth = Q(probe['depth_exact'])
    rows, rhs, bounds = context.cell_constraints(probe['branch'], depth, depth)
    witness = probe['historical_whole_camera']['classification']['witness']
    old = tuple(Q(x['numerator'], x['denominator']) for x in witness)
    x = old[:9] + tuple(Q(1, 100) - t for t in old[9:])
    row_margins = [b - sum((a*x[j] for j, a in row.items()), Q(0))
                   for row, b in zip(rows, rhs)]
    bound_margins = [value for val, (lo, hi) in zip(x, bounds) for value in (val-lo, hi-val)]
    record = {'observation_id': probe['observation_id'], 'side': probe['side'],
              'depth_exact': str(depth), 'transformation': 'd=d; b=1/100-t; ba=1/100-ta',
              'all_A13_necessary_rows_satisfied_exactly': min(row_margins) >= 0,
              'all_A13_bounds_satisfied_exactly': min(bound_margins) >= 0,
              'minimum_row_margin': str(min(row_margins)),
              'minimum_bound_margin': str(min(bound_margins)),
              'witness': [str(value) for value in x]}
    records.append(record)
    print(json.dumps({k:v for k,v in record.items() if k != 'witness'}))
result = {'records': records, 'total_cpu_seconds_including_imports': time.process_time()-START_CPU,
          'invocation': 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B /srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z/diagnostic_witness.py'}
with (ROOT / 'diagnostic_witness.json').open('x') as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
print(json.dumps({'total_cpu_seconds_including_imports': result['total_cpu_seconds_including_imports']}))

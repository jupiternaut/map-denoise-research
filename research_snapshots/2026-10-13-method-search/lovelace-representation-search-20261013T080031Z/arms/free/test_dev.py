"""Development-only scorer; method receives only the sanctioned inputs."""
import json
import os
import sys
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'task_bundle'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import method


def run():
    cpu_start = time.process_time()
    data = json.loads((ROOT/'task_bundle'/'dev.json').read_text())
    truths = {row['observation_id']: row['truth'] for row in data['truths']}
    for budget in (6., 24.):
        for variant, prepare in (('full', method.prepare),
                                 ('drop_material_coupling', method.prepare_ablation)):
            states = {}
            for row in data['observations']:
                oid = row['observation_id']
                states[oid] = state = prepare(row['observation'], budget)
                truth = truths[oid]
                assert state['lower'] <= truth <= state['upper'], state
                assert method.decode(state, truth) == truth
                json.dumps(state, allow_nan=False)
                print(json.dumps({'variant': variant, 'budget': budget,
                                  'observation_id': oid, 'truth': truth,
                                  'lower': state['lower'], 'upper': state['upper'],
                                  'width': state['upper']-state['lower'],
                                  'reason': state['reason'], 'stats': state['stats'],
                                  'cpu_seconds': state['cpu_seconds'],
                                  'wall_seconds': state['wall_seconds']}))
            before, after, differences = [], [], []
            for row in data['tasks']:
                truth = truths[row['observation_id']]
                prediction = method.decode(states[row['observation_id']], row['incumbent'])
                old, new = abs(row['incumbent']-truth), abs(prediction-truth)
                before.append(old)
                after.append(new)
                differences.append(new-old)
            n = len(before)
            print(json.dumps({'summary': True, 'variant': variant, 'budget': budget,
                              'n': n, 'identity_mae': sum(before)/n,
                              'mae': sum(after)/n,
                              'identity_mse': sum(x*x for x in before)/n,
                              'mse': sum(x*x for x in after)/n,
                              'improved': sum(x < -1e-9 for x in differences),
                              'unchanged': sum(abs(x) <= 1e-9 for x in differences),
                              'worsened': sum(x > 1e-9 for x in differences),
                              'cpu_seconds': sum(s['cpu_seconds'] for s in states.values()),
                              'wall_seconds': sum(s['wall_seconds'] for s in states.values())}))
    print(json.dumps({'total_development_test_cpu_seconds': time.process_time()-cpu_start}))


if __name__ == '__main__':
    run()

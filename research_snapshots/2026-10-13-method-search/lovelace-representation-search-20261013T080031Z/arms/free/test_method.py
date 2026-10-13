"""Executable local tests; no hidden or historical data is accessed."""
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

import numpy as np
import method
from forward import areas, render, BRANCHES


def synthetic_observation(depth=597.37, branch='partial'):
    offsets = np.array([.087, 0., -.043])
    colors = np.array([[.2, .65, .42], [.23, .61, .46], [.18, .63, .44]])
    photo_bias = np.array([.003, -.004, .006])
    appearance_bias = np.array([-.004, .005, -.003])
    image = render(depth, branch, offsets, colors).reshape(3, 48, 3)
    observation = {'photo': np.clip(image+photo_bias[:, None, None], 0., 1.).ravel().tolist(),
                   'camera': {'u': ((-np.array([-80., 0., 80.])-offsets-.003)/1000.).tolist(),
                              'Xf': 0., 'Zf': 1000., 'survey_bound': .01,
                              'localization_bound': .01},
                   'appearance': {'coupons': np.clip(colors+appearance_bias[:, None], 0., 1.).tolist(),
                                  'references': (.5+appearance_bias).tolist(),
                                  'bias_bound': .01, 'transfer_bound': .005,
                                  'noise_bound': .005}}
    return observation


def test_area_enclosures():
    rng = np.random.default_rng(10313)
    maximum_violation = 0.
    for _ in range(150):
        zlo, zhi = sorted(rng.uniform(540., 660., 2))
        offset_intervals = [tuple(sorted(rng.uniform(-.2, .2, 2))), (0., 0.),
                            tuple(sorted(rng.uniform(-.2, .2, 2)))]
        cameras = [(c+lo, c+hi) for c, (lo, hi) in
                   zip((-80., 0., 80.), offset_intervals)]
        z = rng.uniform(zlo, zhi)
        offsets = [rng.uniform(lo, hi) for lo, hi in offset_intervals]
        for branch in BRANCHES:
            lo, hi = method._area_enclosures((zlo, zhi), branch, cameras)
            actual = areas(z, branch, offsets)
            violation = max(float(np.max(lo-actual)), float(np.max(actual-hi)))
            maximum_violation = max(maximum_violation, violation)
            assert violation < 2e-10, (branch, violation)
    return maximum_violation


def run():
    started = time.process_time()
    print(json.dumps({'test': 'area_enclosures', 'maximum_violation': test_area_enclosures()}))
    for branch in BRANCHES:
        observation = synthetic_observation(branch=branch)
        state = method.prepare(observation, 1.5)
        assert state['lower'] <= 597.37 <= state['upper'], state
        assert method.decode(state, 597.37) == 597.37
        assert abs(method.decode(state, 555.)-597.37) <= abs(555.-597.37)
        assert abs(method.decode(state, 655.)-597.37) <= abs(655.-597.37)
        json.dumps(state, allow_nan=False)
        print(json.dumps({'test': 'synthetic', 'branch': branch,
                          'lower': state['lower'], 'upper': state['upper'],
                          'reason': state['reason'], 'stats': state['stats'],
                          'cpu_seconds': state['cpu_seconds'],
                          'wall_seconds': state['wall_seconds']}))
    observation = synthetic_observation()
    state = method.prepare(observation, 0.)
    assert (state['lower'], state['upper']) == (540., 660.)
    assert method.decode(state, 611.238) == 611.238
    white = synthetic_observation()
    white['photo'] = [1.]*432
    white['appearance']['coupons'] = [1.]*9
    white['appearance']['references'] = [.5]*3
    state = method.prepare(white, 1.5)
    assert (state['lower'], state['upper']) == (540., 660.)
    assert method.decode(state, 612.987) == 612.987
    print(json.dumps({'test': 'zero_budget_and_unidentifiable', 'passed': True,
                      'white_cpu_seconds': state['cpu_seconds']}))
    print(json.dumps({'total_test_cpu_seconds': time.process_time()-started}))


if __name__ == '__main__':
    run()

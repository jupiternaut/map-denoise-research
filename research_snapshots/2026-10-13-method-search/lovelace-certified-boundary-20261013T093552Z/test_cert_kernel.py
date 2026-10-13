"""A13 bounded developer tests. Only frozen A12 development data are read."""
import copy
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import random
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import cert_kernel as kernel

A12 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')


def exact(value):
    return Q(value['numerator'], value['denominator']) if isinstance(value, dict) else Q(value)


def direct_area(rect, pixel):
    width = min(rect[1], pixel[1])-max(rect[0], pixel[0])
    height = min(rect[3], pixel[3])-max(rect[2], pixel[2])
    return max(Q(0), width)*max(Q(0), height)*750


def direct_geometry(depth, branch, cameras):
    answer = []
    for camera in cameras:
        def project(rect, z):
            return ((Q(rect[0])-camera)/z, (Q(rect[1])-camera)/z,
                    Q(rect[2])/z, Q(rect[3])/z)
        left, right, front = (project(rect, depth) for rect in (kernel.LEFT, kernel.RIGHT, kernel.FRONT))
        rear = project(kernel.REARS[branch], Q(720)) if branch != 'none' else None
        overlap = (max(rear[0], front[0]), min(rear[1], front[1]),
                   max(rear[2], front[2]), min(rear[3], front[3])) if rear is not None else None
        answer.append(tuple((direct_area(left, pixel), direct_area(right, pixel),
                             direct_area(rear, pixel)-direct_area(overlap, pixel) if rear else Q(0))
                            for pixel in kernel.PIXELS))
    return answer


def assert_witness(context, branch, lo, hi, witness):
    assert context.error is None, context.error
    rows, rhs, bounds = context.cell_constraints(branch, lo, hi)
    assert all(a <= x <= b for x, (a, b) in zip(witness, bounds))
    for index, (row, upper) in enumerate(zip(rows, rhs)):
        actual = sum((coefficient*witness[j] for j, coefficient in row.items()), Q(0))
        assert actual <= upper, (index, actual-upper)


def test_rounding_and_signs():
    for source in (Q(1, 3), Q(7, 1000), Q(0), Q(1), Q(-37, 101), Q(8917, 99991)):
        lo, hi = kernel.rounded_interval(float(source))
        assert lo <= source <= hi
    assert kernel.PIXELS[0] == (Q(-1, 5), Q(-3, 20), Q(-2, 25), Q(-4, 75))
    assert kernel.PIXELS[-1] == (Q(3, 20), Q(1, 5), Q(4, 75), Q(2, 25))
    bounds = [(Q(0), Q(1))]*15
    assert kernel.box_margin(({0: Q(1)},), (Q(-1),), bounds, ((0, Q(1)),)) == 1
    assert kernel.box_margin(({0: Q(-1)},), (Q(-2),), bounds, ((0, Q(1)),)) == 1
    assert kernel.box_margin(({0: Q(1)},), (Q(1),), bounds, ((0, Q(1)),)) == -1
    try:
        kernel.box_margin(({0: Q(1)},), (Q(-1),), bounds, ((0, Q(-1)),))
        raise AssertionError('negative multiplier accepted')
    except ValueError:
        pass


def test_enclosures():
    rng = random.Random(1313)
    samples = 0
    for _ in range(20):
        lo, hi = sorted((Q(rng.randint(540000, 660000), 1000),
                         Q(rng.randint(540000, 660000), 1000)))
        intervals = tuple((Q(c)-Q(rng.randint(0, 200), 1000),
                           Q(c)+Q(rng.randint(0, 200), 1000)) for c in (-80, 0, 80))
        for branch in kernel.BRANCHES:
            areas = kernel.area_enclosures(branch, lo, hi, intervals)
            for ratio in (Q(0), Q(1, 3), Q(1)):
                z = lo+(hi-lo)*ratio
                cameras = tuple(a+(b-a)*(1-ratio) for a, b in intervals)
                actual = direct_geometry(z, branch, cameras)
                for v in range(3):
                    for p in range(48):
                        amin, amax = areas[v][p]
                        assert all(a <= x <= b for a, x, b in zip(amin, actual[v][p], amax))
                samples += 1
    return samples


def synthetic_saturated():
    depth, branch = Q(600), 'full'
    cameras = (Q(-80), Q(0), Q(80))
    # Black surface, negative photo bias, independently positive appearance bias.
    b, ba = Q(-1, 100), Q(1, 100)
    areas = direct_geometry(depth, branch, cameras)
    photo = []
    for view in areas:
        for pixel in view:
            observed = max(Q(0), min(Q(1), 1-sum(pixel)+b))
            photo.extend([float(observed)]*3)
    observation = {'photo': photo,
                   'camera': {'u': [.08, 0., -.08], 'Xf': 0., 'Zf': 1000.,
                              'survey_bound': .01, 'localization_bound': .01},
                   'appearance': {'coupons': [.01]*9, 'references': [.51]*3,
                                  'bias_bound': .01, 'transfer_bound': .005, 'noise_bound': .005}}
    assert 0. in photo
    witness = [Q(1)]*9+[b]*3+[ba]*3
    assert_witness(kernel.ExactContext(observation), branch, depth, depth, witness)
    return observation


def run():
    start = time.process_time()
    test_rounding_and_signs()
    samples = test_enclosures()
    saturated = synthetic_saturated()
    zero = kernel.ExactContext(saturated).check_cell('full', 590., 610., time.perf_counter())
    assert not zero['verified'] and zero['reason'] == 'deadline'
    white = copy.deepcopy(saturated)
    white['photo'] = [1.]*432
    white['appearance']['coupons'] = [1.]*9
    white['appearance']['references'] = [.5]*3
    white_context = kernel.ExactContext(white)
    for branch in kernel.BRANCHES:
        assert_witness(white_context, branch, Q(540), Q(660), [Q(0)]*15)
        result = white_context.check_cell(branch, 540., 660., time.perf_counter()+1.)
        assert not result['verified'], result
    facts = json.loads((A12/'data/dev/private/truth.json').read_text())
    observations = json.loads((A12/'data/dev/public.json').read_text())['observations']
    by_id = {row['observation_id']: row['observation'] for row in observations}
    feasible_results, impossible_results = [], []
    for fact in facts:
        depth = exact(fact['truth'])
        fdepth = float(depth)
        lo, hi = math.nextafter(fdepth, -math.inf), math.nextafter(fdepth, math.inf)
        witness = [1-exact(c) for c in fact['view_colors']]
        witness += list(map(exact, fact['actual_view_bias']))
        witness += [Q(1, 100)-exact(t) for t in fact['anchor_tape']['appearance_t']]
        assert witness[9:12] != witness[12:15], 'fixture must distinguish b and ba'
        context = kernel.ExactContext(by_id[fact['observation_id']])
        assert_witness(context, fact['rear_branch'], Q.from_float(lo), Q.from_float(hi), witness)
        for camera, offset, center in zip(context.cameras, fact['actual_offsets'], (-80, 0, 80)):
            assert camera[0] <= center+exact(offset) <= camera[1]
        result = context.check_cell(fact['rear_branch'], lo, hi, time.perf_counter()+1.)
        assert not result['verified'], result
        feasible_results.append({'observation_id': fact['observation_id'], **result})
        if fact['stratum'] == 'contrast':
            farlo, farhi = (540., 541.) if fdepth > 600. else (659., 660.)
            result = context.check_cell(fact['rear_branch'], farlo, farhi, time.perf_counter()+1.)
            assert result['verified'], result
            certificate = result['certificate']
            rows, rhs, bounds = context.cell_constraints(fact['rear_branch'], Q(farlo), Q(farhi))
            weights = tuple((i, Q(q)) for i, q in certificate['multipliers'])
            assert kernel.box_margin(rows, rhs, bounds, weights) == Q(certificate['margin']) > 0
            json.dumps(result, allow_nan=False)
            impossible_results.append({'observation_id': fact['observation_id'],
                                       'support': len(weights), 'wall_seconds': result['wall_seconds']})
    invalid = copy.deepcopy(white)
    invalid['camera']['Zf'] = 999.
    assert not kernel.ExactContext(invalid).check_cell('none', 540., 660., time.perf_counter()+1.)['verified']
    print(json.dumps({'passed': True, 'exact_enclosure_samples': samples,
                      'feasible_dev_cells': len(feasible_results), 'far_cell_certificates': impossible_results,
                      'saturated_photo_and_white_checked': True,
                      'independent_b_and_ba_checked': True, 'zero_budget_checked': True,
                      'total_cpu_seconds': time.process_time()-start}, indent=2))


if __name__ == '__main__':
    run()

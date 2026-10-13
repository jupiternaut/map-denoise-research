"""Bounded A14 tests using only six frozen A12 development worlds."""
from copy import deepcopy
from fractions import Fraction as Q
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import joint_kernel as kernel
import joint_verify as verify

A12 = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z')
sys.path.insert(0, str(A12/'task_bundle'))
_spec = importlib.util.spec_from_file_location('_joint_test_frozen_free', A12/'arms/free/method.py')
free = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(free)


def exact(value):
    return Q(value['numerator'], value['denominator'])


def test_cover():
    for cameras in (((Q(-81), Q(-79)), (Q(0), Q(0)), (Q(79), Q(81))),
                    ((Q(-80), Q(-80)), (Q(0), Q(0)), (Q(80), Q(80)))):
        tiles = kernel.camera_tiles(cameras)
        assert len(tiles) == 4
        for ratio_left in (Q(0), Q(1, 3), Q(1, 2), Q(1)):
            for ratio_right in (Q(0), Q(1, 3), Q(1, 2), Q(1)):
                point = (cameras[0][0]+ratio_left*(cameras[0][1]-cameras[0][0]),
                         Q(0), cameras[2][0]+ratio_right*(cameras[2][1]-cameras[2][0]))
                assert any(all(a <= x <= b for x, (a, b) in zip(point, tile)) for tile in tiles)


def witness_satisfies(rows, rhs, bounds, values):
    assert all(a <= x <= b for x, (a, b) in zip(values, bounds))
    for row, upper in zip(rows, rhs):
        assert sum((a*values[j] for j, a in row.items()), Q(0)) <= upper


def run():
    started, cpu = time.perf_counter(), time.process_time()
    test_cover()
    facts = json.loads((A12/'data/dev/private/truth.json').read_text())
    public = json.loads((A12/'data/dev/public.json').read_text())['observations']
    observations = {row['observation_id']: row['observation'] for row in public}
    certified, true_cells, monotonic_checks, tamper_rejections = [], 0, 0, 0
    for fact in facts:
        observation = observations[fact['observation_id']]
        context = kernel.JointContext(observation)
        assert context.error is None
        truth = exact(fact['truth'])
        lo = math.nextafter(float(truth), -math.inf)
        hi = math.nextafter(float(truth), math.inf)
        witness = [1-exact(c) for c in fact['view_colors']]
        witness += [exact(x) for x in fact['actual_view_bias']]
        witness += [Q(1, 100)-exact(x) for x in fact['anchor_tape']['appearance_t']]
        cameras = tuple(Q(c)+exact(e) for c, e in zip((-80, 0, 80), fact['actual_offsets']))
        containing = [i for i, tile in enumerate(context.tiles)
                      if all(a <= x <= b for x, (a, b) in zip(cameras, tile))]
        assert containing
        for i in containing:
            witness_satisfies(*context.contexts[i].cell_constraints(fact['rear_branch'], Q(lo), Q(hi)), witness)
        result = context.check_cell(fact['rear_branch'], lo, hi, time.perf_counter()+1.)
        assert not result['verified'], result
        true_cells += 1
        whole_rows, whole_rhs, _box = context.whole.cell_constraints(fact['rear_branch'], Q(lo), Q(hi))
        for tile_context in context.contexts:
            rows, rhs, _bounds = tile_context.cell_constraints(fact['rear_branch'], Q(lo), Q(hi))
            assert rhs == whole_rhs and len(rows) == len(whole_rows)
            for row, whole in zip(rows, whole_rows):
                for j in range(9):
                    assert row.get(j, Q(0)) >= whole.get(j, Q(0))
                for j in range(9, 15):
                    assert row.get(j, Q(0)) == whole.get(j, Q(0))
                monotonic_checks += 1
        if fact['stratum'] == 'contrast':
            farlo, farhi = (540., 541.) if float(truth) > 600 else (659., 660.)
            proposal = context.propose_cell(free._encode(observation, True), fact['rear_branch'],
                                            farlo, farhi, 1., free._lp_for_cell)
            assert proposal['status'] == 2 and proposal['tile_statuses'] == [2]*4, proposal
            result = context.check_cell(fact['rear_branch'], farlo, farhi, time.perf_counter()+1.)
            assert result['verified'], result
            cert = result['certificate']
            checked = verify.verify_certificate(observation, cert)
            assert checked['verified'], checked
            assert verify.old.verify_certificate(observation, cert['tiles'][0])['valid'] is False
            corruptions = []
            missing = deepcopy(cert); missing['tiles'].pop(); corruptions.append(missing)
            duplicate = deepcopy(cert); duplicate['tiles'][1] = duplicate['tiles'][0]; corruptions.append(duplicate)
            camera = deepcopy(cert); camera['tiles'][0]['camera_box'][0][0] = '0/1'; corruptions.append(camera)
            margin = deepcopy(cert); margin['tiles'][0]['margin'] = '0/1'; corruptions.append(margin)
            negative = deepcopy(cert); negative['tiles'][0]['multipliers'][0][1] = '-1/1'; corruptions.append(negative)
            depth = deepcopy(cert); depth['tiles'][0]['lo'] = '555/1'; corruptions.append(depth)
            for bad in corruptions:
                assert not verify.verify_certificate(observation, bad)['verified']
                tamper_rejections += 1
            json.dumps(cert, allow_nan=False)
            certified.append({'branch': fact['rear_branch'], 'four_tile_wall_seconds': result['wall_seconds'],
                              'proposal_wall_seconds': proposal['wall_seconds'],
                              'supports': [len(c['multipliers']) for c in cert['tiles']]})
    observation = public[0]['observation']
    context = kernel.JointContext(observation)
    assert not context.check_cell('none', 540., 541., time.perf_counter())['verified']
    encoded = free._encode(observation, True)
    calls = []
    def one_retained(args, branch, lo, hi, budget):
        calls.append(args[-1])
        return SimpleNamespace(status=0)
    outcome = context.propose_cell(encoded, 'none', 540., 541., 1., one_retained)
    assert outcome['status'] == 0 and len(calls) == 1
    original = context.contexts[2].check_cell
    context.contexts[2].check_cell = lambda *args: {'verified': False, 'certificate': None, 'reason': 'forced_unknown',
             'geometry_seconds': 0., 'certificate_search_seconds': 0., 'exact_check_seconds': 0.}
    truth = exact(facts[0]['truth'])
    farlo, farhi = (540., 541.) if float(truth) > 600 else (659., 660.)
    outcome = context.check_cell(facts[0]['rear_branch'], farlo, farhi, time.perf_counter()+1.)
    assert not outcome['verified'] and outcome['certificate'] is None
    context.contexts[2].check_cell = original
    print(json.dumps({'passed': True, 'true_dev_cells_retained': true_cells,
                      'row_monotonicity_checks': monotonic_checks,
                      'four_tile_certificates': certified, 'tamper_rejections': tamper_rejections,
                      'zero_deadline_and_partial_cover_retain': True,
                      'test_cpu_seconds': time.process_time()-cpu,
                      'test_wall_seconds': time.perf_counter()-started}, indent=2))


if __name__ == '__main__':
    run()

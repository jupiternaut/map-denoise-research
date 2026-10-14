"""Bounded A15 unit and real-call-path smoke tests; no confirmation labels."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import reuse_kernel as kernel
import reuse_search as search

DEV = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z/data/dev/public.json')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def structural(state):
    return {key: state[key] for key in ('lower', 'upper', 'cells', 'reason')}


class ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = json.loads(DEV.read_text())['observations'][0]['observation']
        cls.verify = load('_a15_test_frozen_verifier', kernel.A14_ROOT/'joint_verify.py')
        search.warm_load()

    def test_verified_parent_child_and_original_verifier(self):
        context = kernel.JointContext(self.obs, inherit=True)
        parent = context.check_cell('none', 540., 541., time.perf_counter()+3.)
        self.assertTrue(parent['verified'], parent)
        self.assertTrue(self.verify.verify_certificate(self.obs, parent['certificate'])['verified'])
        # A current-child proof must be rechecked without invoking certificate LP.
        with patch.object(kernel.base, 'linprog', side_effect=AssertionError('unexpected new certificate LP')):
            child = context.check_cell('none', 540.25, 540.5, time.perf_counter()+3.)
        self.assertTrue(child['verified'], child)
        self.assertTrue(self.verify.verify_certificate(self.obs, child['certificate'])['verified'])
        counts = context.counters()['counts']
        self.assertEqual(counts['hits'], 4)
        self.assertEqual(counts['strict_descendant_hits'], 4)
        self.assertEqual(counts['repeated_cell_hits'], 0)
        for parent_tile, child_tile in zip(parent['certificate']['tiles'], child['certificate']['tiles']):
            self.assertNotEqual(parent_tile['lo'], child_tile['lo'])
            self.assertEqual(parent_tile['multipliers'], child_tile['multipliers'])

    def test_invalid_parent_falls_back_to_original_checker(self):
        context = kernel.JointContext(self.obs, inherit=True)
        tile = context.contexts[0]
        self.assertTrue(tile.check_cell('none', 540., 541., time.perf_counter()+3.)['verified'])
        tile._verified_parents[0][1]['multipliers'][0][1] = '-1/1'
        calls = tile.reuse_counts['certificate_lp_calls']
        child = tile.check_cell('none', 540.25, 540.5, time.perf_counter()+3.)
        self.assertTrue(child['verified'], child)
        self.assertEqual(tile.reuse_counts['hits'], 0)
        self.assertGreater(tile.reuse_counts['invalid'], 0)
        self.assertEqual(tile.reuse_counts['certificate_lp_calls'], calls+1)

    def test_wrong_scope_branch_and_containment_cannot_hit(self):
        for alteration in ('scope', 'branch', 'outside'):
            with self.subTest(alteration=alteration):
                tile = kernel.JointContext(self.obs, inherit=True).contexts[0]
                self.assertTrue(tile.check_cell('none', 540., 541., time.perf_counter()+3.)['verified'])
                scope, cert = tile._verified_parents[0]
                if alteration == 'scope':
                    tile._verified_parents[0] = (('different-observation', scope[1], scope[2]), cert)
                elif alteration == 'branch':
                    cert['branch'] = 'partial'
                lo, hi = (541., 542.) if alteration == 'outside' else (540.2, 540.3)
                tile.check_cell('none', lo, hi, time.perf_counter()+3.)
                self.assertEqual(tile.reuse_counts['eligible_parents'], 0)
                self.assertEqual(tile.reuse_counts['hits'], 0)

    def test_deadline_retains_unknown(self):
        context = kernel.JointContext(self.obs, inherit=True)
        self.assertTrue(context.check_cell('none', 540., 541., time.perf_counter()+3.)['verified'])
        child = context.check_cell('none', 540.2, 540.3, time.perf_counter()-1.)
        self.assertFalse(child['verified'])
        self.assertIsNone(child['certificate'])
        state = search.prepare(self.obs, 0., hot=True, inherit=True)
        self.assertEqual((state['lower'], state['upper']), (540., 660.))
        self.assertFalse(state['exclusions'])

    def _fake_path(self, hot, unknown_on_third=False):
        tiles = kernel.JointContext(self.obs).tiles
        tile_map = {tuple(tuple(float(x) for x in pair) for pair in tile): i for i, tile in enumerate(tiles)}
        calls, statuses, cell_number = [], [], [0]

        def original(encoded, branch, lo, hi, limit):
            if encoded[-1] == 'whole-marker':
                return SimpleNamespace(status=0)
            tile_id = tile_map[tuple(tuple(x) for x in encoded[-1])]
            calls.append(tile_id)
            return SimpleNamespace(status=(1 if unknown_on_third and cell_number[0] == 2 else 0)
                                   if tile_id == 2 else 2)

        def fake_prepare(_obs, _budget):
            for i in range(3):
                cell_number[0] = i
                reply = search.FREE._lp_for_cell((None, None, None, None, None, 'whole-marker'),
                                                'none', 550.+i, 551.+i, .2)
                statuses.append(reply.status)
            return {'lower': 540., 'upper': 660., 'cells': [], 'reason': 'test'}

        with patch.object(search.FREE, '_lp_for_cell', original), patch.object(search.FREE, 'prepare', fake_prepare):
            state = search.prepare(self.obs, 3., hot=hot, inherit=False)
        return calls, statuses, state

    def test_actual_classify_path_hot_order_and_unknown_semantics(self):
        cold, cold_status, cold_state = self._fake_path(False)
        hot, hot_status, hot_state = self._fake_path(True, unknown_on_third=True)
        self.assertEqual(cold, [0, 1, 2]*3)
        self.assertEqual(hot, [0, 1, 2, 2, 2])
        # An unknown tile returns the original feasible whole answer, as A14 did.
        self.assertEqual(cold_status, [0, 0, 0])
        self.assertEqual(hot_status, [0, 0, 0])
        self.assertEqual(cold_state['inner_counts']['tile_float_lp'], 9)
        self.assertEqual(hot_state['inner_counts']['tile_float_lp'], 5)
        self.assertEqual(hot_state['inner_counts']['hot_hint_unknown_returns'], 1)

    def test_no_switch_real_search_matches_frozen_a14(self):
        # Original source loaded by exact path; supply its kernel explicitly.
        search.OLD.JOINT = kernel.frozen
        frozen_state = search.OLD.prepare(deepcopy(self.obs), 6., 'joint')
        actual = search.prepare(self.obs, 6., tolerance=.025, hot=False, inherit=False)
        self.assertEqual(structural(actual), structural(frozen_state))
        self.assertEqual([x['certificate'] for x in actual['exclusions']],
                         [x['certificate'] for x in frozen_state['exclusions']])
        self.assertEqual(actual['tile_reuse']['counts']['hits'], 0)

    def test_inheritance_no_op_under_unchanged_tree_lifecycle(self):
        off = search.prepare(self.obs, 6., tolerance=.001, hot=False, inherit=False)
        on = search.prepare(self.obs, 6., tolerance=.001, hot=False, inherit=True)
        self.assertEqual(structural(off), structural(on))
        self.assertEqual(on['tile_reuse']['counts']['eligible_parents'], 0)
        self.assertEqual(on['tile_reuse']['counts']['hits'], 0)
        self.assertEqual(off['inner_counts'], on['inner_counts'])


if __name__ == '__main__':
    started, cpu = time.perf_counter(), time.process_time()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReuseTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun,
                      'test_wall_seconds': time.perf_counter()-started,
                      'test_cpu_seconds': time.process_time()-cpu}, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)

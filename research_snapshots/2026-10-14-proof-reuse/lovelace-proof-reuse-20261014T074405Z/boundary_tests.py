"""Unit tests for frontier extraction and exact witness acceptance."""
from fractions import Fraction as Q
from copy import deepcopy
import json
import time
import unittest
import boundary_diagnostic as bd


class BoundaryTests(unittest.TestCase):
    def test_all_branches_tying_hull_selected_and_duplicates_removed(self):
        state = {'lower': 540., 'upper': 660., 'cells': [
            {'branch': 'none', 'lower': 540., 'upper': 550., 'terminal': True},
            {'branch': 'partial', 'lower': 540., 'upper': 560., 'terminal': False},
            {'branch': 'full', 'lower': 650., 'upper': 660., 'terminal': True},
            {'branch': 'none', 'lower': 570., 'upper': 580., 'terminal': True},
            {'branch': 'full', 'lower': 650., 'upper': 660., 'terminal': True}]}
        result = bd.boundary_cells(state)
        self.assertEqual(len(result), 3)
        self.assertEqual({r['branch'] for r in result}, {'none', 'partial', 'full'})
        self.assertEqual(result[1]['sides'], ['lower'])

    def test_cell_touching_both_sides_once(self):
        state = {'lower': 540., 'upper': 660., 'cells': [
            {'branch': 'full', 'lower': 540., 'upper': 660., 'checked': False, 'terminal': False}]}
        result = bd.boundary_cells(state)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['sides'], ['lower', 'upper'])

    def test_inconsistent_saved_hull_rejected(self):
        with self.assertRaises(ValueError):
            bd.boundary_cells({'lower': 541., 'upper': 660., 'cells': [
                {'branch': 'none', 'lower': 540., 'upper': 660.}]})

    def test_exact_not_tolerance_acceptance(self):
        rows, rhs, bounds = [{0: Q(1)}], [Q(1, 3)], [(Q(0), Q(1))]
        self.assertTrue(bd.exact_witness_check(rows, rhs, bounds, (Q(1, 3),)))
        self.assertFalse(bd.exact_witness_check(rows, rhs, bounds, (Q(1, 3)+Q(1, 10**30),)))
        self.assertFalse(bd.exact_witness_check(rows, rhs, bounds, (1/3,)))

    def test_exact_basis_recovers_thin_feasible_vertex(self):
        rows = [{0: Q(1)}, {0: Q(-1)}, {1: Q(1)}, {1: Q(-1)}]
        rhs = [Q(1, 3), -Q(1, 3), Q(2, 7), -Q(2, 7)]
        bounds = [(Q(0), Q(1))]*2
        candidate = bd.rational_basis_candidate(rows, rhs, bounds, [1/3, 2/7], time.perf_counter()+1)
        self.assertEqual(candidate, (Q(1, 3), Q(2, 7)))
        self.assertTrue(bd.exact_witness_check(rows, rhs, bounds, candidate))

    def test_full_four_required_no_earlybreak(self):
        with self.assertRaises(ValueError):
            bd.classify_cell([{'tile_id': 0, 'status': 'exact_necessary_lp_feasible'}])
        tiles = [{'tile_id': i, 'status': 'exact_infeasible'} for i in range(4)]
        self.assertEqual(bd.classify_cell(tiles), 'all_four_exactly_infeasible_offline')
        tiles[3]['status'] = 'unknown'
        self.assertEqual(bd.classify_cell(tiles), 'unresolved_exact_status')
        tiles[1]['status'] = 'exact_necessary_lp_feasible'
        self.assertEqual(bd.classify_cell(tiles), 'necessary_relaxation_retains_at_least_one_tile')

    def test_saved_exact_tile_proof_independent_recheck_and_tamper(self):
        # Frozen public observation + historical proof, no solver invocation.
        bd.load_kernels()
        saved = json.loads((bd.A14/'outputs/formal-v1/jobs/002.result.json').read_text())
        observations = {r['observation_id']: r['observation'] for r in
                        json.loads(bd.PUBLIC.read_text())['observations']}
        wrapper = next(r for r in saved['state']['exclusions'] if r['mode'] == 'tiles')
        proof = wrapper['certificate']['tiles'][0]
        cell = {'branch': wrapper['branch'], 'lo_exact': proof['lo'], 'hi_exact': proof['hi']}
        answer = {'status': 'exact_infeasible', 'certificate': proof,
                  'camera_box': proof['camera_box'], 'observation_sha256': proof['observation_sha256']}
        observation = observations[saved['observation_id']]
        self.assertTrue(bd.independent_verify(observation, cell, 0, answer)['verified'])
        altered = deepcopy(answer)
        altered['certificate']['margin'] = '0/1'
        self.assertFalse(bd.independent_verify(observation, cell, 0, altered)['verified'])
        self.assertFalse(bd.independent_verify(observation, cell, 1, answer)['verified'])


if __name__ == '__main__':
    unittest.main()

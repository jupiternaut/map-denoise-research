"""Handwritten A10 checker tests; no solver, saved formal data, or private reads."""
from copy import deepcopy
from fractions import Fraction as Q
import subprocess
import unittest

import verify_tightening as v


def fixture(impossible_camera=False):
    e = v.enc
    public = {'observation_id': 'manual-public', 'radius': e(Q(1, 5)),
              'baseline': e(80), 'clipped': True, 'observation': [e(1)] * 432}
    frozen = {'appearance_model': 'bounded', 'drift_bound': e(Q(1, 10)),
              'calibration_intervals': [[e(0), e(Q(1, 50))] for _ in range(3)]}
    offsets, survey, noise = (Q(1, 10), Q(0), -Q(1, 10)), Q(1, 200), (Q(1, 200), -Q(1, 200), Q(0))
    anchor = {'observation_id': 'manual-public', 'camera': {
        'u': [e((-c - z - survey - n) / 1000) for c, z, n in zip(v._a9.CENTERS, offsets, noise)],
        'Xf': e(0), 'Zf': e(1000), 'survey_bound': e(Q(1, 100)), 'localization_bound': e(Q(1, 100))},
        'appearance': {'coupons': [e(Q(99, 100))] * 9, 'references': [e(Q(49, 100))] * 3,
                       'bias_bound': e(Q(1, 100)), 'transfer_bound': e(Q(1, 200)), 'noise_bound': e(Q(1, 200))}}
    if impossible_camera:
        anchor['camera']['u'][1] = e(1)
    point = [e(0)] * 12 + [e(Q(1, 50))] * 3
    return public, frozen, anchor, point


def classification(rows, bounds, status='UNKNOWN', point=None):
    """Write a rational one-row proof directly, without invoking an LP."""
    singles = v._proof.single_row_contradictions(rows, bounds)
    cert = None
    if status == 'CERTIFIED_INFEASIBLE':
        index = singles[0]
        normalized, box = v._proof.normalized_problem(rows, bounds)
        coefficients, bound = normalized[index]
        minimum = v._proof.minimum_over_box(coefficients, box)
        cert = {'sparse_weights': [{'row': index, 'weight': v.enc(1)}],
                'aggregate_coefficients': list(map(v.enc, coefficients)),
                'weighted_bound': v.enc(bound), 'box_min': v.enc(minimum), 'margin': v.enc(minimum - bound)}
    return {'status': status, 'certificate': cert, 'witness': point if status == 'FEASIBLE_WITNESS' else None,
            'lp_status': None, 'seconds': 0.0, 'lp_seconds': 0.0, 'single_row_contradiction': bool(singles),
            'single_row_contradiction_rows': singles, 'row_count': len(rows), 'dimension': len(bounds),
            'reason': 'handwritten exact checker fixture'}


def node(ident, parent, root, lo, hi, status, side=None, checks=None, children=None):
    return {'id': ident, 'parent': parent, 'root': root, 'lo': v.enc(lo), 'hi': v.enc(hi),
            'status': status, 'side': side, 'checks': [] if checks is None else checks, 'children': children}


def checks(prepared, branch, arm, lo, hi, status='UNKNOWN', point=None):
    result = []
    for i, tile in enumerate(v.camera_tiles(prepared[3], arm)):
        rows, bounds = v.tile_problem(prepared, branch, lo, hi, tile)
        result.append({'tile_id': i, 'camera_box': v.encoded_camera_box(tile),
                       'classification': classification(rows, bounds, status, point), 'reason': 'classified'})
    return result


def record(prepared, source, trace, schedule, termination, closed=None, arm='camera_tiles', branch='partial'):
    source = v.merged_intervals(source)
    retained = [n for n in trace if n['status'].startswith('retained_')]
    outer = v.merged_intervals((n['lo'], n['hi']) for n in retained)
    return {'branch': branch, 'mode': arm, 'domain': list(map(v.enc, v._a9.DOMAIN)),
            'source_outer_intervals': v._encoded_intervals(source),
            'camera_domain': v.encoded_camera_box(prepared[3]),
            'tiles': [v.encoded_camera_box(t) for t in v.camera_tiles(prepared[3], arm)],
            'roots': list(range(len(source))), 'trace': trace, 'retained_ids': [n['id'] for n in retained],
            'outer_intervals': v._encoded_intervals(outer), 'hull': v._hull(outer),
            'status': 'OUTER_RETAINED' if outer else 'OUTER_EMPTY', 'schedule': schedule,
            'closed_sides': {'left': False, 'right': False} if closed is None else closed,
            'classification_calls': sum(c['classification'] is not None for n in trace for c in n['checks']),
            'processed_nodes': sum(s['action'] == 'process' for s in schedule),
            'pruned_nodes': sum(n['status'] == 'excluded_certificate' for n in trace),
            'split_nodes': sum(n['status'] == 'split' for n in trace), 'max_classifications': 128,
            'min_width': v.enc(v.MIN_WIDTH), 'branch_seconds': 6.0, 'search_budget_seconds': 6.0,
            'termination': termination, 'seconds': 6.1}


def baseline(source, branch='partial'):
    return {'branch': branch, 'outer_intervals': v._encoded_intervals(v.merged_intervals(source))}


def handwritten_tree(status='UNKNOWN'):
    public, frozen, anchor, point = fixture()
    prepared = v._a9._prepare_inputs(public, frozen, anchor, 'both')
    a, d = Q(600), v.MIN_WIDTH
    specs = [(0, None, 0, a, a + 4*d, 'split', 'left', [1, 2]),
             (1, 0, 0, a, a + 2*d, 'split', 'left', [5, 6]),
             (2, 0, 0, a + 2*d, a + 4*d, 'split', 'right', [3, 4]),
             (3, 2, 0, a + 2*d, a + 3*d, 'retained_interior', None, None),
             (4, 2, 0, a + 3*d, a + 4*d, 'retained_tolerance', 'right', None),
             (5, 1, 0, a, a + d, 'retained_tolerance', 'left', None),
             (6, 1, 0, a + d, a + 2*d, 'retained_interior', None, None)]
    trace = [node(i, parent, root, lo, hi, state, side,
                  None if side is None else checks(prepared, 'partial', 'camera_tiles', lo, hi, status, point), kids)
             for i, parent, root, lo, hi, state, side, kids in specs]
    schedule = [{'side': side, 'node_id': i, 'action': 'process'}
                for side, i in [('left', 0), ('right', 2), ('left', 1), ('right', 4), ('left', 5)]]
    source = ((a, a + 4*d),)
    saved = record(prepared, source, trace, schedule, 'boundaries_closed', {'left': True, 'right': True})
    return public, frozen, anchor, baseline(source), saved


class InputGeometryTests(unittest.TestCase):
    def test_exact_target_and_isolated_import_has_no_new_solver(self):
        self.assertTrue(v.check_target())
        proc = subprocess.run([v.sys.executable, '-B', '-c',
            'import sys; import verify_tightening; '
            'assert not any(n in sys.modules for n in '
            '("tightening_solver", "continuous_solver", "anchor_solver", "polytope_filter"))'],
            cwd=v.ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_tiles_cover_nonzero_camera_centers_and_use_explicit_rows(self):
        public, frozen, anchor, point = fixture()
        prepared = v._a9._prepare_inputs(public, frozen, anchor, 'both')
        box = prepared[3]
        tiles = v.camera_tiles(box, 'camera_tiles')
        self.assertEqual(len(tiles), 4)
        self.assertNotEqual(sum(box[0]), 0)
        for l in (box[0][0], sum(box[0]) / 2, box[0][1]):
            for r in (box[2][0], sum(box[2]) / 2, box[2][1]):
                self.assertTrue(any(t[0][0] <= l <= t[0][1] and t[2][0] <= r <= t[2][1] for t in tiles))
        whole, bounds = v.tile_problem(prepared, 'partial', Q(59999, 100), Q(60001, 100), box)
        rebuilt = [v.tile_problem(prepared, 'partial', Q(59999, 100), Q(60001, 100), t) for t in tiles]
        self.assertTrue(any(rows != whole for rows, _ in rebuilt))
        for rows, limits in rebuilt:
            v._proof.verify_witness(rows, limits, point)
        altered = deepcopy(public)
        altered.update(observation_id='irrelevant-label', candidates=[540, 600, 660], truth=600, case='irrelevant')
        self.assertEqual(v.necessary_problem(altered, frozen, anchor, 'partial', 599, 601, tiles[0]),
                         v.necessary_problem(public, frozen, anchor, 'partial', 599, 601, tiles[0]))

    def test_empty_camera_domain_has_constant_exact_contradiction(self):
        public, frozen, anchor, _ = fixture(True)
        prepared = v._a9._prepare_inputs(public, frozen, anchor, 'both')
        for arm in v.ARMS:
            self.assertEqual(v.camera_tiles(prepared[3], arm), (None,))
        rows, bounds = v.tile_problem(prepared, 'none', 590, 610, None)
        proof = classification(rows, bounds, 'CERTIFIED_INFEASIBLE')
        self.assertFalse(v._proof.verify_classification(rows, bounds, proof))
        bad = deepcopy(proof)
        bad['certificate']['margin'] = v.enc(0)
        with self.assertRaises(v.VerificationError):
            v._proof.verify_classification(rows, bounds, bad)


class TreeProofTests(unittest.TestCase):
    def verify(self, bundle):
        public, frozen, anchor, old, saved = bundle
        return v.verify_branch(public, frozen, anchor, saved['mode'], saved['branch'], old, saved)

    def test_unknown_boundary_tree_preserves_all_inner_cells(self):
        bundle = handwritten_tree()
        checked = self.verify(bundle)
        self.assertEqual(checked['classification_calls'], 20)
        self.assertEqual(checked['outer_intervals'], bundle[3]['outer_intervals'])
        self.assertEqual(checked['retained_status_counts'], {'retained_interior': 2, 'retained_tolerance': 2})
        for mutation in ('missing_inner', 'wrong_seam', 'wrong_parent', 'wrong_root', 'wrong_schedule',
                         'wrong_side', 'lost_tile', 'wrong_tile', 'wrong_count', 'deleted_unknown', 'enlarged_hull'):
            changed = deepcopy(bundle)
            saved = changed[-1]
            if mutation == 'missing_inner': saved['trace'].pop()
            elif mutation == 'wrong_seam': saved['trace'][3]['lo'] = v.enc(Q(600) + 2*v.MIN_WIDTH + Q(1, 10000))
            elif mutation == 'wrong_parent': saved['trace'][3]['parent'] = 1
            elif mutation == 'wrong_root': saved['trace'][3]['root'] = 2
            elif mutation == 'wrong_schedule': saved['schedule'][1]['node_id'] = 1
            elif mutation == 'wrong_side': saved['trace'][2]['side'] = 'left'
            elif mutation == 'lost_tile': saved['trace'][0]['checks'].pop()
            elif mutation == 'wrong_tile': saved['trace'][0]['checks'][0]['camera_box'] = saved['camera_domain']
            elif mutation == 'wrong_count': saved['classification_calls'] -= 1
            elif mutation == 'deleted_unknown': saved['trace'][4]['status'] = 'excluded_certificate'
            else: saved['hull'][1] = v.enc(660)
            with self.subTest(mutation=mutation), self.assertRaises(v.VerificationError):
                self.verify(changed)

    def test_exact_outer_witness_is_checked_without_physical_world_claim(self):
        bundle = handwritten_tree('FEASIBLE_WITNESS')
        checked = self.verify(bundle)
        self.assertEqual(checked['classification_status_counts'], {'FEASIBLE_WITNESS': 20})
        broken = deepcopy(bundle)
        broken[-1]['trace'][0]['checks'][2]['classification']['witness'][0] = v.enc(2)
        with self.assertRaises(v.VerificationError): self.verify(broken)
        broken = deepcopy(bundle)
        broken[-1]['trace'][0]['checks'][2]['classification']['status'] = 'FORWARD_WORLD'
        with self.assertRaises(v.VerificationError): self.verify(broken)

    def test_complete_all_tile_certificate_required_for_exclusion(self):
        public, frozen, anchor, _ = fixture()
        public['observation'][:3] = [v.enc(Q(1, 2))] * 3
        prepared = v._a9._prepare_inputs(public, frozen, anchor, 'both')
        source = ((Q(590), Q(610)),)
        trace = [node(0, None, 0, 590, 610, 'excluded_certificate', 'left',
                      checks(prepared, 'partial', 'camera_tiles', 590, 610, 'CERTIFIED_INFEASIBLE'))]
        saved = record(prepared, source, trace, [{'side': 'left', 'node_id': 0, 'action': 'process'}], 'empty')
        bundle = (public, frozen, anchor, baseline(source), saved)
        self.assertEqual(self.verify(bundle)['outer_intervals'], [])
        for mutation in ('omit', 'duplicate', 'certificate', 'unknown'):
            broken = deepcopy(bundle)
            checks_ = broken[-1]['trace'][0]['checks']
            if mutation == 'omit': checks_.pop()
            elif mutation == 'duplicate': checks_[3] = deepcopy(checks_[2])
            elif mutation == 'certificate': checks_[2]['classification']['certificate']['margin'] = v.enc(0)
            else:
                proof = checks_[2]['classification']
                proof.update(status='UNKNOWN', certificate=None)
            with self.subTest(mutation=mutation), self.assertRaises(v.VerificationError): self.verify(broken)

    def test_deadline_before_search_and_mid_tile_keeps_complete_source(self):
        public, frozen, anchor, _ = fixture()
        prepared = v._a9._prepare_inputs(public, frozen, anchor, 'both')
        source = ((Q(590), Q(591)), (Q(610), Q(611)))
        trace = [node(i, None, i, lo, hi, 'retained_deadline') for i, (lo, hi) in enumerate(source)]
        saved = record(prepared, source, trace, [], 'deadline')
        bundle = (public, frozen, anchor, baseline(source), saved)
        self.assertEqual(self.verify(bundle)['outer_intervals'], baseline(source)['outer_intervals'])
        saved['trace'][0]['side'] = 'left'
        saved['trace'][0]['checks'] = checks(prepared, 'partial', 'camera_tiles', 590, 591)
        for entry in saved['trace'][0]['checks'][1:]:
            entry.update(classification=None, reason='not_run_deadline')
        saved['schedule'] = [{'side': 'left', 'node_id': 0, 'action': 'process'}]
        saved.update(classification_calls=1, processed_nodes=1)
        self.assertEqual(self.verify(bundle)['outer_intervals'], baseline(source)['outer_intervals'])
        broken = deepcopy(bundle)
        broken[-1]['trace'][0]['status'] = 'excluded_certificate'
        with self.assertRaises(v.VerificationError): self.verify(broken)

    def test_exhausted_call_budget_retains_every_unprocessed_root(self):
        public, frozen, anchor, _ = fixture(True)
        prepared = v._a9._prepare_inputs(public, frozen, anchor, 'both')
        source = tuple((Q(540) + Q(i, 2), Q(540) + Q(i, 2) + Q(1, 256)) for i in range(129))
        trace = [node(i, None, i, lo, hi, 'retained_call_budget') for i, (lo, hi) in enumerate(source)]
        schedule = []
        for k in range(128):
            ident, side = (k // 2, 'left') if k % 2 == 0 else (128 - k // 2, 'right')
            lo, hi = source[ident]
            trace[ident] = node(ident, None, ident, lo, hi, 'excluded_certificate', side,
                                checks(prepared, 'partial', 'depth', lo, hi, 'CERTIFIED_INFEASIBLE'))
            schedule.append({'side': side, 'node_id': ident, 'action': 'process'})
        saved = record(prepared, source, trace, schedule, 'call_budget', arm='depth')
        bundle = (public, frozen, anchor, baseline(source), saved)
        checked = self.verify(bundle)
        self.assertEqual(checked['classification_calls'], 128)
        self.assertEqual(checked['outer_intervals'], v._encoded_intervals((source[64],)))
        broken = deepcopy(bundle)
        broken[-1]['trace'][64]['checks'] = trace[0]['checks']
        with self.assertRaises(v.VerificationError): self.verify(broken)

    def test_cached_terminal_and_empty_source(self):
        public, frozen, anchor, _ = fixture()
        prepared = v._a9._prepare_inputs(public, frozen, anchor, 'both')
        source = ((Q(600), Q(600) + v.MIN_WIDTH),)
        lo, hi = source[0]
        trace = [node(0, None, 0, lo, hi, 'retained_tolerance', 'left', checks(prepared, 'partial', 'depth', lo, hi))]
        schedule = [{'side': 'left', 'node_id': 0, 'action': 'process'},
                    {'side': 'right', 'node_id': 0, 'action': 'close_cached'}]
        saved = record(prepared, source, trace, schedule, 'boundaries_closed', {'left': True, 'right': True}, arm='depth')
        self.assertEqual(self.verify((public, frozen, anchor, baseline(source), saved))['classification_calls'], 1)
        saved = record(prepared, (), [], [], 'empty')
        self.assertEqual(self.verify((public, frozen, anchor, baseline(()), saved))['outer_intervals'], [])


class SelectionTests(unittest.TestCase):
    def test_independent_hull_action_and_three_branch_union(self):
        outer = ((Q(590), Q(591)), (Q(610), Q(611)))
        for incumbent, selected in ((580, 590), (600, 600), (620, 611)):
            saved = {'arm': 'camera_tiles', 'policy': 'HULL', 'status': 'PROJECTED',
                     'incumbent': v.enc(incumbent), 'selected': v.enc(selected), 'moved': incumbent != selected,
                     'outer_intervals': v._encoded_intervals(outer), 'hull': v._hull(outer),
                     'model_conditional_nonregression': True}
            v.verify_selection(incumbent, outer, 'HULL', saved)
            broken = deepcopy(saved)
            broken['selected'] = v.enc(selected + 1)
            with self.assertRaises(v.VerificationError): v.verify_selection(incumbent, outer, 'HULL', broken)
        branches = [{'branch': b, 'outer_intervals': v._encoded_intervals(x)}
                    for b, x in zip(v.BRANCHES, (outer[:1], outer[1:], ()))]
        self.assertEqual(v.aggregate_branches(branches)['outer_intervals'], v._encoded_intervals(outer))
        with self.assertRaises(v.VerificationError): v.aggregate_branches(branches[:-1])


if __name__ == '__main__':
    unittest.main()

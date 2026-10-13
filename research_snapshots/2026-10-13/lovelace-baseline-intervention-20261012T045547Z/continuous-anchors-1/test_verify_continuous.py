"""Hand-authored A9 checker fixtures: no solver, formal output, or private reads."""
import ast
from collections import deque
from copy import deepcopy
from fractions import Fraction as Q
import subprocess
import unittest

import verify_continuous as v


def fixture(impossible_camera=False, bias_hi=Q(1, 50)):
    enc = v.enc
    public = {'observation_id': 'manual-checker-photo', 'radius': enc(Q(1, 5)),
              'baseline': enc(80), 'clipped': True, 'observation': [enc(1)] * 432}
    frozen = {'appearance_model': 'bounded', 'drift_bound': enc(Q(1, 10)),
              'calibration_intervals': [[enc(0), enc(bias_hi)] for _ in range(3)]}
    offsets, k, errors = (Q(1, 10), Q(0), -Q(1, 10)), Q(1, 200), (Q(1, 200), -Q(1, 200), Q(0))
    anchor = {'observation_id': 'manual-checker-photo', 'camera': {
        'u': [enc((-c - e - k - n) / 1000) for c, e, n in zip(v.CENTERS, offsets, errors)],
        'Xf': enc(0), 'Zf': enc(1000), 'survey_bound': enc(Q(1, 100)), 'localization_bound': enc(Q(1, 100))},
        'appearance': {'coupons': [enc(Q(99, 100))] * 9, 'references': [enc(Q(49, 100))] * 3,
                       'bias_bound': enc(Q(1, 100)), 'transfer_bound': enc(Q(1, 200)), 'noise_bound': enc(Q(1, 200))}}
    if impossible_camera:
        anchor['camera']['u'][1] = enc(1)
    point = [enc(0)] * 12 + [enc(Q(1, 50))] * 3
    return public, frozen, anchor, point


def classification(rows, bounds, status='UNKNOWN', point=None):
    """Construct metadata and a one-row rational proof without an LP oracle."""
    singles = v._proof.single_row_contradictions(rows, bounds)
    certificate = None
    if status == 'CERTIFIED_INFEASIBLE':
        index = singles[0]
        normalized, box = v._proof.normalized_problem(rows, bounds)
        coefficients, bound = normalized[index]
        minimum = v._proof.minimum_over_box(coefficients, box)
        certificate = {'sparse_weights': [{'row': index, 'weight': v.enc(1)}],
                       'aggregate_coefficients': list(map(v.enc, coefficients)),
                       'weighted_bound': v.enc(bound), 'box_min': v.enc(minimum),
                       'margin': v.enc(minimum - bound)}
    return {'status': status, 'certificate': certificate,
            'witness': point if status == 'FEASIBLE_WITNESS' else None,
            'lp_status': None, 'seconds': 0.0, 'lp_seconds': 0.0,
            'single_row_contradiction': bool(singles), 'single_row_contradiction_rows': singles,
            'row_count': len(rows), 'dimension': len(bounds), 'reason': 'handwritten exact checker fixture'}


def node(ident, parent, lo, hi, status, result=None, children=None):
    value = {'id': ident, 'parent': parent, 'lo': v.enc(lo), 'hi': v.enc(hi),
             'status': status, 'classification': result}
    if children is not None:
        value['children'] = children
    return value


def branch_record(trace, termination='tree_complete', budget_seconds=3.0):
    retained = [value for value in trace if value['status'].startswith('retained_')]
    outer = v.merged_intervals((value['lo'], value['hi']) for value in retained)
    return {'domain': list(map(v.enc, v.DOMAIN)), 'trace': trace,
            'retained_ids': [value['id'] for value in retained],
            'outer_intervals': v._encoded_intervals(outer), 'hull': v._hull(outer),
            'status': 'OUTER_RETAINED' if outer else 'OUTER_EMPTY',
            'classified_nodes': sum(value['classification'] is not None for value in trace),
            'pruned_nodes': sum(value['status'] == 'excluded_certificate' for value in trace),
            'max_nodes': 127, 'min_width': v.enc(Q(15, 32)), 'branch_seconds': 3.0,
            'search_budget_seconds': budget_seconds, 'termination': termination, 'seconds': 3.1}


def root_frontier(public, frozen, anchor, arm='both', branch='partial', witness=False):
    rows, bounds = v.necessary_problem(public, frozen, anchor, arm, branch, *v.DOMAIN)
    point = fixture()[3]
    result = classification(rows, bounds, 'FEASIBLE_WITNESS' if witness else 'UNKNOWN', point)
    trace = [node(0, None, 540, 660, 'split', result, [1, 2]),
             node(1, 0, 540, 600, 'retained_deadline'), node(2, 0, 600, 660, 'retained_deadline')]
    return branch_record(trace, 'deadline')


def narrow_path(public, frozen, anchor):
    """One eight-split retained path and eight independently certified siblings."""
    queue, trace, next_id = deque([(0, None, Q(540), Q(660), 0, True)]), [], 1
    prepared = v._prepare_inputs(public, frozen, anchor, 'camera')
    while queue:
        ident, parent, lo, hi, level, on_path = queue.popleft()
        rows, bounds = v._cell_problem(prepared, 'partial', lo, hi)
        if not on_path:
            trace.append(node(ident, parent, lo, hi, 'excluded_certificate', classification(rows, bounds, 'CERTIFIED_INFEASIBLE')))
        elif level == 8:
            trace.append(node(ident, parent, lo, hi, 'retained_tolerance', classification(rows, bounds)))
        else:
            trace.append(node(ident, parent, lo, hi, 'split', classification(rows, bounds), [next_id, next_id + 1]))
            midpoint = (lo + hi) / 2
            queue.append((next_id, ident, lo, midpoint, level + 1, True))
            queue.append((next_id + 1, ident, midpoint, hi, level + 1, False))
            next_id += 2
    return branch_record(trace)


def selection_record(incumbent, intervals, policy, selected, status='PROJECTED'):
    intervals = v.merged_intervals(intervals)
    incumbent, selected = Q(incumbent), Q(selected)
    return {'policy': policy, 'status': status, 'incumbent': v.enc(incumbent),
            'selected': v.enc(selected), 'moved': selected != incumbent,
            'outer_intervals': v._encoded_intervals(intervals), 'hull': v._hull(intervals),
            'model_conditional_nonregression': policy in ('KEEP', 'HULL')}


class GeometryTests(unittest.TestCase):
    def test_exact_target_and_no_new_solver_import(self):
        self.assertTrue(v.check_target())
        tree = ast.parse(v.Path(v.__file__).read_text())
        imported = [alias.name for item in ast.walk(tree) if isinstance(item, ast.Import) for alias in item.names]
        imported += [item.module or '' for item in ast.walk(tree) if isinstance(item, ast.ImportFrom)]
        self.assertFalse(any('solver' in name or 'runner' in name for name in imported))
        isolated = subprocess.run(
            [v.sys.executable, '-B', '-c',
             "import sys; import verify_continuous; "
             "assert not any(name in sys.modules for name in "
             "('continuous_solver', 'anchor_solver', 'point_solver', 'pose_solver', 'polytope_filter'))"],
            cwd=v.ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(isolated.returncode, 0, isolated.stderr)

    def test_full_front_strip_constant_and_exact_enclosure_intersection(self):
        pixel = v._geometry.PIXELS[0]
        self.assertEqual(v.rectangle_lipschitz(v._geometry.FRONT, 80, (-Q(1, 5), Q(1, 5)), pixel), Q(4333))
        intervals = ((Q(-13, 100), Q(-11, 100)), (Q(0), Q(0)), (Q(9, 100), Q(11, 100)))
        for branch in v.BRANCHES:
            low, high = Q(5999, 10), Q(6001, 10)
            enclosure = v.interval_area_enclosure(branch, low, high, intervals)
            crude = v.rectangle_enclosure(branch, low, high, intervals)
            qc, dq = (1 / low + 1 / high) / 2, (1 / low - 1 / high) / 2
            centered = v._a8.area_enclosure(branch, 1 / qc, intervals)
            for index, (bounds, generic, middle) in enumerate(zip(enclosure, crude, centered)):
                view, pixel_number = divmod(index, 48)
                for material, rectangle in enumerate((v._geometry.LEFT, v._geometry.RIGHT, v._geometry.FRONT)):
                    delta = v.rectangle_lipschitz(rectangle, v.CENTERS[view], intervals[view], v._geometry.PIXELS[pixel_number]) * dq
                    self.assertEqual(bounds[2 * material], max(0, generic[2 * material], middle[2 * material] - delta))
                    self.assertEqual(bounds[2 * material + 1], min(1, generic[2 * material + 1], middle[2 * material + 1] + delta))
            for z in (low, Q(600), high):
                for weight in (Q(0), Q(1, 3), Q(1)):
                    offsets = tuple(lo + (hi - lo) * weight for lo, hi in intervals)
                    points = v._geometry.independent_point_areas(branch, z, offsets, Q(80))
                    for point, bounds in zip(points, enclosure):
                        for material, area in enumerate(point):
                            self.assertLessEqual(bounds[2 * material], area)
                            self.assertLessEqual(area, bounds[2 * material + 1])

    def test_zero_width_reduces_to_sealed_all_camera_point_envelope(self):
        for branch in v.BRANCHES:
            depth = Q(9007, 14)
            self.assertEqual(v.interval_area_enclosure(branch, depth, depth),
                             v._a8.area_enclosure(branch, depth, v.DEFAULT_CAMERA_INTERVALS))

    def test_rows_admit_manual_outer_point_and_disabled_sections_are_not_decoded(self):
        public, frozen, anchor, point = fixture()
        for arm in v.ARMS:
            rows, bounds = v.necessary_problem(public, frozen, anchor, arm, 'partial', 590, 610)
            self.assertEqual(len(bounds), 15)
            v.verify_witness(rows, bounds, point)
            modified_public, modified_frozen, modified_anchor = deepcopy(public), deepcopy(frozen), deepcopy(anchor)
            modified_public.update(observation_id='s0-eta-case', truth=625, candidates=[540, 600, 660])
            modified_frozen['branches'] = object()
            modified_anchor['observation_id'] = 'not-a-search-center'
            for key in ('camera', 'appearance'):
                if arm == 'none' or (arm == 'camera' and key == 'appearance') or (arm == 'appearance' and key == 'camera'):
                    modified_anchor[key] = object()
            self.assertEqual((rows, bounds), v.necessary_problem(modified_public, modified_frozen, modified_anchor,
                                                               arm, 'partial', 590, 610))


class TreeTests(unittest.TestCase):
    def test_timeout_retains_entire_domain_with_no_classification(self):
        public, frozen, anchor, _ = fixture()
        record = branch_record([node(0, None, 540, 660, 'retained_deadline')], 'deadline', 0.0)
        result = v.verify_branch(public, frozen, anchor, 'both', 'partial', record)
        self.assertEqual(result['intervals'], (v.DOMAIN,))
        self.assertEqual(result['classified_nodes'], 0)
        record['outer_intervals'] = []
        with self.assertRaises(v.VerificationError):
            v.verify_branch(public, frozen, anchor, 'both', 'partial', record)

    def test_full_certified_domain_and_tampered_certificate(self):
        public, frozen, anchor, _ = fixture(impossible_camera=True)
        rows, bounds = v.necessary_problem(public, frozen, anchor, 'camera', 'none', 540, 660)
        record = branch_record([node(0, None, 540, 660, 'excluded_certificate', classification(rows, bounds, 'CERTIFIED_INFEASIBLE'))])
        result = v.verify_branch(public, frozen, anchor, 'camera', 'none', record)
        self.assertEqual(result['outer_intervals'], [])
        for key, bad in (('margin', v.enc(0)), ('box_min', v.enc(1))):
            changed = deepcopy(record)
            changed['trace'][0]['classification']['certificate'][key] = bad
            with self.assertRaises(v.VerificationError):
                v.verify_branch(public, frozen, anchor, 'camera', 'none', changed)

    def test_original_bias_bounds_are_rebuilt_for_signed_certificate(self):
        public, frozen, anchor, _ = fixture(bias_hi=Q(1, 100))
        public['observation'][:3] = [v.enc(Q(1, 2))] * 3
        rows, bounds = v.necessary_problem(public, frozen, anchor, 'none', 'none', 540, 660)
        record = branch_record([node(0, None, 540, 660, 'excluded_certificate', classification(rows, bounds, 'CERTIFIED_INFEASIBLE'))])
        v.verify_branch(public, frozen, anchor, 'none', 'none', record)
        changed = deepcopy(frozen)
        changed['calibration_intervals'][0][1] = v.enc(Q(1, 50))
        with self.assertRaises(v.VerificationError):
            v.verify_branch(public, changed, anchor, 'none', 'none', record)

    def test_complete_midpoint_tree_and_exact_tolerance_endpoint(self):
        public, frozen, anchor, _ = fixture(impossible_camera=True)
        record = narrow_path(public, frozen, anchor)
        result = v.verify_branch(public, frozen, anchor, 'camera', 'partial', record)
        self.assertEqual(result['intervals'], ((Q(540), Q(540) + Q(15, 32)),))
        self.assertEqual(result['classified_nodes'], 17)
        self.assertEqual(result['pruned_nodes'], 8)
        for mutation in ('missing_child', 'endpoint', 'parent', 'counter', 'retained_id', 'split', 'tolerance', 'termination', 'root'):
            changed = deepcopy(record)
            if mutation == 'missing_child':
                changed['trace'].pop()
            elif mutation == 'endpoint':
                changed['trace'][1]['hi'] = v.enc(Q(600) - Q(1, 1000))
            elif mutation == 'parent':
                changed['trace'][2]['parent'] = 1
            elif mutation == 'counter':
                changed['classified_nodes'] += 1
            elif mutation == 'retained_id':
                changed['retained_ids'].append(16)
            elif mutation == 'split':
                changed['trace'][0]['children'] = [2, 1]
            elif mutation == 'tolerance':
                changed['min_width'] = v.enc(Q(1, 2))
            elif mutation == 'termination':
                changed['termination'] = 'deadline'
            else:
                changed['trace'][0]['lo'] = v.enc(541)
            with self.subTest(mutation=mutation), self.assertRaises(v.VerificationError):
                v.verify_branch(public, frozen, anchor, 'camera', 'partial', changed)

    def test_every_unsearched_cell_survives_full_node_cap(self):
        public, frozen, anchor, _ = fixture(impossible_camera=True)
        queue, trace, next_id, classified = deque([(0, None, Q(540), Q(660))]), [], 1, 0
        prepared = v._prepare_inputs(public, frozen, anchor, 'camera')
        while queue:
            ident, parent, lo, hi = queue.popleft()
            if classified == 127:
                trace.append(node(ident, parent, lo, hi, 'retained_node_budget'))
            else:
                rows, bounds = v._cell_problem(prepared, 'partial', lo, hi)
                trace.append(node(ident, parent, lo, hi, 'split', classification(rows, bounds), [next_id, next_id + 1]))
                midpoint = (lo + hi) / 2
                queue.extend(((next_id, ident, lo, midpoint), (next_id + 1, ident, midpoint, hi)))
                next_id += 2
                classified += 1
        record = branch_record(trace, 'node_budget')
        result = v.verify_branch(public, frozen, anchor, 'camera', 'partial', record)
        self.assertEqual(len(record['trace']), 255)
        self.assertEqual(result['node_budget_leaves'], 128)
        self.assertEqual(result['intervals'], (v.DOMAIN,))
        changed = deepcopy(record)
        changed['trace'][127]['status'] = 'excluded_certificate'
        with self.assertRaises(v.VerificationError):
            v.verify_branch(public, frozen, anchor, 'camera', 'partial', changed)

    def test_positive_evidence_is_outer_only_and_full_witness_is_checked(self):
        public, frozen, anchor, _ = fixture()
        record = root_frontier(public, frozen, anchor, witness=True)
        result = v.verify_branch(public, frozen, anchor, 'both', 'partial', record)
        self.assertEqual(result['classification_status_counts'], {'FEASIBLE_WITNESS': 1})
        changed = deepcopy(record)
        changed['trace'][0]['classification']['witness'][0] = v.enc(Q(11, 10))
        with self.assertRaises(v.VerificationError):
            v.verify_branch(public, frozen, anchor, 'both', 'partial', changed)
        changed = deepcopy(record)
        changed['status'] = 'FORWARD_WITNESS'
        with self.assertRaises(v.VerificationError):
            v.verify_branch(public, frozen, anchor, 'both', 'partial', changed)
        changed = deepcopy(record)
        changed['trace'][0]['classification']['status'] = 'FORWARD_WITNESS'
        with self.assertRaises(v.VerificationError):
            v.verify_branch(public, frozen, anchor, 'both', 'partial', changed)


class SelectionTests(unittest.TestCase):
    def test_keep_hull_union_gap_and_lower_tie(self):
        outer = ((540, 560), (640, 660))
        for policy, selected, status in (('KEEP', 600, 'KEEP'), ('HULL', 600, 'PROJECTED'), ('UNION', 560, 'PROJECTED')):
            record = selection_record(600, outer, policy, selected, status)
            self.assertEqual(v.verify_selection(600, outer, policy, record)['selected'], v.enc(selected))
        changed = selection_record(600, outer, 'UNION', 640)
        with self.assertRaises(v.VerificationError):
            v.verify_selection(600, outer, 'UNION', changed)
        record = selection_record(700, outer, 'HULL', 660)
        v.verify_selection(700, outer, 'HULL', record)
        record = selection_record(600, (), 'UNION', 600, 'SUPPORT_FAILED')
        v.verify_selection(600, (), 'UNION', record)

    def test_disconnected_union_harms_a_truth_inside_support(self):
        outer, incumbent, truth = ((540, 560), (640, 660)), Q(600), Q(650)
        record = selection_record(incumbent, outer, 'UNION', 560)
        verified = v.verify_selection(incumbent, outer, 'UNION', record)
        selected = v.exact(verified['selected'])
        self.assertGreater(abs(selected - truth), abs(incumbent - truth))
        self.assertGreater((selected - truth) ** 2, (incumbent - truth) ** 2)
        self.assertFalse(verified['model_conditional_nonregression'])
        changed = {**record, 'model_conditional_nonregression': True}
        with self.assertRaises(v.VerificationError):
            v.verify_selection(incumbent, outer, 'UNION', changed)

    def test_aggregate_requires_all_three_and_preserves_disconnected_union(self):
        branches = [{'branch': name, 'outer_intervals': v._encoded_intervals(intervals)}
                    for name, intervals in zip(v.BRANCHES, (((540, 560),), ((640, 660),), ()))]
        result = v.aggregate_branches(branches)
        self.assertEqual(result['outer_intervals'], v._encoded_intervals(((540, 560), (640, 660))))
        v.aggregate_branches(branches, result)
        with self.assertRaises(v.VerificationError):
            v.aggregate_branches(branches[:-1])
        with self.assertRaises(v.VerificationError):
            v.aggregate_branches(branches, {**result, 'outer_intervals': v._encoded_intervals(((540, 660),))})

    def test_noncanonical_rational_or_boolean_metadata_cannot_pass(self):
        public, frozen, anchor, _ = fixture()
        record = branch_record([node(0, None, 540, 660, 'retained_deadline')], 'deadline', 0.0)
        changed = deepcopy(record)
        changed['trace'][0]['lo'] = {'numerator': 1080, 'denominator': 2}
        with self.assertRaises(v.VerificationError):
            v.verify_branch(public, frozen, anchor, 'none', 'partial', changed)
        changed = selection_record(600, ((540, 660),), 'HULL', 600)
        changed['moved'] = 0
        with self.assertRaises(v.VerificationError):
            v.verify_selection(600, ((540, 660),), 'HULL', changed)
        changed = selection_record(600, ((540, 660),), 'HULL', 600)
        changed['selected'] = {'numerator': 600.0, 'denominator': 1}
        with self.assertRaises(v.VerificationError):
            v.verify_selection(600, ((540, 660),), 'HULL', changed)


if __name__ == '__main__':
    unittest.main()

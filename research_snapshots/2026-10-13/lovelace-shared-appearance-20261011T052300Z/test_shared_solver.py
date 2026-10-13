from fractions import Fraction as Q
import unittest
from shared_solver import propagate, material_rows, solve_branch, ColoredModel, EPSILON


class SharedTests(unittest.TestCase):
    def test_shared_conflict_individually_feasible(self):
        rows = [('upper', (Q(1),Q(0),Q(0)), Q(1,4)),
                ('lower', (Q(1),Q(0),Q(0)), Q(3,4))]
        result = propagate(rows)
        self.assertTrue(result['excluded'])
        self.assertEqual(result['certificate']['contradiction']['row'], 1)
        self.assertEqual(len(result['certificate']['steps']), 1)

    def test_known_feasible_is_never_excluded(self):
        import random
        rng = random.Random(982)
        for _ in range(100):
            d = tuple(Q(rng.randrange(101),100) for j in range(3))
            rows = []
            for p in range(12):
                coeff = tuple(Q(rng.randrange(101),100) for j in range(3))
                value = sum(a*v for a,v in zip(coeff,d))
                margin = Q(rng.randrange(20),100)
                rows += [('upper',coeff,value+margin),('lower',coeff,value-margin)]
            self.assertFalse(propagate(rows)['excluded'])

    def test_rgb_intersection(self):
        geom = ((Q(1),Q(1),Q(0),Q(0),Q(0),Q(0)),)
        rows = material_rows((Q(1,2)-EPSILON,Q(1,2),Q(1,2)+EPSILON),geom)
        self.assertEqual(rows[0][2], Q(1,2))
        self.assertEqual(rows[1][2], Q(1,2))

    def test_incomplete_contractor_retains_unproved_infeasible(self):
        # Pairwise sums >=1 imply total>=1.5, inconsistent with total<=1.
        # Bound propagation alone sees no contradiction: retaining is correct
        # for a conservative solver, not a claim of joint feasibility.
        rows = [('lower',(Q(1),Q(1),Q(0)),Q(1)),
                ('lower',(Q(1),Q(0),Q(1)),Q(1)),
                ('lower',(Q(0),Q(1),Q(1)),Q(1)),
                ('upper',(Q(1),Q(1),Q(1)),Q(1))]
        self.assertFalse(propagate(rows)['excluded'])

    def test_negative_control_full_prior(self):
        model = ColoredModel('full')
        image = model.point_prediction(Q(600),colors=(Q(1,5),)*3)
        outer = solve_branch(image,model,max_boxes=7)
        self.assertEqual(outer['hull'], outer['domain'])
        self.assertFalse(any(n['status'].startswith('excluded') for n in outer['trace']))

    def test_truth_survives_small_budget(self):
        from continuous_world.interval_solver import decode
        for branch in ('none','partial','full'):
            model = ColoredModel(branch)
            image = model.point_prediction(Q(601),offsets=(Q(1,10),Q(0),Q(-1,5)),colors=(Q(1,4),Q(1,2),Q(3,4)))
            outer = solve_branch(image,model,max_boxes=15)
            kept = set(outer['retained_ids'])
            self.assertTrue(any(decode(n['lo'])<=601<=decode(n['hi']) for n in outer['trace'] if n['id'] in kept))


if __name__ == '__main__':
    unittest.main()

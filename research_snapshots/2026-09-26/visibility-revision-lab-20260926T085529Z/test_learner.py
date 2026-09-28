"""Targeted contracts of the fixed six-arm visibility comparison."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import learner as L
from common import materialize


def fixture(native_score=1., native_loss=2.):
    d = dict(case_id=np.array([0, 0, 1, 1, 1]),
             condition=np.array(['native', 'native', 'plus1', 'plus1', 'plus1']),
             calibration_support=np.array([True, True, True, True, False]),
             losses=np.array([[1., native_loss, 3.], [1., native_loss, 3.],
                              [4., 1., 4.], [4., 1., 4.], [1., 1000., 1000.]]))
    geometry = np.zeros((5, 3, 3))
    geometry[:, 1, 0] = 1.
    geometry[:, 2, 0] = -2.
    scores = np.array([[native_score, -2.], [native_score, -2.],
                       [2., -2.], [2., -2.], [50., -2.]])
    return d, geometry, scores


class LearnerTests(unittest.TestCase):
    def test_feature_prefixes_and_nonmutation(self):
        base = np.arange(3*2*96, dtype=np.float32).reshape(3, 2, 96)
        extra = 1000 + base
        b0, e0 = base.copy(), extra.copy()
        for capacity in L.CAPACITIES:
            for family, width in [('base', 96), ('geometry', 176), ('interaction', 192)]:
                result = L.features_for(base, extra, family+'_'+capacity)
                self.assertEqual(result.shape, (3, 2, width))
                np.testing.assert_array_equal(result[:, :, :96], base)
                np.testing.assert_array_equal(result[:, :, 96:], extra[:, :, :width-96])
        np.testing.assert_array_equal(base, b0)
        np.testing.assert_array_equal(extra, e0)

    def test_same_features_across_capacities(self):
        base = np.zeros((2, 2, 96), np.float32)
        extra = np.ones_like(base)
        for family in L.FAMILIES:
            np.testing.assert_array_equal(L.features_for(base, extra, family+'_shallow'),
                                          L.features_for(base, extra, family+'_rich'))

    def test_base_shallow_parameters_exactly_previous_independent(self):
        self.assertEqual(L.parameters('base_shallow'), L.old_router.PARAMS)
        rich = L.parameters('base_rich')
        self.assertEqual((rich['max_iter'], rich['max_leaf_nodes'], rich['max_depth'],
                          rich['min_samples_leaf']), (160, 31, 6, 40))
        self.assertFalse(rich['early_stopping'])

    def test_fit_pair_preserves_candidate_labels_and_shared_weights(self):
        x = np.arange(4*2*3).reshape(4, 2, 3)
        gains = np.arange(8).reshape(4, 2)-3
        weights = np.array([.2, .7, 1., 4.])
        calls = []
        class Recorder:
            def __init__(self, **params):
                self.params = params
            def fit(self, xx, yy, sample_weight):
                calls.append((xx.copy(), yy.copy(), sample_weight.copy()))
                return self
        with patch.object(L, 'HistGradientBoostingRegressor', Recorder):
            models = L.fit_pair(x, gains, weights, 'geometry_rich')
        self.assertEqual(len(models), 2)
        for i in range(2):
            np.testing.assert_array_equal(calls[i][0], x[:, i])
            np.testing.assert_array_equal(calls[i][1], gains[:, i])
            np.testing.assert_array_equal(calls[i][2], weights)
            self.assertEqual(models[i].params, L.parameters('geometry_rich'))

    def test_predict_uses_own_candidate_features(self):
        x = np.arange(4*2*3).reshape(4, 2, 3)
        class Model:
            def __init__(self, offset): self.offset = offset
            def predict(self, xx): return xx[:, 0] + self.offset
        got = L.predict_pair([Model(-3), Model(7)], x)
        np.testing.assert_array_equal(got, np.column_stack([x[:, 0, 0]-3, x[:, 1, 0]+7]))

    def test_calibration_excludes_unsupported_losses(self):
        d, geometry, scores = fixture()
        row = L.loss_row(d, geometry, scores, threshold=1.)
        self.assertAlmostEqual(row['objective_relative_MSE'], .625)
        self.assertAlmostEqual(row['native_MSE'], 1.)
        self.assertAlmostEqual(row['injected_relative_MSE'], .25)
        self.assertAlmostEqual(row['accepted_fraction'], .5)
        changed = copy.deepcopy(d)
        changed['losses'][-1] = [1e9, 0., 0.]
        self.assertEqual(row, L.loss_row(changed, geometry, scores, threshold=1.))

    def test_balanced_can_protect_native_and_repair_injection(self):
        d, geometry, scores = fixture()
        policies, table = L.calibrate(d, geometry, scores)
        self.assertAlmostEqual(policies['balanced']['objective_relative_MSE'], .625)
        self.assertAlmostEqual(policies['native_priority']['native_MSE'], 1.)
        self.assertTrue(policies['native_priority']['feasible'])
        self.assertEqual(policies['natural']['threshold'], 0.)
        self.assertEqual(sum(row['action']=='keep' for row in table), 1)

    def test_recovery_is_not_silently_native_constrained(self):
        d, geometry, scores = fixture(native_score=3., native_loss=2.)
        policies, _ = L.calibrate(d, geometry, scores)
        self.assertEqual(policies['balanced']['accepted_fraction'], 0.)
        self.assertEqual(policies['balanced']['objective_relative_MSE'], 1.)
        self.assertFalse(policies['native_priority']['feasible'])
        self.assertEqual(policies['native_priority']['action'], 'keep')
        self.assertAlmostEqual(policies['recovery']['injected_relative_MSE'], .25)
        self.assertGreater(policies['recovery']['native_MSE'],
                           policies['recovery']['native_identity_MSE'])

    def test_winner_respects_feasibility_and_fixed_arm_tiebreak(self):
        d, geometry, scores = fixture()
        spec, _ = L.calibrate(d, geometry, scores)
        thresholds = {arm:copy.deepcopy(spec) for arm in L.ARMS}
        self.assertEqual(L.choose_winners(thresholds)['balanced']['arm'], L.ARMS[0])
        thresholds[L.ARMS[0]]['native_priority'].update(feasible=False,
                                                     objective_relative_MSE=-100.)
        winner = L.choose_winners(thresholds)['native_priority']
        self.assertEqual(winner['arm'], L.ARMS[1])
        self.assertTrue(winner['feasible'])

    def test_zero_geometry_cannot_become_fictitious_move(self):
        d, geometry, scores = fixture()
        geometry[:] = 0.
        policies, _ = L.calibrate(d, geometry, scores)
        for policy in L.POLICIES:
            self.assertEqual(policies[policy]['accepted_fraction'], 0.)

    def test_materialized_output_is_exactly_one_existing_candidate(self):
        _, geometry, _ = fixture()
        scores = np.array([[3., 2.], [1., 4.], [-1., -2.], [0., 0.], [4., 4.]])
        route = L.routes(scores, geometry)
        np.testing.assert_array_equal(route, [1, 2, 0, 0, 1])
        output = materialize(geometry, route)
        np.testing.assert_array_equal(output, geometry[np.arange(len(route)), route])

    def test_two_actual_input_layers_have_distinct_front_ownership(self):
        from visibility_features import rasterize, query_support
        p = np.array([[0., 0., 100.], [0., 0., 110.]])
        camera = dict(P=np.array([[100., 0., 32., 0.], [0., 100., 32., 0.],
                                 [0., 0., 1., 0.]]), width=64, height=64)
        query = query_support(p, np.array([0, 1]), rasterize(p, camera))
        np.testing.assert_array_equal(query['known'], [True, True])
        np.testing.assert_array_equal(query['self_front'], [True, False])
        np.testing.assert_array_equal(query['centergap'], [0., 10.])
        np.testing.assert_array_equal(query['fit'], [False, False])


def audit_completed_fit():
    """Read-only real-artifact audit; does not open replay results."""
    import csv, hashlib, json, joblib
    from common import OUT, JOINT, verify_seal
    from train import load_development
    verify_seal(OUT/'training')
    lock = json.loads((OUT/'training/MODEL_LOCK.json').read_text())
    assert lock['dev_scenes'] == [24, 37] and lock['replay_labels_used'] is False
    d, base, extra, geometry = load_development()
    fit = d['calibration_support']
    weights = L.case_weights(d['case_id'][fit], d['e0'][fit])
    gains = d['losses'][fit, :1] - d['losses'][fit, 1:]
    expected_initial = (gains*weights[:, None]).sum(0)/weights.sum()
    checked_folds = checked_policies = checked_models = 0
    initial_error = 0.
    thresholds = json.loads((OUT/'training/THRESHOLDS.json').read_text())
    for arm in L.ARMS:
        folder = OUT/'training/arms'/arm
        verify_seal(folder)
        summary = json.loads((folder/'SUMMARY.json').read_text())
        assert summary['rows'] == len(base) and summary['fit_rows'] == int(fit.sum())
        assert summary['features'] == L.features_for(base, extra, arm).shape[2]
        for fold in summary['folds']:
            train = (d['scene'] == fold['train_scene']) & fit
            test = d['scene'] != fold['train_scene']
            assert not (train & test).any()
            for name, mask in [('train', train), ('test', test)]:
                assert int(mask.sum()) == fold[name+'_rows']
                assert hashlib.sha256(np.flatnonzero(mask).tobytes()).hexdigest() == fold[name+'_row_identity_sha']
            checked_folds += 1
        models = joblib.load(folder/'models.joblib')
        for i, model in enumerate(models):
            assert model.get_params() == L.HistGradientBoostingRegressor(**L.parameters(arm)).get_params()
            delta = abs(float(model._baseline_prediction[0, 0])-expected_initial[i])
            assert delta < 1e-12
            initial_error = max(initial_error, delta)
            checked_models += 1
        with np.load(folder/'OOF.npz') as z: scores = z['scores']
        with (folder/'CALIBRATION.csv').open() as stream:
            rows = [{k:(v if k=='action' else float(v)) for k,v in r.items()}
                    for r in csv.DictReader(stream)]
        for policy, spec in thresholds[arm].items():
            choice = L.routes(scores, geometry, spec['threshold'], spec['action'])
            after = d['losses'][np.arange(len(choice)), choice]
            relative, native, original, injection, accepted = [], [], [], [], []
            for cid in np.unique(d['case_id']):
                indices = (d['case_id']==cid) & fit
                before = d['losses'][indices, 0].mean()
                mse = after[indices].mean()
                relative.append(mse/max(before, 1e-6))
                accepted.append((choice[indices]!=0).mean())
                if d['condition'][indices][0] == 'native':
                    native.append(mse); original.append(before)
                else: injection.append(relative[-1])
            expected = dict(objective_relative_MSE=np.mean(relative), native_MSE=np.mean(native),
                            native_identity_MSE=np.mean(original), injected_relative_MSE=np.mean(injection),
                            accepted_fraction=np.mean(accepted))
            for key,value in expected.items(): assert abs(value-spec[key]) < 1e-12, (arm, policy, key)
            if policy != 'natural':
                feasible = rows
                if policy == 'native_priority':
                    feasible = [r for r in rows if r['native_MSE']<=r['native_identity_MSE']+1e-12
                                and r['injected_relative_MSE']<=.95+1e-12]
                    assert bool(feasible) == spec['feasible']
                if feasible:
                    metric = 'injected_relative_MSE' if policy=='recovery' else 'objective_relative_MSE'
                    selected = min(feasible, key=lambda r:(r[metric], r['accepted_fraction'], -r['threshold']))
                    assert selected['threshold']==spec['threshold'] and selected['action']==spec['action']
                else: assert spec['action']=='keep'
            else: assert spec['threshold']==0 and spec['action']=='threshold'
            checked_policies += 1
    winners = {}
    for policy in L.POLICIES:
        eligible = [a for a in L.ARMS if thresholds[a][policy]['feasible']] or list(L.ARMS)
        key = 'injected_relative_MSE' if policy=='recovery' else 'objective_relative_MSE'
        arm = min(eligible, key=lambda a:(thresholds[a][policy][key], thresholds[a][policy]['accepted_fraction'],
                                        -thresholds[a][policy]['threshold'], L.ARMS.index(a)))
        assert lock['winners'][policy]['arm'] == arm
        winners[policy] = arm
    with np.load(OUT/'training/arms/base_shallow/OOF.npz') as z: baseline = z['scores']
    with np.load(JOINT/'training/OOF_SCORES.npz') as z: old = z['independent_absolute']
    assert np.array_equal(baseline, old)
    return dict(status='PASS', rows=len(base), fit_rows=int(fit.sum()), folds_checked=checked_folds,
                models_checked=checked_models, policies_checked=checked_policies, winners=winners,
                baseline_oof_max_difference=float(abs(baseline-old).max()),
                weighted_initial_gain_max_error=float(initial_error), replay_metrics_read=False)


if __name__ == '__main__': unittest.main()

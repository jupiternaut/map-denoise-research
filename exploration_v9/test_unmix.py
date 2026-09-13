"""Small math/API checks; no main project data or experiment truth loaded."""
import copy
import importlib.util
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

_SPEC = importlib.util.spec_from_file_location('_v9_test_unmix', Path(__file__).with_name('unmix.py'))
M = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(M)


def population_fixture(a=1., n=160):
    rng = np.random.default_rng(912920)
    xy = rng.normal(size=(n//2,2))
    x = np.column_stack([np.ones(n), np.vstack([xy,xy])])
    group = np.repeat([0,1], n//2)
    sign = 2*group-1
    kappa = 2*ndtr(a)-1
    b = 2*np.exp(-a*a/2)/np.sqrt(2*np.pi)
    y = sign*(a*kappa+b)
    local = np.column_stack([x[:,1:], y])
    world = local/1000.
    # Nontrivial ordering exercises world/local row alignment.
    order = np.arange(n)[::-1]
    world = world[order]
    state = dict(world=world.copy(), order=order.copy(), active=np.arange(n),
                 support=np.ones(n,bool), design=x, corrected=y, local=local,
                 weights=np.ones(n), normal=np.array([0.,0.,1.]))
    groups_world = np.empty(n,int); groups_world[order] = group
    artifacts = dict(group_ids=groups_world, candidate_mask=np.ones((n,2),bool),
                     coefficients=np.array([[-a,0.,0.],[a,0.,0.]]))
    return state, artifacts, kappa


class UnmixTest(unittest.TestCase):
    def test_population_offset_recovers_surface_not_mixed_center(self):
        state, artifacts, kappa = population_fixture()
        out, info, result = M.unmix_frozen(state, artifacts, 1., 'offset')
        np.testing.assert_allclose(result['coefficients'][:,0], [-1.,1.], atol=1e-12)
        np.testing.assert_allclose(out[:,2]*1000., 2*artifacts['group_ids']-1, atol=1e-12)
        noise, _ = M._V8.selection_bias(state['design'], artifacts['coefficients'],
                  artifacts['candidate_mask'][state['order']], artifacts['group_ids'][state['order']], 1.)
        corrected = state['corrected']-noise
        self.assertAlmostEqual(corrected[state['active'] < 80].mean(), -kappa)
        self.assertLess(2*kappa,2.)
        self.assertEqual(info['fit_rank'],2)

    def test_population_affine_recovers_zero_slopes(self):
        state, artifacts, _ = population_fixture()
        _, info, result = M.unmix_frozen(state, artifacts, 1., 'affine')
        np.testing.assert_allclose(result['coefficients'], artifacts['coefficients'], atol=1e-12)
        self.assertEqual(info['fit_rank'],6)

    def test_offset_preserves_seed_slopes_exactly(self):
        state, artifacts, _ = population_fixture()
        artifacts['coefficients'][:,1:] = [[.01,.02],[-.04,.03]]
        _, _, result = M.unmix_frozen(state, artifacts, 1., 'offset')
        np.testing.assert_array_equal(result['coefficients'][:,1:], artifacts['coefficients'][:,1:])

    def test_input_immutable_support_and_world_order(self):
        state, artifacts, _ = population_fixture()
        state['support'][0] = False
        before_s, before_a = copy.deepcopy(state), copy.deepcopy(artifacts)
        output, _, result = M.unmix_frozen(state, artifacts, 1.)
        for key in state: np.testing.assert_array_equal(state[key], before_s[key])
        for key in artifacts: np.testing.assert_array_equal(artifacts[key], before_a[key])
        np.testing.assert_array_equal(output[~result['support_mask']], state['world'][~result['support_mask']])
        np.testing.assert_array_equal(result['group_ids'], artifacts['group_ids'])
        np.testing.assert_array_equal(result['active_original_indices'], state['order'][state['active']])

    def test_deterministic(self):
        state, artifacts, _ = population_fixture()
        a = M.unmix_frozen(state, artifacts, 1.)
        b = M.unmix_frozen(state, artifacts, 1.)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2]['coefficients'], b[2]['coefficients'])

    def test_unseen_component_keeps_seed_nullspace(self):
        state, artifacts, _ = population_fixture()
        artifacts['coefficients'] = np.vstack([artifacts['coefficients'], [12.,.5,-.3]])
        artifacts['candidate_mask'] = np.column_stack([artifacts['candidate_mask'], np.zeros(160,bool)])
        for mode in ('offset','affine'):
            _, info, result = M.unmix_frozen(state, artifacts, 1., mode)
            np.testing.assert_array_equal(result['coefficients'][2], artifacts['coefficients'][2])
            self.assertGreater(info['numerical_nullity'],0)

    def test_geometric_nullspace_keeps_unseen_slope(self):
        state, artifacts, _ = population_fixture()
        state['design'][:,2] = 0
        artifacts['coefficients'][:,2] = [.7,-.6]
        _, info, result = M.unmix_frozen(state, artifacts, 1., 'affine')
        np.testing.assert_allclose(result['coefficients'][:,2], [.7,-.6], atol=1e-12)
        self.assertEqual(info['fit_rank'],4)

    def test_low_probability_excluded_and_identity_fallback(self):
        state, artifacts, _ = population_fixture()
        original = M._V8.selection_bias
        def fake(*args):
            noise, details = original(*args)
            details['modeled_selection_probability'][0] = 0.
            details['selection_component_mass'][0] = 0.
            return noise, details
        with patch.object(M._V8, 'selection_bias', fake):
            out, info, result = M.unmix_frozen(state, artifacts, 1.)
        row = state['order'][0]
        np.testing.assert_array_equal(out[row], state['world'][row])
        self.assertEqual(info['low_probability_rows'],1)
        self.assertFalse(result['fit_row_mask'][row])
        self.assertTrue(result['support_mask'][row])

    def test_moment_sse_nonincrease(self):
        state, artifacts, _ = population_fixture()
        state['corrected'] += .07*state['design'][:,1]
        for mode in ('offset','affine'):
            _, info, _ = M.unmix_frozen(state, artifacts, 1., mode)
            self.assertLessEqual(info['weighted_moment_sse_after_mm2'],info['weighted_moment_sse_before_mm2']+1e-12)

    def test_no_truth_api(self):
        self.assertEqual(list(inspect.signature(M.unmix_frozen).parameters),['state','artifacts','sigma_mm','mode'])

    def test_invalid_mode_sigma(self):
        state, artifacts, _ = population_fixture()
        with self.assertRaises(ValueError): M.unmix_frozen(state, artifacts, 0.)
        with self.assertRaises(ValueError): M.unmix_frozen(state, artifacts, 1., 'unknown')

    def test_no_active_rows(self):
        state, artifacts, _ = population_fixture()
        state['active'] = np.empty(0,int); state['support'][:] = False
        output, info, _ = M.unmix_frozen(state, artifacts, 1.)
        np.testing.assert_array_equal(output, state['world'])
        self.assertEqual(info['status'],'UNSUPPORTED')


if __name__ == '__main__':
    unittest.main()

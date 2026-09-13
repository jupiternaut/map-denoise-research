"""Small exposed interface checks; no formal evaluation seeds are generated."""
import inspect
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

import direction_pooling as candidate

REAL = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-transfer-v5-zrtui_f5')


def small_plane():
    rng = np.random.default_rng(2309)
    scan = np.repeat(np.arange(6), 96)
    xy = rng.uniform([-90., -55.], [90., 55.], (len(scan), 2))
    z = xy @ np.array([.03, -.02]) + np.array([-3., 2., 1., -1., 3., -2.])[scan] + rng.normal(0., .7, len(scan))
    return np.c_[xy, z] / 1000. + [1., -2., .5], scan


class DirectionTests(unittest.TestCase):
    def test_legal_boundary_and_read_only_inputs(self):
        self.assertEqual(list(inspect.signature(candidate.estimate).parameters),
                         ['xyz_world_m', 'scan_id', 'sigma_mm', 'variant'])
        world, scan = small_plane(); before = world.copy(); old_scan = scan.copy()
        world.flags.writeable = False; scan.flags.writeable = False
        for variant in candidate.VARIANTS:
            out, info = candidate.estimate(world, scan, 1., variant)
            self.assertEqual(out.shape, world.shape); self.assertTrue(np.isfinite(out).all())
            json.dumps(info, allow_nan=False)
            np.testing.assert_array_equal(world, before); np.testing.assert_array_equal(scan, old_scan)
        with self.assertRaises(ValueError): candidate.estimate(world, scan, 0.)

    def test_difference_replays_frozen_real_output(self):
        for case in ('cy_thin__zero', 'da_junction__normal_translation'):
            with np.load(REAL/'inputs'/(case+'.npz')) as a: world=a['xyz_world']; scan=a['scan_id']
            out, _ = candidate.estimate(world, scan, 2., 'difference')
            with np.load(REAL/'outputs'/(case+'__pool_compatible.npz')) as a:
                np.testing.assert_array_equal(out, a['xyz_world'])

    def test_pca_enters_graph_and_pooling_without_shared_mutation(self):
        world, scan = small_plane()
        isolated_a, isolated_b = candidate._load_private(), candidate._load_private()
        self.assertIsNot(isolated_a._V4._V3, isolated_b._V4._V3)
        old_basis = isolated_b._V4._V3._basis
        with mock.patch.object(candidate, '_pca_basis', wraps=candidate._pca_basis) as hook:
            _, info = candidate.estimate(world, scan, 1., 'pooled_pca')
            self.assertGreaterEqual(hook.call_count, 3)
        self.assertIs(isolated_b._V4._V3._basis, old_basis)
        self.assertEqual(info['basis_call_count'], hook.call_count)
        n=np.asarray(info['normal_world'])
        np.testing.assert_allclose(info['v4_info']['initial_info']['normal_world'], n, atol=1e-12)
        for entry in info['basis_calls']:
            basis=np.array(entry['basis_world'])
            np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
            np.testing.assert_allclose(basis[:,2],n,atol=1e-12)
            self.assertGreater(np.linalg.det(basis),.999999)

    def test_coordinate_point_order_and_normal_only(self):
        world, scan=small_plane()
        rotation=Rotation.from_euler('xyz',[23.,-31.,57.],degrees=True).as_matrix()
        translation=np.array([2.37,-5.11,.79])
        order=np.random.default_rng(451).permutation(len(world))
        for variant in candidate.VARIANTS:
            base, info=candidate.estimate(world,scan,1.,variant)
            rotated,_=candidate.estimate(world@rotation.T+translation,scan,1.,variant)
            np.testing.assert_allclose((rotated-translation)@rotation,base,atol=1e-8,rtol=0.)
            shuffled,_=candidate.estimate(world[order],scan[order],1.,variant)
            np.testing.assert_allclose(shuffled[np.argsort(order)],base,atol=1e-10,rtol=0.)
            n=np.array(info['normal_world']);delta=base-world
            np.testing.assert_allclose(delta-(delta@n)[:,None]*n,0.,atol=1e-12,rtol=0.)
            np.testing.assert_array_equal(base[info['unsupported_point_indices']],world[info['unsupported_point_indices']])

    def test_pca_is_current_input_covariance_and_no_eval_interface(self):
        world,scan=small_plane();centered=(world-world.mean(0))*1000.
        basis,info=candidate._pca_basis(centered,scan,1.)
        _,vectors=np.linalg.eigh(centered.T@centered/len(centered))
        self.assertAlmostEqual(abs(basis[:,2]@vectors[:,0]),1.,places=12)
        self.assertEqual(list(inspect.signature(candidate._pca_basis).parameters),['points','scan','sigma'])
        self.assertNotIn('read_evaluation',Path(candidate.__file__).read_text())
        self.assertNotIn('np.load',Path(candidate.__file__).read_text())


if __name__=='__main__': unittest.main(verbosity=2)

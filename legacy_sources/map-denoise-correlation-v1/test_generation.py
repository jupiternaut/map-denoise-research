"""Independent structural checks for the frozen generated correlation dataset."""
import ast
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

import generate_cases as gen


DATA = Path(__file__).resolve().parent / 'data'


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / 'manifest.json').read_text())
        cls.entries = cls.manifest['entries']
        cls.groups = {}
        for entry in cls.entries:
            cls.groups.setdefault(entry['pair_id'], []).append(entry)

    @staticmethod
    def read(entry):
        with np.load(DATA / entry['input_path'], allow_pickle=False) as source:
            inp = {key: source[key] for key in source.files}
        with np.load(DATA / entry['eval_path'], allow_pickle=False) as source:
            truth = {key: source[key] for key in source.files}
        return inp, truth

    def test_design_cardinality_and_paths(self):
        self.assertEqual(len(self.entries), 162)
        self.assertEqual(len(self.groups), 54)
        self.assertEqual(len({entry['family'] for entry in self.entries}), 6)
        self.assertEqual({entry['seed'] for entry in self.entries}, set(gen.SEEDS))
        self.assertEqual({entry['amplitude_mm'] for entry in self.entries}, {0., 2., 4.})
        self.assertEqual(len({entry['id'] for entry in self.entries}), 162)
        self.assertEqual(len(list((DATA / 'inputs').glob('*.npz'))), 162)
        self.assertEqual(len(list((DATA / 'eval').glob('*.npz'))), 162)
        for members in self.groups.values():
            self.assertEqual({entry['arm'] for entry in members}, {'correlated', 'shuffle_0', 'shuffle_1'})
            self.assertEqual({entry['repeat'] for entry in members}, {-1, 0, 1})
        for entry in self.entries:
            for key in ['input_path', 'eval_path']:
                self.assertTrue((DATA / entry[key]).resolve().is_relative_to(DATA))

    def test_input_has_no_ground_truth(self):
        for entry in self.entries:
            inp, truth = self.read(entry)
            self.assertEqual(set(inp), {'xyz_mm', 'frame', 'sigma_mm'})
            self.assertEqual(set(truth), {'clean_xyz_mm', 'labels', 'perturbation_mm', 'base_xyz_mm'})
            self.assertEqual(inp['xyz_mm'].shape, (768, 3))
            self.assertEqual(inp['frame'].shape, (768,))
            self.assertEqual(truth['perturbation_mm'].shape, (768,))
            self.assertEqual(float(inp['sigma_mm']), 1.)
            np.testing.assert_array_equal(np.unique(inp['frame'], return_counts=True)[1], np.full(8, 96))
            self.assertTrue(np.isfinite(inp['xyz_mm']).all())
            np.testing.assert_array_equal(inp['xyz_mm'][:, :2], truth['clean_xyz_mm'][:, :2])
            np.testing.assert_allclose(inp['xyz_mm'][:, 2] - truth['clean_xyz_mm'][:, 2], truth['perturbation_mm'], rtol=0, atol=2e-15)

    def test_pair_geometry_frames_base_and_error_multisets(self):
        for members in self.groups.values():
            reference = next(entry for entry in members if entry['arm'] == 'correlated')
            reference_inp, reference_truth = self.read(reference)
            for entry in members:
                inp, truth = self.read(entry)
                np.testing.assert_array_equal(inp['frame'], reference_inp['frame'])
                for key in ['clean_xyz_mm', 'labels', 'base_xyz_mm']:
                    np.testing.assert_array_equal(truth[key], reference_truth[key])
                np.testing.assert_array_equal(np.sort(truth['perturbation_mm']), np.sort(reference_truth['perturbation_mm']))
                for layer in np.unique(truth['labels']):
                    take = truth['labels'] == layer
                    np.testing.assert_array_equal(np.sort(truth['perturbation_mm'][take]), np.sort(reference_truth['perturbation_mm'][take]))

    def test_identity_metrics_pair_exact_and_coordinates_roundoff(self):
        for members in self.groups.values():
            reference_stats = None
            reference_coordinate_stats = None
            for entry in members:
                inp, truth = self.read(entry)
                stats = gen.identity_error_stats(truth['perturbation_mm'])
                err = inp['xyz_mm'] - truth['clean_xyz_mm']
                coordinate_stats = np.array([np.mean(abs(err[:, 2])), np.sqrt(np.mean(np.sum(err ** 2, axis=1)))])
                np.testing.assert_allclose(coordinate_stats, list(stats.values()), rtol=0, atol=2e-15)
                if reference_stats is None:
                    reference_stats, reference_coordinate_stats = stats, coordinate_stats
                self.assertEqual(stats, reference_stats)
                np.testing.assert_allclose(coordinate_stats, reference_coordinate_stats, rtol=0, atol=2e-15)

    def test_planar_merged_z_multiset_exact(self):
        for members in self.groups.values():
            if members[0]['group'] == 'curved':
                continue
            reference = np.sort(self.read(members[0])[0]['xyz_mm'][:, 2])
            for entry in members[1:]:
                np.testing.assert_array_equal(np.sort(self.read(entry)[0]['xyz_mm'][:, 2]), reference)

    def test_zero_amplitude_keeps_noise_but_changes_shuffle_array(self):
        for members in self.groups.values():
            if members[0]['amplitude_mm'] != 0.:
                continue
            reference = next(entry for entry in members if entry['arm'] == 'correlated')
            inp, truth = self.read(reference)
            np.testing.assert_array_equal(inp['xyz_mm'], truth['base_xyz_mm'])
            self.assertGreater(np.std(truth['perturbation_mm']), .8)
            self.assertLess(np.std(truth['perturbation_mm']), 1.2)
            for entry in members:
                if entry['arm'] != 'correlated':
                    other, other_truth = self.read(entry)
                    self.assertFalse(np.array_equal(other['xyz_mm'], inp['xyz_mm']))
                    self.assertFalse(np.array_equal(other_truth['perturbation_mm'], truth['perturbation_mm']))

    def test_bias_center_rms_and_reconstruction(self):
        for entry in self.entries:
            inp, truth = self.read(entry)
            bias = np.array(entry['frame_bias_mm'])
            self.assertAlmostEqual(float(bias.mean()), 0., delta=1e-15)
            self.assertAlmostEqual(float(np.sqrt(np.mean(bias ** 2))), entry['amplitude_mm'], delta=1e-15)
            if entry['arm'] == 'correlated':
                eps = np.random.default_rng(np.random.SeedSequence([entry['seed'], 202])).normal(0., 1., 768)
                np.testing.assert_array_equal(truth['perturbation_mm'], eps + bias[inp['frame']])
                base = truth['clean_xyz_mm'].copy()
                base[:, 2] += eps
                np.testing.assert_array_equal(truth['base_xyz_mm'], base)

    def test_geometry_and_epsilon_fixed_across_amplitudes(self):
        grouped = {}
        for entry in self.entries:
            if entry['arm'] == 'correlated':
                grouped.setdefault((entry['family'], entry['seed']), []).append(entry)
        for entries in grouped.values():
            inp0, truth0 = self.read(entries[0])
            for entry in entries[1:]:
                inp, truth = self.read(entry)
                np.testing.assert_array_equal(inp['frame'], inp0['frame'])
                for key in ['clean_xyz_mm', 'labels', 'base_xyz_mm']:
                    np.testing.assert_array_equal(truth[key], truth0[key])

    def test_raycast_first_visible_finite_plate_and_curvature(self):
        for entry in self.entries:
            if entry['arm'] != 'correlated' or entry['amplitude_mm'] != 0.:
                continue
            inp, truth = self.read(entry)
            p, labels = truth['clean_xyz_mm'], truth['labels']
            if entry['group'] == 'raycast':
                gap = entry['gap_mm']
                np.testing.assert_array_equal(p[:, 2], gap * labels)
                self.assertTrue((abs(p[:, 1]) <= 50).all())
                for layer, low, high in [(0, -60., 15.), (1, -15., 60.)]:
                    self.assertTrue(((p[labels == layer, 0] >= low) & (p[labels == layer, 0] <= high)).all())
                cameras = np.asarray(entry['camera_origins_by_frame_mm'])[inp['frame']]
                rear = labels == 0
                at_front = cameras[rear] + ((gap - cameras[rear, 2]) / (p[rear, 2] - cameras[rear, 2]))[:, None] * (p[rear] - cameras[rear])
                occluded = (at_front[:, 0] >= -15) & (at_front[:, 0] <= 60) & (abs(at_front[:, 1]) <= 50)
                self.assertFalse(occluded.any())
            if entry['group'] == 'curved':
                expected = 6. * labels + .0015 * (p[:, 0] ** 2 + .5 * p[:, 1] ** 2)
                np.testing.assert_array_equal(p[:, 2], expected)

    def test_hashes_frozen_source_and_copied_ray_function(self):
        self.assertEqual(gen.sha256(gen.OLD_COMMON), gen.OLD_SHA256)
        for path, expected in self.manifest['source_sha256'].items():
            self.assertEqual(gen.sha256(path), expected)
        for entry in self.entries:
            for kind in ['input', 'eval']:
                self.assertEqual(gen.sha256(DATA / entry[f'{kind}_path']), entry[f'{kind}_sha256'])
        old_tree = ast.parse(gen.OLD_COMMON.read_text())
        old_function = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ray_hits')
        copied_function = ast.parse(inspect.getsource(gen.ray_hits)).body[0]
        self.assertEqual(ast.dump(old_function, include_attributes=False), ast.dump(copied_function, include_attributes=False))

    def test_existing_data_is_not_overwritten(self):
        before = gen.sha256(DATA / 'manifest.json')
        with self.assertRaises(FileExistsError):
            gen.generate_dataset(DATA)
        self.assertEqual(gen.sha256(DATA / 'manifest.json'), before)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GenerationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

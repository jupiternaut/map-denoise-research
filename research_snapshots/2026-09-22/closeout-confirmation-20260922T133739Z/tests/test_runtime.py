"""Portable contract checks and optional read-only V28 archive replay."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'package'))
import unittest
import joblib
import numpy as np
from v28_closeout import apply_arrays, construct, load_models, policy_masks, schema
from v28_closeout.runtime import MODELS, _post_features
from v28_closeout.surfacelet import solve_surfacelets


class RuntimeTests(unittest.TestCase):
    def test_exact_ties_and_threshold(self):
        masks = policy_masks({'pre_A': np.array([-1., 0., 1., 2.]),
                              'post_A': np.array([0., 2., -1., 1.]),
                              'post_B': np.array([0., 2., 3., -1.])})
        np.testing.assert_array_equal(masks['pre_A_keep'], [0, 0, 1, 1])
        np.testing.assert_array_equal(masks['post_A_keep'], [0, 1, 0, 1])
        np.testing.assert_array_equal(masks['post_AB_keep'], [0, 1, 2, 1])

    def test_model_load_and_serialized_repeatability(self):
        models = load_models()
        p = np.arange(36, dtype=float).reshape(-1, 3)
        features = {key: np.zeros((len(p), len(schema()[key])), np.float32)
                    for key, _ in MODELS.values()}
        first = apply_arrays(p, p+1, p-1, features, models)
        second = apply_arrays(p, p+1, p-1, features)
        for group_a, group_b in zip(first, second):
            for key in group_a:
                np.testing.assert_array_equal(group_a[key], group_b[key])
        self.assertEqual(np.count_nonzero(first[2]['random_A_keep']),
                         np.count_nonzero(first[2]['post_A_keep']))
        for name, output in first[0].items():
            mask = first[2][name]
            np.testing.assert_array_equal(output[mask == 0], p[mask == 0])

    def test_invalid_schema_is_rejected(self):
        p = np.zeros((3, 3))
        features = {key: np.zeros((3, len(schema()[key])), np.float32)
                    for key, _ in MODELS.values()}
        features['A_all_post'] = features['A_all_post'][:, :-1]
        with self.assertRaisesRegex(ValueError, 'schema'):
            apply_arrays(p, p, p, features)

    def test_camera_constructor_without_scene_files(self):
        yy, xx = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 100.)])
        intrinsic = np.array([[100., 0., 31.5], [0., 100., 31.5], [0., 0., 1.]])
        def camera(x):
            center = np.array([x, 0., 0.])
            projection = intrinsic @ np.column_stack([np.eye(3), -center])
            return {'image': np.ones((64, 64)), 'P': projection, 'center': center}
        raw, features, state = construct(points, camera(0), [camera(x) for x in (-2, -1, 1, 2)])
        for result in raw.values():
            np.testing.assert_array_equal(result, points)
        for key, names in schema().items():
            if isinstance(names, list):
                self.assertEqual(features[key].shape, (len(points), len(names)))
                self.assertTrue(np.isfinite(features[key]).all())
        self.assertTrue((state['A_all_offset'] == 0).all())
        outputs, _, _ = apply_arrays(points, raw['A_all'], raw['B_all'], features)
        for output in outputs.values():
            np.testing.assert_array_equal(output, points)

    def test_package_has_no_old_workspace_imports(self):
        package = Path(__file__).resolve().parents[1]/'package/v28_closeout'
        for path in package.glob('*.py'):
            content = path.read_text()
            self.assertNotIn('/home/', content, str(path))
            self.assertNotIn('/srv/', content, str(path))
            self.assertNotIn('metric_core', content, str(path))
            self.assertNotIn('load_scene', content, str(path))


@unittest.skipUnless(os.environ.get('V28_SOURCE'), 'V28_SOURCE needed for archive compatibility checks')
class ArchiveReplayTests(unittest.TestCase):
    def test_all_old_fold_predictions_and_decisions(self):
        old = Path(os.environ['V28_SOURCE'])
        cases_checked = rows_checked = 0
        for train, scene in ((24, 37), (37, 24)):
            fold = old/'selector_results'/f'train{train}_test{scene}'
            models = {name: joblib.load(fold/(name+'.joblib')) for name in MODELS}
            cases = sorted(p for p in (old/'real_results').iterdir()
                           if p.is_dir() and p.name.startswith(f'scan{scene}_'))
            for index, case in enumerate(cases):
                with np.load(case/'features.npz', allow_pickle=False) as z:
                    features = {key: z[key] for key, _ in MODELS.values()}
                n = len(features['pre'])
                # Geometry is irrelevant to routing; row identity is preserved.
                p = np.zeros((n, 3))
                _, predictions, masks = apply_arrays(p, p+1, p-1, features, models,
                                                      20260922+100*scene+index)
                with np.load(fold/case.name/'decisions.npz', allow_pickle=False) as archived:
                    for name, prediction in predictions.items():
                        np.testing.assert_array_equal(prediction, archived['pred_'+name])
                    for name, mask in masks.items():
                        np.testing.assert_array_equal(mask, archived[name])
                cases_checked += 1
                rows_checked += n
        self.assertEqual(cases_checked, 24)
        print(f'ARCHIVE_PREDICTIONS: {cases_checked} cases, {rows_checked} rows, bitwise equal', flush=True)

    def test_cached_evidence_to_features_and_offsets(self):
        old = Path(os.environ['V28_SOURCE'])
        for case_name in ('scan24_window_left__native', 'scan37_scissor_cross__native'):
            case = old/'real_results'/case_name
            with np.load(case/'evidence.npz', allow_pickle=False) as z:
                scores = z['scores']
            with np.load(case/'state.npz', allow_pickle=False) as z:
                state = {key: z[key] for key in z.files}
            with np.load(case/'features.npz', allow_pickle=False) as z:
                features = {key: z[key] for key in z.files}
            solutions = solve_surfacelets(scores, state['offsets'])
            ids, weights = state['interpolation_ids'], state['interpolation_weights']
            anchor_pre = features['pre'][state['anchor_ids']]
            for arm in ('A_all', 'B_all', 'A_fit', 'B_fit'):
                full_offset = np.sum(solutions[arm]['offset'][ids]*weights, axis=1)
                np.testing.assert_array_equal(full_offset, state[arm+'_offset'])
                post = _post_features(solutions[arm], anchor_pre)
                full = np.sum(post[ids]*weights[..., None], axis=1).astype(np.float32)
                # Archived anchor input features are float32, unlike the original float64 pre-interpolation values.
                np.testing.assert_allclose(full[:, :8], features[arm+'_post'][:, :8], rtol=2e-7, atol=1e-6)
                np.testing.assert_array_equal(full[:, 8:], features[arm+'_post'][:, 8:])


if __name__ == '__main__':
    unittest.main(verbosity=2)

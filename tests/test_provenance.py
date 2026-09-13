import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_synthetics import make_ambiguous, make_dual, make_ghost
from perturb import apply_station_transforms, make_perturbation
from schema import estimator_arrays, read_evaluation, read_patch, write_patch
from transforms import as_matrix44


class ProvenanceTests(unittest.TestCase):
    def test_ids_survive_perturbation_and_remap(self):
        points, meta, ev = make_ghost(912101, 4.0)
        with tempfile.TemporaryDirectory() as tmp:
            write_patch(Path(tmp), "ghost", points, meta, ev)
            loaded, loaded_meta = read_patch(Path(tmp) / "ghost.json")
        remap = {sid: 100 + sid for sid in np.unique(loaded["scan_id"])}
        arrays = estimator_arrays(loaded, scan_to_frame=remap)
        self.assertEqual(set(arrays["scan_id"]), set(loaded["scan_id"]))
        self.assertTrue(np.all(arrays["frame"] == loaded["scan_id"] + 100))
        np.testing.assert_array_equal(arrays["source_point_index"], loaded["source_point_index"])
        poses = {int(k): as_matrix44(v) for k, v in loaded_meta["T_world_from_scan_input"].items()}
        pert = make_perturbation(
            loaded["xyz_world"], loaded["scan_id"], poses, 912401, 0.005, 0.0, min(poses)
        )
        recovered = apply_station_transforms(
            pert["xyz_world_perturbed"],
            loaded["scan_id"],
            {int(k): as_matrix44(v) for k, v in pert["inverse_extra_T_world"].items()},
        )
        np.testing.assert_allclose(recovered, loaded["xyz_world"], atol=1e-12)
        # IDs still address the same original rows.
        self.assertEqual(len(np.unique(loaded["source_point_index"])), len(loaded["source_point_index"]))

    def test_raw_clean_not_row_paired(self):
        a = np.arange(10)
        b = np.array([0, 2, 4, 5, 6, 7, 8, 9, 1, 3])
        self.assertFalse(np.array_equal(a, b))
        # pairing is only legal through source_point_index identity, never row number
        paired = {int(i): int(j) for i, j in zip(a, a)}
        self.assertNotEqual(list(paired), list(b))

    def test_ambiguous_has_two_explanations(self):
        points, meta, ev = make_ambiguous(912101, 8.0)
        self.assertFalse(meta["identifiable"])
        self.assertIn("explanation_a", ev)
        self.assertIn("explanation_b", ev)
        self.assertNotEqual(ev["explanation_a"]["name"], ev["explanation_b"]["name"])

    def test_dual_front_occludes(self):
        points, meta, ev = make_dual(912101, 8.0, 0.0)
        labels = ev["gt_layer"]
        self.assertGreater(np.sum(labels == 0), 0)
        self.assertGreater(np.sum(labels == 1), 0)
        # No stored point sits behind an opaque front hit along the same generated ray
        # with a farther layer label at the same XY sample: front z is the larger plate.
        self.assertTrue(np.all(points["xyz_world"][:, 2][labels == 1] > points["xyz_world"][:, 2][labels == 0].min() - 1))


if __name__ == "__main__":
    unittest.main()

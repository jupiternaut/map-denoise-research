import io
import struct
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import camera_mapping as m


class MappingMathTests(unittest.TestCase):
    def test_half_affine_and_corner_conversion(self):
        A = m.resize_affine([1554, 1162], [777, 581])
        np.testing.assert_array_equal(A, [[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])
        np.testing.assert_array_equal(A @ m.T_COLMAP_TO_ARRAY,
                                      [[.5, 0, -.5], [0, .5, -.5], [0, 0, 1]])
        raw = np.array([777., 581., 1.])
        np.testing.assert_array_equal(A @ m.T_COLMAP_TO_ARRAY @ raw, [388., 290., 1.])

    def test_pillow_ramp_resize_pixel_centres(self):
        ramp = np.tile(np.arange(16, dtype=np.float32), (12, 1))
        out = np.asarray(Image.fromarray(ramp).resize((8, 6), Image.Resampling.BILINEAR))
        np.testing.assert_allclose(out[2, 1:-1], 2 * np.arange(1, 7) + .5, atol=1e-6)

    def test_fixed_pose_and_independent_projection(self):
        K = np.array([[2892., 0., 777.], [0., 2883., 581.], [0., 0., 1.]])
        R = Rotation.from_euler('xyz', [20, -35, 15], degrees=True).as_matrix()
        C = np.array([593., -120., 373.])
        raw, full = m.corrected_projection(K, R, C)
        A = m.resize_affine([1554, 1162], [777, 581]); half = A @ full
        examples = m.projection_examples(K, R, C, raw, full, half, A)
        np.testing.assert_allclose(examples[0]['manual_half_array_xy'], [388., 290.], atol=1e-12)
        for P in (raw, full, half):
            np.testing.assert_allclose(-np.linalg.solve(P[:, :3], P[:, 3]), C, atol=1e-10)
        _, Rout, Cout = m.decompose(raw)
        np.testing.assert_allclose(Rout, R, atol=1e-12)
        np.testing.assert_allclose(Cout, C, atol=1e-10)

    def test_binary_camera_decoder(self):
        value = struct.pack('<QiiQQdddd', 1, 1, 1, 1554, 1162, 2892., 2883., 777., 581.)
        with patch('pathlib.Path.open', return_value=io.BytesIO(value)):
            camera = m.read_cameras('/not_a_real_file')[1]
        np.testing.assert_array_equal(camera['K'], [[2892., 0, 777.], [0, 2883., 581.], [0, 0, 1]])

    def test_decoder_rejects_distorted_model(self):
        value = struct.pack('<QiiQQ', 1, 1, 4, 1554, 1162)
        with patch('pathlib.Path.open', return_value=io.BytesIO(value)):
            with self.assertRaisesRegex(ValueError, 'PINHOLE'):
                m.read_cameras('/not_a_real_file')

    def test_pose_decoder_uses_2d_tracks_only(self):
        value = struct.pack('<Qidddddddi', 1, 23, 1., 0., 0., 0., 1., 2., 3., 1)
        value += b'0022.png\0' + struct.pack('<Qddq', 1, 12.5, 20.5, 123)
        with patch('pathlib.Path.open', return_value=io.BytesIO(value)):
            pose = m.read_image_metadata('/not_a_real_file')['0022.png']
        np.testing.assert_array_equal(pose['center'], [-1., -2., -3.])
        np.testing.assert_array_equal(pose['obs']['xy'], [[12.5, 20.5]])

    def test_epipolar_corner_array_equivalence(self):
        K = np.array([[1000., 0, 777.], [0, 1100., 581.], [0, 0, 1.]])
        R0 = np.eye(3); R1 = Rotation.from_euler('y', 12, degrees=True).as_matrix()
        C0 = np.zeros(3); C1 = np.array([80., 5., 10.])
        F = m.fundamental(K, R0, C0, K, R1, C1)
        Ki = m.T_COLMAP_TO_ARRAY @ K
        Fi = m.fundamental(Ki, R0, C0, Ki, R1, C1)
        uv = np.array([[650., 500.], [720., 555.]])
        xy = np.array([[775., 500.], [800., 552.]])
        np.testing.assert_allclose(m.residual(F, uv, xy), m.residual(Fi, uv - .5, xy - .5), atol=1e-12)


if __name__ == '__main__':
    unittest.main()

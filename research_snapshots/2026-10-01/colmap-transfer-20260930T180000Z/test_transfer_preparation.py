"""Pre-run synthetic-only and input-identity tests. No GT or MVS execution."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent


def camera_tests():
    import prepare_transfer as p
    plan = json.loads((ROOT / 'PLAN.json').read_text())
    assert set(plan['scenes']) == {'118', '122'}
    total = 0
    for scene in plan['scenes'].values():
        for roi in scene['rois']:
            expected, _ = p.query_pixels(roi['box_xyxy_original'])
            assert np.array_equal(expected, roi['pixel_xy'])
            assert len(set(map(tuple, expected))) == 128
            total += len(expected)
        for camera in scene['cameras'].values():
            P = np.array(camera['P']); C = np.array(camera['center'])
            K = np.array(camera['K_half']); raw = np.array(camera['K_raw'])
            assert np.allclose(K[0, 2], raw[0, 2] / 2 - .5)
            uv = np.array([[0, 0], [388, 290], [776, 580]], float)
            rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(P[:, :3]).T
            xyz = C + np.array([600., 700., 800.])[:, None] * rays
            projected = xyz @ P[:, :3].T + P[:, 3]
            assert np.max(np.abs(projected[:, :2] / projected[:, 2:] - uv)) < 1e-9
    assert total == 512
    print('PASS camera/query checks: 10 cameras, 512 fixed unique per-ROI queries')


def cpu_tests():
    import cpu_baseline as c
    P = np.column_stack((np.eye(3), np.zeros(3)))
    pixels = np.array([[0., 0.], [1., 2.], [3., 4.], [5., 6.]])
    scores = np.array([[.1, np.nan, .1, .1], [.9, np.nan, .5, .9], [.5, np.nan, .2, .3]])
    variance = np.array([.1, .1, .1, 0.])
    xyz, valid, depth, best = c.prediction(scores, variance, np.array([10., 11., 12.]), pixels, P)
    assert np.array_equal(valid, [True, False, False, False])
    assert np.isnan(xyz[~valid]).all() and np.isnan(depth[~valid]).all()
    assert abs(depth[0] - (11. + 1. / 6.)) < 1e-12
    assert np.allclose(xyz[0], [0, 0, depth[0]])
    # Omitting other pixels cannot change any independent patch score.
    y, x = np.mgrid[:64, :64]
    image = ((x * 17 + y * 31 + x * y) % 256).astype(np.uint8)
    cameras = [P.copy() for _ in range(4)]
    grids = np.array([[16., 16.], [24., 24.], [32., 32.]])
    a, av = c.plane_scores(image, [image] * 4, P, cameras, grids, np.array([10., 11., 12.]))
    b, bv = c.plane_scores(image, [image] * 4, P, cameras, grids[[2, 0]], np.array([10., 11., 12.]))
    assert np.array_equal(a[:, [2, 0]], b, equal_nan=True)
    assert np.array_equal(av[[2, 0]], bv, equal_nan=True)
    print('PASS synthetic CPU tests: thresholds, invalid mask, parabola, pixel-wise score equivalence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('camera', 'cpu'))
    args = parser.parse_args()
    camera_tests() if args.phase == 'camera' else cpu_tests()

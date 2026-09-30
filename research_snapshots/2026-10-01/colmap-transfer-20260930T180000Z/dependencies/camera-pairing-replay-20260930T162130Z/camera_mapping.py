"""M0: deterministic photo camera mapping, with no GT or points3D access.

Decoder and epipolar equations reuse the reviewed old camera diagnosis.
R/C remain the supplied world_mat rig in mm. Raw COLMAP corner coordinates
are converted to integer-index array centres before the PIL resize affine.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket
import struct

import numpy as np
from PIL import Image
from scipy.linalg import rq
from scipy.spatial.transform import Rotation

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/camera-pairing-replay-20260930T162130Z')
OLD = Path('/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z')
DATA = ROOT.parent.parent / 'datasets'
DOCS = {
    'colmap_projection': 'https://colmap.github.io/cameras.html#projection',
    'colmap_observations': 'https://colmap.github.io/database.html#keypoints-and-descriptors',
    'pillow_coordinates': 'https://pillow.readthedocs.io/en/latest/handbook/concepts.html#coordinate-system',
}
T_COLMAP_TO_ARRAY = np.array([[1., 0., -.5], [0., 1., -.5], [0., 0., 1.]])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def check_target():
    if socket.gethostname() != 'liekkas' or Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Wrong host or mapping workspace')


def exact_read(f, size):
    value = f.read(size)
    if len(value) != size:
        raise ValueError('Truncated camera/2D metadata')
    return value


def read_cameras(path):
    result = {}
    with Path(path).open('rb') as f:
        count, = struct.unpack('<Q', exact_read(f, 8))
        for _ in range(count):
            cid, model, width, height = struct.unpack('<iiQQ', exact_read(f, 24))
            if model != 1 or cid <= 0 or cid in result:
                raise ValueError('Only unique positive PINHOLE camera IDs supported')
            fx, fy, cx, cy = struct.unpack('<dddd', exact_read(f, 32))
            if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy, width, height) <= 0:
                raise ValueError('Invalid camera parameters')
            result[cid] = dict(width=width, height=height, model='PINHOLE',
                               K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]))
        if f.read(1):
            raise ValueError('Trailing camera bytes')
    return result


def read_image_metadata(path):
    """Read poses and stored 2D observations/track IDs, never 3D coordinates."""
    result = {}
    with Path(path).open('rb') as f:
        count, = struct.unpack('<Q', exact_read(f, 8))
        for _ in range(count):
            data = struct.unpack('<idddddddi', exact_read(f, 64))
            name = bytearray()
            while True:
                c = exact_read(f, 1)
                if c == b'\0':
                    break
                name.extend(c)
            name = name.decode()
            size, = struct.unpack('<Q', exact_read(f, 8))
            obs = np.frombuffer(exact_read(f, size * 24),
                                dtype=[('xy', '<f8', (2,)), ('id', '<i8')]).copy()
            q = np.array(data[1:5])
            if not np.isclose(q @ q, 1., atol=1e-5) or name in result:
                raise ValueError('Invalid/duplicate stored pose')
            R = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
            t = np.array(data[5:8])
            result[name] = dict(image_id=data[0], camera_id=data[8], R=R,
                                center=-R.T @ t, obs=obs)
        if f.read(1):
            raise ValueError('Trailing pose/2D bytes')
    return result


def decompose(P):
    K, R = rq(P[:, :3])
    D = np.diag(np.where(np.diag(K) < 0, -1., 1.))
    K, R = K @ D, D @ R
    K /= K[2, 2]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-10) or np.linalg.det(R) < 0:
        raise ValueError('Not a proper canonical world_mat rotation')
    return K, R, -np.linalg.solve(P[:, :3], P[:, 3])


def resize_affine(full_wh, half_wh):
    sx, sy = np.asarray(half_wh, float) / np.asarray(full_wh, float)
    return np.array([[sx, 0., (sx - 1.) / 2.], [0., sy, (sy - 1.) / 2.], [0., 0., 1.]])


def corrected_projection(K_colmap, R, center):
    raw = K_colmap @ np.column_stack((R, -R @ center))
    return raw, T_COLMAP_TO_ARRAY @ raw


def project(P, X):
    h = np.r_[X, 1.] @ P.T
    return h[:2] / h[2]


def skew(t):
    x, y, z = t
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def fundamental(K0, R0, C0, K1, R1, C1):
    return np.linalg.solve(K1.T, skew(R1 @ (C0 - C1)) @ (R1 @ R0.T)) @ np.linalg.inv(K0)


def residual(F, uv, xy):
    a, b = np.column_stack((uv, np.ones(len(uv)))), np.column_stack((xy, np.ones(len(xy))))
    Fa, Fb = a @ F.T, b @ F
    return np.abs(np.sum(b * Fa, 1)) / np.maximum(
        np.sqrt(np.square(Fa[:, :2]).sum(1) + np.square(Fb[:, :2]).sum(1)), 1e-30)


def stats(values):
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty or nonfinite epipolar residuals')
    return dict(zip(('min', 'median', 'q95', 'max'), np.quantile(values, [0, .5, .95, 1]).tolist()))


def projection_examples(K, R, C, raw, full, half, A):
    examples = []
    for xyz in ([0., 0., 700.], [30., -20., 650.], [-55., 40., 800.]):
        xyz = np.array(xyz)
        X = C + R.T @ xyz
        manual_raw = np.array([K[0, 0] * xyz[0] / xyz[2] + K[0, 2],
                               K[1, 1] * xyz[1] / xyz[2] + K[1, 2]])
        manual_full = manual_raw - .5
        manual_half = (manual_full + .5) * np.diag(A)[:2] - .5
        uv_full, uv_half = project(full, X), project(half, X)
        ray = np.linalg.solve(full[:, :3], np.r_[uv_full, 1.])
        back = C + xyz[2] * ray
        errors = [np.linalg.norm(project(raw, X) - manual_raw), np.linalg.norm(uv_full - manual_full),
                  np.linalg.norm(uv_half - manual_half), np.linalg.norm(back - X)]
        if max(errors) > 1e-8:
            raise ValueError('Independent projection/backprojection example failed')
        examples.append(dict(camera_xyz_mm=xyz.tolist(), fixture_world_xyz_mm=X.tolist(),
            manual_raw_colmap_xy=manual_raw.tolist(), manual_full_array_xy=manual_full.tolist(),
            manual_half_array_xy=manual_half.tolist(), matrix_full_array_xy=uv_full.tolist(),
            matrix_half_array_xy=uv_half.tolist(), errors_raw_full_half_px_back_mm=errors))
    return examples


def provenance_records():
    photos_path = DATA / 'loss-alignment-v23/DOWNLOAD_RETRY_1789286634053014484.json'
    binary_path = DATA / 'loss-alignment-v23/CALIBRATION_DOWNLOAD_1789286845432733581.json'
    npz24_path = DATA / 'real-closure-v21/CAMERA_PROBE.json'
    npz37_path = DATA / 'reconstruction-v22-scan37/DOWNLOAD_MANIFEST.json'
    photos, binary = load(photos_path), load(binary_path)
    npz24, npz37 = load(npz24_path), load(npz37_path)
    entries = {e['path']: e for e in photos['files']}
    entries.update({e['path']: e for e in binary['files'] if Path(e['path']).name != 'points3D.bin'})
    entries[npz24['path']] = npz24
    entries.update({e['path']: e for e in npz37['files'] if e.get('kind') == 'camera'})
    return (photos_path, binary_path, npz24_path, npz37_path), entries, photos['url']


def build_mapping():
    check_target()
    plan = load(OLD / 'PLAN.json')
    records, provenance, archive_url = provenance_records()
    sources = {}
    def bind(path):
        path = Path(path).resolve()
        if path.name.lower() in ('points3d.bin', 'points3d.txt') or path.suffix.lower() in ('.ply', '.pcd'):
            raise PermissionError('GT/3D geometry forbidden in M0')
        digest = sha(path)
        sources[str(path)] = digest
        if str(path) in provenance and provenance[str(path)]['sha256'] != digest:
            raise ValueError('Acquisition manifest hash mismatch: ' + str(path))
        return digest
    for path in [Path(__file__), ROOT / 'test_camera_mapping.py', ROOT / 'AGENTS.md',
                 ROOT / 'refine-logs/EXPERIMENT_PLAN.md', OLD / 'PLAN.json',
                 OLD / 'evaluation/diagnose_camera_pairing.py', *records]:
        bind(path)
    for path in plan['source_files']:
        if bind(path) != plan['source_files'][path]:
            raise ValueError('Old plan input identity changed')
    mapping = dict(schema='camera_mapping.v1.array_center', status='FROZEN', host='liekkas', root=str(ROOT),
        source_run=str(OLD), coordinate_units='mm', gt_accessed=False, points3D_accessed=False,
        pose_fit=False, intrinsic_fit=False, sources=sources, primary_coordinate_sources=DOCS,
        source_metadata=[str(p) for p in records],
        photo_and_npz_archive_url=archive_url,
        provenance_caveat='Acquisition member/hash records and stored PINHOLE 2D tracks identify the operational pairing. Original sensor/rectification/crop recipe and full SfM provenance are unavailable. No crop inferred.',
        pixel_convention=dict(raw_colmap='upper-left pixel centre (0.5,0.5)', array='upper-left pixel centre (0,0)',
            T_colmap_to_full_array=T_COLMAP_TO_ARRAY.tolist(),
            full_array_to_half_array='(u+0.5)*sx-0.5, (v+0.5)*sy-0.5',
            raw_colmap_to_half_array='sx*u_colmap-0.5, sy*v_colmap-0.5',
            old_projection='Unchanged old matrix as used by integer-index sampler; do not additionally shift W0.'),
        scenes={})
    for sid, spec in plan['scenes'].items():
        binary_root = Path(spec['images_dir']).parent / 'sparse/0'
        cameras_file, poses_file = binary_root / 'cameras.bin', binary_root / 'images.bin'
        bind(cameras_file); bind(poses_file); bind(spec['camera_file'])
        cameras, poses = read_cameras(cameras_file), read_image_metadata(poses_file)
        scene = dict(camera_file=spec['camera_file'], image_dir=spec['images_dir'],
                     camera_binary=str(cameras_file), pose_2d_binary=str(poses_file),
                     registered_images=len(poses), views={}, epipolar_pairs=[])
        models = {}
        with np.load(spec['camera_file'], allow_pickle=False) as z:
            for name in [spec['reference']] + spec['Q'] + spec['C'] + spec['H']:
                old = spec['cameras'][name]
                path = Path(old['image_path']); bind(path)
                if name not in poses or provenance[str(path)]['member'] != f'DTU/scan{sid}/images/{name}':
                    raise ValueError('Image name/member identity mismatch')
                pose = poses[name]; camera = cameras[pose['camera_id']]
                with Image.open(path) as im:
                    wh = list(im.size)
                hh = [old['width'], old['height']]
                if wh != [camera['width'], camera['height']] or hh != [wh[0] // 2, wh[1] // 2]:
                    raise ValueError('Image/camera/resize dimensions disagree')
                index = int(Path(name).stem)
                Pold = z[f'world_mat_{index}'][:3].astype(float)
                Kold, R, Ccalc = decompose(Pold)
                C = np.array(old['center'], float)
                if not np.allclose(C, Ccalc, rtol=0, atol=1e-8):
                    raise ValueError('Old plan world centre mismatch')
                A = resize_affine(wh, hh)
                if not np.allclose(A @ Pold, old['P'], rtol=0, atol=1e-8):
                    raise ValueError('Old P half-res identity mismatch')
                raw, full = corrected_projection(camera['K'], R, C)
                half = A @ full
                Cbinary = (z[f'scale_mat_{index}'].astype(float) @ np.r_[pose['center'], 1])[:3]
                rdiff = float(Rotation.from_matrix(R @ pose['R'].T).magnitude() * 180 / np.pi)
                cdiff = float(np.linalg.norm(C - Cbinary))
                if rdiff > 1e-3 or cdiff > .01:
                    raise ValueError('Stored camera poses do not identify the same fixed rig')
                scene['views'][name] = dict(P_full_old=Pold.tolist(), P_full_corrected=full.tolist(),
                    P_half_old=old['P'], P_half_corrected=half.tolist(), center_mm=old['center'],
                    image_path=str(path), full_size_wh=wh, half_size_wh=hh,
                    P_full_raw_colmap_corrected=raw.tolist(), K_raw_colmap=camera['K'].tolist(),
                    K_full_array=(T_COLMAP_TO_ARRAY @ camera['K']).tolist(), K_full_old=Kold.tolist(),
                    R_world_to_camera=R.tolist(), T_colmap_to_full_array=T_COLMAP_TO_ARRAY.tolist(),
                    A_full_array_to_half_array=A.tolist(), A_raw_colmap_to_half_array=(A @ T_COLMAP_TO_ARRAY).tolist(),
                    world_mat_index=index, colmap_image_id=pose['image_id'], colmap_camera_id=pose['camera_id'],
                    camera_model=camera['model'], distortion_parameters=[],
                    image_source=provenance[str(path)], camera_source=provenance[str(cameras_file)],
                    world_mat_source=provenance[spec['camera_file']],
                    identity_checks=dict(rotation_delta_degrees=rdiff, center_delta_mm=cdiff),
                    independent_projection_examples=projection_examples(camera['K'], R, C, raw, full, half, A))
                models[name] = (Kold, R, C, T_COLMAP_TO_ARRAY @ camera['K'], camera['K'])
        ref = spec['reference']; a = poses[ref]['obs']; a = a[a['id'] >= 0]
        for group in ('Q', 'C', 'H'):
            for name in spec[group]:
                b = poses[name]['obs']; b = b[b['id'] >= 0]
                ids, ia, ib = np.intersect1d(a['id'], b['id'], return_indices=True)
                uvraw, xyraw = a['xy'][ia], b['xy'][ib]
                uv, xy = uvraw - .5, xyraw - .5
                K0, R0, C0, Ki0, Kr0 = models[ref]
                K1, R1, C1, Ki1, Kr1 = models[name]
                oldres = residual(fundamental(K0, R0, C0, K1, R1, C1), uv, xy)
                newres = residual(fundamental(Ki0, R0, C0, Ki1, R1, C1), uv, xy)
                rawres = residual(fundamental(Kr0, R0, C0, Kr1, R1, C1), uvraw, xyraw)
                if not np.allclose(newres, rawres, rtol=0, atol=1e-9):
                    raise ValueError('Corner/array epipolar coordinate equivalence failed')
                scene['epipolar_pairs'].append(dict(reference=ref, source=name, group=group,
                    registered_track_pairs=len(ids), coordinate_basis='full-resolution integer-index array centres',
                    old_sampson_px=stats(oldres), corrected_sampson_px=stats(newres),
                    raw_colmap_equivalent_max_abs_difference=float(np.max(np.abs(newres - rawres)))))
        mapping['scenes'][sid] = scene
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError('Source changed during M0')
    return mapping


def write_exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify():
    check_target()
    mapping_path = ROOT / 'CAMERA_MAPPING.json'
    mapping, plan = load(mapping_path), load(ROOT / 'PLAN_CORRECTED.json')
    if plan['camera_mapping_sha256'] != sha(mapping_path) or mapping['status'] != 'FROZEN':
        raise ValueError('Mapping hash/status mismatch')
    for path, digest in mapping['sources'].items():
        if sha(path) != digest:
            raise ValueError('Mapping source hash mismatch: ' + path)
    if plan['root'] != str(ROOT) or plan['source_run'] != str(OLD):
        raise ValueError('Plan root mismatch')
    for sid, scene in mapping['scenes'].items():
        for name, view in scene['views'].items():
            cam = plan['scenes'][sid]['cameras'][name]
            if cam['P'] != view['P_half_corrected'] or cam['center'] != view['center_mm']:
                raise ValueError('Corrected plan/mapping camera identity mismatch')
    print('MAPPING_VERIFIED', sha(mapping_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.build == args.verify:
        parser.error('Choose exactly one of --build/--verify')
    if args.verify:
        return verify()
    if any((ROOT / p).exists() for p in ('CAMERA_MAPPING.json', 'PLAN_CORRECTED.json')):
        raise FileExistsError('Frozen M0 outputs already exist; refusing overwrite')
    mapping = build_mapping()
    write_exclusive(ROOT / 'CAMERA_MAPPING.json', mapping)
    plan = copy.deepcopy(load(OLD / 'PLAN.json'))
    plan.update(schema='camera_pairing_plan.v1.array_center', root=str(ROOT), source_run=str(OLD),
        camera_mapping=str(ROOT / 'CAMERA_MAPPING.json'), camera_mapping_sha256=sha(ROOT / 'CAMERA_MAPPING.json'),
        projection_authority='CAMERA_MAPPING.json P_half_corrected; do not overwrite from old world_mat',
        calibration_condition='Fixed provided world_mat R/C in mm; source-associated PINHOLE K, COLMAP corner-to-array conversion; no GT fit; full pose provenance unverified',
        pixel_convention=mapping['pixel_convention'], source_files=dict(mapping['sources']))
    plan['source_files'][str(ROOT / 'CAMERA_MAPPING.json')] = plan['camera_mapping_sha256']
    for sid, spec in plan['scenes'].items():
        spec['projection_authority'] = plan['projection_authority']
        for name, cam in spec['cameras'].items():
            cam['P'] = mapping['scenes'][sid]['views'][name]['P_half_corrected']
    for case in plan['cases']:
        relative = Path(case['output_dir']).relative_to(OLD)
        case['output_dir'] = str(ROOT / relative)
    write_exclusive(ROOT / 'PLAN_CORRECTED.json', plan)
    verify()
    print('M0_COMPLETE', len(mapping['scenes']), sum(len(v['views']) for v in mapping['scenes'].values()))


if __name__ == '__main__':
    main()

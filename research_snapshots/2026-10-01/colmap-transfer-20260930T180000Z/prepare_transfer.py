"""Freeze supplied cameras and photo-only fixed query grids, without GT."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from pathlib import Path
import socket

import numpy as np
from PIL import Image

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z')
MAPPING = Path('/srv/slam-research/grf/map-denoise/runs/camera-pairing-replay-20260930T162130Z/camera_mapping.py')
spec = importlib.util.spec_from_file_location('_frozen_camera_mapping', MAPPING)
mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapping)
ROIS = {
    118: [('upper_fold', [480, 240, 1040, 680]), ('base_ridge', [480, 760, 1040, 1000])],
    122: [('feather', [520, 200, 960, 560]), ('book_edge', [500, 600, 1060, 860])],
}
REFERENCE = '0022.png'
Q = ['0016.png', '0035.png', '0017.png', '0007.png']


def save(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def query_pixels(box):
    x0, y0, x1, y1 = box
    xx, yy = np.meshgrid(np.arange(x0 / 2 + 4, x1 / 2 - 4, 8., dtype=np.float64),
                         np.arange(y0 / 2 + 4, y1 / 2 - 4, 8., dtype=np.float64))
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    order = sorted(range(len(pixels)), key=lambda i: hashlib.sha256(pixels[i].tobytes()).digest())[:128]
    selected = pixels[order]
    if len(selected) != 128 or not np.array_equal(selected, selected.astype(np.int64)):
        raise ValueError('Expected 128 integer grid points')
    return selected.astype(np.int64), len(pixels)


def build():
    if socket.gethostname() != 'liekkas' or Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Wrong target identity')
    if (ROOT / 'PLAN.json').exists() or (ROOT / 'CAMERA_CHECKS.json').exists():
        raise FileExistsError('Camera/query plan already frozen')
    manifest_path = ROOT / 'INPUT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = {r['path']: r for r in manifest['files']}
    source_files = {str(path): mapping.sha(path) for path in
                    (Path(__file__), MAPPING, ROOT / 'AGENTS.md', manifest_path,
                     ROOT / 'SCENE_SELECTION.json')}
    for filename, entry in inputs.items():
        if mapping.sha(filename) != entry['sha256'] or Path(filename).stat().st_size != entry['bytes']:
            raise RuntimeError('Acquired input changed: ' + filename)
        source_files[filename] = entry['sha256']
    scenes, checks = {}, {}
    for sid, rois in ROIS.items():
        folder = ROOT / 'inputs' / f'scan{sid}'
        raw_cameras = mapping.read_cameras(folder / 'sparse/0/cameras.bin')
        poses = mapping.read_image_metadata(folder / 'sparse/0/images.bin')
        cameras, scene_checks = {}, {}
        with np.load(folder / 'cameras.npz', allow_pickle=False) as z:
            for name in [REFERENCE] + Q:
                path = folder / 'images' / name
                pose = poses[name]
                cam = raw_cameras[pose['camera_id']]
                index = int(Path(name).stem)
                _, R, center = mapping.decompose(z[f'world_mat_{index}'][:3].astype(float))
                center_binary = (z[f'scale_mat_{index}'].astype(float) @ np.r_[pose['center'], 1.])[:3]
                rdelta = float(mapping.Rotation.from_matrix(R @ pose['R'].T).magnitude() * 180 / np.pi)
                cdelta = float(np.linalg.norm(center - center_binary))
                if rdelta > 1e-3 or cdelta > .01:
                    raise ValueError('Raw pose and physical world_mat disagree')
                with Image.open(path) as image:
                    width, height = image.size
                if (width, height) != (cam['width'], cam['height']) or (width, height) != (1554, 1162):
                    raise ValueError('Frozen image dimensions changed')
                Kraw = cam['K'].copy()
                Khalf = Kraw.copy()
                Khalf[:2] *= .5
                Khalf[:2, 2] -= .5
                P = Khalf @ np.column_stack((R, -R @ center))
                raw, full = mapping.corrected_projection(Kraw, R, center)
                affine = mapping.resize_affine([width, height], [width // 2, height // 2])
                if not np.allclose(P, affine @ full, rtol=0, atol=1e-9):
                    raise ValueError('Camera half-array coordinate convention mismatch')
                examples = mapping.projection_examples(Kraw, R, center, raw, full, P, affine)
                cameras[name] = dict(P=P.tolist(), R=R.tolist(), center=center.tolist(),
                    K_half=Khalf.tolist(), K_raw=Kraw.tolist(), image_path=str(path),
                    width=width // 2, height=height // 2, world_mat_index=index,
                    camera_id=pose['camera_id'], image_id=pose['image_id'])
                scene_checks[name] = dict(rotation_delta_degrees=rdelta, center_delta_mm=cdelta,
                    image_source_member=inputs[str(path)]['member'], projection_examples=examples,
                    pixel_convention='K_raw first two rows /2 then principal point minus0.5; integer array uv',
                    source_pose_scale='scale_mat maps COLMAP normalized center to supplied world_mat mm')
        locked_rois = []
        for name, box in rois:
            pixels, grid_count = query_pixels(box)
            locked_rois.append(dict(id=f'scan{sid}_{name}', box_xyxy_original=box,
                pixel_xy=pixels.tolist(), grid_count=grid_count, query_count=len(pixels),
                selection='first128 SHA256(float64[x,y].tobytes()) ascending; photo ROI fixed before scores'))
        scenes[str(sid)] = dict(reference=REFERENCE, Q=Q, cameras=cameras, rois=locked_rois,
                               images_dir=str(folder / 'images'), camera_file=str(folder / 'cameras.npz'))
        checks[str(sid)] = scene_checks
    plan = dict(host='liekkas', root=str(ROOT), scenes=scenes, coordinate_units='mm',
                fixed_pixel_count=512, source_files=source_files,
                gt_accessed=False, points3D_accessed=False, pose_fit=False,
                note='New same-family DTU scenes; ROIs photo-selected once by root before depth scoring')
    save(ROOT / 'PLAN.json', plan)
    save(ROOT / 'CAMERA_CHECKS.json', dict(status='PASS', scenes=checks,
         plan_sha256=mapping.sha(ROOT / 'PLAN.json'), gt_accessed=False,
         limitation='Algebraic and supplied-rig consistency; not independently recalibrated hardware'))
    print('CAMERAS_AND_PIXELS_FROZEN', 2, 10, 512, flush=True)


if __name__ == '__main__':
    build()

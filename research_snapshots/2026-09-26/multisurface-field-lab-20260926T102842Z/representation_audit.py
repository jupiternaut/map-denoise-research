"""Read-only, reference-free audit of sealed shared-field representation.

Reads only inference arrays/metadata and original camera metadata. No laser,
evaluation masks, native parent, geometry errors, or photograph pixels are read.
Run only after all three inference scene seals exist.
"""

import json
from pathlib import Path
import struct
import time

import numpy as np
from scipy.spatial.transform import Rotation

import common as c
from v28_closeout.direct_evidence import _camera


EXPECTED_POOL = ['keep', 'old_A', 'old_B', 'half_A', 'half_B', 'minus6',
                 'minus3', 'plus3', 'plus6', 'field_K1', 'field_K2a', 'field_K2b']


def _read(stream, fmt):
    return struct.unpack('<'+fmt, stream.read(struct.calcsize('<'+fmt)))


def cameras_metadata_only(scene, view_records):
    """Read COLMAP intrinsics/extrinsics, deliberately never points3D.bin."""
    folder = c.DATA / 'closeout-confirmation-v1' / 'inputs' / f'scan{scene}'
    calibration = folder / 'cameras.npz'
    camera_file = folder / 'sparse/0/cameras.bin'
    image_file = folder / 'sparse/0/images.bin'
    with np.load(calibration, allow_pickle=False) as z:
        inverse = np.linalg.inv(z['scale_mat_0'].astype(float))
    intrinsics = {}
    with camera_file.open('rb') as stream:
        for _ in range(_read(stream, 'Q')[0]):
            cid, model, width, height = _read(stream, 'iiQQ')
            if model not in (0, 1):
                raise ValueError('unsupported calibrated pinhole camera')
            parameters = _read(stream, 'ddd' if model == 0 else 'dddd')
            fx, fy, cx, cy = (parameters[0], parameters[0], parameters[1], parameters[2]) if model == 0 else parameters
            intrinsics[cid] = (np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]), width, height)
    requested = {record['reference'] for record in view_records.values()}
    cameras = {}
    with image_file.open('rb') as stream:
        for _ in range(_read(stream, 'Q')[0]):
            values = _read(stream, 'idddddddi')
            encoded = bytearray()
            while True:
                byte = stream.read(1)
                if byte == b'\0':
                    break
                if not byte:
                    raise EOFError('camera image name')
                encoded.extend(byte)
            tracks = _read(stream, 'Q')[0]
            stream.seek(tracks*24, 1)
            name = encoded.decode()
            if name not in requested:
                continue
            rotation = Rotation.from_quat([values[2], values[3], values[4], values[1]]).as_matrix()
            translation = np.asarray(values[5:8])
            intrinsic, width, height = intrinsics[values[8]]
            projection = intrinsic @ np.column_stack([rotation, translation]) @ inverse
            center = -np.linalg.solve(projection[:, :3], projection[:, 3])
            sx, sy = (width//2)/width, (height//2)/height
            resize = np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
            cameras[name] = _camera(dict(P=resize@projection, center=center,
                                         image=np.zeros((1, 1))))
    if set(cameras) != requested:
        raise AssertionError('missing requested reference camera metadata')
    return cameras, [calibration, camera_file, image_file]


def _fraction(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def audit_case(case, camera, seal, sources):
    for filename in ('POOL.npz', 'EVIDENCE.npz', 'FIELDS.json', 'META.json'):
        path = case/filename
        digest = c.sha(path)
        if seal['files'][str(path.relative_to(case.parent))] != digest:
            raise AssertionError('case differs from scene seal: '+str(path))
        sources[str(path)] = digest
    with np.load(case/'POOL.npz', allow_pickle=False) as z:
        pool, names = z['points'], z['candidate_names'].tolist()
    if names != EXPECTED_POOL:
        raise AssertionError('unexpected candidate order')
    keys = ('uv', 'depth', 'pool_offsets', 'pool_common_count', 'field_offsets',
            'field_common_count', 'k1_support', 'k2_support', 'k2_responsibility',
            'patch_index', 'point_wta', 'single_field', 'multi_field',
            'multi_field_visibility', 'multi_field_graph')
    with np.load(case/'EVIDENCE.npz', allow_pickle=False) as z:
        evidence = {key:z[key] for key in keys}
    fields = json.loads((case/'FIELDS.json').read_text())
    meta = json.loads((case/'META.json').read_text())
    n = len(pool)
    if pool.shape != (n, 12, 3) or meta['rows'] != n:
        raise AssertionError('row or pool shape mismatch')
    p = pool[:, 0]
    relative = p-camera.center
    rays = relative/np.linalg.norm(relative, axis=1, keepdims=True)
    camera_points = relative@camera.matrix.T
    depth = camera_points[:, 2]
    uv = camera_points[:, :2]/depth[:, None]
    zrate = rays@camera.matrix[2]
    depth_error = float(np.max(np.abs(depth-evidence['depth'])))
    uv_error = float(np.max(np.abs(uv-evidence['uv'])))
    if depth_error > 1e-8 or uv_error > 1e-8 or np.any(zrate <= 0):
        raise AssertionError('independent camera reconstruction mismatch')
    all_offsets = np.column_stack([evidence['pool_offsets'], evidence['field_offsets'][:, 1:]])
    expected_pool = p[:, None, :] + all_offsets[:, :, None]*rays[:, None, :]
    ray_error = float(np.max(np.abs(pool-expected_pool)))
    if ray_error > 1e-8:
        raise AssertionError('POOL is not described by stored ray offsets')

    unknown = evidence['pool_common_count'] < 2
    if (not np.array_equal(pool[unknown, 9:], np.repeat(p[unknown, None], 3, axis=1))
            or np.any(evidence['field_offsets'][unknown] != 0)
            or np.any(evidence['point_wta'][unknown] != 0)):
        raise AssertionError('unknown pool evidence changed input geometry')
    support1, support2 = evidence['k1_support'], evidence['k2_support']
    for support, columns in ((support1, [9]), (support2, [10, 11])):
        if not np.array_equal(pool[~support][:, columns],
                              np.repeat(p[~support, None], len(columns), axis=1)):
            raise AssertionError('unsupported field did not retain input')
    fresh_unknown = evidence['field_common_count'] < 2
    for arm, support in (('single_field', support1), ('multi_field', support2),
                         ('multi_field_visibility', support2), ('multi_field_graph', support2)):
        if np.any(evidence[arm][fresh_unknown | ~support] != 0):
            raise AssertionError('unsupported final field arm did not KEEP')

    coefficient_error = 0.
    reconstructed = 0
    raw_offsets = np.full((n, 3), np.nan)
    patch_checks = []
    for family, support, columns in (('K1', support1, [0]), ('K2', support2, [1, 2])):
        for patch in fields[family]:
            ids = np.flatnonzero((evidence['patch_index'] == patch['patch']) & support)
            if patch['status'] != 'fit':
                if len(ids):
                    raise AssertionError('failed patch contains supported rows')
                continue
            if not len(ids):
                continue
            normalized = (evidence['uv'][ids]-np.asarray(patch['centre_uv']))/32.
            u, v = normalized.T
            basis = np.column_stack([np.ones(len(ids)), u, v, u*u, u*v, v*v])
            beta = np.asarray(patch['coefficients'])
            if beta.shape != (len(columns), 6):
                raise AssertionError('surface is not represented by six shared coefficients')
            zc = patch['z_center_mm']
            predicted_depth = 1./(1./zc+(basis@beta.T)/zc**2)
            unbounded = (predicted_depth-evidence['depth'][ids, None])/zrate[ids, None]
            raw_offsets[np.ix_(ids, columns)] = unbounded
            bounded = np.clip(unbounded, -6., 6.)
            positions = p[ids, None, :] + bounded[:, :, None]*rays[ids, None, :]
            observed = pool[ids][:, np.asarray(columns)+9]
            error = float(np.max(np.abs(positions-observed)))
            if error > 1e-8:
                raise AssertionError('six-coefficient field reconstruction mismatch')
            coefficient_error = max(coefficient_error, error)
            reconstructed += len(ids)*len(columns)
            patch_checks.append(dict(family=family, patch=patch['patch'], rows=len(ids),
                                     max_world_coordinate_difference_mm=error))
    supported_matrix = np.column_stack([support1, support2, support2])
    if np.any(~np.isfinite(raw_offsets[supported_matrix])):
        raise AssertionError('supported field missing coefficient reconstruction')
    lower = evidence['pool_offsets'][:, :3].min(axis=1)
    upper = evidence['pool_offsets'][:, :3].max(axis=1)
    field_offsets = evidence['field_offsets'][:, 1:]
    outside = ((field_offsets < lower[:, None]-1e-8)
               | (field_offsets > upper[:, None]+1e-8))
    separation = np.linalg.norm(pool[support2, 10]-pool[support2, 11], axis=1)
    camera_separation = separation*zrate[support2]
    responsibility = evidence['k2_responsibility']
    if (not np.allclose(responsibility[support2].sum(axis=1), 1., atol=1e-12)
            or np.any(responsibility[~support2] != 0)):
        raise AssertionError('invalid layer responsibility mass')
    near_order = field_offsets[support2, 1:].argsort(axis=1)
    near_far_mass = np.take_along_axis(responsibility[support2], near_order, axis=1).sum(axis=0)
    counts = dict(rows=n, k1_supported=int(support1.sum()), k2_supported=int(support2.sum()),
        unknown_pool_rows=int(unknown.sum()), unknown_fresh_rows=int(fresh_unknown.sum()),
        unsupported_k1_rows=int((~support1).sum()), unsupported_k2_rows=int((~support2).sum()),
        coincident_k2_rows=int(np.sum(separation < .1)),
        outside_old_interval_count=(outside & supported_matrix).sum(axis=0).tolist(),
        supported_field_count=supported_matrix.sum(axis=0).tolist(),
        raw_clip_trigger_count=((np.abs(raw_offsets)>6.) & supported_matrix).sum(axis=0).tolist(),
        at_clip_limit_count=((np.abs(field_offsets)>=6.-1e-10) & supported_matrix).sum(axis=0).tolist(),
        local_layer_responsibility_mass=responsibility[support2].sum(axis=0).tolist(),
        near_far_responsibility_mass=near_far_mass.tolist(),
        reconstructed_layer_rows=reconstructed)
    record = dict(case=case.name, scene=int(case.name.split('_')[0][4:]),
        condition=case.name.split('__')[1], counts=counts,
        k2_ray_separation_median_mm=float(np.median(separation)) if len(separation) else None,
        k2_ray_separation_p90_mm=float(np.quantile(separation, .9)) if len(separation) else None,
        k2_camera_depth_separation_median_mm=float(np.median(camera_separation)) if len(separation) else None,
        k2_camera_depth_separation_p90_mm=float(np.quantile(camera_separation, .9)) if len(separation) else None,
        k2_coincident_fraction=_fraction(counts['coincident_k2_rows'], counts['k2_supported']),
        unknown_fallback_exact=True, unsupported_field_fallback_exact=True,
        unsupported_output_routes_keep=True,
        max_camera_depth_difference_mm=depth_error, max_uv_difference_px=uv_error,
        max_stored_offset_world_difference_mm=ray_error,
        max_coefficient_world_difference_mm=coefficient_error, patch_checks=patch_checks)
    return record, separation, camera_separation


def main():
    c.check_host()
    started = time.monotonic()
    roots = [c.OUT/'inference'/f'scan{scene}' for scene in c.SCENES]
    if not all((folder/'SEALED.json').is_file() for folder in roots):
        raise RuntimeError('all three scene seals required before this audit')
    seals = {scene:c.verify_seal(folder) for scene,folder in zip(c.SCENES,roots)}
    sources = {str(folder/'SEALED.json'):c.sha(folder/'SEALED.json') for folder in roots}
    records = []
    ray_separations = {condition:[] for condition in c.CONDS}
    depth_separations = {condition:[] for condition in c.CONDS}
    for scene in c.SCENES:
        views_path = c.RESERVED/'evidence'/f'scan{scene}'/'VIEWS.json'
        views = json.loads(views_path.read_text())
        sources[str(views_path)] = c.sha(views_path)
        if sources[str(views_path)] != seals[scene]['source'][str(views_path)]:
            raise AssertionError('view metadata differs from inference source')
        cameras, calibration_files = cameras_metadata_only(scene, views)
        for path in calibration_files:
            digest = c.sha(path)
            if digest != seals[scene]['source'][str(path)]:
                raise AssertionError('camera metadata differs from inference source')
            sources[str(path)] = digest
        folder = c.OUT/'inference'/f'scan{scene}'
        cases = sorted(path for path in folder.glob('scan*__*') if path.is_dir())
        if len(cases) != 20:
            raise AssertionError('twenty sealed inference cases expected per scene')
        for case in cases:
            roi = case.name.split('__')[0]
            record, ray_sep, depth_sep = audit_case(case, cameras[views[roi]['reference']],
                                                   seals[scene], sources)
            records.append(record)
            ray_separations[record['condition']].append(ray_sep)
            depth_separations[record['condition']].append(depth_sep)
            print('REPRESENTATION AUDITED', case.name, flush=True)
    condition_summary = {}
    scalar_counts = ('rows', 'k1_supported', 'k2_supported', 'unknown_pool_rows',
                     'unknown_fresh_rows', 'unsupported_k1_rows', 'unsupported_k2_rows',
                     'coincident_k2_rows', 'reconstructed_layer_rows')
    vector_counts = ('outside_old_interval_count', 'supported_field_count',
                     'raw_clip_trigger_count', 'at_clip_limit_count',
                     'local_layer_responsibility_mass', 'near_far_responsibility_mass')
    for condition in c.CONDS:
        rr = [record for record in records if record['condition'] == condition]
        if len(rr) != 12:
            raise AssertionError('twelve ROI records required per condition')
        count = {key:sum(record['counts'][key] for record in rr) for key in scalar_counts}
        count.update({key:np.asarray([record['counts'][key] for record in rr]).sum(axis=0).tolist()
                      for key in vector_counts})
        ray_sep = np.concatenate(ray_separations[condition])
        depth_sep = np.concatenate(depth_separations[condition])
        denominator = np.asarray(count['supported_field_count'])
        fractions = {}
        for key in ('outside_old_interval_count', 'raw_clip_trigger_count', 'at_clip_limit_count'):
            fractions[key.replace('_count', '_fraction')] = [
                _fraction(num, den) for num, den in zip(count[key], denominator)]
        condition_summary[condition] = dict(counts=count, **fractions,
            k2_ray_separation_median_mm=float(np.median(ray_sep)) if len(ray_sep) else None,
            k2_ray_separation_p90_mm=float(np.quantile(ray_sep,.9)) if len(ray_sep) else None,
            k2_camera_depth_separation_median_mm=float(np.median(depth_sep)) if len(depth_sep) else None,
            k2_camera_depth_separation_p90_mm=float(np.quantile(depth_sep,.9)) if len(depth_sep) else None,
            k2_coincident_fraction=_fraction(count['coincident_k2_rows'],count['k2_supported']),
            local_layer_responsibility_fraction=[_fraction(x,count['k2_supported']) for x in count['local_layer_responsibility_mass']],
            near_far_responsibility_fraction=[_fraction(x,count['k2_supported']) for x in count['near_far_responsibility_mass']])
    if any(c.sha(Path(path)) != digest for path,digest in sources.items()):
        raise AssertionError('source changed during representation audit')
    result = dict(status='PASS', scene_count=3, cases=len(records),
        reference_access=False, photographs_read=False, sparse_points_read=False,
        condition_summary=condition_summary, records=records, source_sha256=sources,
        seconds=time.monotonic()-started,
        aggregation='pooled eligible rows per condition; descriptive representation counts, not scene-balanced accuracy metrics',
        field_order=['K1','K2a','K2b'],
        separation='post-clipping same-ray separation, not physical normal layer thickness',
        local_layer_labels='patch-local, arbitrary; near/far masses additionally order actual predicted ray depth per row',
        reconstruction_scope='supported core rows in fitted patches only; unknown/failed fits checked separately for exact fallback',
        max_coefficient_world_difference_mm=max(record['max_coefficient_world_difference_mm'] for record in records),
        max_stored_offset_world_difference_mm=max(record['max_stored_offset_world_difference_mm'] for record in records))
    c.save_json(c.ROOT/'REPRESENTATION_AUDIT.json', result)
    lines = ['# Reference-free representation audit', '',
        'Status: PASS. All 60 cases were read only after all three scene seals existed.',
        'No laser, evaluation support, native parent, geometry errors, photograph pixels or sparse 3D points were read.', '',
        '| Condition | K2 ray gap median / p90 (mm) | Gap < 0.1 mm | Field outside old interval K1 / K2a / K2b | Raw >6mm clip rate K1 / K2a / K2b |',
        '|---|---:|---:|---:|---:|']
    for condition in c.CONDS:
        cell = condition_summary[condition]
        fmt = lambda values:'/'.join('NA' if x is None else f'{100*x:.2f}%' for x in values)
        lines.append(f"| {condition} | {cell['k2_ray_separation_median_mm']:.4f} / {cell['k2_ray_separation_p90_mm']:.4f} | {100*cell['k2_coincident_fraction']:.2f}% | {fmt(cell['outside_old_interval_fraction'])} | {fmt(cell['raw_clip_trigger_fraction'])} |")
    lines += ['', 'Counts pool eligible rows; they are not the scene-balanced geometry-quality score.',
        'Gap is the clipped separation along a reference-camera ray, not physical layer thickness.',
        'Layer IDs are local to patches; JSON also reports near/far-ordered posterior mass.',
        'Every supported core prediction was independently reconstructed from its patch coefficients,',
        'converted through original camera metadata, clipped to ±6mm, and checked against saved POOL geometry.',
        f"Maximum reconstructed world-coordinate discrepancy: {result['max_coefficient_world_difference_mm']:.3g} mm.",
        'Unknown evidence and unsupported fits retain input exactly; unsupported output labels are KEEP.',
        'These checks establish a real shared-parameter implementation, not correct physical surface identity or improved accuracy.', '']
    with (c.ROOT/'REPRESENTATION_AUDIT.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print('REPRESENTATION AUDIT PASS', len(records), 'cases', flush=True)


if __name__ == '__main__':
    main()

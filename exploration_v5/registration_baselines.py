"""Official Open3D GICP in a stated pairwise-to-anchor wrapper.

The wrapper chooses a measured anchor and a translation gauge, not reference
geometry. It adds no point-level denoiser. Not a multiway BA implementation.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial import cKDTree

PARAMETERS = {'epsilon': 0.001, 'max_iterations_per_scale': 30,
              'relative_fitness': 1e-6, 'relative_rmse': 1e-6,
              'radius_multipliers': [3.0, 1.5], 'sigma_spacing_floor': 4.0,
              'minimum_points': 8, 'minimum_correspondences': 6}


def estimate(xyz_world_m, scan_id, sigma_mm, variant='gicp'):
    started = time.perf_counter()
    p = np.asarray(xyz_world_m, dtype=float)
    f = np.asarray(scan_id)
    sigma = float(sigma_mm)
    if variant != 'gicp': raise ValueError('only gicp is implemented')
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('finite nonempty N by 3 world metres required')
    if f.shape != (len(p),) or f.dtype.kind not in 'iu':
        raise ValueError('one integer scan ID per point required')
    if not np.isfinite(sigma) or sigma <= 0: raise ValueError('positive sigma_mm required')
    ids, scans, counts = np.unique(f, return_inverse=True, return_counts=True)
    anchor = int(np.argmax(counts))  # smallest ID on a count tie
    info = dict(method='official_open3d_gicp_wrapper', variant=variant,
                scan_ids=ids.tolist(), scan_counts=counts.tolist(), anchor_scan_id=int(ids[anchor]),
                anchor_rule='largest measured point count; smallest scan ID on ties',
                parameters=dict(PARAMETERS), input_units='m', postfilter='none',
                scope='pairwise source-to-fixed-anchor GICP; not full multiway BA',
                gauge='single measured-centroid translation on successfully linked component; anchor rotation retained',
                point_count=len(p), supported_fraction=0., pairs=[])
    if len(ids) < 2 or counts[anchor] < PARAMETERS['minimum_points']:
        info.update(status='UNSUPPORTED', reason='not enough scans/anchor points', seconds=time.perf_counter()-started)
        return p.copy(), info
    import open3d as o3d
    info['open3d_version'] = o3d.__version__
    origin = p.mean(axis=0)
    local = p-origin
    spacing = []
    for sid in range(len(ids)):
        q = local[scans == sid]
        if len(q) > 1:
            d = cKDTree(q).query(q, k=2)[0][:, 1]
            spacing.extend(d[d > 1e-12].tolist())
    base = max(PARAMETERS['sigma_spacing_floor']*sigma/1000.,
               float(np.median(spacing)) if spacing else 0.)
    radii = np.asarray(PARAMETERS['radius_multipliers'])*base
    info.update(origin_world_m=origin.tolist(), measured_spacing_median_m=float(np.median(spacing)) if spacing else None,
                radii_m=radii.tolist())
    clouds = []
    for sid in range(len(ids)):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(local[scans == sid].copy())
        clouds.append(cloud)
    transforms = np.repeat(np.eye(4)[None], len(ids), axis=0)
    linked = np.zeros(len(ids), dtype=bool)
    linked[anchor] = True
    registration = o3d.pipelines.registration
    for sid in range(len(ids)):
        if sid == anchor: continue
        record = dict(source_scan_id=int(ids[sid]), target_scan_id=int(ids[anchor]), stages=[])
        if counts[sid] < PARAMETERS['minimum_points']:
            record['status'] = 'INSUFFICIENT_POINTS'
            info['pairs'].append(record)
            continue
        transform = np.eye(4)
        for radius in radii:
            result = registration.registration_generalized_icp(
                clouds[sid], clouds[anchor], float(radius), transform,
                registration.TransformationEstimationForGeneralizedICP(PARAMETERS['epsilon']),
                registration.ICPConvergenceCriteria(relative_fitness=PARAMETERS['relative_fitness'],
                    relative_rmse=PARAMETERS['relative_rmse'], max_iteration=PARAMETERS['max_iterations_per_scale']))
            transform = np.asarray(result.transformation).copy()
            if not np.isfinite(transform).all(): raise FloatingPointError('nonfinite official GICP transform')
            if not np.allclose(transform[:3,:3].T@transform[:3,:3], np.eye(3), atol=1e-6):
                raise FloatingPointError('GICP output not rigid')
            record['stages'].append(dict(radius_m=float(radius), fitness=float(result.fitness),
                inlier_rmse_m=float(result.inlier_rmse), correspondences=len(result.correspondence_set),
                transform_centered=transform.tolist()))
        ok = len(result.correspondence_set) >= PARAMETERS['minimum_correspondences'] and result.fitness > 0.
        record['status'] = 'APPLY' if ok else 'NO_CORRESPONDENCE_IDENTITY'
        record['raw_final_transform_centered'] = transform.tolist()
        if ok:
            transforms[sid] = transform
            linked[sid] = True
        info['pairs'].append(record)
    output = local.copy()
    for sid in np.flatnonzero(linked):
        take = scans == sid
        transform = transforms[sid]
        output[take] = local[take]@transform[:3,:3].T + transform[:3,3]
    component = linked[scans]
    # Input-defined coordinate convention, NOT alignment to the unperturbed cloud.
    common_shift = -(output[component]-local[component]).mean(axis=0) if linked.sum()>1 else np.zeros(3)
    output[component] += common_shift
    output += origin
    output[~component] = p[~component]
    if linked.sum() <= 1: output = p.copy()
    info.update(status='APPLY' if linked.sum()>1 else 'UNSUPPORTED',
                reason=None if linked.sum()>1 else 'no successful pair',
                successful_scan_ids=ids[linked].tolist(),
                supported_fraction=float(component.mean()) if linked.sum()>1 else 0.,
                applied_raw_transforms_centered=transforms.tolist(),
                gauge_common_translation_m=common_shift.tolist(),
                output_centroid_edit_mm=((output-p).mean(axis=0)*1000.).tolist(),
                seconds=time.perf_counter()-started)
    return output, info

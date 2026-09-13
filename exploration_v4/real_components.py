"""Frozen real-cloud component interventions; references never enter estimators."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from paths import PATCHES, RUNS, OLD_CHECKPOINT, require_liekkas
from schema import read_patch
from adapt import build_adapter, to_operator_mm, from_operator_mm

V3 = PROJECT / 'exploration_v3'
FORMAL_V3 = RUNS / 'exploration-v3-lnx049yd'
METHODS = ('identity', 'fast', 'graph_local', 'graph_shared',
           'measure_balanced', 'measure_unbalanced')
MODES = ('zero', 'normal_translation', 'tangent_translation', 'small_rotation')
SIGMA_MM = 2.0
TARGET_RMS_MM = 3.5
SEED = 912501


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def save_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(plain(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms_m(vectors):
    return float(np.sqrt(np.mean(np.sum(np.asarray(vectors) ** 2, axis=1))))


def decomposed_mm(delta, normal):
    delta = np.asarray(delta, float)
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    scalar = delta @ normal
    tangent = delta - scalar[:, None] * normal
    return {'xyz_rms_mm': 1000. * rms_m(delta),
            'normal_rms_mm': float(1000. * np.sqrt(np.mean(scalar ** 2))),
            'tangent_rms_mm': 1000. * rms_m(tangent)}


def scan_coefficients(scan_id, seed):
    ids, frame = np.unique(scan_id, return_inverse=True)
    counts = np.bincount(frame)
    if len(ids) < 2:
        raise ValueError('weighted zero-mean interventions require at least two scans')
    raw = np.random.default_rng(seed).normal(size=len(ids))
    raw -= np.average(raw, weights=counts)
    scale = np.sqrt(np.average(raw ** 2, weights=counts))
    if scale < 1e-12:
        raise ValueError('degenerate intervention coefficients')
    return ids, frame, counts, raw / scale


def construct_input(reference, scan_id, basis, mode, target_rms_mm=TARGET_RMS_MM, seed=SEED):
    """Evaluator-side known interventions on the exact same measured points."""
    reference = np.asarray(reference, float)
    basis = np.asarray(basis, float)
    if mode not in MODES:
        raise ValueError(mode)
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-9):
        raise ValueError('construction basis must be orthonormal')
    ids, frame, counts, coeff = scan_coefficients(scan_id, seed)
    matrices = np.repeat(np.eye(4)[None], len(ids), axis=0)
    output = reference.copy()
    angles = np.zeros(len(ids))
    centers = np.array([reference[frame == f].mean(axis=0) for f in range(len(ids))])
    n = basis[:, 2]
    tangent = basis[:, 0]
    rotation_axis = .6 * basis[:, 0] + .8 * basis[:, 1]
    target = float(target_rms_mm) / 1000.
    if target <= 0:
        raise ValueError('target RMS must be positive')
    if mode in ('normal_translation', 'tangent_translation'):
        axis = n if mode == 'normal_translation' else tangent
        vectors = target * coeff[:, None] * axis
        output += vectors[frame]
        matrices[:, :3, 3] = vectors
    elif mode == 'small_rotation':
        def at_angle(scale, return_matrices=False):
            trial = reference.copy()
            transforms = np.repeat(np.eye(4)[None], len(ids), axis=0)
            for f in range(len(ids)):
                r = Rotation.from_rotvec(rotation_axis * coeff[f] * scale).as_matrix()
                t = centers[f] - r @ centers[f]
                mask = frame == f
                # Centered form keeps the numerical mean displacement small.
                trial[mask] = (reference[mask] - centers[f]) @ r.T + centers[f]
                transforms[f, :3, :3] = r
                transforms[f, :3, 3] = t
            return (trial, transforms) if return_matrices else trial
        low = 0.
        high = np.deg2rad(5.) / np.max(np.abs(coeff))
        if rms_m(at_angle(high) - reference) < target:
            raise ValueError('3.5 mm RMS cannot be reached within the frozen 5 degree rotation bound')
        for _ in range(55):
            mid = .5 * (low + high)
            if rms_m(at_angle(mid) - reference) < target:
                low = mid
            else:
                high = mid
        scale = .5 * (low + high)
        output, matrices = at_angle(scale, return_matrices=True)
        angles = np.degrees(coeff * scale)
    delta = output - reference
    actual = decomposed_mm(delta, n)
    mean_per_scan = np.array([delta[frame == f].mean(axis=0) for f in range(len(ids))])
    info = dict(mode=mode, seed=seed, target_rms_mm=0. if mode == 'zero' else target_rms_mm,
                actual=actual, ids=ids, counts=counts, scan_coefficients=coeff,
                coefficient_weighted_mean=float(np.average(coeff, weights=counts)),
                coefficient_weighted_rms=float(np.sqrt(np.average(coeff ** 2, weights=counts))),
                construction_basis_world=basis, construction_normal_world=n,
                construction_tangent_world=tangent, construction_rotation_axis_world=rotation_axis,
                construction_axis_role='PCA of original measured XYZ; injection definition, not a true surface normal',
                scan_centers_world=centers, extra_world_transforms=matrices,
                rotation_angles_deg=angles, max_abs_rotation_deg=float(np.max(np.abs(angles))),
                mean_displacement_per_scan_mm=mean_per_scan * 1000.,
                all_point_mean_displacement_mm=delta.mean(axis=0) * 1000.,
                gauge='point-count-weighted zero-mean displacement; rotations additionally have zero mean in each scan',
                core_point_count=len(reference), scan_bias_translation_max_mm=float(np.max(np.linalg.norm(mean_per_scan, axis=1))) * 1000.)
    expected = 0. if mode == 'zero' else target_rms_mm
    if abs(actual['xyz_rms_mm'] - expected) > 1e-7:
        raise AssertionError('actual intervention RMS calibration failed')
    if np.linalg.norm(delta.mean(axis=0)) > 1e-10:
        raise AssertionError('weighted displacement gauge failed')
    return output, info


def reachability_diagnostic(delta, scan_id, candidate_axis):
    """Project KNOWN injected error onto the candidate's allowed scan-axis space.

    This is evaluator-only and never a candidate output. It concerns restoring
    this intervention to the original measured cloud, not denoising real truth.
    """
    normal = np.asarray(candidate_axis, float)
    normal /= np.linalg.norm(normal)
    ids, frame = np.unique(scan_id, return_inverse=True)
    counts = np.bincount(frame)
    heights = np.asarray(delta) @ normal
    coefficients = np.array([heights[frame == f].mean() for f in range(len(ids))])
    coefficients -= np.average(coefficients, weights=counts)
    removable = coefficients[frame, None] * normal
    remaining = delta - removable
    return {
        'injection_along_candidate_axis_rms_mm': float(1000. * np.sqrt(np.mean(heights ** 2))),
        'best_scan_constant_axis_removable_rms_mm': 1000. * rms_m(removable),
        'best_scan_constant_axis_remaining_rms_mm': 1000. * rms_m(remaining),
        'oracle_projection_scan_coefficients_mm': (coefficients * 1000.).tolist(),
        'projection_role': 'evaluation-only injection-space diagnostic; not a candidate or real geometry bound',
    }


def evaluate_output(output, current, reference, zero_output, axis):
    row = {}
    for prefix, delta in [('recovery', output - reference), ('zero_edit', zero_output - reference),
                          ('self_response', output - zero_output), ('input_edit', output - current),
                          ('injected', current - reference)]:
        row.update({prefix + '_' + key: value for key, value in decomposed_mm(delta, axis).items()})
    edit = np.linalg.norm(output-current, axis=1) * 1000.
    row.update(recovery_centroid_error_mm=float(1000.*np.linalg.norm((output-reference).mean(axis=0))),
               input_edit_p95_mm=float(np.quantile(edit, .95)),
               moved_fraction=float(np.mean(edit > 1e-6)),
               unchanged_exact=bool(np.array_equal(output, current)),
               independent_real_geometry='not_measured')
    return row


def load_exact(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != Path(path).resolve():
        raise RuntimeError('loaded source identity mismatch')
    return module


def invoke(method, current_xyz, scan_id, sigma_mm, modules):
    """Only this copied legal payload enters any candidate estimator."""
    xyz = np.array(current_xyz, dtype=float, copy=True)
    scan = np.array(scan_id, dtype=np.int64, copy=True)
    if method == 'identity':
        return xyz, {'status': 'IDENTITY', 'method': method}
    if method == 'fast':
        adapter = build_adapter(xyz)
        result, info = modules['old'].estimate('fast', to_operator_mm(xyz, adapter), scan, sigma_mm)
        info = dict(info, normal_world=adapter['normal_world'].tolist(),
                    runner_axis_source='current legal input PCA; attached after estimate')
        return from_operator_mm(result, adapter), info
    if method.startswith('graph_'):
        return modules['graph'].estimate(xyz, scan, sigma_mm,
                                         variant='graph' if method == 'graph_shared' else 'local_only')
    return modules['measure'].estimate(xyz, scan, sigma_mm,
                                       variant='balanced' if method == 'measure_balanced' else 'unbalanced')


def support_metadata(info, scan_id):
    ids, counts = np.unique(scan_id, return_counts=True)
    support = info.get('supported_fraction')
    if support is None and 'supported_scans' in info:
        support = float(np.average(np.asarray(info['supported_scans'], float), weights=counts))
    return dict(method_status=info.get('status', 'NOT_REPORTED'),
                method_reason=info.get('reason'), supported_point_fraction=support,
                supported_scan_fraction=info.get('supported_scan_fraction'),
                support_role='reported applicability, not independently validated accuracy',
                scan_bias_mm=info.get('bias_mm'), method_scan_ids=info.get('scan_ids', info.get('frame_ids', ids.tolist())),
                transport_mass=info.get('effective_transport_mass'),
                usable_transport_mass=info.get('usable_transport_mass'))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(plain(v)) if isinstance(v, (dict, list, np.ndarray)) else v
                             for k, v in row.items()})


def sources_and_protection(modules):
    sources = [HERE/'REAL_PROTOCOL.md', Path(__file__), HERE/'test_real_components.py',
               V3/'graph_surface.py', V3/'measure_surface.py']
    sources += [PROJECT/name for name in ('paths.py', 'schema.py', 'hashutil.py', 'adapt.py', 'transforms.py')]
    sources += [Path(p) for p in modules['old'].source_hashes()]
    sources = sorted(set(sources))
    protected = set(sources)
    protected.update(p for p in PATCHES.rglob('*') if p.is_file())
    protected.update(FORMAL_V3/name for name in ('RESULTS.csv', 'SUMMARY.json', 'SOURCE_MANIFEST.json'))
    return sources, {str(p): digest(p) for p in sorted(protected)}


def summarise(rows):
    result = []
    for mode in MODES:
        for method in METHODS:
            subset = [r for r in rows if r['mode'] == mode and r['method'] == method and r['ok']]
            record = dict(mode=mode, method=method, expected_patches=6, successful_patches=len(subset))
            for key in ('recovery_xyz_rms_mm', 'recovery_normal_rms_mm', 'recovery_tangent_rms_mm',
                        'zero_edit_xyz_rms_mm', 'self_response_xyz_rms_mm', 'input_edit_xyz_rms_mm',
                        'candidate_axis_abs_cos', 'best_scan_constant_axis_remaining_rms_mm',
                        'supported_point_fraction'):
                values = [r[key] for r in subset if r.get(key) is not None]
                if values:
                    record['median_' + key] = float(np.median(values))
                    record['mean_' + key] = float(np.mean(values))
            result.append(record)
    return result


def write_readout(dest, summary, rows):
    lines = ['# V4 真实测量分量诊断', '',
             '参考是未注入的原测量云，不是独立几何真值；下表仅衡量干预恢复。', '',
             f"{summary['successful_rows']}/{summary['rows']} 输出成功；24 输入来自 6 固定片区、2 场景。三个非零模式均按全部点标定至 3.5 mm XYZ RMS。", '',
             '|模式|方法|恢复 XYZ 中位 RMS mm|构造法向 mm|构造切向 mm|零注入改写 mm|',
             '|---|---|---:|---:|---:|---:|']
    for r in summary['aggregate']:
        if not r['successful_patches']:
            continue
        lines.append('|'+ '|'.join([r['mode'], r['method']] + [f"{r['median_'+k]:.6f}" for k in
                    ('recovery_xyz_rms_mm', 'recovery_normal_rms_mm', 'recovery_tangent_rms_mm', 'zero_edit_xyz_rms_mm')])+'|')
    lines += ['', '完整逐片区分数、方法支持率、扫描偏差和候选轴投影见 RESULTS.csv；输入构造见 evaluation/，未传给方法。', '',
              '旋转绕每扫描自身点质心，平均位移为零，因此每扫描一个常量轴向平移无法恢复其空间变化。候选轴的可达空间投影只作评价诊断，不列为候选胜者。', '',
              '绝对世界坐标评分，没有逐扫描、逐层或参考对齐。法向由输入测量估计，仅定义注入坐标，不是真值法向。不同估计器使用的方向可能不同，RESULTS.csv 中保留轴余弦与可恢复投影。', '',
              '低零注入改写和低自响应不证明去噪；小投影剩余误差也不证明实际估计器成功。measure 单独输出只有逐扫描纠偏，graph/fast 含点级去噪，不作功能无差别排名。', '',
              f"历史保护文件 {summary['protected_file_count']} 项前后相同。单线程 CPU 实测本运行 wall={summary['elapsed_s']:.3f}s；其他并行任务可能争用，不作性能排名。Peak RSS 仅本 Python 进程。", '',
              '旧 V3 混合扰动保持原样：'+str(FORMAL_V3/'RESULTS.csv'), '',
              '本轮为公开开发观察，没有独立真实几何收益结论，没有新增独立场景或未见确认数据。']
    with (dest/'READOUT.md').open('x') as handle:
        handle.write('\n'.join(lines)+'\n')


def run():
    require_liekkas()
    started = time.perf_counter()
    modules = {'old': load_exact(OLD_CHECKPOINT/'operators.py', '_v4_real_old'),
               'graph': load_exact(V3/'graph_surface.py', '_v4_real_graph'),
               'measure': load_exact(V3/'measure_surface.py', '_v4_real_measure')}
    sources, before = sources_and_protection(modules)
    dest = Path(tempfile.mkdtemp(prefix='real-components-v4-', dir=RUNS))
    print(str(dest), flush=True)
    for dirname in ('source', 'inputs', 'outputs', 'evaluation'):
        (dest/dirname).mkdir()
    save_json(dest/'PROTECTED_BEFORE.json', before)
    manifest = {}
    for i, source in enumerate(sources):
        snapshot = dest/'source'/f'{i:02d}_{source.name}'
        shutil.copyfile(source, snapshot)
        manifest[str(source)] = {'sha256': digest(source), 'snapshot': str(snapshot)}
    save_json(dest/'SOURCE_MANIFEST.json', manifest)
    warm = modules['old'].warmup(('identity', 'fast'))
    patches = sorted(PATCHES.glob('*/*.json'))
    if len(patches) != 6:
        raise RuntimeError('expected exactly the six frozen real patches')
    first, _ = read_patch(patches[0])
    for method in METHODS:
        if method not in ('identity', 'fast'):
            t = time.perf_counter()
            invoke(method, first['xyz_world'], first['scan_id'], SIGMA_MM, modules)
            warm['method_warmup_s'][method] = time.perf_counter()-t
    save_json(dest/'WARMUP.json', warm)
    rows, constructions = [], []
    for patch_index, file in enumerate(patches):
        points, metadata = read_patch(file)
        reference = points['xyz_world']
        scan_id = points['scan_id']
        source_ids = points['source_point_index']
        construction_basis = build_adapter(reference)['basis']
        construction_axis = construction_basis[:, 2]
        baseline_outputs, baseline_axes = {}, {}
        for mode in MODES:
            case = file.stem+'__'+mode
            current, construction = construct_input(reference, scan_id, construction_basis, mode, seed=SEED+patch_index)
            construction.update(case=case, source=str(file), scene=metadata['scene'])
            constructions.append(construction)
            save_json(dest/'evaluation'/f'{case}.json', construction)
            with (dest/'evaluation'/f'{case}.npz').open('xb') as handle:
                np.savez_compressed(handle, reference_measured_xyz_world=reference, injected_delta_world=current-reference)
            with (dest/'inputs'/f'{case}.npz').open('xb') as handle:
                np.savez_compressed(handle, xyz_world=current, scan_id=scan_id, source_point_index=source_ids)
            for method in METHODS:
                row = dict(case=case, patch=file.stem, scene=metadata['scene'], mode=mode, method=method,
                           n_points=len(current), n_scans=len(np.unique(scan_id)), sigma_mm=SIGMA_MM,
                           source=str(file), target_injected_rms_mm=construction['target_rms_mm'],
                           max_abs_injected_rotation_deg=construction['max_abs_rotation_deg'],
                           max_scan_mean_injected_translation_mm=construction['scan_bias_translation_max_mm'])
                t = time.perf_counter()
                try:
                    output, info = invoke(method, current, scan_id, SIGMA_MM, modules)
                    method_seconds = time.perf_counter()-t
                    output = np.asarray(output, dtype=float)
                    if output.shape != reference.shape or not np.isfinite(output).all():
                        raise ValueError('output must preserve point count and contain finite XYZ')
                    candidate_axis = info.get('normal_world')
                    if candidate_axis is not None:
                        candidate_axis = np.asarray(candidate_axis, float)
                        candidate_axis /= np.linalg.norm(candidate_axis)
                    if mode == 'zero':
                        baseline_outputs[method] = output.copy()
                        baseline_axes[method] = candidate_axis
                    if method not in baseline_outputs:
                        raise RuntimeError('same-method zero-injection baseline failed')
                    row.update(evaluate_output(output, current, reference, baseline_outputs[method], construction_axis))
                    row.update(support_metadata(info, scan_id))
                    row['candidate_axis_available'] = candidate_axis is not None
                    if candidate_axis is not None:
                        row['candidate_axis_abs_cos'] = float(abs(candidate_axis @ construction_axis))
                        row.update(reachability_diagnostic(current-reference, scan_id, candidate_axis))
                        row['edit_orthogonal_to_candidate_axis_rms_mm'] = decomposed_mm(output-current, candidate_axis)['tangent_rms_mm']
                        baseline_axis = baseline_axes.get(method)
                        if baseline_axis is not None:
                            row['candidate_axis_vs_zero_abs_cos'] = float(abs(candidate_axis @ baseline_axis))
                    _, dense = np.unique(scan_id, return_inverse=True)
                    scan_means = np.array([(output-current)[dense == f].mean(axis=0) for f in range(dense.max()+1)])
                    row['within_scan_nonconstant_edit_rms_mm'] = 1000.*rms_m(output-current-scan_means[dense])
                    row.update(ok=True, method_seconds=method_seconds,
                               exact_fallback=bool(np.array_equal(output, current)),
                               point_order_contract='same input row and source ID; estimators preserve row order')
                    path = dest/'outputs'/f'{case}__{method}.npz'
                    with path.open('xb') as handle:
                        np.savez_compressed(handle, xyz_world=output, scan_id=scan_id, source_point_index=source_ids)
                    row.update(output=str(path), output_sha256=digest(path))
                    save_json(path.with_suffix('.json'), {'info': info, 'row': row})
                except Exception as exc:
                    row.update(ok=False, error=repr(exc), method_seconds=time.perf_counter()-t)
                    save_json(dest/'outputs'/f'{case}__{method}.error.json',
                              {'error': repr(exc), 'traceback': traceback.format_exc()})
                rows.append(row)
            print(case+': '+str(sum(r['ok'] for r in rows[-len(METHODS):]))+'/'+str(len(METHODS)), flush=True)
    write_csv(dest/'RESULTS.csv', rows)
    save_json(dest/'CONSTRUCTIONS.json', constructions)
    after = {p: digest(p) for p in before}
    save_json(dest/'PROTECTED_AFTER.json', after)
    if before != after:
        raise RuntimeError('protected inputs or sources changed')
    summary = dict(run_dir=str(dest), rows=len(rows), successful_rows=sum(r['ok'] for r in rows),
                   input_count=len(constructions), real_patches=6, real_scenes=2,
                   target_nonzero_xyz_rms_mm=TARGET_RMS_MM, sigma_mm=SIGMA_MM,
                   methods=METHODS, modes=MODES, protected_files_unchanged=True, protected_file_count=len(before),
                   elapsed_s=time.perf_counter()-started, method_total_s=sum(r['method_seconds'] for r in rows),
                   peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   memory_scope='this Python process high-water RSS; not simultaneous multi-run pipeline RAM',
                   timing_scope='single-thread calls including preprocessing; concurrent task contention possible; no performance ranking',
                   threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   python=sys.executable, aggregate=summarise(rows),
                   previous_mixed_intervention_results=str(FORMAL_V3/'RESULTS.csv'),
                   evidence='development intervention recovery to measured clouds; no independent geometry GT')
    save_json(dest/'SUMMARY.json', summary)
    write_readout(dest, summary, rows)
    print(json.dumps({'run_dir':str(dest), 'rows':len(rows), 'ok':summary['successful_rows'], 'seconds':summary['elapsed_s']}), flush=True)
    return dest


if __name__ == '__main__':
    run()

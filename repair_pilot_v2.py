"""Append-only repair run: rescore 444 saved outputs and compute current-input arm.

No new solver or hyperparameter search. Run with the existing Open3D environment.
"""
from __future__ import annotations
import argparse
import csv
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

from adapt import build_adapter, to_operator_mm, from_operator_mm, adapter_payload
from evaluate_v2 import synthetic_geometry, point_rms_mm, reported_model_k
from generate_synthetics import generate_all
from hashutil import sha256_file, dump_json
from paths import PROJECT, RUNS, PATCHES, PILOT, SYNTHETICS, OLD_CHECKPOINT, require_liekkas
from schema import read_patch, read_evaluation, validate_pose_consistency, pose_table, fill_rays
from perturb import make_perturbation

METHODS = ('identity', 'xyz_mixture', 'fast', 'open3d_icp_then_xyz')
SIGMAS = (2., 5.)
SEEDS = (912401, 912409, 912419)


def old_manifest():
    paths = [PROJECT / 'PILOT_RESULTS.csv']
    for root in (PILOT, PATCHES, SYNTHETICS):
        paths.extend(p for p in root.rglob('*') if p.is_file())
    paths.extend([OLD_CHECKPOINT / 'operators.py', OLD_CHECKPOINT / 'generate_cases.py',
        Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py'),
        Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/baseline/scalar_reference.py')])
    return {str(p): sha256_file(p) for p in sorted(paths)}


def case_files(root):
    return sorted(Path(root).glob('*/*.json'))


def saved_output(case, sigma, method):
    file = PILOT / 'outputs' / f'{case}__sig{sigma}__{method}.npz'
    with np.load(file) as a:
        world = a['xyz_world'].copy()
    meta = json.loads(file.with_suffix('.json').read_text())
    return world, meta, file


def visible_perturbation(points, meta, kind, seed):
    change = make_perturbation(points['xyz_world'], points['scan_id'], pose_table(meta),
        seed, .005, 0. if kind == 'trans' else np.deg2rad(.05), int(points['scan_id'].min()))
    visible = {k: np.array(v, copy=True) for k, v in points.items()}
    visible['xyz_world'] = change['xyz_world_perturbed']
    current_meta = dict(meta)
    current_meta['T_world_from_scan_input'] = change['current_T_world_from_scan']
    for sid, T in pose_table(current_meta).items():
        visible['scanner_origin_world'][visible['scan_id'] == sid] = T[:3, 3]
    fill_rays(visible)
    validate_pose_consistency(visible, current_meta)
    return visible, change


def compute_current(ops, points, sigma):
    # No reference cloud, evaluation metadata, or baseline adapter argument.
    adapter = build_adapter(points['xyz_world'])
    xyz_mm = to_operator_mm(points['xyz_world'], adapter)
    outputs = []
    for method in METHODS:
        start = time.perf_counter()
        try:
            out, info = ops.estimate(method, xyz_mm, points['scan_id'], sigma)
            world = from_operator_mm(out, adapter)
            if world.shape != points['xyz_world'].shape or not np.isfinite(world).all():
                raise ValueError('invalid method output')
            outputs.append((method, world, info, time.perf_counter()-start, None))
        except Exception as exc:
            outputs.append((method, None, {}, time.perf_counter()-start, repr(exc)))
    return outputs, adapter


def real_metrics(world, reference, baseline):
    b = (baseline - reference) * 1000.
    d = (world - baseline) * 1000.
    total = point_rms_mm(world, reference)
    b_rms = point_rms_mm(baseline, reference)
    d_rms = point_rms_mm(world, baseline)
    cross = float(2 * np.mean(np.sum(b * d, axis=1)))
    return dict(recovery_vs_measured_reference_rms_mm=total,
                zero_injection_edit_rms_mm=b_rms,
                response_vs_own_baseline_rms_mm=d_rms,
                interaction_mm2=cross,
                decomposition_error_mm2=abs(total**2-b_rms**2-d_rms**2-cross))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def run():
    require_liekkas()
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-parent', type=Path, default=RUNS)
    args = parser.parse_args()
    args.out_parent.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix='repair-v2-', dir=args.out_parent))
    start = time.perf_counter()
    before = old_manifest()
    dump_json(dest / 'OLD_MANIFEST_BEFORE.json', before)
    shutil.copyfile(PROJECT / 'REPAIR_PROTOCOL_V2.md', dest / 'PROTOCOL.md')
    source_dir = dest / 'source'; source_dir.mkdir()
    for source in PROJECT.glob('*.py'): shutil.copyfile(source, source_dir / source.name)
    shutil.copytree(PROJECT / 'tests', source_dir / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    synth = generate_all(dest / 'synthetics')
    rows = []; input_checks = []
    # Reuse outputs: no estimator sees the new evaluation surface descriptions.
    for file in case_files(dest / 'synthetics'):
        p, m = read_patch(file); ev = read_evaluation(file)
        oldfile = SYNTHETICS / file.parent.name / file.name
        oldp, _ = read_patch(oldfile); oldev = read_evaluation(oldfile)
        for key in ('xyz_world', 'scan_id', 'source_point_index'):
            np.testing.assert_array_equal(p[key], oldp[key])
        np.testing.assert_array_equal(ev['gt_clean_xyz_world'], oldev['gt_clean_xyz_world'])
        validate_pose_consistency(p, m)
        input_checks.append({'case': file.stem, 'world_input_identical': True,
                             'source_pose_contract_valid': True})
        for method in METHODS:
            world, saved, origin = saved_output(file.stem, 1., method)
            row = dict(split='synthetic', case=file.stem, method=method, sigma_mm=1.,
                       family=m['family'], gap_mm=m['gap_mm'], bias_rms_mm=m['bias_rms_mm'],
                       seed=m['seed'], execution='reused_v1', ok=True,
                       source_output=str(origin), source_output_sha256=sha256_file(origin))
            row.update(synthetic_geometry(world, saved['info'], ev)); rows.append(row)
    dump_json(dest / 'SYNTHETIC_INPUT_CHECKS.json', input_checks)
    sys.path.insert(0, str(OLD_CHECKPOINT))
    import operators
    operators.warmup(METHODS)
    out_dir = dest / 'outputs'; out_dir.mkdir()
    for file in case_files(PATCHES):
        p, m = read_patch(file); validate_pose_consistency(p, m)
        reference = p['xyz_world']
        baselines = {}
        variants = [('unpert', None)] + [(kind, seed) for kind in ('trans', 'trans_rot') for seed in SEEDS]
        for kind, seed in variants:
            if kind == 'unpert':
                visible = p; actual = 0.; case = file.stem + '_unpert'
            else:
                visible, perturb = visible_perturbation(p, m, kind, seed)
                actual = perturb['actual_point_delta_rms_mm']
                case = f'{file.stem}_{kind}_s{seed}'
                dump_json(dest / 'evaluation' / f'{case}.json',
                          {k:v for k,v in perturb.items() if k != 'xyz_world_perturbed'})
            for sigma in SIGMAS:
                current, adapter = compute_current(operators, visible, sigma)
                for mode in ('fixed_reference_diagnostic', 'current_input'):
                    for method, newworld, newinfo, seconds, error in current:
                        origin = None
                        if mode == 'fixed_reference_diagnostic':
                            world, saved, origin = saved_output(case, sigma, method)
                            info = saved['info']; err = None
                        else:
                            world, info, err = newworld, newinfo, error
                        row = dict(split='real', case=case, patch_id=file.stem, scene=m['scene'],
                                   method=method, sigma_mm=sigma, perturbation=kind, seed=seed,
                                   adapter_mode=mode, execution='reused_v1' if origin else 'computed_v2',
                                   ok=err is None, error=err, actual_point_delta_rms_mm=actual,
                                   independent_real_geometry='not_measured')
                        if origin:
                            row.update(source_output=str(origin), source_output_sha256=sha256_file(origin))
                        else:
                            row['method_seconds'] = seconds
                        if err is None:
                            if kind == 'unpert': baselines[(mode, method, sigma)] = world
                            base = baselines.get((mode, method, sigma))
                            if base is not None: row.update(real_metrics(world, reference, base))
                            row['reported_model_k'] = reported_model_k(info)
                            if not origin:
                                target = out_dir / f'{case}__sig{sigma}__{method}.npz'
                                with target.open('xb') as h: np.savez_compressed(h, xyz_world=world)
                                dump_json(target.with_suffix('.json'),
                                    {'adapter': adapter_payload(adapter),
                                     'info': {k:v for k,v in info.items() if k != 'backend_info'},
                                     'seconds': seconds})
                                row['source_output'] = str(target)
                                row['source_output_sha256'] = sha256_file(target)
                        rows.append(row)
    write_csv(dest / 'RESULTS_V2.csv', rows)
    after = old_manifest(); dump_json(dest / 'OLD_MANIFEST_AFTER.json', after)
    if before != after: raise RuntimeError('old inputs/results changed during repair run')
    summary = dict(run_dir=str(dest), rows=len(rows), ok=sum(r['ok'] for r in rows),
                   reused_outputs=sum(r['execution']=='reused_v1' for r in rows),
                   newly_computed_outputs=sum(r['execution']=='computed_v2' for r in rows),
                   old_files_checked=len(before), old_files_unchanged=True,
                   elapsed_s=time.perf_counter()-start,
                   peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   new_method_seconds=sum(r.get('method_seconds', 0.) for r in rows),
                   synthetic_cases=synth['n_cases'], real_scenes=2, real_patches=6,
                   protocol_sha256=sha256_file(dest/'PROTOCOL.md'))
    dump_json(dest / 'SUMMARY.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': run()

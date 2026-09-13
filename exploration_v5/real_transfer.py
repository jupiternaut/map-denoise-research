"""V5 transfer to byte-identical V4 real inputs; no reference reaches estimators."""
from __future__ import annotations

import argparse
import csv
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
V3, V4 = PROJECT/'exploration_v3', PROJECT/'exploration_v4'
sys.path.insert(0, str(PROJECT))
from paths import RUNS, require_liekkas
from adapt import build_adapter
from hashutil import sha256_array, sha256_file

OLD_RUN = RUNS/'real-components-v4-h3dnauhf'
PATCH_NAMES = ('cy_junction', 'cy_thin', 'cy_wall', 'da_junction', 'da_thin', 'da_wall')
MODES = ('zero', 'normal_translation', 'tangent_translation', 'small_rotation')
METHODS = ('identity', 'pool_independent', 'pool_compatible', 'shared_group_slope',
           'node_intercepts', 'graph_shared', 'measure_unbalanced', 'official_gicp')
SIGMA_MM = 2.0


def load_exact(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == Path(path).resolve()
    return module


_EVAL = load_exact(V4/'real_components.py', '_v5_real_frozen_evaluation')


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def save_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(plain(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def load_cases(root=OLD_RUN):
    """Read exact serialized interventions, never regenerate them."""
    root = Path(root)
    expected = {f'{patch}__{mode}' for patch in PATCH_NAMES for mode in MODES}
    if {p.stem for p in (root/'inputs').glob('*.npz')} != expected:
        raise ValueError('the exact 24 frozen input files are required')
    cases = []
    for patch in PATCH_NAMES:
        zero = None
        for mode in MODES:
            name = f'{patch}__{mode}'
            with np.load(root/'inputs'/f'{name}.npz', allow_pickle=False) as a:
                current = {k: a[k].copy() for k in ('xyz_world', 'scan_id', 'source_point_index')}
            xyz, scan, source = (current[k] for k in ('xyz_world', 'scan_id', 'source_point_index'))
            if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                raise ValueError('invalid frozen coordinates')
            if scan.shape != (len(xyz),) or source.shape != scan.shape or scan.dtype.kind not in 'iu':
                raise ValueError('invalid frozen point IDs')
            construction = json.loads((root/'evaluation'/f'{name}.json').read_text())
            with np.load(root/'evaluation'/f'{name}.npz', allow_pickle=False) as a:
                reference = a['reference_measured_xyz_world'].copy()
                delta = a['injected_delta_world'].copy()
            if construction['mode'] != mode or not np.array_equal(xyz-reference, delta):
                raise ValueError('input/evaluator mismatch')
            if mode == 'zero': zero = dict(current, reference=reference)
            if not np.array_equal(reference, zero['reference']): raise ValueError('reference changed between modes')
            if not np.array_equal(scan, zero['scan_id']) or not np.array_equal(source, zero['source_point_index']):
                raise ValueError('point identities changed between modes')
            if mode == 'zero' and not np.array_equal(xyz, reference): raise ValueError('zero input is not original measured cloud')
            actual = _EVAL.decomposed_mm(delta, construction['construction_normal_world'])
            if abs(actual['xyz_rms_mm']-(0. if mode == 'zero' else 3.5)) > 1e-7:
                raise ValueError('existing RMS intervention contract failed')
            cases.append(dict(case=name, patch=patch, mode=mode, **current,
                              reference=reference, construction=construction))
    return cases


def load_modules():
    return {'pool': load_exact(V4/'surface_pooling.py', '_v5_real_pool'),
            'slope': load_exact(HERE/'slope_pooling.py', '_v5_real_slope'),
            'graph': load_exact(V3/'graph_surface.py', '_v5_real_graph'),
            'measure': load_exact(V3/'measure_surface.py', '_v5_real_measure'),
            'gicp': load_exact(HERE/'registration_baselines.py', '_v5_real_gicp')}


def invoke(method, current_xyz, scan_id, sigma_mm, modules):
    """This copied legal payload is the complete estimator boundary."""
    xyz = np.array(current_xyz, dtype=float, copy=True)
    scan = np.array(scan_id, dtype=np.int64, copy=True)
    if method == 'identity': return xyz, {'method': 'identity', 'status': 'IDENTITY'}
    if method == 'pool_independent': return modules['pool'].estimate(xyz, scan, sigma_mm, variant='independent')
    if method in ('pool_compatible', 'shared_group_slope', 'node_intercepts'):
        variant = 'v4_compatible' if method == 'pool_compatible' else method
        return modules['slope'].estimate(xyz, scan, sigma_mm, variant=variant)
    if method == 'graph_shared': return modules['graph'].estimate(xyz, scan, sigma_mm, variant='graph')
    if method == 'measure_unbalanced': return modules['measure'].estimate(xyz, scan, sigma_mm, variant='unbalanced')
    if method == 'official_gicp': return modules['gicp'].estimate(xyz, scan, sigma_mm, variant='gicp')
    raise ValueError(method)


def candidate_axis(info):
    axis = info.get('normal_world')
    if axis is None: return None
    a = np.asarray(axis, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-12: return None
    return a/np.linalg.norm(a)


def applicability(info, scan_id):
    """Never infer an acceptance mask from whether a point moved."""
    n = len(scan_id); mask = None; role = 'fraction only; point mask unavailable'
    if 'unsupported_point_indices' in info:
        indices = np.asarray(info['unsupported_point_indices'], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= n) or len(np.unique(indices)) != len(indices):
            raise ValueError('invalid unsupported point indices')
        mask = np.ones(n, dtype=bool); mask[indices] = False
        role = 'candidate-reported accepted original input rows'
    elif 'supported_scans' in info:
        ids = info.get('scan_ids', np.unique(scan_id).tolist())
        table = dict(zip(ids, info['supported_scans']))
        mask = np.asarray([bool(table[int(sid)]) for sid in scan_id])
        role = 'whole-scan applicability, not point-level confidence'
    elif 'successful_scan_ids' in info:
        mask = np.isin(scan_id, info['successful_scan_ids']) if info.get('status') == 'APPLY' else np.zeros(n, bool)
        role = 'successfully linked registration component, not point-level inliers'
    fraction = info.get('supported_fraction')
    if fraction is None and mask is not None: fraction = float(mask.mean())
    if mask is not None and fraction is not None and abs(float(mask.mean())-float(fraction)) > 1e-12:
        raise ValueError('support mask/fraction mismatch')
    return mask, dict(supported_point_fraction=fraction, support_mask_available=mask is not None,
                      support_role=role+'; applicability does not certify accuracy',
                      support_mask_sha256=sha256_array(mask) if mask is not None else None,
                      candidate_reported_support_mask_sha256=info.get('support_mask_sha256'),
                      method_status=info.get('status', 'NOT_REPORTED'), method_reason=info.get('reason'),
                      scan_bias_mm=info.get('bias_mm'), transport_mass=info.get('effective_transport_mass'))


def score(output, case, zero_output, axis, zero_axis):
    current, reference = case['xyz_world'], case['reference']
    construction_axis = np.asarray(case['construction']['construction_normal_world'])
    row = _EVAL.evaluate_output(output, current, reference, zero_output, construction_axis)
    pca = build_adapter(current)['normal_world']
    row.update(current_input_pca_normal_world=pca.tolist(), candidate_axis_available=axis is not None,
               current_pca_vs_construction_abs_cos=float(abs(pca @ construction_axis)),
               input_axis_role='diagnostic PCA of current input; not substituted into estimators')
    if axis is not None:
        cosine = float(np.clip(abs(axis @ pca), 0., 1.))
        row.update(candidate_axis_world=axis.tolist(), candidate_axis_vs_input_pca_abs_cos=cosine,
                   candidate_axis_vs_input_pca_angle_deg=float(np.degrees(np.arccos(cosine))),
                   candidate_axis_abs_cos=float(abs(axis @ construction_axis)),
                   edit_orthogonal_to_candidate_axis_rms_mm=_EVAL.decomposed_mm(output-current, axis)['tangent_rms_mm'])
        row.update(_EVAL.reachability_diagnostic(current-reference, case['scan_id'], axis))
        row['projection_scope'] = 'known injection constant-axis diagnostic; not a bound on pooling or GICP'
        if zero_axis is not None: row['candidate_axis_vs_zero_abs_cos'] = float(abs(axis @ zero_axis))
    _, frame = np.unique(case['scan_id'], return_inverse=True)
    edits = output-current
    means = np.array([edits[frame == f].mean(axis=0) for f in range(frame.max()+1)])
    row['within_scan_nonconstant_edit_rms_mm'] = 1000.*_EVAL.rms_m(edits-means[frame])
    return row


def aggregate(rows):
    fields = ('recovery_xyz_rms_mm', 'recovery_normal_rms_mm', 'recovery_tangent_rms_mm',
              'zero_edit_xyz_rms_mm', 'self_response_xyz_rms_mm', 'input_edit_xyz_rms_mm',
              'candidate_axis_vs_input_pca_angle_deg', 'supported_point_fraction', 'method_seconds')
    groups, paired = [], []
    for mode in MODES:
        for method in METHODS:
            group = [r for r in rows if r['mode'] == mode and r['method'] == method and r['ok']]
            item = dict(mode=mode, method=method, expected_patches=6, successful_patches=len(group))
            for key in fields:
                z = [r[key] for r in group if r.get(key) is not None]
                if z: item[key] = {'n':len(z), 'mean':float(np.mean(z)), 'median':float(np.median(z))}
            groups.append(item)
        lookup = {(r['patch'], r['method']):r for r in rows if r['mode'] == mode and r['ok']}
        for base in ('pool_compatible', 'pool_independent', 'node_intercepts'):
            for metric in ('recovery_xyz_rms_mm', 'recovery_normal_rms_mm', 'recovery_tangent_rms_mm'):
                records = []
                for patch in PATCH_NAMES:
                    if (patch, base) in lookup and (patch, 'shared_group_slope') in lookup:
                        gain = lookup[patch, base][metric]-lookup[patch, 'shared_group_slope'][metric]
                        records.append({'patch':patch, 'gain':gain})
                if records:
                    z = np.asarray([x['gain'] for x in records])
                    paired.append(dict(mode=mode, baseline=base, candidate='shared_group_slope', metric=metric,
                                       n=len(z), improved=int(np.sum(z>1e-9)), tied=int(np.sum(abs(z)<=1e-9)),
                                       worse=int(np.sum(z < -1e-9)), mean_gain=float(z.mean()),
                                       median_gain=float(np.median(z)), cases=records))
    return {'groups':groups, 'paired':paired}


def source_paths():
    sources = [HERE/n for n in ('real_transfer.py', 'test_real_transfer.py', 'REAL_PROTOCOL.md',
                                'slope_pooling.py', 'registration_baselines.py', 'BASELINE_NOTES.md')]
    sources += [V4/n for n in ('surface_pooling.py', 'real_components.py')]
    sources += [V3/n for n in ('graph_surface.py', 'measure_surface.py')]
    sources += [PROJECT/n for n in ('adapt.py', 'transforms.py', 'paths.py', 'schema.py', 'hashutil.py')]
    return sources


def verify_run(dest):
    """Reload and independently recompute saved decompositions, no estimator calls."""
    dest = Path(dest)
    rows = list(csv.DictReader((dest/'RESULTS.csv').open()))
    if len(rows) != 192 or len({(r['case'],r['method']) for r in rows}) != 192:
        raise AssertionError('expected 192 distinct outcomes')
    cases = {c['case']:c for c in load_cases(dest)}
    good = {(r['case'],r['method']):r for r in rows if r['ok']=='True'}
    worst = 0.; checked = 0; masks = 0
    for row in good.values():
        case = cases[row['case']]
        path = Path(row['output'])
        if sha256_file(path) != row['output_sha256']: raise AssertionError('output hash mismatch')
        with np.load(path, allow_pickle=False) as a:
            out = a['xyz_world']
            assert out.shape == case['xyz_world'].shape and np.isfinite(out).all()
            for key in ('scan_id','source_point_index'): np.testing.assert_array_equal(a[key], case[key])
            moved = np.linalg.norm(out-case['xyz_world'], axis=1)*1000. > 1e-6
            np.testing.assert_array_equal(a['moved_mask'], moved)
            if 'support_mask' in a:
                support = a['support_mask']; masks += 1
                assert support.shape == (len(out),) and support.dtype == np.bool_
                assert abs(float(support.mean())-float(row['supported_point_fraction'])) < 1e-12
                np.testing.assert_array_equal(out[~support], case['xyz_world'][~support])
        zero_row = good[(row['patch']+'__zero',row['method'])]
        with np.load(zero_row['output'], allow_pickle=False) as a: zero = a['xyz_world']
        axis = np.asarray(case['construction']['construction_normal_world'], dtype=float)
        axis /= np.linalg.norm(axis)
        for prefix, delta in [('recovery',out-case['reference']), ('zero_edit',zero-case['reference']),
                              ('self_response',out-zero), ('input_edit',out-case['xyz_world']),
                              ('injected',case['xyz_world']-case['reference'])]:
            normal = delta @ axis; tangent = delta-normal[:,None]*axis
            numbers = {'xyz_rms_mm':float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))*1000.),
                       'normal_rms_mm':float(np.sqrt(np.mean(normal**2))*1000.),
                       'tangent_rms_mm':float(np.sqrt(np.mean(np.sum(tangent**2,axis=1)))*1000.)}
            for suffix,value in numbers.items():
                error = abs(value-float(row[prefix+'_'+suffix])); worst=max(worst,error); checked+=1
                if error>1e-8: raise AssertionError((row['case'],row['method'],prefix,suffix,error))
    before = json.loads((dest/'PROTECTED_BEFORE.json').read_text())
    after = json.loads((dest/'PROTECTED_AFTER.json').read_text())
    if before != after or any(sha256_file(Path(p)) != h for p,h in before.items()):
        raise AssertionError('protected inputs or sources changed')
    for item in json.loads((dest/'SOURCE_MANIFEST.json').read_text()).values():
        if sha256_file(Path(item['snapshot'])) != item['sha256']: raise AssertionError('snapshot hash mismatch')
    input_manifest = json.loads((dest/'INPUT_MANIFEST.json').read_text())
    for item in input_manifest:
        if sha256_file(Path(item['original'])) != item['sha256'] or sha256_file(Path(item['copy'])) != item['sha256']:
            raise AssertionError('frozen intervention bytes changed')
    return dict(status='PASS', rows=len(rows), successful_rows=len(good),
                numeric_scores_recomputed=checked, maximum_score_error_mm=worst,
                exact_point_id_and_shape_checks=len(good), saved_support_masks_checked=masks,
                unchanged_original_run=True, protected_files=len(before),
                estimator_rerun=False, input_files_byte_identical=True)


def write_readout(dest, summary, stats):
    lines = ['# V5 真实输入迁移', '',
             '本表是恢复已有注入量到原测量云，不是独立真实几何精度。低改写/低自响应也不证明去噪。', '',
             f"{summary['successful_rows']}/{summary['rows']} 输出成功；复用同一24输入、6片区、2场景；sigma=2 mm。", '',
             '|模式|方法|XYZ恢复中位RMS mm|构造法向 mm|构造切向 mm|零注入改写 mm|',
             '|---|---|---:|---:|---:|---:|']
    for r in stats['groups']:
        if r['successful_patches']:
            values = [f"{r[k]['median']:.6f}" for k in ('recovery_xyz_rms_mm','recovery_normal_rms_mm','recovery_tangent_rms_mm','zero_edit_xyz_rms_mm')]
            lines.append('|'+ '|'.join([r['mode'],r['method']]+values)+'|')
    lines += ['', '逐片区结果、支持率、当前输入PCA与候选轴分歧见 RESULTS.csv；配对增量见 AGGREGATES.json。',
              '逐点moved_mask与候选支持mask分开保存。旧接口没有逐点mask时明确缺失，不能用低移动量补造。',
              'GICP是官方求解器的输入锚站包装，无点级去噪；原始变换和输入定义平移规范在输出JSON中。',
              'measure仍为原3步扫描常量纠偏。功能与pool/graph不同，不作无差别去噪排名。',
              '未用参考或构造法向重置算法，没有逐扫描/逐层/参考对齐，没有新增独立真实场景。',
              f"总wall {summary['elapsed_s']:.3f}s；方法调用累计 {summary['method_total_s']:.3f}s。单线程CPU但其他任务可能争用。", 
              '源码、整个V4原run和输入hash前后检查；VERIFICATION.json重读所有输出并独立复算分解分数。']
    with (dest/'READOUT.md').open('x') as f: f.write('\n'.join(lines)+'\n')


def run():
    require_liekkas(); started=time.perf_counter()
    sources = source_paths()
    if not all(p.is_file() for p in sources): raise FileNotFoundError('all frozen V5 module/source files must exist')
    source_before = {str(p):sha256_file(p) for p in sources}
    before = dict(source_before)
    before.update({str(p):sha256_file(p) for p in OLD_RUN.rglob('*') if p.is_file()})
    import_start=time.perf_counter(); modules=load_modules(); import_seconds=time.perf_counter()-import_start
    cases=load_cases()
    dest=Path(tempfile.mkdtemp(prefix='real-transfer-v5-',dir=RUNS)); print(dest,flush=True)
    for name in ('inputs','evaluation','outputs','source'): (dest/name).mkdir()
    save_json(dest/'PROTECTED_BEFORE.json',before)
    manifest={}
    for index,p in enumerate(sources):
        snapshot=dest/'source'/f'{index:02d}_{p.name}';shutil.copyfile(p,snapshot)
        manifest[str(p)]={'sha256':source_before[str(p)],'snapshot':str(snapshot)}
    save_json(dest/'SOURCE_MANIFEST.json',manifest)
    input_manifest=[]
    for case in cases:
        for folder,ext in (('inputs','.npz'),('evaluation','.npz'),('evaluation','.json')):
            old=OLD_RUN/folder/(case['case']+ext);new=dest/folder/old.name
            shutil.copyfile(old,new);input_manifest.append(dict(original=old,copy=new,sha256=sha256_file(old)))
    save_json(dest/'INPUT_MANIFEST.json',input_manifest)
    save_json(dest/'CONFIGURATION.json',dict(methods=METHODS,modes=MODES,sigma_mm=SIGMA_MM,
              original_run=OLD_RUN,inputs=24,expected_outputs=192,protocol_sha256=sha256_file(HERE/'REAL_PROTOCOL.md'),
              import_seconds=import_seconds,frozen_before_estimator_calls=True))
    warm=[]
    for method in METHODS:
        t=time.perf_counter()
        try:
            invoke(method,cases[0]['xyz_world'],cases[0]['scan_id'],SIGMA_MM,modules)
            warm.append(dict(method=method,ok=True,seconds=time.perf_counter()-t))
        except Exception as exc: warm.append(dict(method=method,ok=False,error=repr(exc),seconds=time.perf_counter()-t))
    save_json(dest/'WARMUP.json',warm)
    rows=[];zero_outputs={};zero_axes={}
    for case in cases:
        for method in METHODS:
            row=dict(case=case['case'],patch=case['patch'],mode=case['mode'],method=method,
                     scene=case['construction']['scene'],n_points=len(case['xyz_world']),
                     n_scans=len(np.unique(case['scan_id'])),sigma_mm=SIGMA_MM,
                     original_input=str(OLD_RUN/'inputs'/(case['case']+'.npz')),
                     independent_real_geometry='not_measured')
            t=time.perf_counter()
            try:
                out,info=invoke(method,case['xyz_world'],case['scan_id'],SIGMA_MM,modules)
                elapsed=time.perf_counter()-t;out=np.asarray(out,dtype=float)
                if out.shape!=case['xyz_world'].shape or not np.isfinite(out).all():raise ValueError('invalid output')
                key=(case['patch'],method);axis=candidate_axis(info)
                if case['mode']=='zero':zero_outputs[key]=out.copy();zero_axes[key]=axis
                if key not in zero_outputs:raise RuntimeError('same-method zero output unavailable; self response cannot be scored')
                row.update(score(out,case,zero_outputs[key],axis,zero_axes[key]))
                support,support_info=applicability(info,case['scan_id']);row.update(support_info)
                row.update(ok=True,method_seconds=elapsed,point_order_contract='original row and source ID',
                           source_point_id_sha256=sha256_array(case['source_point_index']))
                moved=np.linalg.norm(out-case['xyz_world'],axis=1)*1000.>1e-6
                arrays=dict(xyz_world=out,scan_id=case['scan_id'],source_point_index=case['source_point_index'],moved_mask=moved)
                if support is not None:arrays['support_mask']=support
                path=dest/'outputs'/f"{case['case']}__{method}.npz"
                with path.open('xb') as f:np.savez_compressed(f,**arrays)
                row.update(output=str(path),output_sha256=sha256_file(path))
                save_json(path.with_suffix('.json'),{'info':info,'row':row})
            except Exception as exc:
                row.update(ok=False,error=repr(exc),method_seconds=time.perf_counter()-t)
                save_json(dest/'outputs'/f"{case['case']}__{method}.error.json",dict(error=repr(exc),traceback=traceback.format_exc()))
            rows.append(row)
        print(case['case']+': '+str(sum(r['ok'] for r in rows[-len(METHODS):]))+'/'+str(len(METHODS)),flush=True)
        if any(sha256_file(Path(p))!=h for p,h in source_before.items()):raise RuntimeError('frozen source changed during run; preserve partial run')
    _EVAL.write_csv(dest/'RESULTS.csv',rows)
    after={p:sha256_file(Path(p)) for p in before};save_json(dest/'PROTECTED_AFTER.json',after)
    if before!=after:raise RuntimeError('protected original run/source changed')
    stats=aggregate(rows);save_json(dest/'AGGREGATES.json',stats)
    verification=verify_run(dest);save_json(dest/'VERIFICATION.json',verification)
    summary=dict(run_dir=dest,rows=len(rows),successful_rows=sum(r['ok'] for r in rows),inputs=24,
                 original_world_patches=6,scenes=2,methods=METHODS,source_run=OLD_RUN,
                 sigma_mm=SIGMA_MM,elapsed_s=time.perf_counter()-started,
                 method_total_s=sum(r['method_seconds'] for r in rows),import_seconds=import_seconds,
                 warmup_seconds=sum(r['seconds'] for r in warm),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 memory_scope='this Python process high-water RSS, not combined concurrent-task RAM',
                 threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                 python=sys.executable,protected_files=len(before),historical_run_unchanged=True,
                 verified=verification['status']=='PASS',evidence='same-real-input intervention recovery; no independent geometry truth')
    save_json(dest/'SUMMARY.json',summary);write_readout(dest,summary,stats)
    print(json.dumps(plain(summary),indent=2),flush=True)
    return dest


if __name__=='__main__':
    require_liekkas();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',type=Path)
    args=parser.parse_args()
    if args.verify_only:print(json.dumps(verify_run(args.verify_only),indent=2))
    else:run()

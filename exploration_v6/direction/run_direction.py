"""Frozen real direction comparison plus predeclared public synthetic diagnostic."""
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
import direction_pooling as candidate

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
from adapt import build_adapter
from evaluate_v2 import synthetic_geometry
from hashutil import sha256_file, sha256_array
from paths import RUNS, require_liekkas

REAL_RUN = RUNS / 'real-transfer-v5-zrtui_f5'
SYNTH_RUN = RUNS / 'exploration-v5-development-vdbkdouk'
SEEDS = (912101, 912113, 912127)
METHODS = candidate.VARIANTS


def load_exact(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path.resolve()
    return module


REAL = load_exact(PROJECT/'exploration_v5'/'real_transfer.py', '_v6_real_evaluator')
STRUCTURE = load_exact(PROJECT/'exploration_v3'/'metrics.py', '_v6_structure_evaluator')


def save_json(path, value):
    REAL.save_json(path, value)


def read_eval_npz(path):
    result = {}
    with np.load(path, allow_pickle=False) as a:
        for key in a.files:
            if key == 'json': result.update(json.loads(bytes(a[key]).decode()))
            else: result[key] = a[key].copy()
    return result


def load_inputs():
    cases = []
    for c in REAL.load_cases(REAL_RUN):
        name = c['case']
        cases.append(dict(c, split='real', sigma_mm=2.,
                          original_input=REAL_RUN/'inputs'/(name+'.npz'),
                          original_evaluation=REAL_RUN/'evaluation'/(name+'.npz'),
                          original_evaluation_json=REAL_RUN/'evaluation'/(name+'.json')))
    for seed in SEEDS:
        for gap in (0, 2, 4, 8):
            for bias in (0, 4):
                name = f'ghost_s{seed}_b{bias}' if gap == 0 else f'dual_g{gap}_s{seed}_b{bias}'
                meta_path = SYNTH_RUN/'inputs'/(name+'.json')
                record = json.loads(meta_path.read_text())
                assert record['sampling'] == 'full' and record['case'] == record['base_case'] == name
                assert record['meta']['seed'] == seed and record['sigma'] == 1.
                assert np.array_equal(record['rotation'], np.eye(3)) and np.array_equal(record['translation'], np.zeros(3))
                source_meta = Path(record['source'])
                source = json.loads(source_meta.read_text())
                original_points = Path(source['npz'])
                evaluation = original_points.parent/'evaluation'/(source['patch_id']+'.eval.npz')
                with np.load(SYNTH_RUN/'inputs'/(name+'.npz'), allow_pickle=False) as a:
                    xyz, scans = a['xyz_world'].copy(), a['scan_id'].copy()
                with np.load(original_points, allow_pickle=False) as a:
                    np.testing.assert_array_equal(xyz, a['xyz_world'])
                    np.testing.assert_array_equal(scans, a['scan_id'])
                    source_ids = a['source_point_index'].copy()
                ev = read_eval_npz(evaluation)
                assert ev['identifiable'] and len(ev['gt_clean_xyz_world']) == len(xyz)
                cases.append(dict(case=name, split='synthetic', patch=name, mode='public_full',
                                  xyz_world=xyz, scan_id=scans, source_point_index=source_ids,
                                  sigma_mm=1., gap_mm=gap, bias_rms_mm=bias, seed=seed,
                                  evaluation=ev, original_input=SYNTH_RUN/'inputs'/(name+'.npz'),
                                  original_evaluation=evaluation, original_input_json=meta_path,
                                  original_source_meta=source_meta, original_source_points=original_points))
    assert len(cases) == 48
    return cases


def point_rms(delta):
    return float(np.sqrt(np.mean(np.sum(np.asarray(delta)**2, axis=1))) * 1000.)


def axis_diagnostics(delta, output_edit, normal, support, injection_axis):
    """Evaluation only: the output's allowed direction never enters a selector."""
    result = {}
    if normal is None:
        if support.any(): raise ValueError('supported output requires a recorded direction')
        perpendicular = np.asarray(delta)
        off_axis = np.asarray(output_edit)
    else:
        n = np.asarray(normal, float); n = n / np.linalg.norm(n)
        perpendicular = delta - (delta@n)[:, None]*n
        off_axis = output_edit - (output_edit@n)[:, None]*n
        m = np.asarray(injection_axis, float); m = m / np.linalg.norm(m)
        result['candidate_axis_vs_injection_deg'] = float(np.degrees(np.arccos(np.clip(abs(n@m), 0., 1.))))
    restricted = perpendicular.copy()
    restricted[~support] = delta[~support]
    result.update(axis_pointwise_lower_bound_xyz_rms_mm=point_rms(perpendicular),
                  support_axis_lower_bound_xyz_rms_mm=point_rms(restricted),
                  output_off_axis_rms_mm=point_rms(off_axis),
                  axis_bound_scope='pointwise XYZ recovery to original measured cloud; evaluator only, not clean truth or a normal-only score')
    return result


def real_scores(out, c, zero_output, normal, support):
    axis = c['construction']['construction_normal_world']
    row = REAL._EVAL.evaluate_output(out, c['xyz_world'], c['reference'], zero_output, axis)
    row.update(axis_diagnostics(c['xyz_world']-c['reference'], out-c['xyz_world'], normal, support, axis))
    current_pca = build_adapter(c['xyz_world'])['normal_world']
    row['current_input_pca_axis_world'] = current_pca.tolist()
    if normal is not None:
        n = np.asarray(normal, float); n /= np.linalg.norm(n)
        row['candidate_axis_vs_current_pca_deg'] = float(np.degrees(np.arccos(np.clip(abs(n@current_pca), 0., 1.))))
    return row


def aggregates(rows):
    good = [r for r in rows if r['ok']]
    buckets = [('real', mode) for mode in REAL.MODES] + [('synthetic', tag) for tag in ('all', 'single', 'double', 'g2', 'g4', 'g8')]
    real_keys = ('recovery_xyz_rms_mm', 'recovery_normal_rms_mm', 'recovery_tangent_rms_mm',
                 'zero_edit_xyz_rms_mm', 'self_response_xyz_rms_mm', 'input_edit_xyz_rms_mm',
                 'axis_pointwise_lower_bound_xyz_rms_mm', 'support_axis_lower_bound_xyz_rms_mm',
                 'candidate_axis_vs_injection_deg', 'candidate_axis_vs_current_pca_deg')
    synth_keys = ('surface_accuracy_mean_mm', 'surface_accuracy_rms_mm', 'surface_accuracy_p95_mm',
                  'matched_point_rms_mm', 'fitted_gap_at_same_xy_error_mm', 'source_group_gap_error_mm',
                  'source_group_gap_retention', 'reference_sample_coverage_1mm', 'input_edit_xyz_rms_mm')
    groups, paired = [], []
    for split, bucket in buckets:
        selected = [r for r in good if r['split'] == split and (r['mode'] == bucket if split == 'real' else
                    bucket == 'all' or (bucket == 'single' and r['gap_mm'] == 0) or
                    (bucket == 'double' and r['gap_mm'] > 0) or bucket == 'g'+str(r['gap_mm']))]
        keys = real_keys if split == 'real' else synth_keys
        for method in METHODS:
            subset = [r for r in selected if r['method'] == method]
            item = dict(split=split, bucket=bucket, method=method, n_cases=len(subset), cases=[r['case'] for r in subset])
            for key in (*keys, 'supported_fraction', 'method_seconds'):
                values = [float(r[key]) for r in subset if r.get(key) is not None]
                if values: item[key] = dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)))
            groups.append(item)
        lookup = {(r['case'], r['method']):r for r in selected}
        metrics = ('recovery_xyz_rms_mm', 'zero_edit_xyz_rms_mm') if split == 'real' else ('surface_accuracy_mean_mm', 'fitted_gap_at_same_xy_error_mm', 'matched_point_rms_mm')
        for metric in metrics:
            records=[]
            for case in sorted({r['case'] for r in selected}):
                if all((case,m) in lookup and lookup[case,m].get(metric) is not None for m in METHODS):
                    records.append(dict(case=case, gain=lookup[case,'difference'][metric]-lookup[case,'pooled_pca'][metric]))
            if records:
                z=np.array([r['gain'] for r in records])
                paired.append(dict(split=split, bucket=bucket, metric=metric, n=len(z), cases=records,
                                   improved=int(np.sum(z>1e-9)), tied=int(np.sum(abs(z)<=1e-9)),
                                   worse=int(np.sum(z < -1e-9)), mean_gain=float(z.mean()), median_gain=float(np.median(z))))
    return dict(groups=groups, paired=paired, sign='positive gain = difference minus PCA, lower is better')


def sources():
    paths = list(HERE.glob('*.py')) + [HERE/'PROTOCOL.md']
    paths += [PROJECT/p for p in ('exploration_v5/slope_pooling.py', 'exploration_v5/real_transfer.py',
              'exploration_v4/surface_pooling.py', 'exploration_v4/real_components.py',
              'exploration_v3/graph_surface.py', 'exploration_v3/metrics.py', 'adapt.py', 'transforms.py',
              'schema.py', 'paths.py', 'hashutil.py', 'evaluate_v2.py')]
    return sorted(set(paths))


def protected_paths(cases):
    files = set(sources())
    for root in (REAL_RUN, SYNTH_RUN): files.update(p for p in root.rglob('*') if p.is_file())
    for c in cases:
        files.update(v for k,v in c.items() if k.startswith('original_') and isinstance(v, Path))
    return sorted(files)


def snapshot_inputs(dest, cases):
    manifest=[];registry=[]
    for c in cases:
        folder=dest/c['split']; (folder/'inputs').mkdir(parents=True,exist_ok=True); (folder/'evaluation').mkdir(exist_ok=True)
        name=c['case'];entry={k:c[k] for k in ('case','split','patch','mode','sigma_mm')}
        for key in ('gap_mm','bias_rms_mm','seed'):
            if key in c:entry[key]=c[key]
        targets={'original_input':folder/'inputs'/(name+'.npz'),
                 'original_evaluation':folder/'evaluation'/(name+'.npz')}
        if c['split']=='real': targets['original_evaluation_json']=folder/'evaluation'/(name+'.json')
        else:
            (folder/'provenance').mkdir(exist_ok=True)
            targets.update(original_input_json=folder/'inputs'/(name+'.json'),
                           original_source_meta=folder/'provenance'/(name+'.json'),
                           original_source_points=folder/'provenance'/(name+'.npz'))
        for key,target in targets.items():
            original=c[key];shutil.copyfile(original,target)
            manifest.append(dict(original=str(original),copy=str(target),sha256=sha256_file(original)))
            entry[key.removeprefix('original_')]=str(target)
        registry.append(entry)
    save_json(dest/'INPUT_MANIFEST.json',manifest);save_json(dest/'CASES.json',registry)


def write_readout(dest, summary, stats):
    lines=['# V6 A: current-input direction replacement','',
           'PCA uses current observations only. Real numbers describe recovery to an original measured cloud, not independent geometry truth.',
           f"{summary['successful_rows']}/{summary['rows']} formal outputs; 24 existing real + 24 exposed synthetic inputs, two methods.",'',
           '|Real intervention|Direction|XYZ recovery median mm|Normal mm|Tangent mm|Zero edit mm|Support mean|',
           '|---|---|---:|---:|---:|---:|---:|']
    for g in stats['groups']:
        if g['split']=='real' and g['n_cases']:
            keys=('recovery_xyz_rms_mm','recovery_normal_rms_mm','recovery_tangent_rms_mm','zero_edit_xyz_rms_mm')
            values=[f"{g[k]['median']:.6f}" for k in keys]
            lines.append('|'+ '|'.join([g['bucket'],g['method'],*values,f"{g['supported_fraction']['mean']:.6f}"])+'|')
    lines+=['','|Public synthetic subset|Direction|Surface MAE mean mm|Matched RMS mm|Same-XY gap error mm|',
            '|---|---|---:|---:|---:|']
    for g in stats['groups']:
        if g['split']=='synthetic' and g['n_cases']:
            values=[f"{g[k]['mean']:.6f}" if k in g else 'NA' for k in ('surface_accuracy_mean_mm','matched_point_rms_mm','fitted_gap_at_same_xy_error_mm')]
            lines.append('|'+ '|'.join([g['bucket'],g['method'],*values])+'|')
    lines+=['','All case-paired wins, ties and regressions are in AGGREGATES.json and RESULTS.csv.',
            'Single-axis pointwise bounds are evaluator-only XYZ-to-measurement bounds; they are not normal-only or true-surface accuracy bounds.',
            'The injection axis was defined using original-measurement PCA. Better recovery alone cannot validate the physical normal.',
            'Direction replacement also recomputes bias, cells, assignments and support; this is a matched upstream intervention, not fixed-association attribution.',
            'All outputs, actual support masks and separate movement masks are retained. The difference arm must exactly replay all 48 prior compatible outputs.',
            f"Method total {summary['method_total_s']:.3f}s; wall {summary['elapsed_s']:.3f}s. Single CPU process under possible concurrent contention.",
            'No new confirmation seeds, downloads, dependencies, GPU or algorithm edits in older versions.']
    with (dest/'READOUT.md').open('x') as f:f.write('\n'.join(lines)+'\n')


def run():
    require_liekkas();started=time.perf_counter();cases=load_inputs()
    protected={str(p):sha256_file(p) for p in protected_paths(cases)}
    dest=Path(tempfile.mkdtemp(prefix='direction-v6-',dir=RUNS));print(dest,flush=True)
    (dest/'source').mkdir();(dest/'outputs').mkdir()
    save_json(dest/'PROTECTED_BEFORE.json',protected)
    manifest={}
    for p in sources():
        relative=p.relative_to(PROJECT);target=dest/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);manifest[str(p)]=dict(snapshot=str(target),sha256=protected[str(p)])
    save_json(dest/'SOURCE_MANIFEST.json',manifest);snapshot_inputs(dest,cases)
    save_json(dest/'CONFIGURATION.json',dict(methods=METHODS,real_inputs=24,public_synthetic_inputs=24,
              expected_outputs=96,real_sigma_mm=2.,synthetic_sigma_mm=1.,synthetic_seeds=SEEDS,
              protocol_sha256=sha256_file(HERE/'PROTOCOL.md'),frozen_before_formal_estimation=True))
    warm=[]
    for method in METHODS:
        t=time.perf_counter()
        candidate.estimate(cases[0]['xyz_world'],cases[0]['scan_id'],2.,method)
        warm.append(dict(method=method,seconds=time.perf_counter()-t))
    save_json(dest/'WARMUP.json',warm)
    rows=[];zeros={}
    for c in cases:
        for method in METHODS:
            row={k:c[k] for k in ('case','split','patch','mode','sigma_mm')}
            row.update(method=method,n_points=len(c['xyz_world']),n_scans=len(np.unique(c['scan_id'])),
                       original_input=str(c['original_input']),original_input_sha256=protected[str(c['original_input'])],
                       evaluation=str(dest/c['split']/'evaluation'/(c['case']+'.npz')),
                       evaluation_json=str(dest/'real'/'evaluation'/(c['case']+'.json')) if c['split']=='real' else None)
            for k in ('seed','gap_mm','bias_rms_mm'):
                if k in c:row[k]=c[k]
            t=time.perf_counter()
            try:
                out,info=candidate.estimate(c['xyz_world'],c['scan_id'],c['sigma_mm'],method)
                elapsed=time.perf_counter()-t
                support=np.ones(len(out),bool);support[np.asarray(info['unsupported_point_indices'],int)]=False
                moved=np.linalg.norm(out-c['xyz_world'],axis=1)*1000.>1e-6
                path=dest/'outputs'/(c['split']+'__'+c['case']+'__'+method+'.npz')
                with path.open('xb') as f:np.savez_compressed(f,xyz_world=out,scan_id=c['scan_id'],
                     source_point_index=c['source_point_index'],support_mask=support,moved_mask=moved)
                row.update(output=str(path),output_sha256=sha256_file(path),method_seconds=elapsed,
                           supported_fraction=float(support.mean()),support_mask_sha256=sha256_array(support),
                           moved_fraction=float(moved.mean()),input_edit_xyz_rms_mm=point_rms(out-c['xyz_world']),
                           normal_world=info.get('normal_world'),basis_call_count=info['basis_call_count'],
                           initialization_sha256=info['initialization_sha256'],grouping_sha256=info['grouping_sha256'],
                           frozen_state_sha256=info['frozen_state_sha256'])
                if c['split']=='real':
                    if c['mode']=='zero':zeros[c['patch'],method]=out.copy()
                    row.update(real_scores(out,c,zeros[c['patch'],method],info.get('normal_world'),support))
                else:
                    row.update(synthetic_geometry(out,info,c['evaluation']))
                    row.update(STRUCTURE.structure_metrics(out,c['evaluation']))
                row['ok']=True
                save_json(path.with_suffix('.json'),dict(info=info,row=row))
            except Exception as exc:
                row.update(ok=False,error=repr(exc),method_seconds=time.perf_counter()-t)
                save_json(dest/'outputs'/(c['split']+'__'+c['case']+'__'+method+'.error.json'),
                          dict(error=repr(exc),traceback=traceback.format_exc(),row=row))
            rows.append(row)
        print(c['split']+'/'+c['case']+': '+str(sum(r['ok'] for r in rows[-2:]))+'/2',flush=True)
        if any(sha256_file(Path(p))!=v['sha256'] for p,v in manifest.items()):
            raise RuntimeError('frozen source changed; preserve partial run')
    REAL._EVAL.write_csv(dest/'RESULTS.csv',rows)
    after={p:sha256_file(Path(p)) for p in protected};save_json(dest/'PROTECTED_AFTER.json',after)
    if after!=protected:raise RuntimeError('protected historical files changed')
    stats=aggregates(rows);save_json(dest/'AGGREGATES.json',stats)
    from verify_direction import verify
    verification=verify(dest);save_json(dest/'VERIFICATION.json',verification)
    summary=dict(run_dir=dest,rows=len(rows),successful_rows=sum(r['ok'] for r in rows),
                 elapsed_s=time.perf_counter()-started,method_total_s=sum(r['method_seconds'] for r in rows),
                 warmup_seconds=sum(r['seconds'] for r in warm),protected_files=len(protected),
                 peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 memory_scope='this Python process only, not all concurrent processes',
                 threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                 python=sys.executable,verification=verification['status'])
    save_json(dest/'SUMMARY.json',summary);write_readout(dest,summary,stats)
    print(json.dumps(REAL.plain(summary),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();run()

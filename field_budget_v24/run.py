"""Reuse frozen V22 fields; construct and seal all arrays before reference scoring."""
from pathlib import Path
import concurrent.futures as futures
import csv
import hashlib
import json
import resource
import shutil
import socket
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

from field_ops import construct, FIELDS, BUDGETS, SHUFFLE_SEEDS
from confidence import diagnose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = Path('/srv/slam-research/grf/map-denoise/runs/reconstruction-v22-qayc8gft')
RUNS = CACHE.parent
DATA = Path('/srv/slam-research/grf/map-denoise/datasets')
REFERENCES = {
    'scan24': DATA/'published-outputs-v2-reference/stl024_total.ply',
    'scan37': DATA/'reconstruction-v22-scan37/stl037_total.ply',
}
sys.path.insert(0, '/srv/slam-research/grf/map-denoise/tools/published-plyfile')
from plyfile import PlyData


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False,
                  default=lambda x: str(x) if isinstance(x, Path) else np.asarray(x).tolist())


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def reference(path):
    a = PlyData.read(str(path))['vertex'].data
    return np.column_stack([a[k] for k in ('x', 'y', 'z')]).astype(float)


def prepare(dest):
    cases = []
    for phase, scene in [('development', 'scan24'), ('confirmation', 'scan37')]:
        for b in read(CACHE/f'{phase}_SEALED.json'):
            job = b['job']; case = job['case']
            assert sha(job['input']) == job['sha256']
            x = np.load(job['input']); cached = {}
            for r in b['records']:
                assert r['status'] == 'OK', (case, r)
                assert sha(r['path']) == r['sha256']
                cached[r['method']] = np.load(r['path'])
            alpha = np.asarray(read(CACHE/'diagnostics'/f'{case}.json')['operator']['consensus']['alpha'])
            np.testing.assert_allclose(cached['multiscale_consensus']-x,
                                       alpha[:, None]*(cached['multiscale_full']-x), atol=1e-10, rtol=0)
            input_path = dest/'inputs'/f'{case}.npy'
            np.save(input_path, x)
            alpha_path = dest/'inputs'/f'{case}__alpha.npy'; np.save(alpha_path, alpha)
            evaluation_path = dest/'evaluation'/f'{case}.npz'
            shutil.copyfile(CACHE/'evaluation'/f'{case}.npz', evaluation_path)
            records, diagnostic = construct(x, cached, alpha)
            records = [dict(method=f'native__{name}', kind='native', field=name,
                            allocation='cached', status='OK', output=z)
                       for name, z in cached.items()] + records
            for rec in records:
                out = rec.pop('output')
                if out is None: continue
                assert out.shape == x.shape and np.isfinite(out).all()
                path = dest/'outputs'/f'{case}__{rec["method"]}.npy'; np.save(path, out)
                move = np.linalg.norm(out-x, axis=1)
                rec.update(path=str(path), sha256=sha(path),
                           displacement_rms_mm=float(np.sqrt(np.mean(move**2))),
                           displacement_p95_mm=float(np.quantile(move, .95)),
                           displacement_max_mm=float(move.max()),
                           moved_fraction=float(np.mean(move > 1e-7)))
            cases.append(dict(case=case, scene=scene, evidence_role='exposed_diagnostic',
                              legacy_phase=phase, input=str(input_path), alpha=str(alpha_path),
                              evaluation=str(evaluation_path), original_input=job['input'],
                              original_diagnostic=str(CACHE/'diagnostics'/f'{case}.json'),
                              input_sha256=sha(input_path), alpha_sha256=sha(alpha_path),
                              evaluation_sha256=sha(evaluation_path), diagnostic=diagnostic,
                              records=records))
            print('SEALED_CASE', case, len(records), flush=True)
    save(dest/'SEALED.json', dict(version=24, cache=str(CACHE), cases=cases,
                                 protocol_sha256=sha(HERE/'PROTOCOL.md'),
                                 construction_used_reference=False))
    return cases


def score_case(case, ref, tree, dest):
    x = np.load(case['input']); alpha = np.load(case['alpha'])
    ev = np.load(case['evaluation']); keep = ev['input_mask']; ids = ev['reference_ids']
    assert keep.dtype == bool and keep.shape == (len(x),) and keep.any() and len(ids)
    rows = []; distances = {}
    for rec in case['records']:
        row = {k: v for k, v in rec.items() if k not in ('path', 'sha256')}
        row.update(case=case['case'], scene=case['scene'], n_input=int(keep.sum()), n_reference=len(ids))
        if rec['status'] != 'OK': rows.append(row); continue
        assert sha(rec['path']) == rec['sha256']
        output = np.load(rec['path'])
        a = tree.query(output[keep], workers=1)[0]
        b = cKDTree(output).query(ref[ids], workers=1)[0]
        precision = float(np.mean(a <= 1.)); recall = float(np.mean(b <= 1.))
        row.update(accuracy_mm=float(a.mean()), p95_mm=float(np.quantile(a, .95)),
                   completeness_mm=float(b.mean()), precision=precision, recall=recall,
                   fscore=2*precision*recall/(precision+recall) if precision+recall else 0.)
        rows.append(row); distances[rec['method']] = a
    np.savez_compressed(dest/'distances'/f'{case["case"]}.npz', **distances)
    probes = {name: distances[name] for name in ('probe_t0.5', 'probe_t1')}
    confidence = diagnose(alpha[keep], distances['native__identity'], probes)
    confidence.update(case=case['case'], scene=case['scene'], n_scored_points=int(keep.sum()),
                      evidence_role='exposed_fixed_action_diagnostic')
    # Preserve per-point values only for independent arithmetic checks, not scene counts.
    np.savez_compressed(dest/'confidence'/f'{case["case"]}.npz', alpha=alpha[keep],
                        baseline_errors=distances['native__identity'],
                        probe_t05_errors=probes['probe_t0.5'], probe_t1_errors=probes['probe_t1'])
    save(dest/'confidence'/f'{case["case"]}.json', confidence)
    print('SCORED_CASE', case['case'], len(rows), flush=True)
    return rows, confidence


def paired(candidate, control):
    keys = sorted(set(candidate) & set(control))
    if not keys: return dict(n=0)
    a = np.array([candidate[k]['accuracy_mm'] for k in keys])
    b = np.array([control[k]['accuracy_mm'] for k in keys])
    gain = b-a
    rng = np.random.default_rng(92424)
    boots = gain[rng.integers(0, len(gain), size=(2000, len(gain)))].mean(axis=1)
    return dict(n=len(keys), mean_gain_mm=float(gain.mean()), median_gain_mm=float(np.median(gain)),
                relative_gain_percent=float(100*(1-a.mean()/b.mean())),
                within_scene_patch_bootstrap_ci95_mm=np.quantile(boots, [.025, .975]).tolist(),
                wins=int(np.sum(gain > 1e-10)), ties=int(np.sum(abs(gain) <= 1e-10)),
                losses=int(np.sum(gain < -1e-10)),
                recall_difference_pp=float(100*np.mean([candidate[k]['recall']-control[k]['recall'] for k in keys])),
                fscore_difference=float(np.mean([candidate[k]['fscore']-control[k]['fscore'] for k in keys])))


def summarize(rows, confidence):
    metrics = ('accuracy_mm', 'p95_mm', 'completeness_mm', 'recall', 'fscore',
               'displacement_rms_mm', 'displacement_max_mm', 'moved_fraction')
    summaries = []; comparisons = []; confidence_summary = []
    for scene in ('scan24', 'scan37'):
        rr = [r for r in rows if r['scene'] == scene and r['status'] == 'OK']
        index = {(r['case'], r['method']): r for r in rr}
        cases = sorted(set(r['case'] for r in rr))
        def method(name): return {c: index[(c, name)] for c in cases if (c, name) in index}
        identity = method('native__identity')
        for name in sorted(set(r['method'] for r in rr)):
            selected = list(method(name).values()); first = selected[0]
            entry = dict(scene=scene, method=name, n=len(selected), kind=first['kind'])
            for k in metrics: entry[k] = float(np.mean([r[k] for r in selected]))
            for k in ('field', 'allocation', 'requested_budget_mm', 'seed'):
                if k in first: entry[k] = first[k]
            if 'actual_budget_mm' in first:
                entry.update(actual_budget_mean_mm=float(np.mean([r['actual_budget_mm'] for r in selected])),
                             duplicate_budget_patches=sum(r['duplicate_budget'] for r in selected),
                             uninformative_patches=sum(not r['direction_informative'] for r in selected),
                             max_normalization_factor=max(r['normalization_factor'] for r in selected))
            entry.update(paired(method(name), identity)); summaries.append(entry)
        for requested in BUDGETS:
            prefix = f'budget{requested:g}__'
            contrasts = [('local_plane64__uniform', 'quadratic64__uniform'),
                         ('multiscale_full__uniform', 'quadratic64__uniform'),
                         ('multiscale_full__alpha', 'multiscale_full__uniform')]
            for a, b in contrasts:
                comparisons.append(dict(scene=scene, requested_budget_mm=requested,
                                        candidate=a, control=b, **paired(method(prefix+a), method(prefix+b))))
            shuffle = {}
            for c in cases:
                group = [index[(c, prefix+f'multiscale_full__shuffle_{s}')]
                         for s in SHUFFLE_SEEDS if (c, prefix+f'multiscale_full__shuffle_{s}') in index]
                if len(group) == len(SHUFFLE_SEEDS):
                    shuffle[c] = {k: float(np.mean([r[k] for r in group])) for k in metrics}
            comparisons.append(dict(scene=scene, requested_budget_mm=requested,
                                    candidate='multiscale_full__alpha', control='mean_of_5_shuffles_per_patch',
                                    **paired(method(prefix+'multiscale_full__alpha'), shuffle)))
        for probe in ('probe_t0.5', 'probe_t1'):
            cc = [c['probes'][probe] for c in confidence if c['scene'] == scene]
            rho = [c['spearman_rank_correlation'] for c in cc if c['spearman_rank_correlation'] is not None]
            confidence_summary.append(dict(scene=scene, probe=probe, n_patches=len(cc), n_defined_rho=len(rho),
                                           mean_patch_spearman=float(np.mean(rho)) if rho else None,
                                           median_patch_spearman=float(np.median(rho)) if rho else None,
                                           positive_rho_patches=sum(r > 0 for r in rho)))
    return summaries, comparisons, confidence_summary


def main():
    assert socket.gethostname() == 'liekkas', 'wrong target host'
    started = time.perf_counter()
    dest = Path(tempfile.mkdtemp(prefix='field-budget-v24-', dir=RUNS))
    print('RUN', dest, flush=True)
    for name in ('inputs', 'outputs', 'evaluation', 'distances', 'confidence', 'source'):
        (dest/name).mkdir()
    historic = [p for p in ROOT.rglob('*') if p.is_file() and HERE not in p.parents
                and '.git' not in p.parts and '__pycache__' not in p.parts]
    historic += [p for p in CACHE.rglob('*') if p.is_file()]
    lock = {str(p): sha(p) for p in sorted(set(historic))}
    save(dest/'HISTORY_LOCK.json', lock)
    sources = [HERE/n for n in ('PROTOCOL.md', 'field_ops.py', 'confidence.py', 'run.py',
                                'test_field_ops.py', 'test_confidence.py')]
    source_lock = {str(p): sha(p) for p in sources}
    save(dest/'SOURCE_LOCK.json', source_lock)
    for p in sources: shutil.copyfile(p, dest/'source'/p.name)
    cases = prepare(dest)
    construction_seconds = time.perf_counter()-started
    # This is the first point at which the independent reference is opened.
    save(dest/'REFERENCE_MANIFEST.json', {scene: dict(path=str(p), sha256=sha(p)) for scene, p in REFERENCES.items()})
    rows = []; confidence = []; eval_start = time.perf_counter()
    for scene in ('scan24', 'scan37'):
        ref = reference(REFERENCES[scene]); tree = cKDTree(ref)
        with futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(score_case, case, ref, tree, dest) for case in cases if case['scene'] == scene]
            for f in futures.as_completed(pending):
                r, c = f.result(); rows.extend(r); confidence.append(c)
        del tree, ref
    rows.sort(key=lambda r: (r['scene'], r['case'], r['method']))
    confidence.sort(key=lambda r: r['case'])
    save(dest/'RESULTS.json', rows); write_csv(dest/'RESULTS.csv', rows)
    save(dest/'CONFIDENCE.json', confidence)
    summaries, comparisons, confidence_summary = summarize(rows, confidence)
    save(dest/'SUMMARIES.json', summaries); write_csv(dest/'SUMMARIES.csv', summaries)
    save(dest/'COMPARISONS.json', comparisons); write_csv(dest/'COMPARISONS.csv', comparisons)
    save(dest/'CONFIDENCE_SUMMARY.json', confidence_summary)
    changed = [p for p, h in lock.items() if not Path(p).exists() or sha(p) != h]
    changed_source = [p for p, h in source_lock.items() if sha(p) != h]
    save(dest/'RUN_SUMMARY.json', dict(n_patches=len(cases), n_outputs=len(rows),
                                      failed_outputs=sum(r['status'] != 'OK' for r in rows),
                                      construction_and_prelock_seconds=construction_seconds,
                                      evaluation_analysis_and_integrity_seconds=time.perf_counter()-eval_start,
                                      total_seconds=time.perf_counter()-started,
                                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                      threads=4, gpu_used=False, cached_fit_cost_excluded=True,
                                      historical_files_checked=len(lock), changed_historical_files=changed,
                                      changed_source_files=changed_source,
                                      evidence_role='both_scenes_exposed_diagnostic',
                                      default_promoted=False))
    assert not changed and not changed_source
    print('DONE', dest, json.dumps(read(dest/'RUN_SUMMARY.json')), flush=True)


if __name__ == '__main__': main()

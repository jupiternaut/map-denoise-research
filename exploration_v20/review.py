"""Independent arithmetic and sealed-output checks; no fitting or selection changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in ('', None)]
    return float(np.mean(vals)) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    dest = args.run
    results, largest, checked, folds_checked = {}, 0., 0, 0
    for phase in ('exposed','confirmation'):
        sealed = read(dest/f'{phase}_SEALED_BEFORE_GT.json')
        rows = read(dest/f'{phase}_ROWS.json')
        index = {(r['case'],r.get('method')):r for r in rows}
        assert all(r['status']=='OK' for r in sealed), 'retain failed cases; no positive conclusion'
        for bundle in sealed:
            job = bundle['job']
            assert sha(job['input'])==job['input_sha256']
            assert sha(job['evaluation'])==job['evaluation_sha256']
            assert sha(job['cache'])==job['cache_sha256']
            cache = read(job['cache'])
            with np.load(job['input'], allow_pickle=False) as inp:
                observed = inp['xyz_world']*1000.
            with np.load(job['evaluation'], allow_pickle=False) as gt:
                reference = gt['gt_clean_xyz_world']*1000.
                labels, levels = gt['gt_layer'], gt['true_means_mm']
            x = np.asarray(cache['x'])
            for record in bundle['records']:
                assert sha(record['output'])==record['sha256']
                model = read(record['output'])
                points = observed.copy()
                if not model.get('identity'):
                    points[:,2] = model['prediction']
                    surfaces = np.asarray(model['means'])[None,:] + (x[:,1:]@np.asarray(model['slope']))[:,None]
                    posterior = np.asarray(model['posterior'])
                    np.testing.assert_allclose(posterior.sum(1),1.,rtol=0,atol=1e-9)
                    assert posterior.shape == surfaces.shape
                    assert np.min(posterior)>=0
                    np.testing.assert_allclose(points[:,2],surfaces[np.arange(len(x)),model['groups']],rtol=0,atol=1e-8)
                d = abs(points[:,2,None]-levels)
                metrics = dict(surface_mae_mm=float(d.min(1).mean()),
                               balanced_source_mae_mm=float(np.mean([d[labels==k,k].mean() for k in np.unique(labels)])),
                               matched_rms_mm=float(np.sqrt(np.mean(np.sum((points-reference)**2,axis=1)))),
                               coverage_1mm=float(np.mean(cKDTree(points).query(reference)[0]<=1.+1e-9)))
                if job['gap']:
                    metrics['source_gap_error_mm'] = float(abs(points[labels==1,2].mean()-points[labels==0,2].mean()-job['gap']))
                row = index[(job['case'],record['method'])]
                for key, value in metrics.items():
                    diff = abs(value-float(row[key]))
                    largest = max(largest,diff)
                    assert diff<1e-8,(job['case'],record['method'],key,diff)
                checked += 1
            diagnostic = read(dest/'diagnostics'/f'{job["case"]}__repaired_gate.json')
            if not diagnostic.get('single_passthrough'):
                gate = diagnostic['insample_gate']
                for key in ('means','slope','gate','feature_center','feature_scale'):
                    np.testing.assert_allclose(gate[key],cache['baseline'][key],rtol=0,atol=1e-8)
                for fold in diagnostic['crossfit']['training']:
                    tr, va = set(fold['train_indices']), set(fold['validation_indices'])
                    assert not tr&va and tr|va==set(range(len(x)))
                    assert set(fold['full_diagnostic']['pool'])=={'S','C','L','R'}
                    assert len(fold['model']['prediction'])==len(tr)
                    folds_checked += 1
        summary = {}
        for subset_name, predicate in [('all',lambda r:True),('dual',lambda r:r['gap']>0),('single',lambda r:r['gap']==0)]:
            selected = [r for r in rows if predicate(r)]
            summary[subset_name] = {method:dict(n=len(rr),**{k:mean(rr,k) for k in ('surface_mae_mm','balanced_source_mae_mm','matched_rms_mm','source_gap_error_mm','coverage_1mm')})
                                   for method in sorted({r['method'] for r in selected})
                                   for rr in [[r for r in selected if r['method']==method]]}
        choices = read(dest/f'{phase}_ORACLE_DIAGNOSTIC.json')
        for choice in choices:
            candidate_rows = [r for r in rows if r['case']==choice['case'] and r['method'].startswith('candidate_')]
            winner = min(candidate_rows,key=lambda r:(r['balanced_source_mae_mm'],r['method']))
            assert winner==choice['oracle']
            assert choice['headroom_source_mae_mm']>=-1e-10
        summary['oracle_dual'] = dict(n=sum(c['gap']>0 for c in choices),
            G_mm=mean([c for c in choices if c['gap']>0],'headroom_source_mae_mm'),
            **{k:mean([c['oracle'] for c in choices if c['gap']>0],k) for k in ('balanced_source_mae_mm','surface_mae_mm','matched_rms_mm','source_gap_error_mm','coverage_1mm')})
        summary['per_seed_dual'] = []
        for seed in sorted({r['seed'] for r in rows}):
            for method in ('v18_map','old_insample_gate_map','old_crossfit_gate_map','repaired_insample_map','repaired_crossfit_map'):
                rr=[r for r in rows if r['seed']==seed and r['gap']>0 and r['method']==method]
                if rr:summary['per_seed_dual'].append(dict(seed=seed,method=method,n=len(rr),source_mae_mm=mean(rr,'balanced_source_mae_mm'),surface_mae_mm=mean(rr,'surface_mae_mm')))
        results[phase]=summary
    for manifest in ('SOURCE_LOCK.json','OLD_FILES_BEFORE.json'):
        assert all(sha(path)==value for path,value in read(dest/manifest).items())
    result = dict(checked_outputs=checked,checked_training_folds=folds_checked,max_metric_difference=largest,
                  source_and_history_hashes_unchanged=True,results=results,
                  limitation='Arithmetic validation only; GT-selected oracle is not a deployed method.')
    with (dest/'INDEPENDENT_REVIEW.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(dict(checked_outputs=checked,checked_folds=folds_checked,max_difference=largest,
                          confirmation_dual=results['confirmation']['dual'],
                          oracle=results['confirmation']['oracle_dual']),indent=2))


if __name__=='__main__':
    main()

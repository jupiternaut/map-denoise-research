"""One bounded fit from archived V28 labels; never open scenes or references."""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import time
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'package'))
from v28_closeout.runtime import MODELS, PARAMS, schema


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v28', required=True, type=Path)
    args = parser.parse_args()
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('expected host liekkas')
    old = args.v28.resolve(strict=True)
    if old == ROOT or old in ROOT.parents:
        raise ValueError('source/output paths must be separate')
    dest = ROOT/'package/v28_closeout/models'
    if dest.exists() or (ROOT/'TRAINING_LOCK.json').exists():
        raise FileExistsError('release already prepared; do not overwrite')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'package').rglob('*'))
               if p.is_file()}
    sources['prepare_release.py'] = sha(__file__)
    old_sources = {name: sha(old/name) for name in (
        'selector_pipeline.py', 'surfacelet.py', 'direct_evidence.py',
        'graph_field.py', 'run_real.py', 'SELECTOR_PROTOCOL.md', 'PROTOCOL.md')}
    selector_seal = json.loads((old/'selector_results/SEALED.json').read_text())
    raw_seal = json.loads((old/'real_results/SEALED.json').read_text())
    data = {name: [[], []] for name in MODELS}
    records = []
    definition = schema()
    for scene, test in ((24, 37), (37, 24)):
        fold = old/'selector_results'/f'train{scene}_test{test}'
        fold_lock = json.loads((fold/'MODEL_LOCK.json').read_text())
        if fold_lock['params'] != PARAMS:
            raise AssertionError('original parameter mismatch')
        files = sorted(fold.glob('*_training_labels.npz'))
        if len(files) != 12:
            raise AssertionError('expected 12 archived training cases per scene')
        for label_path in files:
            case = label_path.name.removesuffix('_training_labels.npz')
            if not case.startswith(f'scan{scene}_'):
                raise AssertionError('unexpected training scene')
            feature_path = old/'real_results'/case/'features.npz'
            schema_path = feature_path.with_name('FEATURE_SCHEMA.json')
            label_hash, feature_hash = sha(label_path), sha(feature_path)
            if selector_seal['files'][str(label_path.relative_to(old/'selector_results'))] != label_hash:
                raise AssertionError('archived labels changed')
            if raw_seal['files'][str(feature_path.relative_to(old/'real_results'))] != feature_hash:
                raise AssertionError('archived features changed')
            if json.loads(schema_path.read_text()) != definition:
                raise AssertionError('feature schema changed')
            with np.load(label_path, allow_pickle=False) as labels, np.load(feature_path, allow_pickle=False) as features:
                ids = labels['row_ids']
                if len(ids) == 0 or len(np.unique(ids)) != len(ids):
                    raise AssertionError('invalid training row IDs')
                samples = {'row_ids': array_sha(ids)}
                for name, (feature_key, target_key) in MODELS.items():
                    x, y = features[feature_key][ids], labels[target_key]
                    if x.shape != (len(ids), len(definition[feature_key])) or y.shape != (len(ids),):
                        raise AssertionError('training shape mismatch')
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise AssertionError('nonfinite training data')
                    data[name][0].append(x)
                    data[name][1].append(y)
                    samples[name] = {'x_sha256': array_sha(x), 'y_sha256': array_sha(y)}
                records.append({'scene': scene, 'case': case, 'rows': int(len(ids)),
                                'labels': str(label_path.relative_to(old)), 'labels_sha256': label_hash,
                                'features': str(feature_path.relative_to(old)), 'features_sha256': feature_hash,
                                'samples': samples})
    deps = {name: importlib.metadata.version(name) for name in
            ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')}
    total = sum(record['rows'] for record in records)
    lock = {'status': 'pre_fit_frozen', 'host': socket.gethostname(), 'source_workspace': str(old),
            'output_workspace': str(ROOT), 'training_scenes': [24, 37], 'rows_per_model': total,
            'rows_by_scene': {str(s): sum(r['rows'] for r in records if r['scene'] == s) for s in (24, 37)},
            'case_order': [r['case'] for r in records], 'records': records, 'parameters': PARAMS,
            'sources_sha256': sources, 'v28_sources_sha256': old_sources,
            'feature_schema_sha256': sha(ROOT/'package/v28_closeout/FEATURE_SCHEMA.json'),
            'dependencies': deps, 'python': platform.python_version(), 'threads': 1, 'gpu': False,
            'primary': 'post_A_keep', 'secondary': 'post_AB_keep', 'threshold': 0,
            'no_new_gt_access': True, 'no_scene_query': True, 'no_hyperparameter_search': True,
            'evidence_status': 'New unified fit for future confirmation; not V28 fold performance.'}
    save(ROOT/'TRAINING_LOCK.json', lock)
    dest.mkdir()
    models = {}
    started = time.monotonic()
    with threadpool_limits(limits=1):
        for name, (xs, ys) in data.items():
            x, y = np.concatenate(xs), np.concatenate(ys)
            model = HistGradientBoostingRegressor(**PARAMS).fit(x, y)
            joblib.dump(model, dest/(name+'.joblib'))
            # JSON is a human-auditable model specification, not a second model fit.
            spec = {'name': name, 'estimator': 'sklearn.ensemble.HistGradientBoostingRegressor',
                    'params': model.get_params(), 'n_features': int(model.n_features_in_),
                    'n_iter': int(model.n_iter_), 'training_rows': len(x),
                    'training_x_sha256': array_sha(x), 'training_y_sha256': array_sha(y),
                    'feature_key': MODELS[name][0], 'target': MODELS[name][1],
                    'artifact_sha256': sha(dest/(name+'.joblib'))}
            save(dest/(name+'.json'), spec)
            models[name] = {'sha256': spec['artifact_sha256'], 'spec_sha256': sha(dest/(name+'.json')),
                            'n_features': int(model.n_features_in_), 'rows': len(x)}
            print(name, len(x), x.shape[1], 'fitted', flush=True)
    if any(sha(old/name) != digest for name, digest in old_sources.items()):
        raise AssertionError('old sources changed')
    manifest = {'models': models, 'training_lock_sha256': sha(ROOT/'TRAINING_LOCK.json'),
                'feature_schema_sha256': lock['feature_schema_sha256'], 'dependencies': deps,
                'wall_seconds': time.monotonic()-started, 'primary': 'post_A_keep',
                'secondary': 'post_AB_keep', 'development_fit_only': True}
    save(dest/'MODEL_LOCK.json', manifest)
    print(json.dumps({'rows_per_model': total, 'rows_by_scene': lock['rows_by_scene'],
                      'wall_seconds': manifest['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()

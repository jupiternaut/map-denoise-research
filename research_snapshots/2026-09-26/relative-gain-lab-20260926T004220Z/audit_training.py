"""Check prepared development arrays directly against read-only archived artifacts."""
from pathlib import Path
import hashlib
import json
import socket
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert socket.gethostname() == 'liekkas'
    manifest = json.loads((ROOT/'data/TRAIN_MANIFEST.json').read_text())
    raw_seal = json.loads((OLD/'real_results/SEALED.json').read_text())['files']
    label_seal = json.loads((OLD/'selector_results/SEALED.json').read_text())['files']
    counts, maximum = [], 0.0
    with np.load(ROOT/'data/train.npz', allow_pickle=False) as data:
        n = len(data['gain'])
        assert n == 79594 and data['X'].shape == (n, 64)
        assert set(np.unique(data['scene'])) == {24, 37}
        assert set(np.unique(data['condition'])) == {'native', 'minus3', 'plus3'}
        assert len(manifest['records']) == 24 and len(np.unique(data['case_id'])) == 24
        assert all(np.isfinite(data[key]).all() for key in ('X', 'gain', 'e0', 'e1'))
        assert np.array_equal(data['gain'], data['e0'] - data['e1'])
        assert (data['e0'] >= 0).all() and (data['e1'] >= 0).all()
        assert int(data['calibration_support'].sum()) == 78598
        for record in manifest['records']:
            case, scene = record['case'], record['scene']
            selected = data['case_id'] == record['case_id']
            assert selected.sum() == record['rows']
            other = 37 if scene == 24 else 24
            path = OLD/'selector_results'/f'train{scene}_test{other}'/(case+'_training_labels.npz')
            with np.load(path, allow_pickle=False) as labels:
                assert np.array_equal(data['row_id'][selected], labels['row_ids'])
                difference = float(np.max(np.abs(data['gain'][selected] - labels['A_gain'])))
                maximum = max(maximum, difference)
                assert difference <= 1e-8
            with np.load(OLD/'real_results'/case/'features.npz', allow_pickle=False) as features:
                assert np.array_equal(data['X'][selected], features['A_all_post'][data['row_id'][selected]])
            native_case = record['roi'] + '__native'
            with np.load(OLD/'evaluation'/native_case/'row_diagnostics.npz', allow_pickle=False) as diag:
                support = diag['input_support'][data['row_id'][selected]]
                assert np.array_equal(data['calibration_support'][selected], support)
            assert (data['scene'][selected] == scene).all()
            assert (data['condition'][selected] == record['condition']).all()
            counts.append({'case': case, 'rows': int(selected.sum()), 'calibration_rows': int(support.sum()),
                           'archived_gain_max_abs_difference': difference})
    checked = 0
    for name, recorded in manifest['source_sha256'].items():
        path = Path(name)
        if path.is_relative_to(OLD/'real_results'):
            expected = raw_seal[str(path.relative_to(OLD/'real_results'))]
        elif path.is_relative_to(OLD/'selector_results'):
            expected = label_seal[str(path.relative_to(OLD/'selector_results'))]
        else:
            continue
        actual = sha(path)
        assert actual == recorded == expected, str(path)
        checked += 1
    report = {'status': 'PASS', 'training_rows': n, 'calibration_rows': 78598,
              'cases': counts, 'archived_gain_max_abs_difference': maximum,
              'archived_source_hash_checks': checked,
              'training_npz_sha256': sha(ROOT/'data/train.npz'),
              'method': 'independent direct comparison to archived row IDs, X, gain, native support, sealed hashes'}
    with (ROOT/'AUDIT_TRAINING.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}))


if __name__ == '__main__':
    main()

"""Read-only checks of stored rendering/matching/decision artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
report = json.loads((root/'results/results.json').read_text())
protocol = json.loads((root/'PROTOCOL.json').read_text())
assert hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest() == report['protocol_hash']
assert len(report['cases']) == 12
for case in report['cases']:
    case_root = root/'results'/f'{case["scenario"]}_{case["seed"]}'
    rendered = np.load(case_root/'rendered.npz')
    measured = np.load(case_root/'observations.npz')
    assert rendered['photos'].shape == (3, 721, 3)
    assert rendered['layers'][0,351] == 1
    assert abs(rendered['physical_x'][0,351]+.3) < 1e-12
    # The pair set is exactly the thresholded measured full source matrix.
    np.testing.assert_array_equal(measured['all_source_pairs'], np.argwhere(measured['pair_ncc'] >= .78))
    star = case['arms']['star']
    common = case['arms']['common_track']
    star_pairs = {(p['u1'],p['u2']) for p in star['pairs']}
    assert all((p['u1'],p['u2']) in star_pairs and p['ncc12'] >= .78 for p in common['pairs'])
    for arm in [star, common]:
        assert arm['accept_b'] == (arm['gain_bounds'] is not None and arm['gain_bounds'][0] > 0)
        assert arm['true_target_covered'] == any(a <= 6 <= b for a,b in arm['intervals'])
    if case['scenario'] == 'occlusion_wrong_layer':
        assert case['posthoc_target_visible'] == [True,True,False]
        assert common['pairs'][0]['posthoc_source_layers'] == [1,0]
    if case['scenario'] == 'common_camera_error':
        assert all(p['posthoc_same_true_target'] for p in common['pairs'])
        assert not common['true_target_covered']
print('PASS: all 12 stored scenes, full measured source match sets, nesting, decisions, and identity diagnostics checked.')

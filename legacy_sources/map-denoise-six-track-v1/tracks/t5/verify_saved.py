"""Read-only validation of saved final points, branch masks, and source hashes."""
import hashlib
import json
from pathlib import Path
import numpy as np
RUN=Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t5')
data=json.loads((RUN/'results.json').read_text())
for filename,expected in data['source_hashes'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,filename
comparisons=0; decisions=0
for record in data['comparisons']:
    case,arm=record['case'],record['arm']
    ref=np.load(RUN/f'{case}-reference.npy',allow_pickle=False)
    got=np.load(RUN/f'{case}-{arm}.npy',allow_pickle=False)
    assert ref.shape==got.shape and np.isfinite(got).all()
    delta=np.linalg.norm(ref-got,axis=1)
    assert not len(delta) or float(delta.max())<1e-9
    with np.load(RUN/f'{case}-reference-branches.npz') as left,np.load(RUN/f'{case}-{arm}-branches.npz') as right:
        assert left.files==right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key],right[key]); decisions+=left[key].size
    comparisons+=1
assert all(x['branch_mismatches']==0 for x in json.loads((RUN/'residual-checks.json').read_text()))
assert all(x['passed'] for x in json.loads((RUN/'inherited-unit-tests.json').read_text()))
print(json.dumps({'saved_output_comparisons':comparisons,'saved_binary_decisions_compared':decisions,'source_hashes_match':True,'status':'PASS'}))

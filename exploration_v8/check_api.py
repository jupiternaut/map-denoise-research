"""Compare full legal-input API to cached outputs; no evaluator or truth reads."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
import numpy as np
from compensation import estimate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();assert socket.gethostname()=='liekkas'
    records=json.loads((run/'OUTPUTS_SEALED_BEFORE_GT.json').read_text())
    selected=[r for r in records if r['mode']=='conditional' and r['budget']==6
              and r['case'] in ('dual_g2_s912101_b0','ghost_s912101_b0')]
    assert len(selected)==2
    result=[]
    for r in selected:
        with np.load(r['input'],allow_pickle=False) as p:
            world=p['xyz_world'].copy();scans=p['scan_id'].copy()
        before=world.copy();start=time.perf_counter()
        xyz,info,artifacts=estimate(world,scans,1.,budget=6)
        elapsed=time.perf_counter()-start
        np.testing.assert_array_equal(before,world)
        with np.load(r['output'],allow_pickle=False) as p:
            max_error_mm=float(np.max(abs(xyz-p['xyz_world']))*1000)
            np.testing.assert_allclose(xyz,p['xyz_world'],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(artifacts['support_mask'],p['support_mask'])
        result.append(dict(case=r['case'],max_error_mm=max_error_mm,full_api_seconds=elapsed,
                           input_unchanged=True,output_support_matches=True))
    obj=dict(cases=result,truth_fields_used=[],
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             note='two-case integration check, not a timing benchmark; includes search and discarded old refit')
    with (run/'API_CHECK.json').open('x') as f:json.dump(obj,f,indent=2)
    print(json.dumps(obj,indent=2))

if __name__=='__main__':main()

"""Seal current V11 source and replay full legal-input API against confirmation."""
import json,sys,tempfile,socket
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'exploration_v9'))
from run_v9 import RUNS,read,sha,save
from api import estimate

def main():
    assert socket.gethostname()=='liekkas'
    base=RUNS/'spatial-v11-confirmation-xjro7mgm';case='dual_g2_s9131011_b4'
    data=read(base/'inputs'/(case+'.npz'))
    out,info,_=estimate(data['xyz_world'],data['scan_id'],1.,iterations=12)
    np.testing.assert_array_equal(out,read(base/'outputs'/(case+'__spatial_free_i12.npz'))['xyz_world'])
    target=Path(tempfile.mkdtemp(prefix='spatial-v11-checkpoint-',dir=RUNS));(target/'source').mkdir()
    import shutil
    hashes={str(p):sha(p) for p in HERE.iterdir() if p.is_file()}
    for p in hashes:shutil.copyfile(p,target/'source'/Path(p).name)
    save(target/'CHECKPOINT.json',dict(host=socket.gethostname(),source_sha256=hashes,full_api_exact_replay=True,
        full_api_seconds=info['full_seconds'],claim='same-family synthetic geometry improvement; real transfer not established',
        confirmation=str(base),audit=str(RUNS/'spatial-v11-audit-sec9reh1/AUDIT.json')))
    print(target)
if __name__=='__main__':main()

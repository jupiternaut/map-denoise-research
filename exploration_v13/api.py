"""Legal-input experimental API; does not overwrite a source map."""
import importlib.util,time
from pathlib import Path
import numpy as np
from atlas import prepare,decode
ROOT=Path(__file__).resolve().parents[1]

def estimate(xyz_world_m,scan_id,sigma_mm,mode='atlas_tied',neighbors=256):
    spec=importlib.util.spec_from_file_location('_atlas_api_v7',ROOT/'exploration_v7/algorithm/reassociation.py')
    v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
    start=time.perf_counter();state=v7.freeze(xyz_world_m,scan_id,sigma_mm)
    if 'design' not in state:return np.array(xyz_world_m,copy=True),dict(status='UNSUPPORTED'),{}
    p=prepare(state,sigma_mm);out,info,art=decode(state,p,sigma_mm,mode,neighbors)
    info.update(full_api_seconds=time.perf_counter()-start,status='EXPERIMENTAL_OUTPUT',
        scope='No demonstrated overall win on real geometry; local reassociation cannot invent extra global components.')
    return out,info,art

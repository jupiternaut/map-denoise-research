"""Prototype NPZ API. This is not a validated general real-map filter."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
from spatial_filter import filter_frozen
ROOT=Path(__file__).resolve().parents[1]

def estimate(xyz_world_m,scan_id,sigma_mm,method='spatial_free',iterations=12):
    spec=importlib.util.spec_from_file_location('_v11_api_v7',ROOT/'exploration_v7/algorithm/reassociation.py')
    v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
    start=time.perf_counter();state=v7.freeze(xyz_world_m,scan_id,sigma_mm)
    if 'design' not in state:
        out=np.array(xyz_world_m,copy=True)
        return out,dict(status='UNSUPPORTED',full_seconds=time.perf_counter()-start),dict(support_mask=np.zeros(len(out),bool))
    out,info,art=filter_frozen(state,sigma_mm,method,iterations)
    info.update(status='EXPERIMENTAL_OUTPUT',full_seconds=time.perf_counter()-start,
        warning='Confirmed only on a parallel-layer synthetic family; real-patch transfer not established.')
    art.pop('xyz_world',None)
    return out,info,art

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sigma-mm',type=float,required=True);parser.add_argument('--iterations',type=int,choices=(12,36),default=12)
    parser.add_argument('--method',choices=('spatial_free','spatial_fixed','global_free'),default='spatial_free');a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.input,allow_pickle=False) as f:
        xyz=f['xyz_world'].copy();scan=f['scan_id'].copy()
    out,info,art=estimate(xyz,scan,a.sigma_mm,a.method,a.iterations)
    with a.output.open('xb') as f:np.savez_compressed(f,xyz_world=out,scan_id=scan,metadata_json=np.frombuffer(json.dumps(info).encode(),dtype=np.uint8),**art)
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()

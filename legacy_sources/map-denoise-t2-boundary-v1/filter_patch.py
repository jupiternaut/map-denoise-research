"""Usable local patch filter: explicit specializations, no truth-based routing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from common import Input
from run_suite import load_module, plain

ROOT=Path(__file__).resolve().parent
MODES={
    'fast':(ROOT/'baseline/scalar_reference.py','scalar_profile_hard'),
    'fixed-map':(ROOT/'candidate/proposal.py','framewise_proposal_hard'),
    'tilted-map':(ROOT/'baseline/plane_profile.py','plane_profile_map'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=MODES,default='fast')
    a=p.parse_args();meta_path=a.output.with_suffix('.json')
    if a.output.resolve()==a.input.resolve() or a.output.exists() or meta_path.exists():
        raise FileExistsError('Use a new output path; input and previous output are preserved.')
    with np.load(a.input,allow_pickle=False) as f:
        xyz=np.asarray(f['xyz_mm'],float);frame=np.asarray(f['frame'],int)
        roi=np.asarray(f['roi'],int) if 'roi' in f else np.zeros(len(xyz),int)
        sigma=float(f['sigma_mm']);bound=float(f['bias_bound_mm']) if 'bias_bound_mm' in f else 8*sigma
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('Finite N by 3 xyz_mm required')
    if frame.shape!=(len(xyz),) or roi.shape!=(len(xyz),) or not sigma>0:raise ValueError('Invalid frame, roi or sigma')
    # Keep frame provenance but make integer indexing compact.
    frame_ids,frame=np.unique(frame,return_inverse=True)
    inp=Input(xyz.copy(),frame,roi,sigma,bound)
    path,method=MODES[a.mode];sys.path.insert(0,str(path.parent));module=load_module('patch_cli',path)
    out,bias,info=module.estimate(inp,method)
    if out.shape!=xyz.shape or not np.isfinite(out).all():raise FloatingPointError('Invalid filter output')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:np.savez_compressed(f,xyz_mm=out,frame=frame_ids[frame],roi=roi,bias_mm=bias,frame_ids=frame_ids)
    metadata={'input':str(a.input.resolve()),'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'mode':a.mode,'method':method,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'units':'mm','points':len(xyz),'mean_displacement_mm':np.linalg.norm(out-xyz,axis=1).mean(),
        'scope':'locally approximately parallel surfaces; frame provenance and supplied noise scale required',
        'info':info}
    with meta_path.open('x') as f:json.dump(plain(metadata),f,indent=2)
    print(json.dumps({'output':str(a.output.resolve()),'mode':a.mode,'points':len(out)}))


if __name__=='__main__':main()

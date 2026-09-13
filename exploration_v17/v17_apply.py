"""Reusable estimator entry point, without benchmark baselines or evaluator data."""
from pathlib import Path
import sys,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import v17_operator as op

def estimate(design,height_mm,sigma_mm,selector='adaptive_bic'):
    if selector not in ('adaptive_bic','adaptive_cv'):raise ValueError(selector)
    pool=op.fit_pool(design,height_mm,sigma_mm)
    cv=op.cross_validation(design,height_mm,sigma_mm) if selector=='adaptive_cv' else None
    scores=cv['scores'] if cv is not None else {f:m['bic'] for f,m in pool.items()}
    family=op.choose(scores)
    return dict(pool[family],selected_family=family,selection_method=selector),dict(scores=scores,cv=cv)

def filter_local(points_mm,sigma_mm,selector='adaptive_bic'):
    """Local coordinate frame supplied by caller; modifies Z only, not arbitrary world normals."""
    points_mm=np.asarray(points_mm,float)
    if points_mm.ndim!=2 or points_mm.shape[1]!=3:raise ValueError('N x 3 points in a supplied local frame, millimetres')
    x=np.c_[np.ones(len(points_mm)),points_mm[:,:2]/50.]
    model,diagnostic=estimate(x,points_mm[:,2],sigma_mm,selector)
    output=points_mm.copy();output[:,2]=model['prediction']
    return output,model,diagnostic

def main():
    p=argparse.ArgumentParser(description='Filter a local-frame millimetre point cloud, not an arbitrary world-coordinate map.')
    p.add_argument('input',help='NPZ with local_xyz_mm array');p.add_argument('output');p.add_argument('--sigma-mm',type=float,required=True)
    p.add_argument('--selector',choices=['adaptive_bic','adaptive_cv'],default='adaptive_bic');args=p.parse_args()
    with np.load(args.input,allow_pickle=False) as f:points=f['local_xyz_mm'].copy()
    out,m,dg=filter_local(points,args.sigma_mm,args.selector)
    with Path(args.output).open('xb') as f:np.savez_compressed(f,local_xyz_mm=out,k=m['k'],means=m['means'],slope=m['slope'],
        groups=m['groups'],selected_family=m['selected_family'],sigma_mm=args.sigma_mm)
    print('family',m['selected_family'],'k',m['k'],'points',len(out))

if __name__=='__main__':main()

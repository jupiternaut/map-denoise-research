"""Matched information diagnostics, preserving the global kNN and update rule.

The original association-only oracle re-queried k points within each component,
which changes footprint (especially on cylinders). These additional arms mask
the SAME global kNN instead. This prevents interpreting footprint inflation as
the isolated effect of knowing association. Never deployable results.
"""
import argparse,json,sys,tempfile,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/home/grf/Documents/Codex/2026-09-10/map-denoise-v0')
from parallel_geometry_v5.association.operator import _planes,_spacing
from geometry import evaluate

def matched(points, labels=None, normals=None):
    p=points.copy();spacing=_spacing(p);floor=.05*spacing
    for _ in range(2):
        dist,ix=cKDTree(p).query(p,k=min(48,len(p)),workers=1)
        w=np.exp(-.5*(dist/np.maximum(.5*dist[:,-1,None],floor))**2)
        if labels is not None:w*=labels[ix]==labels[:,None]
        center,normal,values=_planes(p[ix],w)
        if normals is not None:normal=normals
        h=np.einsum('nkj,nj->nk',p[ix]-p[:,None,:],normal)
        bw=w*np.exp(-.5*(h/(.5*spacing))**2)
        delta=.8*normal*((bw*h).sum(axis=1)/np.maximum(bw.sum(axis=1),1e-30))[:,None]
        good=values[:,1]>np.maximum(values[:,2]*1e-10,floor**2*1e-5)
        delta[~good]=0
        length=np.linalg.norm(delta,axis=1)
        delta*=np.minimum(1.,2*spacing/np.maximum(length,floor))[:,None]
        p+=delta
    return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    source=Path(a.run);dest=Path(tempfile.mkdtemp(prefix='matched-oracles-',dir=source.parent));(dest/'outputs').mkdir()
    mf=json.loads((source/'manifest.json').read_text());rows=[]
    arms={'matched_no_oracle':(False,False),'matched_association_oracle':(True,False),
          'matched_normal_oracle':(False,True),'matched_both_oracle':(True,True)}
    (dest/'manifest.json').write_text(json.dumps({'source_run':str(source),'arms':arms,'scope':'diagnostics only; fixed global kNN, bandwidth and update'},indent=2))
    for file in sorted((source/'outputs').glob('*-input.npz')):
        case=file.name.removesuffix('-input.npz');family,condition,seed=case.rsplit('-',2)
        cfg=next(c for c in mf['conditions'] if c['name']==condition)
        with np.load(file) as d:p=d['xyz'];rot=d['rotation']
        with np.load(source/'outputs'/f'{case}-evaluator.npz') as d:
            ref=d['reference'];rl=d['ref_labels'];labels=d['labels'];normals=d['normals']@rot.T
        for arm,(assoc,norm) in arms.items():
            st=time.perf_counter();out=matched(p,labels if assoc else None,normals if norm else None);duration=time.perf_counter()-st
            path=dest/'outputs'/f'{case}-{arm}.npy';np.save(path,out,allow_pickle=False)
            row={'case':case,'family':family,'condition':condition,'seed':int(seed[1:]),'arm':arm,'oracle':assoc or norm,
                 'status':'ok','elapsed_s':duration,'metrics':evaluate(family,np.load(path)@rot,cfg['gap'],ref,rl,len(p))}
            if arm=='matched_no_oracle':
                old=np.load(source/'outputs'/f'{case}-old_A_disabled.npy')
                row['max_delta_from_prior_matched_baseline_m']=float(np.max(np.linalg.norm(out-old,axis=1)))
            rows.append(row)
    (dest/'results.json').write_text(json.dumps({'records':rows},indent=2))
    print(json.dumps({'matched_oracles':str(dest),'rows':len(rows),'no_oracle_max_delta_m':max(r.get('max_delta_from_prior_matched_baseline_m',0) for r in rows)}),flush=True)
if __name__=='__main__':main()

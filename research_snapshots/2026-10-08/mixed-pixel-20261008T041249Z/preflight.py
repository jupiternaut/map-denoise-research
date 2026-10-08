"""Three specified one-world timing probes, no geometric scoring or tuning."""
import time
import resource
import numpy as np
from fixtures import make_world,render_world,cameras
from predictor import estimate_aux,predict_curve
from run import ROOT,load,dump,now,combine_curves

def main():
    p=load(ROOT/'METHOD.json');grid=np.arange(300.,1001.,2.);cams=cameras();rows=[]
    for mechanism in ('flat_contrast','textured_boundary','flat_equal'):
        t=time.perf_counter();images=render_world(make_world(mechanism,1.65,1103),cams)
        rendering=time.perf_counter()-t;aux=[];fits=[]
        for train,ev in p['folds']:
            t=time.perf_counter();aux.append(estimate_aux(images[0],images[train],cams[0],cams[train],p));fits.append(time.perf_counter()-t)
        t=time.perf_counter()
        curves=[predict_curve(a,cams[0],cams[ev],images[ev],grid,540,p) for a,(_,ev) in zip(aux,p['folds'])]
        scan=time.perf_counter()-t;t=time.perf_counter();loss,counts,valid=combine_curves(curves);pool=time.perf_counter()-t
        rows.append(dict(mechanism=mechanism,render_seconds=rendering,aux_fit_seconds=fits,
                         two_fold_scan_seconds=scan,pooling_seconds=pool,pixel_counts=counts.tolist(),
                         auxiliary_valid=[bool(a['valid']) for a in aux],sigma_valid=[bool(a['sigma_valid']) for a in aux]))
    dump(ROOT/'preflight/RESULTS.json',dict(created=now(),rows=rows,
        estimated_two_stage_seconds=36*float(np.mean([sum(r['aux_fit_seconds'])+8*r['two_fold_scan_seconds'] for r in rows])),
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='timing and contracts only; no geometry scoring'))
    print(__import__('json').dumps(rows))
if __name__=='__main__':main()

from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import t
METHODS=('old_constant64','old_spatial36','constant_only','linear_only','rbf_only','adaptive_bic','adaptive_cv')
METRICS=('surface_mae_mm','balanced_source_mae_mm','matched_rms_mm','source_gap_error_mm','fitted_gap_error_mm','coverage_1mm','reference_sample_coverage_1mm')

def aggregate(rows):
    result={}
    for m in METHODS:
        rr=[r for r in rows if r['method']==m];du=[r for r in rr if r['gap']];si=[r for r in rr if not r['gap']]
        out={key:float(np.mean([r[key] for r in du if r.get(key) is not None])) for key in METRICS if any(r.get(key) is not None for r in du)}
        out.update(dual_n=len(du),dual_k1=sum(r['model_k']==1 for r in du),single_n=len(si),single_false_split=sum(r['model_k']==2 for r in si),
            dual_selected_family={f:sum(r['selected_family']==f for r in du) for f in ('S','C','L','R')},
            single_surface_mae_mm=float(np.mean([r['surface_mae_mm'] for r in si])) if si else None)
        result[m]=out
    return result

def paired(rows,a,b,metric):
    x={(r['seed'],r['case']):r[metric] for r in rows if r['method']==a}
    y={(r['seed'],r['case']):r[metric] for r in rows if r['method']==b}
    assert x.keys()==y.keys()
    seeds=sorted({k[0] for k in x});means=np.array([np.mean([x[k]-y[k] for k in x if k[0]==s]) for s in seeds])
    half=float(t.ppf(.975,len(means)-1)*means.std(ddof=1)/np.sqrt(len(means))) if len(means)>1 else 0.
    return dict(a=a,b=b,metric=metric,mean_a_minus_b=float(means.mean()),ci95=[float(means.mean()-half),float(means.mean()+half)],
                seeds=seeds,seed_differences=means.tolist(),scope='paired seed means, fixed condition grid; not real-scene confidence')

def main(dest):
    summary={}
    for phase in ('development','confirmation','bridge'):
        path=dest/f'{phase}_ROWS.json'
        if not path.exists():continue
        rows=json.loads(path.read_text());assert all(r['status']=='OK' for r in rows)
        summary[phase]=dict(overall=aggregate(rows),slices={})
        dual=[r for r in rows if r['gap']]
        for key in ('dependence','gap','sigma'):
            for value in sorted({r[key] for r in dual if key in r}):
                rr=[r for r in dual if r.get(key)==value]
                summary[phase]['slices'][f'{key}={value}']=aggregate(rr)
        comparisons=[]
        for a,b in [('old_spatial36','linear_only'),('old_spatial36','adaptive_bic'),('old_spatial36','adaptive_cv'),('linear_only','adaptive_bic'),('linear_only','adaptive_cv'),('old_constant64','constant_only')]:
            for metric in ('surface_mae_mm','balanced_source_mae_mm','source_gap_error_mm'):
                comparisons.append(paired(dual,a,b,metric))
        summary[phase]['paired']=comparisons
    out=dest/'SUMMARY.json'
    if out.exists():raise FileExistsError(out)
    with out.open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps({phase:s['overall'] for phase,s in summary.items()},indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))

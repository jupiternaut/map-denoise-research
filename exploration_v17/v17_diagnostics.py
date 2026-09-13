"""Read-only post-confirmation diagnostics; never change models or selection."""
from pathlib import Path
import sys,json
import numpy as np
from v17_summary import paired

def main(dest):
    result={}
    for phase in ('confirmation','bridge'):
        rows=json.loads((dest/f'{phase}_ROWS.json').read_text());dual=[r for r in rows if r['gap']]
        data=dict(paired_rms=paired(dual,'old_spatial36','adaptive_bic','matched_rms_mm'),single_cv=[],lambda_paired=[])
        for r in rows:
            if r['gap'] or r['method']!='adaptive_cv' or r['model_k']!=2:continue
            with np.load(dest/'outputs'/f'{r["case"]}__adaptive_cv.npz',allow_pickle=False) as f:
                gap=float(np.ptp(f['means']))
                groups=f['groups'];sizes=np.bincount(groups,minlength=2).tolist()
            data['single_cv'].append(dict(case=r['case'],fitted_local_gap_mm=gap,group_sizes=sizes,surface_mae_mm=r['surface_mae_mm']))
        if phase=='confirmation':
            for lam in (0.,.5,1.):
                rr=[r for r in dual if r['dependence']==lam]
                for metric in ('surface_mae_mm','balanced_source_mae_mm','matched_rms_mm','source_gap_error_mm'):
                    data['lambda_paired'].append(dict(dependence=lam,**paired(rr,'old_spatial36','adaptive_bic',metric)))
        a={r['case']:r for r in dual if r['method']=='old_spatial36'};b={r['case']:r for r in dual if r['method']=='adaptive_bic'}
        data['rms_increase_cases']=sorted([dict(case=c,old_rms_mm=a[c]['matched_rms_mm'],new_rms_mm=b[c]['matched_rms_mm'],
            difference_mm=b[c]['matched_rms_mm']-a[c]['matched_rms_mm'],old_k=a[c]['model_k'],new_k=b[c]['model_k'],family=b[c]['selected_family']) for c in a],key=lambda r:r['difference_mm'],reverse=True)
        result[phase]=data
    with (dest/'POSTHOC_DIAGNOSTICS.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({p:dict(paired_rms=v['paired_rms'],top_rms_increase=v['rms_increase_cases'][:3],single_cv=v['single_cv']) for p,v in result.items()},indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))

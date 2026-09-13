"""Posthoc objective audit: did the new pool miss the old spatial incumbent? No refits/selection edits."""
from pathlib import Path
import sys,json
import numpy as np
from v17_audit import load,nll

def main(dest):
    result={}
    for phase in ('confirmation','bridge'):
        records=json.loads((dest/f'{phase}_SEALED_BEFORE_GT.json').read_text());rows=[]
        for r in records:
            if r['method']!='old_spatial36':continue
            art=load(r['output'])
            if int(art['k'])!=2:continue
            dg=json.loads(Path(r['diagnostic']).read_text());new=dg['pool']['R']
            if phase=='bridge':x,y=art['design'],art['corrected']
            else:
                inp=load(r['input']);x,y=inp['design'],inp['height_mm']
            old=dict(art,family='R',feature_center=new['feature_center'],feature_scale=new['feature_scale'])
            cost_old=float(nll(old,x,y,r['sigma']).sum()+.05*np.sum(art['gate'][1:]**2))
            cost_new=float(new['nll']+.5*new['ridge_penalty'])
            rows.append(dict(case=r['case'],seed=r['seed'],gap=r['gap'],sigma=r['sigma'],
                old_objective=cost_old,new_objective=cost_new,new_minus_old=cost_new-cost_old))
        result[phase]=dict(n_old_double=len(rows),missed_by_more_than_1e_minus6=sum(r['new_minus_old']>1e-6 for r in rows),
            missed_by_more_than_1=sum(r['new_minus_old']>1 for r in rows),rows=sorted(rows,key=lambda r:r['new_minus_old'],reverse=True),
            scope='Same smooth objective evaluated at already saved parameters. No warmstart added, no output changed; next-development hypothesis only.')
    with (dest/'INCUMBENT_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({p:{**{k:v for k,v in d.items() if k!='rows'},'top':d['rows'][:5]} for p,d in result.items()},indent=2))

if __name__=='__main__':main(Path(sys.argv[1]))

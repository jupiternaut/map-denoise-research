"""Post-seal gate and paired-error accounting. Does not change predictions."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'evaluation/ROWS.json').read_text())
params=json.loads((ROOT/'METHOD.json').read_text())
lookup={(r['id'],r['arm']):r for r in rows}
records=[]
for r in rows:
    with np.load(r['curve_file']) as c:
        idx=int(np.argmin(abs(c['grid']-r['true_depth'])))
        mass=c['mass']>=params['min_mass']
        ess=c['ess']>=params['min_ess']-1e-10
        ref=c['ref_std']>=params['std_min']
        src=np.all(c['source_std']>=params['std_min'],axis=0)
        valid=np.all(c['valid_samples'] | (c['weights'][None,:,:]<=1e-14),axis=(0,2))
        counts=dict(total=len(mass),mass=int(mass.sum()),mass_ess=int((mass&ess).sum()),mass_ess_ref=int((mass&ess&ref).sum()),
                    mass_ess_ref_src=int((mass&ess&ref&src).sum()),mass_ess_ref_src_valid=int((mass&ess&ref&src&valid).sum()),accepted=int(c['accepted'].sum()))
        records.append(dict(id=r['id'],name=r['name'],seed=r['seed'],arm=r['arm'],selected_depth=r['selected_depth'],error=r['absolute_error'],empty=r['empty'],
          gate_survival=counts,maximum_mass=float(c['mass'].max()),maximum_ess=float(c['ess'].max()),
          at_true=dict(mass=float(c['mass'][idx]),ess=float(c['ess'][idx]),count=int(c['count'][idx]),ref_std=float(c['ref_std'][idx]),source_std=c['source_std'][:,idx].tolist(),
                       scores=[float(v) if np.isfinite(v) else None for v in c['scores'][:,idx]],qualified=bool(c['qualified'][idx]))))
comparisons=[]
for arm,baseline in [('oracle_component','oracle_majority'),('oracle_component','full9'),('estimated_footprint','estimated_center')]:
    for group in ('ring','other','all'):
        chosen=[r for r in rows if r['arm']==arm and (group=='all' or r['name'].startswith('ring')==(group=='ring'))]
        dif=[r['absolute_error']-lookup[r['id'],baseline]['absolute_error'] for r in chosen]
        comparisons.append(dict(arm=arm,baseline=baseline,group=group,n=len(chosen),mean_error_change=float(np.mean(dif)),improved=sum(v<-1e-9 for v in dif),
                                worsened=sum(v>1e-9 for v in dif),unchanged=sum(abs(v)<=1e-9 for v in dif)))
payload=dict(posthoc=True,records=records,comparisons=comparisons)
with (ROOT/'evaluation/GATE_DIAGNOSTIC.json').open('x') as f:json.dump(payload,f,indent=2,allow_nan=False)
print(json.dumps(dict(comparisons=comparisons,ring_component=[r for r in records if r['arm']=='oracle_component' and r['name'].startswith('ring')],
                     estimator_empty_reasons=[r for r in records if r['arm']=='estimated_footprint']),indent=2))

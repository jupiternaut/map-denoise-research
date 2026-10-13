"""Descriptive post-score residual accounting; not an inference method."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
folder=ROOT/'outputs/formal-v1'
rows=list(csv.DictReader((folder/'METRICS.csv').open()))
err=defaultdict(float); counts=defaultdict(int); groups={}
for r in rows:
    if r['budget_seconds']=='24.0' and r['arm'] in ('certified','legacy'):
        key=(r['observation_id'],r['arm'])
        err[key]+=float(r['final_se']); counts[key]+=1
        groups[r['observation_id']]={k:r[k] for k in ('branch','stratum','negative_control')}
records=[{'observation_id':oid,'certified_minus_legacy_mse':
          (err[(oid,'certified')]-err[(oid,'legacy')])/counts[(oid,'certified')],**labels}
         for oid,labels in groups.items()]
negative=sum(err[(o,'certified')] for o,g in groups.items() if g['negative_control']=='True')
total=sum(err[(o,'certified')] for o in groups)
result={'scope':'post-hoc descriptive analysis of frozen predictions; no parameter selection',
        'worlds':records,'certified_better':sum(x['certified_minus_legacy_mse'] < -1e-12 for x in records),
        'certified_worse':sum(x['certified_minus_legacy_mse'] > 1e-12 for x in records),
        'same':sum(abs(x['certified_minus_legacy_mse']) <=1e-12 for x in records),
        'negative_control_fraction_of_certified_squared_error':negative/total}
with (folder/'PAIRED_RESIDUALS.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='worlds'}))

"""Same-process cost context against existing vectorized classical bilateral code."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import csv
import importlib.util
import json
import numpy as np
from accelerated import OLD
from run import RUN, measured, write_json

path=OLD.parents[2]/'parallel_geometry_v4'/'baselines'/'operators.py'
spec=importlib.util.spec_from_file_location('t5_bilateral_context',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
data=json.loads((RUN/'results.json').read_text())
records=[]
for record in data['records']:
    if record['arm']!='reference' or 'metrics' not in record: continue
    case=record['case']; p=np.load(RUN/f'{case}-input.npy')
    timing=[]
    def fn(p,cfg): return mod.denoise(p,{'method':'bilateral_normal'})
    for _ in range(3): out,t=measured(fn,p);timing.append(t)
    np.save(RUN/f'{case}-bilateral-cost-context.npy',out,allow_pickle=False)
    compiled=next(x for x in data['records'] if x['case']==case and x['arm']=='fused')
    t0=float(np.median([t['seconds'] for t in record['timings']]))
    t1=float(np.median([t['seconds'] for t in compiled['timings']]))
    tb=float(np.median([t['seconds'] for t in timing]))
    records.append({'case':case,'bilateral_median_s':tb,'reference_median_s':t0,'fused_median_s':t1,
      'reference_over_bilateral':t0/tb,'fused_over_bilateral':t1/tb,'bilateral_timings':timing})
result={'scope':'cost-only context; existing classical self-implementation, NOT an official strong baseline reproduction; different default parameters and models',
 'bilateral_config':{'method':'bilateral_normal','k':32,'iterations':2,'step':1.0},
 'mixture_config':data['config'],'records':records,
 'aggregate_reference_over_bilateral':sum(r['reference_median_s'] for r in records)/sum(r['bilateral_median_s'] for r in records),
 'aggregate_fused_over_bilateral':sum(r['fused_median_s'] for r in records)/sum(r['bilateral_median_s'] for r in records)}
write_json(RUN/'bilateral-cost-context.json',result)
with (RUN/'equivalence-and-cost.csv').open('x',newline='') as stream:
    fields=['case','arm','max_output_delta_m','rms_output_delta_m','choose_two_mismatches','split_mismatches','reference_median_s','candidate_median_s','speedup']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for record in data['comparisons']:
        row=record.copy();m=row.pop('branch_mismatches');row['choose_two_mismatches']=m['choose_two'];row['split_mismatches']=m['split'];writer.writerow(row)
print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

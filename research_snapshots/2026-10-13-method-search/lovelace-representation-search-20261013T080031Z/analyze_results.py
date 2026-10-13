"""Post-prediction diagnostics only; never imported by the methods or runner."""
import csv
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def main():
    started=time.perf_counter()
    folder=ROOT/'outputs/formal-v1'
    ready=read(folder/'PREDICTIONS_READY.json')
    summary=read(folder/'SUMMARY.json')
    facts={f['observation_id']:f for f in read(ROOT/'data/holdout/private/truth.json')}
    by={}
    for receipt in ready['receipts']:
        result=read(ROOT/receipt['result'])
        by[(receipt['arm'],receipt['budget_seconds'],receipt['observation_id'])]=result
    pairs={};coverage={};detail=[]
    for budget in (6.,24.):
        for a,b in (('fixed','free'),('library','free'),('fixed','library')):
            group=[]
            for oid,f in facts.items():
                truth=f['truth']['numerator']/f['truth']['denominator']
                first=by[(a,budget,oid)];second=by[(b,budget,oid)]
                pm={p['task_id']:p for p in first['predictions']}
                for p in second['predictions']:
                    prior=pm[p['task_id']]
                    ae0=abs(prior['selected']-truth);ae1=abs(p['selected']-truth)
                    group.append({'task_id':p['task_id'],'observation_id':oid,
                       'correct':abs(p['incumbent']-truth)<=1e-9,'error_first':ae0,'error_second':ae1,
                       'delta_squared_error':ae1**2-ae0**2})
            nonzero=[r for r in group if not r['correct']]
            pairs[f'{a}_to_{b}@{budget:g}']={'shifted_n':len(nonzero),
               'better':sum(r['error_first']-r['error_second']>1e-9 for r in nonzero),
               'same':sum(abs(r['error_first']-r['error_second'])<=1e-9 for r in nonzero),
               'worse':sum(r['error_second']-r['error_first']>1e-9 for r in nonzero),
               'mean_delta_se':sum(r['delta_squared_error'] for r in nonzero)/len(nonzero),
               'rows':group}
        for arm in ('fixed','library','free'):
            records=[]
            for oid,f in facts.items():
                truth=f['truth']['numerator']/f['truth']['denominator']
                r=by[(arm,budget,oid)];s=r['state']
                hull=None if s is None else ([s['lower'],s['upper']] if arm=='free' else s.get('hull'))
                record={'observation_id':oid,'branch':f['rear_branch'],'stratum':f['stratum'],
                  'negative_control':f['negative_control'],'truth':truth,'hull':hull,
                  'covered':hull is not None and hull[0]-1e-9<=truth<=hull[1]+1e-9,
                  'width':None if hull is None else hull[1]-hull[0],
                  'prepare_seconds':r['cost'].get('prepare_seconds'),
                  'failure':r['prepare_failure']}
                records.append(record);detail.append({'arm':arm,'budget':budget,**record})
            coverage[f'{arm}@{budget:g}']={'n':len(records),'covered':sum(r['covered'] for r in records),
              'mean_width':sum(r['width'] for r in records if r['width'] is not None)/sum(r['width'] is not None for r in records)}
    supplemental=[]
    sr=read(ROOT/'outputs/supplement-v1/PREDICTIONS_READY.json')
    for receipt in sr['receipts']:
        r=read(ROOT/receipt['result']);full=by[('free',24.,r['observation_id'])]
        truth=facts[r['observation_id']]['truth'];truth=truth['numerator']/truth['denominator']
        fullp={p['task_id']:p for p in full['predictions']}
        supplemental.append({'observation_id':r['observation_id'],
          'branch':facts[r['observation_id']]['rear_branch'],
          'full_mse':sum((p['selected']-truth)**2 for p in full['predictions'])/7,
          'ablated_mse':sum((p['selected']-truth)**2 for p in r['predictions'])/7,
          'prediction_difference_count':sum(abs(p['selected']-fullp[p['task_id']]['selected'])>1e-9 for p in r['predictions']),
          'max_prediction_delta':max(abs(p['selected']-fullp[p['task_id']]['selected']) for p in r['predictions']),
          'full_process_seconds':next(x['process_seconds'] for x in ready['receipts'] if x['arm']=='free' and x['budget_seconds']==24 and x['observation_id']==r['observation_id']),
          'ablated_process_seconds':receipt['process_seconds']})
    result={'evaluation_type':'simulation_only','pairs':pairs,'coverage':coverage,
       'world_details':detail,'supplement':supplemental,'seconds':time.perf_counter()-started}
    with (folder/'ANALYSIS.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'coverage':coverage,'supplement':supplemental,'seconds':result['seconds']}))

if __name__=='__main__':main()

"""Post-score paired comparison; never imported by the inference worker."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load(p): return json.loads(p.read_text())

def main():
    folder=ROOT/'outputs/formal-v1'
    summary=load(folder/'SUMMARY.json')
    ready=load(folder/'PREDICTIONS_READY.json')
    outcomes={}
    states={}
    for receipt in ready['receipts']:
        r=load(ROOT/receipt['result'])
        key=(r['arm'],r['budget_seconds'],r['observation_id'])
        states[key]=r['state']
        outcomes[key]={x['task_id']:x['selected'] for x in r['predictions']}
    records={}
    for budget in (6.,24.):
        numeric=summary[f'numeric@{budget:g}']
        certified=summary[f'certified@{budget:g}']
        legacy=summary[f'legacy@{budget:g}']
        chosen=[(key,state) for key,state in states.items() if key[0]=='certified' and key[1]==budget]
        attempts=[x for _,state in chosen if state is not None for x in state['certificate_attempts']]
        exclusions=[x for _,state in chosen if state is not None for x in state['exclusions']]
        hull_same=prediction_same=0
        for key,state in chosen:
            nk=('numeric',budget,key[2]); n=states[nk]
            if state is not None and n is not None:
                hull_same += (state['lower']==n['lower'] and state['upper']==n['upper'])
            prediction_same += sum(value==outcomes[nk][task] for task,value in outcomes[key].items())
        timing_keys=('geometry_seconds','certificate_search_seconds','exact_check_seconds','wall_seconds','cpu_seconds')
        records[f'{budget:g}']={
            'worlds':len(chosen),'certificate_attempts':len(attempts),'verified_exclusions':len(exclusions),
            'attempt_reasons':dict(Counter(x['reason'] for x in attempts)),
            'exact_same_hulls_vs_numeric':hull_same,'exact_same_predictions_vs_numeric':prediction_same,
            'exclusions_by_branch':dict(Counter(x['branch'] for x in exclusions)),
            'proof_support_sizes':dict(Counter(len(x['certificate']['multipliers']) for x in exclusions)),
            'proof_component_totals':{key:sum(x.get(key,0) for x in attempts) for key in timing_keys},
            'search_component_totals':{key:sum(state['component_seconds'][key] for _,state in chosen if state is not None)
                for key in ('floating_proposal_seconds','context_seconds','certification_seconds')},
            'legacy_over_certified_process_ratio':legacy['cost']['process_seconds']/certified['cost']['process_seconds'],
            'certified_over_numeric_process_ratio':certified['cost']['process_seconds']/numeric['cost']['process_seconds'],
            'certified_vs_legacy_shifted_mse_difference':certified['shifted']['mse']-legacy['shifted']['mse'],
            'certified_vs_legacy_shifted_mse_percent':100*(certified['shifted']['mse']/legacy['shifted']['mse']-1),
            'mean_process_seconds':{a:summary[f'{a}@{budget:g}']['cost']['process_seconds']/18 for a in ('numeric','certified','legacy')},
            'internal_fallbacks':dict(Counter(state.get('reason','missing') if state else 'worker_failed' for _,state in chosen))}
    diagnostic=load(ROOT/'outputs/diagnostic-v1/RESULTS.json')['records']
    records['dev_same_proposals']={
        'worlds':len(diagnostic),'attempts':sum(x['attempted'] for x in diagnostic),
        'verified':sum(x['verified'] for x in diagnostic),
        'same_hulls':sum(x['verified_replay_hull']==[x['numeric_state']['lower'],x['numeric_state']['upper']] for x in diagnostic),
        'total_wall_seconds':sum(x['seconds'] for x in diagnostic),
        'reasons':dict(Counter(p['reason'] for x in diagnostic for p in x['proofs']))}
    with (folder/'ANALYSIS.json').open('x') as f: json.dump(records,f,indent=2,allow_nan=False)
    print(json.dumps(records,indent=2))

if __name__=='__main__': main()

"""Descriptive paired comparisons after saved predictions and primary scoring."""
from collections import Counter,defaultdict
import csv
from pathlib import Path
from experiment import load,save,ROOT

def main():
    folder=ROOT/'outputs/formal-v1'
    summary=load(folder/'SUMMARY.json');ready=load(folder/'PREDICTIONS_READY.json')
    rows=list(csv.DictReader((folder/'METRICS.csv').open()))
    indexed={(r['arm'],float(r['budget_seconds']),r['task_id']):r for r in rows}
    comparisons={}
    for other,budget in (('base',6.),('fine',6.),('legacy',6.),('legacy',24.)):
        counts=Counter();byworld=defaultdict(lambda:[0.,0.])
        for r in rows:
            if r['arm']!='joint' or r['correct_input']=='True':continue
            o=indexed[(other,budget,r['task_id'])]
            x,y=float(r['final_se']),float(o['final_se'])
            counts['better' if x<y-1e-9 else 'worse' if x>y+1e-9 else 'same']+=1
            byworld[r['observation_id']][0]+=x;byworld[r['observation_id']][1]+=y
        worldcounts=Counter('better' if x<y-1e-9 else 'worse' if x>y+1e-9 else 'same' for x,y in byworld.values())
        a=summary['joint@6'];b=summary[f'{other}@{budget:g}']
        comparisons[f'joint_vs_{other}@{budget:g}']={'shifted_input_counts':dict(counts),
            'world_sse_counts':dict(worldcounts),'mse_reduction_percent':100*(1-a['shifted']['mse']/b['shifted']['mse']),
            'baseline_over_joint_cold_time':b['mean_cold_seconds']/a['mean_cold_seconds']}
    states={};proofs=defaultdict(Counter);timings=defaultdict(Counter);reasons=defaultdict(Counter)
    for receipt in ready['receipts']:
        o=load(ROOT/receipt['result']);s=o['state'];label=f"{o['arm']}@{o['budget_seconds']:g}"
        if o['arm']=='legacy' or s is None:continue
        states[(o['arm'],o['observation_id'])]=s
        reasons[label][s['reason']]+=1
        for x in s['certificate_attempts']:
            mode=x.get('mode','whole');proofs[label][mode+'_attempts']+=1
            proofs[label][mode+'_accepted']+=bool(x['verified'])
        for k,v in s['component_seconds'].items():timings[label][k]+=v
    inclusion=[]
    observation_ids=sorted({load(ROOT/r['result'])['observation_id'] for r in ready['receipts']})
    for ident in observation_ids:
        missing=[arm for arm in ('base','fine','joint') if (arm,ident) not in states]
        if missing:
            inclusion.append({'observation_id':ident,'status':'unavailable','missing_arms':missing})
            continue
        base,fine,joint=(states[(arm,ident)] for arm in ('base','fine','joint'))
        inclusion.append({'observation_id':ident,'status':'available','base_width':base['upper']-base['lower'],
            'fine_width':fine['upper']-fine['lower'],'joint_width':joint['upper']-joint['lower'],
            'joint_subset_base':base['lower']<=joint['lower']<=joint['upper']<=base['upper'],
            'joint_subset_fine':fine['lower']<=joint['lower']<=joint['upper']<=fine['upper']})
    result={'comparisons':comparisons,'proof_counts':{k:dict(v) for k,v in proofs.items()},
        'component_seconds':{k:dict(v) for k,v in timings.items()},
        'termination':{k:dict(v) for k,v in reasons.items()},'hull_comparisons':inclusion,
        'batch_wall_seconds':ready['seconds'],
        'total_child_cpu_seconds':sum(r['process_cpu_seconds'] for r in ready['receipts'])}
    save(folder/'ANALYSIS.json',result)
    print(__import__('json').dumps(result,indent=2))
if __name__=='__main__':main()

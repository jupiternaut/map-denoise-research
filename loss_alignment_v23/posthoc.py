"""Post-result range/support diagnosis, no new fitting or confirmation claims."""
import sys,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
import run
def main():
    dest=Path(sys.argv[1]);load=lambda p:json.loads(Path(p).read_text());bs=load(dest/'SEALED_BEFORE_GT.json');rows=load(dest/'RESULTS.json');lookup={(r['case'],r['method']):r for r in rows};result={}
    for scene in (24,37):
        group=[b for b in bs if b['scene']==scene];stats={}
        for pool in ('all','no_offsets'):
            choices=[];drops=[];harmed=0;changed=0
            for b in group:
                costs={m:l for m,l in b['photo_loss'].items() if l is not None and (pool=='all' or not m.startswith('offset'))};choice=min(costs,key=lambda m:(costs[m],m!='identity',m));row=lookup[b['job']['case'],choice];base=lookup[b['job']['case'],'identity'];choices.append(row)
                drops.append(1-costs[choice]/costs['identity'] if costs['identity'] else 0)
                if choice!='identity':changed+=1;harmed+=row['accuracy_mm']>base['accuracy_mm']
            stats[pool]=dict(accuracy_mm=float(np.mean([r['accuracy_mm'] for r in choices])),recall=float(np.mean([r['recall'] for r in choices])),choices=dict(Counter(r['method'] for r in choices)),mean_relative_photo_loss_drop=float(np.mean(drops)),changed=changed,harmed=harmed)
        ref=run.xyz(run.DATA/'published-outputs-v2-reference/stl024_total.ply' if scene==24 else run.DATA/'reconstruction-v22-scan37/stl037_total.ply');tree=cKDTree(ref);aligned=[]
        for b in group:
            case=b['job']['case'];support=np.load(dest/'support'/f'{case}.npz')['support'].sum(0)>=2;core=np.load(run.OLD/'evaluation'/f'{case}.npz')['input_mask'];mask=core&support
            if not mask.any():aligned.append(dict(case=case,n=0));continue
            values={m:float(tree.query(np.load(dest/'outputs'/f'{case}__{m}.npy')[mask])[0].mean()) for m in ('identity','photo_selector')}
            aligned.append(dict(case=case,n=int(mask.sum()),**values))
        valid=[r for r in aligned if r['n']];stats['same_accuracy_and_photo_support']=dict(rows=aligned,identity_mm=float(np.mean([r['identity'] for r in valid])),photo_selector_mm=float(np.mean([r['photo_selector'] for r in valid])))
        result[str(scene)]=stats
    run.save(dest/'POSTHOC.json',dict(scope='Exposed post-result action-range and common-support analysis; no upgraded model or new confirmation',scenes=result));print(json.dumps(result))
if __name__=='__main__':main()

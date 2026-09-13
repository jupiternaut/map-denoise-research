"""Development attribution after exposing the reference's orientation counts.

All cases retained. A fixed larger-neighborhood old A is needed before crediting
the new hypothesis search. No changes to the already frozen reference source.
"""
import argparse,json,sys,tempfile,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/grf/Documents/Codex/2026-09-10/map-denoise-v0')
from parallel_geometry_v5.association.operator import denoise as old_a
from local_operator import denoise
from geometry import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    src=Path(a.run);dest=Path(tempfile.mkdtemp(prefix='scale-ablation-',dir=src.parent));(dest/'outputs').mkdir()
    mf=json.loads((src/'manifest.json').read_text());rows=[]
    arms={'pursuit_k48':{'ks':(48,)},'pursuit_k96':{'ks':(96,)},'old_A_k96_em16':{'k':96,'em_iterations':16,'fallback':'bilateral'}}
    (dest/'manifest.json').write_text(json.dumps({'source_run':str(src),'arms':arms,'scope':'exposed development attribution; same output evaluator'},indent=2))
    print(json.dumps({'scale_ablation':str(dest)}),flush=True)
    for file in sorted((src/'outputs').glob('*-input.npz')):
        case=file.name.removesuffix('-input.npz');family,condition,seed=case.rsplit('-',2)
        cfg=next(c for c in mf['conditions'] if c['name']==condition)
        with np.load(file) as d:p=d['xyz'];rot=d['rotation']
        with np.load(src/'outputs'/f'{case}-evaluator.npz') as d:ref=d['reference'];rl=d['ref_labels']
        for arm,config in arms.items():
            st=time.perf_counter();out,diag=(old_a(p,config) if arm.startswith('old_A') else denoise(p,config=config));duration=time.perf_counter()-st
            path=dest/'outputs'/f'{case}-{arm}.npy';np.save(path,out,allow_pickle=False)
            row={'case':case,'family':family,'condition':condition,'seed':int(seed[1:]),'arm':arm,'oracle':False,'status':'ok','elapsed_s':duration,
                 'metrics':evaluate(family,np.load(path)@rot,cfg['gap'],ref,rl,len(p)),'diagnostics':diag}
            rows.append(row)
            with open(dest/'records.jsonl','a') as fp:fp.write(json.dumps(row)+'\n')
        print(json.dumps({'case':case,'rows':len(rows)}),flush=True)
    (dest/'results.json').write_text(json.dumps({'records':rows},indent=2))
    print(json.dumps({'completed':str(dest),'rows':len(rows)}),flush=True)
if __name__=='__main__':main()

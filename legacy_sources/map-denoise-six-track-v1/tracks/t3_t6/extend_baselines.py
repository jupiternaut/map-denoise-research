"""Predeclared development amendment: small-radius MLS plus official PathNet.

The first two MLS radii straddle layers on resolved plates. To avoid winning
against only oversmoothed radii, evaluate k=8/12/16 as well on EVERY case.
No per-case selection is a deployed method. PathNet uses the same XYZ only.
Run after run.py so at most one computing process belongs to this track.
"""
import argparse,hashlib,json,os,sys,tempfile,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/grf/Documents/Codex/2026-09-10/map-denoise-v0')
from parallel_geometry_v4.baselines.operators import denoise as baseline
from parallel_geometry_v5.learned_baselines.pathnet_worker import infer
from geometry import evaluate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    source=Path(args.run);manifest=json.loads((source/'manifest.json').read_text())
    dest=Path(tempfile.mkdtemp(prefix='external-extension-',dir=source.parent));(dest/'outputs').mkdir()
    protocol={'source_run':str(source),'arms':['pcl_mls_k8','pcl_mls_k12','pcl_mls_k16','pathnet_official'],
              'scope':'same 24 development cases; no independent confirmation; no tuning after extension',
              'reason':'small-radius external baseline protection plus existing official learned weights',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'threads':1}
    (dest/'manifest.json').write_text(json.dumps(protocol,indent=2));records=[]
    print(json.dumps({'extension':str(dest)}),flush=True)
    for file in sorted((source/'outputs').glob('*-input.npz')):
        case=file.name.removesuffix('-input.npz');family,condition,seed=case.rsplit('-',2)
        cfg=next(c for c in manifest['conditions'] if c['name']==condition)
        with np.load(file) as d:p=d['xyz'];rot=d['rotation']
        with np.load(source/'outputs'/f'{case}-evaluator.npz') as d:ref=d['reference'];labels=d['ref_labels']
        for arm in protocol['arms']:
            row={'case':case,'family':family,'condition':condition,'seed':int(seed[1:]),'arm':arm,'oracle':False}
            start=time.perf_counter()
            try:
                if arm=='pathnet_official':out,diag=infer(p,device='cuda',iterations=2,batch_size=32,threads=1)
                else:out,diag=baseline(p,{'method':'pcl_mls','k':int(arm.rsplit('k',1)[1]),'polynomial_order':2})
                duration=time.perf_counter()-start;path=dest/'outputs'/f'{case}-{arm}.npy'
                np.save(path,out,allow_pickle=False)
                met=evaluate(family,np.load(path,allow_pickle=False)@rot,cfg['gap'],ref,labels,len(p))
                row.update(status='ok',elapsed_s=duration,metrics=met,diagnostics=diag)
            except Exception as e:row.update(status='error',elapsed_s=time.perf_counter()-start,error=repr(e))
            records.append(row)
            with open(dest/'records.jsonl','a') as fp:fp.write(json.dumps(row)+'\n')
        print(json.dumps({'case':case,'completed':len(records)}),flush=True)
    (dest/'results.json').write_text(json.dumps({'manifest':protocol,'records':records},indent=2))
    print(json.dumps({'completed_extension':str(dest),'errors':sum(r['status']!='ok' for r in records)}),flush=True)
if __name__=='__main__':main()

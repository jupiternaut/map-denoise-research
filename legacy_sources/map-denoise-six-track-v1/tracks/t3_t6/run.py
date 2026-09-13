"""Frozen development protocol; output saved BEFORE evaluator reads it."""
import argparse, csv, hashlib, json, os, platform, resource, sys, tempfile, time
from pathlib import Path
import numpy as np

OLD=Path('/home/grf/Documents/Codex/2026-09-10/map-denoise-v0')
sys.path.insert(0,str(OLD))
from parallel_geometry_v4.baselines.operators import denoise as baseline
from parallel_geometry_v5.association.operator import denoise as old_a
from geometry import FAMILIES, sample, rotation, evaluate
from local_operator import denoise as reference, oracle_association, oracle_normal

CONDITIONS=[{'name':'resolved','gap':.009,'sigma':.0012,'balance':.5},
            {'name':'overlap','gap':.005,'sigma':.0018,'balance':.5},
            {'name':'imbalanced','gap':.007,'sigma':.0015,'balance':.8}]
ARMS=['identity','bilateral','old_A','old_A_disabled','pcl_mls_k32','pcl_mls_k64',
      'pursuit_reference','pursuit_disabled','oracle_association','oracle_normal']

def execute(arm,p,labels,normals):
    if arm=='identity':return p.copy(),{'requires_ground_truth':False}
    if arm=='bilateral':return baseline(p,{'method':'bilateral_normal','k':48,'iterations':2,'step':.8})
    if arm.startswith('old_A'):return old_a(p,{'fallback':'bilateral','bic_gain':1e6 if arm.endswith('disabled') else 6.})
    if arm.startswith('pcl_mls'):return baseline(p,{'method':'pcl_mls','k':int(arm.rsplit('k',1)[1]),'polynomial_order':2})
    if arm.startswith('pursuit'):return reference(p,split=not arm.endswith('disabled'))
    if arm=='oracle_association':return oracle_association(p,labels)
    if arm=='oracle_normal':return oracle_normal(p,normals)
    raise ValueError(arm)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=1800);ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--root',default='/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t3_t6')
    args=ap.parse_args();base=Path(args.root);base.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='development-',dir=base));(run/'outputs').mkdir()
    manifest={'scope':'all cases are development; procedural CAD-like, NOT ABC/real scanned CAD',
        'no_frame_bias':'zero frame bias fixed in every arm; only IID additive XYZ noise',
        'oracle_policy':'diagnostic only, excluded from deployable comparisons',
        'conditions':CONDITIONS,'arms':ARMS,'n':args.n,'seeds':args.seeds,'host':platform.node(),
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
        'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
    print(json.dumps({'run':str(run),'cases':len(FAMILIES)*len(CONDITIONS)*args.seeds}),flush=True)
    for fi,family in enumerate(FAMILIES):
      for ci,c in enumerate(CONDITIONS):
       for seed in range(args.seeds):
        case=f'{family}-{c["name"]}-s{seed}';rng=np.random.default_rng(48000+fi*1000+ci*100+seed)
        clean,normals,labels=sample(family,args.n,c['gap'],c['balance'],rng)
        rot=rotation(7800+fi*11+seed);noisy=(clean+rng.normal(0,c['sigma'],clean.shape))@rot.T
        normal_world=normals@rot.T
        ref,_,ref_labels=sample(family,18000,c['gap'],.5,np.random.default_rng(94000+fi*100+ci))
        np.savez_compressed(run/'outputs'/f'{case}-input.npz',xyz=noisy,rotation=rot)
        # Evaluator-only truth stored in a different file, never passed to ordinary algorithms.
        np.savez_compressed(run/'outputs'/f'{case}-evaluator.npz',clean=clean,labels=labels,normals=normals,reference=ref,ref_labels=ref_labels)
        for arm in ARMS:
          started=time.perf_counter();row={'case':case,'family':family,'condition':c['name'],'seed':seed,'arm':arm,'oracle':arm.startswith('oracle')}
          try:
            out,diag=execute(arm,noisy.copy(),labels,normal_world)
            duration=time.perf_counter()-started
            output_path=run/'outputs'/f'{case}-{arm}.npy';np.save(output_path,out,allow_pickle=False)
            saved=np.load(output_path,allow_pickle=False)
            metrics=evaluate(family,saved@rot,c['gap'],ref,ref_labels,len(noisy))
            row.update({'status':'ok','elapsed_s':duration,'metrics':metrics,'diagnostics':diag})
          except Exception as error:
            row.update({'status':'error','elapsed_s':time.perf_counter()-started,'error':repr(error)})
          rows.append(row)
          with open(run/'records.jsonl','a') as fp:fp.write(json.dumps(row)+'\n')
        print(json.dumps({'case':case,'last_seconds':rows[-1]['elapsed_s'],'completed_rows':len(rows)}),flush=True)
    (run/'results.json').write_text(json.dumps({'manifest':manifest,'records':rows,'parent_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
    flat=[]
    for r in rows:
        x={k:v for k,v in r.items() if k not in ('metrics','diagnostics')}
        x.update({k:v for k,v in r.get('metrics',{}).items() if not isinstance(v,dict)});flat.append(x)
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with open(run/'metrics.csv','w') as fp:
        writer=csv.DictWriter(fp,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    print(json.dumps({'completed_run':str(run),'rows':len(rows),'failed':sum(r['status']!='ok' for r in rows)}),flush=True)

if __name__=='__main__':main()

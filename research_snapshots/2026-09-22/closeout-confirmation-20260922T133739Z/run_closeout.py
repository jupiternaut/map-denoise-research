"""Construct and seal new scene outputs before any evaluator reference access."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,time,resource,socket,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
from v28_closeout import construct,apply_arrays
from scene_adapter import load_scene,rois,camera,sha,save,write_ply
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1')
ARMS=('identity','A_all','B_all','post_A_keep','post_AB_keep','random_A_keep','A_fit_reserved')
CONDITIONS={'native':0.,'minus1':-1.,'plus1':1.,'minus3':-3.,'plus3':3.}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',type=int,required=True,choices=[40,55,65,69]);args=parser.parse_args()
    if socket.gethostname()!='liekkas':raise RuntimeError('host mismatch')
    if args.scene!=40 and not (ROOT/'adaptation/scan40/SEALED.json').exists():raise RuntimeError('adaptation output not ready')
    audit=json.loads((ROOT/'PREPARATION_AUDIT.json').read_text())
    for name in ('EXPERIMENT_PROTOCOL.md','TRAINING_LOCK.json','package/RELEASE_MANIFEST.json'):
        if sha(ROOT/name)!=audit['sha256'][name]:raise RuntimeError('frozen source mismatch')
    manifest=json.loads((DATA/'INPUT_MANIFEST.json').read_text())
    relevant=[r for r in manifest['extracted'] if f'/scan{args.scene}/' in r['path']]
    for r in relevant:
        if sha(Path(r['path']))!=r['sha256']:raise RuntimeError('input hash mismatch')
    group='adaptation' if args.scene==40 else 'confirmation'
    out=ROOT/group/f'scan{args.scene}';out.mkdir(parents=True,exist_ok=False)
    sources=['scene_adapter.py','run_closeout.py','EXPERIMENT_PROTOCOL.md','TRAINING_LOCK.json']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'package/v28_closeout').rglob('*') if p.is_file()]
    hashes={p:sha(ROOT/p) for p in sources}
    save(out/'LOCK.json',dict(scene=args.scene,source=hashes,input_manifest_sha256=sha(DATA/'INPUT_MANIFEST.json'),
                             conditions=CONDITIONS,arms=ARMS,reference_access=False))
    start=time.monotonic();scene=load_scene(DATA/'inputs'/f'scan{args.scene}')
    save(out/'CALIBRATION.json',scene['audit']);specs=rois(scene,args.scene,out);records=[];cache={}
    for roi in specs:
        if roi['status']!='READY':records.append(roi);continue
        original=scene['points'][np.load(out/f"roi{roi['index']}_native_ids.npy")]
        for n in roi['views']:
            if n not in cache:cache[n]=camera(scene['views'][n])
        cameras=[cache[n] for n in roi['views']];ref=cameras[0]
        ray=original-ref['center'];norm=np.linalg.norm(ray,axis=1,keepdims=True)
        if (norm<1e-12).any():raise RuntimeError('zero ray')
        ray/=norm
        for condition,bias in CONDITIONS.items():
            case=roi['id']+'__'+condition;dest=out/case;dest.mkdir();tick=time.monotonic()
            p=original+bias*ray
            print('BUILD',case,'rows',len(p),flush=True)
            raw,features,state=construct(p,ref,cameras[1:]);selected,pred,masks=apply_arrays(p,raw['A_all'],raw['B_all'],features,seed=20260922)
            outputs={**raw,**selected}
            for arm in ARMS:write_ply(dest/(arm+'.ply'),outputs[arm])
            np.savez_compressed(dest/'FEATURES.npz',**features)
            np.savez_compressed(dest/'DECISIONS.npz',**masks,**{'pred_'+k:v for k,v in pred.items()})
            metrics={arm:dict(moved_fraction=float(np.mean(np.linalg.norm(outputs[arm]-p,axis=1)>1e-7)),
                             move_rms_mm=float(np.sqrt(np.mean(np.sum((outputs[arm]-p)**2,axis=1))))) for arm in ARMS}
            record=dict(case=case,condition=condition,scene=args.scene,roi=roi,rows=len(p),views=roi['views'],
                        image_hashes={n:sha(scene['views'][n]['path']) for n in roi['views']},
                        wall_seconds=time.monotonic()-tick,arms=metrics,reference_access=False)
            save(dest/'CONSTRUCTION.json',record);records.append(record)
            print('DONE',case,round(record['wall_seconds'],2),'s',flush=True)
    if {p:sha(ROOT/p) for p in sources}!=hashes:raise RuntimeError('construction source changed')
    save(out/'SUMMARY.json',dict(records=records,wall_seconds=time.monotonic()-start,
                               peak_own_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu_used=False))
    save(out/'SEALED.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},source=hashes))
    print('SEALED',args.scene,flush=True)
if __name__=='__main__':main()

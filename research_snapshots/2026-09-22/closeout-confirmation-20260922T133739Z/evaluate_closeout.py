"""Evaluator only: verify sealed output, then score common native support.

Reference access is confined to this program. Constructors never import it.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,csv,time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.io import loadmat
from scipy.spatial import cKDTree
from scene_adapter import sha,save,in_box
ROOT=Path(__file__).resolve().parent
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1')
ARMS=('identity','A_all','B_all','post_A_keep','post_AB_keep','random_A_keep','A_fit_reserved')
CONDS=('native','minus1','plus1','minus3','plus3')
def points(path):
    p=np.asarray(o3d.io.read_point_cloud(str(path)).points).copy()
    if not len(p) or not np.isfinite(p).all():raise ValueError('invalid PLY')
    return p
def observed(p,obs):
    ijk=np.around((p-np.asarray(obs['BB']).reshape(-1,3)[0])/float(np.asarray(obs['Res']).ravel()[0])).astype(int)
    valid=np.all((ijk>=0)&(ijk<np.array(obs['ObsMask'].shape)),axis=1);keep=np.zeros(len(p),bool)
    keep[valid]=obs['ObsMask'][tuple(ijk[valid].T)].astype(bool)
    return keep
def voxel(p):
    if not len(p):return p
    c=o3d.geometry.PointCloud();c.points=o3d.utility.Vector3dVector(p)
    return np.asarray(c.voxel_down_sample(.8).points).copy()
def dist(p,q):return cKDTree(q).query(p,workers=1)[0] if len(q) else np.full(len(p),1000.)
def metrics(q,ref):
    if not len(q):return dict(E_sym_mm=1000.,accuracy_mm=1000.,completeness_mm=1000.,precision_1mm=0.,recall_1mm=0.,fscore_1mm=0.,p95_mm=1000.,status='EMPTY_OUTPUT')
    a,b=dist(q,ref),dist(ref,q);pr=float(np.mean(a<=1));re=float(np.mean(b<=1))
    return dict(E_sym_mm=float((a.mean()+b.mean())/2),accuracy_mm=float(a.mean()),completeness_mm=float(b.mean()),
        precision_1mm=pr,recall_1mm=re,fscore_1mm=2*pr*re/(pr+re) if pr+re else 0.,p95_mm=float(np.quantile(a,.95)),status='SCORED')
def main():
    out=ROOT/'evaluation';out.mkdir(exist_ok=False);start=time.monotonic();seals={}
    for sid in (40,55,65,69):
        folder=ROOT/('adaptation' if sid==40 else 'confirmation')/f'scan{sid}'
        seal=json.loads((folder/'SEALED.json').read_text())
        for name,expected in seal['files'].items():
            if sha(folder/name)!=expected:raise RuntimeError('output seal mismatch')
        seals[str(sid)]=sha(folder/'SEALED.json')
    manifest=json.loads((DATA/'REFERENCE_MANIFEST.json').read_text())
    if manifest['method_seals_before_retrieval']!=seals:raise RuntimeError('reference/output sequence mismatch')
    for r in manifest['records']:
        if sha(Path(r['path']))!=r['sha256']:raise RuntimeError('reference hash mismatch')
    save(out/'LOCK.json',dict(source={n:sha(ROOT/n) for n in ('evaluate_closeout.py','EXPERIMENT_PROTOCOL.md')},
                             method_seals=seals,reference_manifest_sha256=sha(DATA/'REFERENCE_MANIFEST.json'),
                             support='native point rows fixed across every condition and arm',voxel_mm=.8))
    rows=[];checks=[]
    for sid in (40,55,65,69):
        folder=ROOT/('adaptation' if sid==40 else 'confirmation')/f'scan{sid}'
        base=DATA/'evaluation_only'/f'scan{sid}'
        laser=points(base/f'stl{sid:03d}_total.ply');obs=loadmat(base/f'ObsMask{sid}_10.mat')
        for roi in json.loads((folder/'ROIS.json').read_text()):
            if roi['status']!='READY':continue
            native=points(folder/(roi['id']+'__native')/'identity.ply');lo,hi=np.array(roi['lo']),np.array(roi['hi'])
            support=in_box(native,lo,hi)&observed(native,obs)
            ref=voxel(laser[in_box(laser,lo,hi)&observed(laser,obs)])
            if not support.any() or not len(ref):raise RuntimeError('empty evaluator support')
            np.save(out/(roi['id']+'_native_support.npy'),support)
            for condition in CONDS:
                case=roi['id']+'__'+condition;casepath=folder/case
                original=points(casepath/'identity.ply');before=dist(original,ref)
                with np.load(casepath/'DECISIONS.npz') as decision_file:
                    accepted={a:float(np.mean(decision_file[a]!=0)) for a in ARMS if a in decision_file}
                for arm in ARMS:
                    q=points(casepath/(arm+'.ply'))
                    if q.shape!=native.shape:raise RuntimeError('row contract broken')
                    after=dist(q,ref);delta=after[support]-before[support]
                    gain=before[support]**2-after[support]**2
                    keep=in_box(q,lo,hi)&observed(q,obs);q_eval=voxel(q[keep]);m=metrics(q_eval,ref)
                    movement=np.linalg.norm(q-original,axis=1)
                    row=dict(scene=sid,role='adaptation' if sid==40 else 'confirmation',roi=roi['id'],condition=condition,arm=arm,
                        n_rows=len(q),n_source=int(support.sum()),n_output_eval=len(q_eval),n_reference=len(ref),
                        source_MSE_mm2=float(np.mean(after[support]**2)),source_MAE_mm=float(np.mean(after[support])),
                        source_p95_mm=float(np.quantile(after[support],.95)),improved_fraction=float(np.mean(delta<-.1)),
                        harmed_fraction=float(np.mean(delta>.1)),unchanged_band_fraction=float(np.mean(abs(delta)<=.1)),
                        benefit_sum_mm2=float(np.maximum(gain,0).sum()),harm_sum_mm2=float(np.maximum(-gain,0).sum()),
                        accepted_fraction=accepted.get(arm,0. if arm=='identity' else 1. if arm in ('A_all','B_all') else None),
                        moved_fraction=float(np.mean(movement>1e-7)),move_RMS_mm=float(np.sqrt(np.mean(movement**2))),
                        lost_source_fraction=float(np.mean(~keep[support])),**m)
                    rows.append(row)
                    if arm=='identity' and condition=='native':
                        a=o3d.geometry.PointCloud();a.points=o3d.utility.Vector3dVector(q[support])
                        b=o3d.geometry.PointCloud();b.points=o3d.utility.Vector3dVector(ref)
                        independent=np.asarray(a.compute_point_cloud_distance(b));err=float(np.max(abs(independent-after[support])))
                        if err>1e-9:raise RuntimeError('NN independent check failed')
                        checks.append(dict(roi=roi['id'],max_abs_mm=err))
            print('SCORED',roi['id'],int(support.sum()),'rows',flush=True)
    with (out/'METRICS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    numeric=('source_MSE_mm2','source_MAE_mm','E_sym_mm','recall_1mm','fscore_1mm','source_p95_mm','improved_fraction','harmed_fraction','moved_fraction')
    for sid in (40,55,65,69):
        summary[str(sid)]={c:{a:{k:float(np.mean([r[k] for r in rows if r['scene']==sid and r['condition']==c and r['arm']==a])) for k in numeric} for a in ARMS} for c in CONDS}
    pooled={c:{a:{k:float(np.mean([summary[str(s)][c][a][k] for s in (55,65,69)])) for k in numeric} for a in ARMS} for c in CONDS}
    save(out/'SUMMARY.json',dict(per_scene=summary,confirmation=pooled,rows=len(rows),independent_NN_checks=checks,wall_seconds=time.monotonic()-start))
    save(out/'SEALED.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}))
    print('EVALUATION_SEALED',len(rows),flush=True)
if __name__=='__main__':main()

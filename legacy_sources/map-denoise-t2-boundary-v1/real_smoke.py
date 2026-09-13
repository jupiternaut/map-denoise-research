"""Execution-only real Oxford ROI input smoke. No geometry truth is available.

Six exact prior NPZ inputs are read-only. Approximate normals come from those
files; all methods receive the same local-coordinate input, fixed uncalibrated
sigma 10 mm and bias-scale assumption 50 mm. No frames or points are dropped.
"""
import hashlib,json,platform,sys,tempfile,time
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parent
OLD=Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2')
sys.path.insert(0,str(OLD));sys.path.insert(0,str(WORK/'baseline'))
from experiment import Input,estimate as old_estimate
from scalar_reference import estimate as scalar_estimate
from plane_profile import estimate as plane_estimate

SOURCE=Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600/t4/run-selwrbu7')
NAMES=tuple(f'{side}-{radius}-sample.npz' for side in ('left','center','right') for radius in ('0.15','0.30'))
DEST=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/real-input-smoke')
METHODS=('joint_forced','scalar_profile_hard','plane_profile_map')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    paths=[SOURCE/name for name in NAMES]
    if not all(p.is_file() for p in paths):raise FileNotFoundError('A designated input is missing; no substitution allowed')
    if platform.node()!='liekkas':raise RuntimeError('Wrong host')
    DEST.mkdir(parents=True,exist_ok=True);run=Path(tempfile.mkdtemp(prefix='execution-',dir=DEST));(run/'outputs').mkdir()
    manifest={'scope':'real-input execution/output checks only; no GT or accuracy inference',
              'host':platform.node(),'inputs':{str(p):sha(p) for p in paths},
              'methods':METHODS,'sigma_mm':10.,'bias_bound_mm':50.,
              'parameter_scope':'fixed same assumptions as prior smoke; not estimated/calibrated sensor parameters',
              'source_sha256':{str(p):sha(p) for p in (Path(__file__),OLD/'experiment.py',WORK/'baseline/scalar_reference.py',WORK/'baseline/plane_profile.py')}}
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2));records=[]
    print(json.dumps({'run':str(run)}),flush=True)
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            xyz=np.asarray(data['points'],dtype=float);normal=np.asarray(data['normal'],dtype=float).reshape(3)
            original_frame=np.asarray(data['frame_index'])
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():raise ValueError('Invalid source XYZ')
        if original_frame.shape!=(len(xyz),) or not np.isfinite(normal).all() or np.linalg.norm(normal)<=0:raise ValueError('Invalid source metadata')
        normal/=np.linalg.norm(normal)
        tangent=np.array([1.,0.,0.]) if abs(normal[0])<.8 else np.array([0.,1.,0.])
        tangent-=normal*np.dot(tangent,normal);tangent/=np.linalg.norm(tangent)
        basis=np.stack((tangent,np.cross(normal,tangent),normal),axis=1);origin=xyz.mean(axis=0)
        local=(xyz-origin)@basis*1000.;frame_ids,frame=np.unique(original_frame,return_inverse=True)
        counts=np.bincount(frame);inp=Input(local,frame,np.zeros(len(xyz),dtype=int),10.,50.)
        roundtrip=float(np.max(np.linalg.norm(local/1000.@basis.T+origin-xyz,axis=1))*1000.)
        # Record the input adaptation separately. Normal/ROI are observed coarse
        # metadata, not verified true surface labels or true normals.
        np.savez_compressed(run/'outputs'/f'{path.stem}-adapted-input.npz',xyz_mm=local,frame=frame,
                            original_frame_ids=frame_ids,basis=basis,origin_m=origin,normal=normal,
                            sigma_mm=10.,bias_bound_mm=50.)
        for method in METHODS:
            row={'input':str(path),'input_sha256':manifest['inputs'][str(path)],'method':method,
                 'n_input':len(xyz),'n_frames':len(frame_ids),'points_per_frame':counts.tolist(),
                 'minimum_points_per_frame':int(counts.min()),'frame_reindex_only':True,
                 'roundtrip_max_mm':roundtrip,'has_geometry_ground_truth':False}
            started=time.perf_counter()
            try:
                fn=old_estimate if method=='joint_forced' else (scalar_estimate if method=='scalar_profile_hard' else plane_estimate)
                output,bias,info=fn(inp,method);elapsed=time.perf_counter()-started
                world=np.asarray(output)/1000.@basis.T+origin
                dest=run/'outputs'/f'{path.stem}__{method}.npz'
                np.savez_compressed(dest,points=world,frame_index=original_frame,bias_mm=bias,original_frame_ids=frame_ids)
                with np.load(dest,allow_pickle=False) as saved:reloaded=saved['points'];sb=saved['bias_mm']
                finite=bool(np.isfinite(reloaded).all() and np.isfinite(sb).all())
                same_count=len(reloaded)==len(xyz);displacement=np.linalg.norm(reloaded-xyz,axis=1)*1000.
                row.update({'execution':'ok','status':info['status'],'n_output':len(reloaded),'same_point_count':same_count,
                    'all_finite':finite,'elapsed_s':elapsed,'selected_k':info.get('k'),
                    'optimizer_success':info.get('optimizer_success'),'mean_displacement_mm':float(displacement.mean()),
                    'median_displacement_mm':float(np.median(displacement)),'max_displacement_mm':float(displacement.max()),
                    'rms_displacement_mm':float(np.sqrt(np.mean(displacement**2))),
                    'output':str(dest),'output_sha256':sha(dest),'info':info})
            except Exception as e:row.update({'execution':'error','elapsed_s':time.perf_counter()-started,'error':repr(e)})
            records.append(row)
            with open(run/'records.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(path.name,'done',flush=True)
    manifest['source_inputs_unchanged']=all(sha(p)==manifest['inputs'][str(p)] for p in paths)
    (run/'results.json').write_text(json.dumps({'manifest':manifest,'records':records},indent=2))
    print(json.dumps({'completed_run':str(run),'outputs':sum(r['execution']=='ok' for r in records),
                      'errors':sum(r['execution']=='error' for r in records),'inputs_unchanged':manifest['source_inputs_unchanged']}),flush=True)
if __name__=='__main__':main()

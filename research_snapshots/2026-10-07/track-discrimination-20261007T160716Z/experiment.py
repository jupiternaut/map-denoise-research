"""Frozen-image evidence -> fixed candidate selection -> sealed NN evaluation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
import csv
import hashlib
import json
import socket
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from theory.kernel import gain_bounds_world

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'plane-support-20261007T084206Z'
ANCESTOR=ROOT.parent/'surface-evidence-20261001T140058Z/evaluation'
NEW_ARMS=('star_full','star','cycle')
OLD_ARMS=('incumbent','photo_U11','joint_interval','official_double')


def now():return datetime.now(timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def csvread(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def csvwrite(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def verify(seal):
    for p,h in load(seal)['files'].items():assert sha(p)==h,p


def observation_verified():
    seal=load(ROOT/'observation/SEAL.json')
    for f,h in seal['sha256'].items():assert sha(ROOT/'observation'/f)==h,f
    assert sha(ROOT/'REQUESTS.json')==seal['request_sha256']
    assert sha(ROOT/'PROTOCOL.json')==seal['protocol_sha256']
    for p,h in seal['input_photo_sha256'].items():assert sha(p)==h,p


def freeze():
    assert socket.gethostname()=='liekkas'
    observation_verified()
    paths=[ROOT/p for p in ['experiment.py','test_integration.py','theory/kernel.py',
          'AGENTS.md','PROTOCOL.json','REQUESTS.json','observation/TRACKS.json',
          'observation/SCANS.json','observation/extract.py','observation/LOCK.json',
          'observation/SEAL.json','refine-logs/EXPERIMENT_PLAN_20261007T160716Z.md']]
    paths.append(OLD/'DECISIONS.json')
    dump(ROOT/'RUN_LOCK.json',dict(created_at=now(),files={str(p):sha(p) for p in paths}))
    print('FROZEN',len(paths))


def reject_evaluation_reads():
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]).lower()
            if any(s in p for s in ['/evaluation/','candidate_distances','point_metrics','evaluation_truth','stl118','stl122','.ply']):
                raise PermissionError('Inference cannot open evaluation: '+p)
    sys.addaudithook(audit)


def choose(objects,current_id,C,r,intervals,gain_floor=0.0):
    lookup={o['candidate_id']:o for o in objects}
    if current_id not in lookup:
        return dict(selected_candidate_id=current_id,reason='NO_CURRENT_OUTPUT',scores=[])
    if not intervals:
        return dict(selected_candidate_id=current_id,reason='EMPTY_UNKNOWN',scores=[])
    current=lookup[current_id]['xyz_mm']
    scores=[]
    for o in objects:
        lo,hi=gain_bounds_world(current,o['xyz_mm'],C,r,intervals)
        scores.append(dict(candidate_id=o['candidate_id'],gain_low_mm2=lo,gain_high_mm2=hi))
    eligible=[s for s in scores if s['candidate_id']!=current_id and s['gain_low_mm2']>gain_floor]
    if not eligible:
        return dict(selected_candidate_id=current_id,reason='NO_UNIFORM_MODEL_GAIN',scores=scores)
    winner=sorted(eligible,key=lambda s:(-s['gain_low_mm2'],s['candidate_id']))[0]
    return dict(selected_candidate_id=winner['candidate_id'],reason='POSITIVE_GAIN_ON_ASSUMED_SET',scores=scores)


def infer():
    start=time.monotonic();reject_evaluation_reads();verify(ROOT/'RUN_LOCK.json');observation_verified()
    proto=load(ROOT/'PROTOCOL.json');req=load(ROOT/'REQUESTS.json')
    tracks={(r['roi'],r['query']):r for r in load(ROOT/'observation/TRACKS.json')['rows']}
    original=load(OLD/'DECISIONS.json')['rows']
    outputs=[]
    for row in original:
        key=(row['roi'],row['query']);t=tracks[key]
        c=req['scenes'][str(row['scene'])]['cameras'][req['scenes'][str(row['scene'])]['reference']]
        ray=np.linalg.solve(np.array(c['K_half']),[*row['pixel_xy'],1.])
        ray=ray/ray[2];ray=np.array(c['R']).T@ray
        objs=[dict(candidate_id=o['candidate_id'],xyz_mm=o['xyz_mm']) for o in row['objects']]
        current=row['baseline_id']
        if current is None:current=-1 if any(o['candidate_id']==-1 for o in objs) else None
        dec={arm:choose(objs,current,c['center'],ray.tolist(),t[arm+'_intervals'],proto['decision']['gain_floor_mm2']) for arm in NEW_ARMS}
        outputs.append(dict(scene=row['scene'],roi=row['roi'],query=row['query'],primary=row['primary'],
            current_id=current,objects=objs,center=c['center'],ray=ray.tolist(),
            intervals={a:t[a+'_intervals'] for a in NEW_ARMS},decisions=dec))
    assert len(outputs)==21 and sum(o['candidate_id']!=-1 for r in outputs for o in r['objects'])==105
    dump(ROOT/'DECISIONS.json',dict(created_at=now(),rows=outputs,gt_accessed=False,
        model_gain_not_nn_gain=True,wallclock_seconds=time.monotonic()-start))
    files=[ROOT/'DECISIONS.json',ROOT/'RUN_LOCK.json',ROOT/'observation/SEAL.json']
    dump(ROOT/'PREDICTIONS_SEALED.json',dict(created_at=now(),files={str(p):sha(p) for p in files}))
    print(json.dumps({a:dict(Counter(r['decisions'][a]['reason'] for r in outputs)) for a in NEW_ARMS}))


def metrics(points,photo):
    primary=[p for p in points if p['primary']];assert len(primary)==484
    rois=sorted({p['roi'] for p in primary})
    ds={r:np.array([p['distance_mm'] for p in primary if p['roi']==r]) for r in rois}
    rm={r:float(np.mean(v*v)) for r,v in ds.items()}
    comparisons=np.array([[photo[(p['roi'],p['query'])]['distance_mm'],p['distance_mm']] for p in primary])
    orig=np.array([[p['incumbent_distance_mm'],p['distance_mm']] for p in primary])
    delta=comparisons[:,1]-comparisons[:,0]
    return dict(arm=points[0]['arm'],primary=len(primary),mse_mm2=float(np.mean(list(rm.values()))),
        mae_mm=float(np.mean([v.mean() for v in ds.values()])),roi_mse=rm,
        scene_mse={s:float(np.mean([v for r,v in rm.items() if r.startswith('scan'+s+'_')])) for s in ['118','122']},
        severe_gt5=int(np.sum(comparisons[:,1]>5)),improved_vs_photo=int(np.sum(delta<-1e-9)),
        worsened_vs_photo=int(np.sum(delta>1e-9)),unchanged_vs_photo=int(np.sum(abs(delta)<=1e-9)),
        good_le1=int(np.sum(comparisons[:,1]<=1)),new_harm_vs_photo=int(np.sum((comparisons[:,0]<=1)&(comparisons[:,1]>1))),
        new_harm_vs_original=int(np.sum((orig[:,0]<=1)&(orig[:,1]>1))),
        finite=sum(p['distance_mm'] is not None for p in points),missing_finite=sum(not p['primary'] and p['distance_mm'] is not None for p in points))


def evaluate():
    verify(ROOT/'RUN_LOCK.json');verify(ROOT/'PREDICTIONS_SEALED.json');verify(OLD/'evaluation/SEALED.json')
    oldrows=csvread(OLD/'evaluation/POINT_METRICS.csv')
    original={(r['roi'],int(r['query'])):r for r in csvread(ANCESTOR/'POINT_METRICS.csv') if r['arm']=='incumbent'}
    candidates={(r['roi'],int(r['query']),int(r['candidate_id'])):r for r in csvread(ANCESTOR/'CANDIDATE_DISTANCES.csv')}
    events={(r['roi'],r['query']):r for r in load(ROOT/'DECISIONS.json')['rows']}
    cc=ic=0
    for key,r in events.items():
        for o in r['objects']:
            target=original[key] if o['candidate_id']==-1 else candidates[(*key,o['candidate_id'])]
            np.testing.assert_array_equal(o['xyz_mm'],[float(target[k]) for k in ['x_mm','y_mm','z_mm']])
            cc+=o['candidate_id']!=-1;ic+=o['candidate_id']==-1
    assert cc==105 and ic==18
    def convert(p):
        return dict(arm=p['arm'],roi=p['roi'],query=int(p['query']),primary=p['primary']=='True',
            selected=int(p['selected']) if p['selected'] else None,reason=p['reason'],
            distance_mm=float(p['distance_mm']) if p['distance_mm'] else None,
            incumbent_distance_mm=float(p['incumbent_distance_mm']) if p['incumbent_distance_mm'] else None)
    allrows=[convert(p) for p in oldrows if p['arm'] in OLD_ARMS]
    photo={(p['roi'],p['query']):p for p in allrows if p['arm']=='photo_U11'};assert len(photo)==512
    transitions=[]
    for arm in NEW_ARMS:
        for key,p in photo.items():
            q=dict(p,arm=arm)
            if key in events:
                dec=events[key]['decisions'][arm];idx=dec['selected_candidate_id']
                q['selected']=idx;q['reason']=dec['reason']
                if idx is not None:
                    target=original[key] if idx==-1 else candidates[(*key,idx)]
                    q['distance_mm']=float(target['distance_mm']) if target['distance_mm'] else None
            allrows.append(q)
            if q['distance_mm']!=p['distance_mm']:
                transitions.append(dict(arm=arm,roi=key[0],query=key[1],primary=p['primary'],
                    photo_mm=p['distance_mm'],output_mm=q['distance_mm'],selected=q['selected'],reason=q['reason']))
    result=[metrics([p for p in allrows if p['arm']==a],photo) for a in (*OLD_ARMS,*NEW_ARMS)]
    mm={m['arm']:m for m in result}
    success=mm['cycle']['mse_mm2']<min(mm['star']['mse_mm2'],mm['photo_U11']['mse_mm2']) and mm['cycle']['new_harm_vs_photo']==0
    evaluation_inputs=[OLD/'evaluation/POINT_METRICS.csv',ANCESTOR/'POINT_METRICS.csv',ANCESTOR/'CANDIDATE_DISTANCES.csv']
    dump(ROOT/'evaluation/RESULTS.json',dict(created_at=now(),metrics=result,primary_success=success,
        candidate_coordinates_verified=cc,incumbent_coordinates_verified=ic,transitions=transitions,
        scope='Two exposed scenes; fixed 21 residual requests/105 candidates, 484 original-valid of512; no new confirmation',
        evaluation_type='real_gt_reused_exact_coordinates',metric='nearest original DTU laser vertex distance; no new GT ray or physical layer labels',
        inputs={str(p):sha(p) for p in evaluation_inputs}))
    csvwrite(ROOT/'evaluation/POINT_METRICS.csv',allrows)
    csvwrite(ROOT/'evaluation/TRANSITIONS.csv',transitions,fields=['arm','roi','query','primary','photo_mm','output_mm','selected','reason'])
    files=[p for p in (ROOT/'evaluation').iterdir() if p.is_file()]
    dump(ROOT/'evaluation/SEALED.json',dict(created_at=now(),files={str(p):sha(p) for p in files}))
    print(json.dumps(dict(primary_success=success,metrics=result,transitions=transitions),indent=2))


if __name__=='__main__':
    assert socket.gethostname()=='liekkas'
    {'freeze':freeze,'infer':infer,'evaluate':evaluate}[sys.argv[1]]()

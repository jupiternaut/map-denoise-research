"""Stage-isolated mixed-pixel development experiment. No truth in ordinary."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
from baseline import cameras,full9_intervals,choose,support

ROOT=Path(__file__).resolve().parent
PLAN=Path('/srv/slam-research/grf/map-denoise/plans/mixed-pixel-study-20261008T035954Z/refine-logs/EXPERIMENT_PLAN.md')
HISTORY=[Path('/srv/slam-research/grf/map-denoise/runs/footprint-support-20261008T025757Z'),
         Path('/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z')]

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(clean(value),indent=2,allow_nan=False).encode()
    with path.open('xb') as f:f.write(payload)
    return hashlib.sha256(payload).hexdigest()
def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=io.BytesIO();np.savez_compressed(b,**arrays);payload=b.getvalue()
    with path.open('xb') as f:f.write(payload)
    return hashlib.sha256(payload).hexdigest()
def pack_aux(aux,path):
    arrays={k:v for k,v in aux.items() if isinstance(v,np.ndarray)}
    scalars={k:clean(v) for k,v in aux.items() if k not in arrays}
    arrays['_metadata_json']=np.asarray(json.dumps(scalars,allow_nan=False))
    return save_npz(path,**arrays)
def unpack_aux(path):
    with np.load(path,allow_pickle=False) as f:
        out={k:f[k].copy() for k in f.files if k!='_metadata_json'}
        if '_metadata_json' in f:out.update(json.loads(str(f['_metadata_json'].item())))
    for k in ('mode','valid','sigma_valid','sigma','background_depth'):
        if k in out and isinstance(out[k],np.ndarray) and out[k].shape==():out[k]=out[k].item()
    if out.get('background_depth') is None:out['background_depth']=np.inf
    return out

def verify_lock():
    lock=load(ROOT/'LOCK.json')
    for p,h in lock['sources'].items():assert sha(p)==h,p
    for p,h in lock['inputs'].items():assert sha(p)==h,p
    return lock

def prepare():
    assert socket.gethostname()=='liekkas'
    assert not (ROOT/'LOCK.json').exists()
    e0=load(ROOT/'e0/RESULTS.json')
    # E0 implementation provides explicit check list; no inference from filename.
    assert e0.get('passed',e0.get('pass',e0.get('all_passed',False))), 'E0 not passed'
    sources={str(p):sha(p) for p in ROOT.glob('*.py')}
    sources.update({str(p):sha(p) for p in (ROOT/'METHOD.json',ROOT/'AGENTS.md',ROOT/'INTERFACE.md',PLAN)})
    for p in ROOT.glob('*NOTES.md'):sources[str(p)]=sha(p)
    historical={str(p):sha(p) for d in HISTORY for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    from fixtures import create_e1
    create_e1(ROOT/'data')
    inputs={str(p):sha(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    dump(ROOT/'LOCK.json',dict(created=now(),host=socket.gethostname(),sources=sources,inputs=inputs,
        historical_hashes=historical,plan_sha256=sha(PLAN),e0_sha256=sha(ROOT/'e0/RESULTS.json')))
    print(json.dumps(dict(stage='prepared',input_files=len(inputs),historical_files=len(historical))),flush=True)

def guard(reads,write_dirs,allow_oracle=False):
    """Audited Python path boundary, not an OS sandbox."""
    exact={str(Path(p).resolve()) for p in reads}
    outputs=[Path(p).resolve() for p in write_dirs]
    runtime=[Path(sys.prefix).resolve(),Path('/usr/lib'),Path('/lib')]
    events=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();flags=args[2]
        writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        reading=not writing or bool(flags&os.O_RDWR)
        if writing and not any(p.is_relative_to(d) for d in outputs):raise PermissionError('undeclared write '+str(p))
        if reading:
            code=(p.suffix in ('.py','.pyc','.so') and
                  (p.parent==ROOT or any(p.is_relative_to(d) for d in runtime)))
            if str(p) not in exact and not code:raise PermissionError('undeclared read '+str(p))
            events.append(str(p))
    sys.addaudithook(hook)
    return events

def combine_curves(curves):
    counts=np.asarray([c['pixel_count'] for c in curves],float)
    valid=all(bool(c['valid']) for c in curves) and counts.sum()>0
    loss=np.sum(np.stack([c['loss'] for c in curves])*counts[:,None],axis=0)/max(1.,counts.sum())
    return loss,counts,valid

def prediction_stage(oracle=False):
    from predictor import estimate_aux,predict_curve
    lock=verify_lock()
    params=load(ROOT/'METHOD.json');grid=np.arange(params['grid'][0],params['grid'][1]+.1,params['grid'][2])
    inp=load(ROOT/'data/observed/inputs.json')
    outdir=ROOT/('oracle_stage' if oracle else 'ordinary_stage')
    outdir.mkdir(exist_ok=True)
    reads=[ROOT/'METHOD.json',ROOT/'data/observed/inputs.json']
    reads += [Path(r['image_file']) for r in inp]
    if oracle:
        oinputs=load(ROOT/'data/oracle/manifest.json')
        omap={r['id']:r for r in oinputs}
        ordinary=load(ROOT/'ordinary_stage/OBSERVATIONS.json')
        sigmas={r['id']:r for r in ordinary['auxiliary']}
        reads += [ROOT/'data/oracle/manifest.json',ROOT/'ordinary_stage/OBSERVATIONS.json']
        reads += [Path(r['aux_file']) for r in oinputs]
    # Warm archive reader before entering the declared input boundary.
    with np.load(inp[0]['image_file']) as f:assert f['images'].shape==(3,128,128)
    eventlog=guard(reads,[outdir],allow_oracle=oracle)
    blocked=[]
    for p in [ROOT/'data/truth/metadata.json']+([] if oracle else [ROOT/'data/oracle/manifest.json']):
        try:
            with p.open():pass
        except PermissionError:blocked.append(str(p))
        else:raise AssertionError('truth boundary did not reject')
    rows=[];auxrecords=[];file_hashes={}
    for n,item in enumerate(inp):
        start=time.perf_counter()
        with np.load(item['image_file']) as f:images=f['images'].copy()
        cams=cameras(item['cameras']);fid=item['id']
        aux=[];fit_times=[]
        for fold,(train_id,eval_id) in enumerate(params['folds']):
            t=time.perf_counter()
            a=unpack_aux(omap[fid]['aux_file']) if oracle else estimate_aux(images[0],images[train_id],cams[0],cams[train_id],params)
            aux.append(a);fit_times.append(time.perf_counter()-t)
            afile=outdir/f'{fid}_aux{fold}.npz'
            file_hashes[str(afile)]=pack_aux(a,afile)
        sigma_values=[float(a.get('sigma',np.nan)) for a in aux]
        sigma_valid=all(bool(a.get('sigma_valid',False)) and np.isfinite(s) and s>=1 for a,s in zip(aux,sigma_values))
        if oracle:
            sigma_values=sigmas[fid]['sigma_values']
            sigma_valid=bool(sigmas[fid]['sigma_valid'])
        if not oracle:
            nb=full9_intervals(images,cams,grid)
            nfile=outdir/f'{fid}_N.npz'
            file_hashes[str(nfile)]=save_npz(nfile,grid=grid,scores=nb['scores'],accepted=nb['accepted'])
            rows.append(dict(id=fid,arm='N',incumbent=None,intervals=nb['intervals'],curve_file=str(nfile),raw_valid=True))
        auxrecords.append(dict(id=fid,sigma_values=sigma_values,sigma_valid=sigma_valid,
            auxiliary_valid=[bool(a.get('valid',False)) for a in aux],
            metadata=[clean(a.get('metadata',{})) for a in aux],fit_seconds=fit_times))
        for mode,inc in [('dynamic',params['incumbents'][1])]+[('fixed',x) for x in params['incumbents']]:
            curve_start=time.perf_counter()
            curves=[predict_curve(a,cams[0],cams[ev],images[ev],grid,inc,params,mode=mode)
                    for a,(_,ev) in zip(aux,params['folds'])]
            loss,counts,valid=combine_curves(curves)
            sigma2=float(np.sum(counts*np.square(sigma_values))/counts.sum()) if sigma_valid and counts.sum()>0 else np.nan
            norm=loss/sigma2
            finite=np.isfinite(loss)
            flat=bool(finite.all() and np.ptp(loss)<=params['flat_tolerance'])
            accepted=np.zeros(len(grid),bool);reason='ok'
            if not valid:reason='auxiliary_or_prediction_invalid'
            elif not sigma_valid:reason='scale_unavailable'
            elif not finite.all():reason='nonfinite_prediction'
            elif flat:reason='flat_curve'
            else:accepted=(norm<=params['residual_abs_max'])&(norm-norm.min()<=params['residual_excess_max'])
            intervals=support.support_intervals(grid,accepted,padding=params['padding_mm'])
            if reason=='ok' and not intervals:reason='threshold_empty'
            arm=('O' if oracle else 'E')+('D' if mode=='dynamic' else 'F')
            cfile=outdir/f'{fid}_{arm}_{int(inc)}.npz'
            file_hashes[str(cfile)]=save_npz(cfile,grid=grid,loss=loss,normalized_loss=norm,
                fold_loss=np.stack([c['loss'] for c in curves]),pixel_counts=counts,
                accepted=accepted,sigma_values=np.asarray(sigma_values,dtype=float))
            rows.append(dict(id=fid,arm=arm,incumbent=None if mode=='dynamic' else inc,
                intervals=intervals,curve_file=str(cfile),raw_valid=valid,scale_valid=sigma_valid,
                reason=reason,flat=flat,pixel_counts=counts.tolist(),
                fold_reasons=[c.get('reason') for c in curves],seconds=time.perf_counter()-curve_start))
        print(json.dumps(dict(stage='oracle' if oracle else 'ordinary',done=n+1,total=len(inp),seconds=time.perf_counter()-start)),flush=True)
    payload=dict(created=now(),rows=rows,auxiliary=auxrecords,blocked_reads=blocked,
        read_log=sorted(set(eventlog)),source_lock_sha256=hashlib.sha256(json.dumps(lock['sources'],sort_keys=True).encode()).hexdigest())
    file_hashes[str(outdir/'OBSERVATIONS.json')]=dump(outdir/'OBSERVATIONS.json',payload)
    dump(outdir/'SEAL.json',dict(created=now(),files=file_hashes))

def infer():
    verify_lock();params=load(ROOT/'METHOD.json')
    rows=[]
    for name in ('ordinary_stage','oracle_stage'):
        seal=load(ROOT/name/'SEAL.json')
        for p,h in seal['files'].items():assert sha(p)==h,p
        observations=load(ROOT/name/'OBSERVATIONS.json')['rows']
        for o in observations:
            for inc in params['incumbents']:
                if o['incumbent'] is not None and o['incumbent']!=inc:continue
                rows.append(dict(**o,initial_depth=inc,**choose(o['intervals'],params['candidates'],inc)))
    for item in load(ROOT/'data/observed/inputs.json'):
        for inc in params['incumbents']:
            rows.append(dict(id=item['id'],arm='K',initial_depth=inc,selected_depth=inc,support_mean=None,estimated_squared_gain=0.,intervals=[]))
    digest=dump(ROOT/'decisions/PREDICTIONS.json',rows)
    dump(ROOT/'decisions/SEAL.json',dict(created=now(),sha256=digest,
        observation_seals={name:sha(ROOT/name/'SEAL.json') for name in ('ordinary_stage','oracle_stage')}))
    print(json.dumps(dict(stage='infer',decisions=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','ordinary','oracle','infer'])
    stage=p.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='ordinary':prediction_stage(False)
    elif stage=='oracle':prediction_stage(True)
    else:infer()

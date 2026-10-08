"""Manually gated CPU experiment: no evaluation truth in score/decide stages."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'mixed-pixel-20261008T041249Z'
PLAN = Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/planning/decision-interface-20261008/refine-logs/EXPERIMENT_PLAN.md')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec); spec.loader.exec_module(out)
    return out

predictor = module('frozen_predictor', OLD/'predictor.py')
baseline = module('frozen_baseline', OLD/'baseline.py')

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def observation_inputs(data_root):
    data_root=Path(data_root)
    rows=load(data_root/'observed/inputs.json')
    for r in rows:
        p=Path(r['image_file'])
        r['image_file']=str(p if p.is_absolute() else (data_root/p).resolve())
    return rows
def clean(x):
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
def dump(p, x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(clean(x),f,indent=2,allow_nan=False)
    return sha(p)
def savez(p, **kwargs):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f: np.savez_compressed(f,**kwargs)
    return sha(p)
def seal(directory):
    d=Path(directory)
    return dump(d/'SEAL.json',dict(created=now(),files={str(p):sha(p) for p in sorted(d.rglob('*')) if p.is_file() and p.name!='SEAL.json'}))
def verify(p):
    for name,h in load(p)['files'].items(): assert sha(name)==h,name
def source_files():
    out=list(ROOT.glob('*.py'))+[ROOT/'AGENTS.md',ROOT/'INTERFACE.md',ROOT/'PROTOCOL_SUPPLEMENT.md',ROOT/'FIXTURE_SPEC.md',ROOT/'METHOD.json',PLAN]
    out += [OLD/'predictor.py',OLD/'baseline.py',OLD/'METHOD.json',baseline.SUPPORT]
    return out

def prepare_calibration():
    from new_fixtures import create_dataset
    assert socket.gethostname()=='liekkas'
    assert load(ROOT/'B0/RESULTS.json')['passed']
    assert not (ROOT/'SOURCE_LOCK.json').exists()
    history={str(p):sha(p) for name in ('mixed-pixel-20261008T041249Z','footprint-support-20261008T025757Z','surface-owned-support-20261008T022918Z') for p in (ROOT.parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    dump(ROOT/'SOURCE_LOCK.json',dict(created=now(),files={str(p):sha(p) for p in source_files()},history=history))
    start=time.perf_counter();create_dataset(ROOT/'calibration/data','calibration')
    records=observation_inputs(ROOT/'calibration/data')
    truth={r['id']:r for r in load(ROOT/'calibration/data/truth/metadata.json')}
    tasks=[]
    for r in records:
        # Known calibration labels define one extra evaluation location only.
        grid=np.unique(np.r_[np.arange(450.,901.,2.),np.arange(450.,901.,15.),truth[r['id']]['true_depth']])
        tasks.append(dict(**r,task_id=r['id'],incumbent=600.,candidates=np.arange(450.,901.,15.).tolist(),grid=grid.tolist()))
    dump(ROOT/'calibration/TASKS.json',tasks)
    dump(ROOT/'calibration/GENERATION.json',dict(seconds=time.perf_counter()-start,objects=len(records),known_geometry_for_calibration=True))
    seal(ROOT/'calibration/data')

def guard_no_truth(output):
    events=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        flags=args[2]; reading=not flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) or flags&os.O_RDWR
        if reading:
            if ('truth' in p.parts or 'oracle' in p.parts or 'evaluation' in p.parts or 'oracle_stage' in p.parts):
                raise PermissionError('evaluation input forbidden '+str(p))
            if p.is_relative_to(ROOT.parent):events.append(str(p))
    sys.addaudithook(hook)
    return events

def score(stage):
    verify(ROOT/'SOURCE_LOCK.json')
    tasks=load(ROOT/stage/'TASKS.json');out=ROOT/stage/'scores';out.mkdir(parents=True)
    records=[];auxrecords=[];last_id=None;aux=None
    # Warm zip reader and imports before semantic input guard. No truth is read.
    with np.load(tasks[0]['image_file']) as f: assert f['images'].shape==(3,128,128)
    events=guard_no_truth(out);blocked=False
    try:load(ROOT/stage/'data/truth/metadata.json')
    except PermissionError:blocked=True
    assert blocked
    start=time.perf_counter()
    for i,item in enumerate(tasks):
        t=time.perf_counter();assert sha(item['image_file'])==item['sha256']
        with np.load(item['image_file']) as f: images=f['images'].copy()
        cams=baseline.cameras(item['cameras']);grid=np.asarray(item['grid'])
        if last_id!=item['id']:
            aux=[predictor.estimate_aux(images[0],images[tr],cams[0],cams[tr]) for tr in (1,2)]
            ar=dict(id=item['id'],sigma_values=[a['sigma'] for a in aux],sigma_valid=[a['sigma_valid'] for a in aux],auxiliary_valid=[a['valid'] for a in aux],metadata=[a['metadata'] for a in aux])
            auxrecords.append(ar)
            for fold,a in enumerate(aux):
                arrays={k:v for k,v in a.items() if isinstance(v,np.ndarray)}
                arrays['_metadata_json']=np.asarray(json.dumps(clean({k:v for k,v in a.items() if k not in arrays})))
                savez(out/f"{item['id']}_aux{fold}.npz",**arrays)
            last_id=item['id']
        modes=['dynamic'] if stage=='calibration' else ['dynamic','fixed']
        for mode in modes:
            curves=[predictor.predict_curve(a,cams[0],cams[ev],images[ev],grid,item['incumbent'],mode=mode) for a,ev in zip(aux,(2,1))]
            counts=np.array([c['pixel_count'] for c in curves],float)
            loss=np.sum(np.stack([c['loss'] for c in curves])*counts[:,None],axis=0)/max(1.,counts.sum())
            arm='ED' if mode=='dynamic' else 'EF';path=out/f"{item['task_id']}_{arm}.npz"
            savez(path,grid=grid,loss=loss,fold_loss=np.stack([c['loss'] for c in curves]),pixel_counts=counts,sigma_values=np.array([a['sigma'] if a['sigma_valid'] else np.nan for a in aux]))
            records.append(dict(id=item['id'],task_id=item['task_id'],arm=arm,incumbent=item['incumbent'],candidates=item['candidates'],curve_file=str(path),raw_valid=all(c['valid'] for c in curves),pixel_counts=counts))
        nb=baseline.full9_intervals(images,cams,grid);path=out/f"{item['task_id']}_N.npz"
        savez(path,grid=grid,scores=nb['scores'],accepted=nb['accepted'])
        # Irregular-grid intervals intentionally derived by common physical cells later.
        records.append(dict(id=item['id'],task_id=item['task_id'],arm='N',incumbent=item['incumbent'],candidates=item['candidates'],curve_file=str(path),raw_valid=True))
        print(json.dumps(dict(stage=stage,done=i+1,total=len(tasks),seconds=time.perf_counter()-t)),flush=True)
    dump(out/'OBSERVATIONS.json',dict(rows=records,auxiliary=auxrecords,seconds=time.perf_counter()-start,blocked_truth_read=blocked,read_log=sorted(set(events))))
    seal(out)

def calibrate():
    from decisions import acceptance,distribution,crps,combine_scale
    verify(ROOT/'SOURCE_LOCK.json');verify(ROOT/'calibration/scores/SEAL.json')
    data=load(ROOT/'calibration/scores/OBSERVATIONS.json')['rows']
    truths={r['id']:r['true_depth'] for r in load(ROOT/'calibration/data/truth/metadata.json')}
    ell=[];spreads=[]
    for r in data:
        with np.load(r['curve_file']) as z:
            if r['arm']=='ED' and r['raw_valid']:
                idx=np.flatnonzero(np.isclose(z['grid'],truths[r['id']],rtol=0,atol=1e-9));assert len(idx)==1
                value=float(z['loss'][idx[0]])
                if np.isfinite(value):ell.append(dict(id=r['id'],mse=value))
            elif r['arm']=='N':
                l=1-np.mean(z['scores'],axis=0)
                valid_l=l[np.isfinite(l)]
                if len(valid_l)>1:spreads.append(float(np.percentile(valid_l,90)-np.percentile(valid_l,10)))
    assert ell and spreads,'calibration has no valid curves'
    sigma2=max(1.,float(np.percentile([r['mse'] for r in ell],80,method='linear')))
    nscale=max(1e-8,float(np.median(spreads)))
    temps=[.25,.5,1.,2.,4.];checks={arm:{str(t):[] for t in temps} for arm in ('ED','N')};exclusions=[]
    for r in data:
        with np.load(r['curve_file']) as z:
            g=z['grid'];l=z['loss'] if r['arm']=='ED' else 1-np.mean(z['scores'],axis=0)
            sc=combine_scale(z['sigma_values'],z['pixel_counts'],sigma2)[0] if r['arm']=='ED' else nscale
            a=acceptance(g,l,sc) if r['arm']=='ED' else z['accepted']
            valid_l=l[np.isfinite(l)]
            finite=bool(len(valid_l)>1) if r['arm']=='N' else bool(np.isfinite(l).all())
            flat=finite and np.ptp(valid_l)<=1e-10+1e-10*np.max(np.abs(valid_l))
            if not r['raw_valid'] or not finite or flat or not a.any():
                exclusions.append(dict(id=r['id'],arm=r['arm'],raw_valid=r['raw_valid'],finite=finite,flat=flat,accepted=int(a.sum())));continue
            for t in temps:
                q=distribution(g,l,sc,t,a)
                checks[r['arm']][str(t)].append(crps(g,q,truths[r['id']]))
    assert all(len(checks[a]['0.25']) for a in checks),'no supported calibration objects'
    means={a:{t:float(np.mean(v)) for t,v in tv.items()} for a,tv in checks.items()}
    chosen={a:min(temps,key=lambda t:(means[a][str(t)],t)) for a in means}
    dump(ROOT/'calibration/CALIBRATION.json',dict(created=now(),sigma_cal2=sigma2,temperature=chosen['ED'],full9_scale=nscale,full9_temperature=chosen['N'],ell=ell,full9_spreads=spreads,crps=means,counts={a:len(checks[a]['0.25']) for a in checks},exclusions=exclusions,score_seal=sha(ROOT/'calibration/scores/SEAL.json')))
    seal(ROOT/'calibration')
    print(json.dumps(dict(sigma_cal2=sigma2,temperature=chosen,counts={a:len(checks[a]['0.25']) for a in checks})))

def predictions(stage):
    from decisions import decide,combine_scale
    verify(ROOT/'SOURCE_LOCK.json');verify(ROOT/'calibration/SEAL.json')
    cal=load(ROOT/'calibration/CALIBRATION.json');rows=[]
    if stage=='replay':
        verify(OLD/'ordinary_stage/SEAL.json')
        observations=load(OLD/'ordinary_stage/OBSERVATIONS.json')['rows']
    else:
        assert load(ROOT/'replay/evaluation/GATE.json')['passed']
        verify(ROOT/'confirmation/scores/SEAL.json')
        observations=load(ROOT/'confirmation/scores/OBSERVATIONS.json')['rows']
    events=guard_no_truth(ROOT/stage)
    for obs in observations:
        incs=[540.,600.,660.] if stage=='replay' else [obs['incumbent']]
        for inc in incs:
            if stage=='replay' and obs['incumbent'] is not None and obs['incumbent']!=inc:continue
            candidates=[450.,540.,600.,660.,900.] if stage=='replay' else obs['candidates']
            with np.load(obs['curve_file']) as z:
                grid=z['grid'];isn=obs['arm']=='N'
                loss=1-np.mean(z['scores'],axis=0) if isn else z['loss']
                if isn:
                    variants=[('N_'+r,r,cal['full9_scale'],cal['full9_temperature'],z['accepted'],[]) for r in ('P','M','Mraw','R')]
                else:
                    variants=[]
                    for scname in ('S0','S1'):
                        sc,sources=combine_scale(z['sigma_values'],z['pixel_counts'],cal['sigma_cal2'] if scname=='S1' else None)
                        rules=('P','M','R') if stage=='replay' else (('P',) if scname=='S0' and obs['arm']=='ED' else (('M','R') if scname=='S1' else ()))
                        for rule in rules:variants.append((obs['arm']+'_'+scname+'_'+rule,rule,sc,cal['temperature'],None,sources))
                    if stage=='replay':
                        sc,sources=combine_scale(z['sigma_values'],z['pixel_counts'],cal['sigma_cal2'])
                        variants.append((obs['arm']+'_S1_U','U',sc,cal['temperature'],None,sources))
                    if stage=='replay' or obs['arm']=='ED':variants.append((obs['arm']+'_Mraw','Mraw',np.nan,1.,None,[]))
                for arm,rule,sc,t,a,sources in variants:
                    historical=obs.get('intervals') if stage=='replay' and (arm=='N_P' or arm.endswith('_S0_P')) else None
                    result=decide(grid,loss,candidates,inc,rule=rule,sigma2=sc,temperature=t,raw_valid=obs['raw_valid'],accepted=a,historical_intervals=historical)
                    rows.append(dict(id=obs['id'],task_id=obs.get('task_id'),arm=arm,initial_depth=inc,curve_file=obs['curve_file'],scale=sc,scale_sources=sources,candidates=candidates,**result))
                if isn:rows.append(dict(id=obs['id'],task_id=obs.get('task_id'),arm='KEEP',initial_depth=inc,selected_depth=inc,reason='identity',move=False,candidates=candidates))
    dump(ROOT/stage/'decisions/PREDICTIONS.json',rows)
    dump(ROOT/stage/'decisions/INPUTS.json',dict(calibration_sha=sha(ROOT/'calibration/CALIBRATION.json'),source_lock=sha(ROOT/'SOURCE_LOCK.json'),read_log=sorted(set(events))))
    seal(ROOT/stage/'decisions');print(json.dumps(dict(stage=stage,decisions=len(rows))))

def summary(rows):
    out=[]
    for arm,kind in sorted({(r['arm'],r['initial_kind']) for r in rows}):
        rr=[r for r in rows if r['arm']==arm and r['initial_kind']==kind]
        out.append(dict(arm=arm,initial_kind=kind,n=len(rr),mae=np.mean([r['error'] for r in rr]),mse=np.mean([r['error']**2 for r in rr]),improved=sum(r['change'] < -1e-9 for r in rr),harmed=sum(r['change']>1e-9 for r in rr),unchanged=sum(abs(r['change'])<=1e-9 for r in rr),practical_harm=sum(r['change']>7.5 for r in rr),move=sum(r['move'] for r in rr),rejected=sum(r['reason'] in ('raw_invalid','nonfinite_curve','flat_curve','scale_unavailable','threshold_empty') for r in rr),regret=np.mean([r['regret'] for r in rr])))
    return clean(out)

def evaluate(stage):
    verify(ROOT/stage/'decisions/SEAL.json')
    truth=load(OLD/'data/truth/metadata.json' if stage=='replay' else ROOT/'confirmation/data/truth/metadata.json')
    tm={r['id']:r for r in truth}
    inputs=observation_inputs(OLD/'data' if stage=='replay' else ROOT/'confirmation/data')
    tensors={}
    for r in inputs:
        with np.load(r['image_file']) as z:tensors[r['id']]=hashlib.sha256(z['images'].tobytes()).hexdigest()
    rows=load(ROOT/stage/'decisions/PREDICTIONS.json')
    for r in rows:
        tr=tm[r['id']]['true_depth']; initial=r['initial_depth']; err=abs(r['selected_depth']-tr)
        r.update(true_depth=tr,error=err,initial_error=abs(initial-tr),change=err-abs(initial-tr),mechanism=tm[r['id']]['mechanism'],tensor=tensors[r['id']],initial_kind='correct' if abs(initial-tr)<1e-9 else ('minus' if initial<tr else 'plus'),regret=err-min(abs(x-tr) for x in r['candidates']))
    out=ROOT/stage/'evaluation';dump(out/'ROWS.json',rows)
    overall=summary(rows);strata={m:summary([r for r in rows if r['mechanism']==m]) for m in sorted({r['mechanism'] for r in rows})}
    unique=[];seen=set()
    for r in rows:
        key=(r['tensor'],r['arm'],r['initial_depth'])
        if key not in seen:unique.append(r);seen.add(key)
    dump(out/'SUMMARY.json',dict(overall=overall,by_mechanism=strata,unique_tensors=summary(unique),distinct_tensors=len(set(tensors.values()))))
    with (out/'RESULTS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(overall[0]));writer.writeheader();writer.writerows(overall)
    main=[r for r in rows if r['arm']=='ED_S1_M']; mp={(r['id'],r['initial_depth']):r for r in main}
    nr=[r for r in rows if r['arm']=='N_P']; successes=[r for r in nr if r['error']<=1e-9 and r['initial_error']>1e-9]
    lost=[r for r in successes if mp[(r['id'],r['initial_depth'])]['error']>1e-9]
    checks=dict(correct_no_harm=all(r['error']<=1e-9 for r in main if r['initial_kind']=='correct'),minus_better_keep=np.mean([r['change'] for r in main if r['initial_kind']=='minus'])<0,plus_better_keep=np.mean([r['change'] for r in main if r['initial_kind']=='plus'])<0,retain_full9_success=len(lost)==0,equal_no_move=not any(r['move'] for r in main if r['mechanism']=='flat_equal'))
    if stage=='confirmation':
        nm_success=[r for r in rows if r['arm']=='N_M' and r['error']<=1e-9 and r['initial_error']>1e-9]
        checks['retain_matched_full9_M_success']=all(mp[(r['id'],r['initial_depth'])]['error']<=1e-9 for r in nm_success)
    dump(out/'GATE.json',dict(passed=all(checks.values()),checks=checks,full9_successes=len(successes),lost_full9_successes=len(lost),lost_rows=[dict(id=r['id'],initial_depth=r['initial_depth']) for r in lost]))
    if stage=='confirmation':
        rng=np.random.default_rng(42001);contrasts=[]
        for kind in ('correct','minus','plus'):
            for mech in ('all','flat_contrast','flat_equal','textured_boundary','textured_single'):
                for comparator in ('EF_S1_M','N_M','N_P','KEEP'):
                    aa={r['id']:r['error'] for r in rows if r['arm']=='ED_S1_M' and r['initial_kind']==kind and (mech=='all' or r['mechanism']==mech)}
                    bb={r['id']:r['error'] for r in rows if r['arm']==comparator and r['initial_kind']==kind and (mech=='all' or r['mechanism']==mech)}
                    delta=np.array([bb[i]-aa[i] for i in sorted(aa)])
                    boots=np.mean(delta[rng.integers(0,len(delta),size=(10000,len(delta)))],axis=1)
                    contrasts.append(dict(initial_kind=kind,mechanism=mech,comparator=comparator,n=len(delta),mae_gain=delta.mean(),ci95=np.percentile(boots,[2.5,97.5])))
        dump(out/'PAIRED_BOOTSTRAP.json',contrasts)
    seal(out)
    print(json.dumps(dict(stage=stage,gate=checks,main=[r for r in overall if r['arm'] in ('KEEP','ED_S1_M','N_M','N_P')]),indent=2))

def prepare_confirmation():
    assert load(ROOT/'replay/evaluation/GATE.json')['passed'],'B2 gate failed: confirmation forbidden'
    verify(ROOT/'SOURCE_LOCK.json');verify(ROOT/'calibration/SEAL.json')
    from new_fixtures import create_dataset
    create_dataset(ROOT/'confirmation/data','confirmation')
    truth={r['id']:r for r in load(ROOT/'confirmation/data/truth/metadata.json')}
    tasks=[]
    for r in observation_inputs(ROOT/'confirmation/data'):
        # Producer builds independent calls; the predictor never sees other incumbents.
        for j,offset in enumerate((-60.,0.,60.)):
            inc=truth[r['id']]['true_depth']+offset;candidates=np.unique(np.r_[np.arange(450.,901.,15.),inc])
            grid=np.unique(np.r_[np.arange(450.,901.,2.),candidates])
            tasks.append(dict(**r,task_id=f"t{len(tasks):04d}",incumbent=inc,candidates=candidates,grid=grid))
    dump(ROOT/'confirmation/TASKS.json',tasks);seal(ROOT/'confirmation/data')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare-calibration','score-calibration','calibrate','replay','evaluate-replay','prepare-confirmation','score-confirmation','confirm','evaluate-confirmation'])
    c=ap.parse_args().command
    started=now();stage_start=time.perf_counter()
    dump(ROOT/'phase_events'/f'{c}_started.json',dict(stage=c,status='running',started=started))
    try:
        if c=='prepare-calibration':prepare_calibration()
        elif c=='score-calibration':score('calibration')
        elif c=='calibrate':calibrate()
        elif c=='replay':predictions('replay')
        elif c=='evaluate-replay':evaluate('replay')
        elif c=='prepare-confirmation':prepare_confirmation()
        elif c=='score-confirmation':score('confirmation')
        elif c=='confirm':predictions('confirmation')
        else:evaluate('confirmation')
    except Exception as exc:
        dump(ROOT/'phase_events'/f'{c}_failed.json',dict(stage=c,status='failed',started=started,finished=now(),seconds=time.perf_counter()-stage_start,error=repr(exc)))
        raise
    dump(ROOT/'phase_events'/f'{c}_completed.json',dict(stage=c,status='completed',started=started,finished=now(),seconds=time.perf_counter()-stage_start))

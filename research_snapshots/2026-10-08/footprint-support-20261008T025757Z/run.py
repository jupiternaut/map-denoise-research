"""Separate data export, truth-free scoring, oracle scoring, choice, evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime,timezone
import numpy as np
from kernel import cameras,score,select
from estimator import estimate_fields

ROOT=Path(__file__).resolve().parent
OLD=Path('/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z')
OLDM=OLD/'mechanism'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)


def guard(allowed_reads,output_dirs):
    """Ordinary Python open guard: exact reads and separate write-only outputs."""
    allowed={str(Path(p).resolve()) for p in allowed_reads}
    outputs=[Path(p).resolve() for p in output_dirs]
    events=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        reading=not writing or bool(flags & os.O_RDWR) or isinstance(mode,str) and '+' in mode
        if reading and str(path) not in allowed:raise PermissionError('undeclared read: '+str(path))
        if writing and not any(path.is_relative_to(d) for d in outputs):raise PermissionError('undeclared write: '+str(path))
        if reading:events.append(str(path))
    sys.addaudithook(hook)
    return events


def verify_source_lock(observation_only=False):
    lock=load(ROOT/'LOCK.json')
    for p,h in lock['source_hashes'].items():
        if observation_only and not (p.endswith('.py') or p in {str(ROOT/'METHOD.json'),str(ROOT/'AGENTS.md'),str(ROOT/'refine-logs/EXPERIMENT_PLAN.md')}):continue
        assert sha(p)==h,p
    assert sha(ROOT/'OBS_INPUTS.json')==lock['observation_manifest_sha256']
    return lock


def prepare():
    assert socket.gethostname()=='liekkas'
    from oracle import ownership_fields
    old_lock=load(OLDM/'RENDER_LOCK.json')
    assert sha(OLDM/'FIXTURES.json')==old_lock['fixture_manifest_sha256']
    assert sha(OLDM/'run_mechanism.py')==old_lock['renderer_sha256']
    assert sha(OLDM/'PROTOCOL.json')==old_lock['protocol_sha256']
    metas=load(OLDM/'FIXTURES.json')
    for d in ('observed','oracle_inputs','curves','decisions','evaluation'):(ROOT/d).mkdir(exist_ok=True)
    inputs=[];truth=[];source_hashes={}
    for name in ('AGENTS.md','METHOD.json','CANDIDATES.json','kernel.py','run.py','estimator.py','oracle.py','test_kernel.py','test_estimator.py','test_oracle.py','refine-logs/EXPERIMENT_PLAN.md'):
        source_hashes[str(ROOT/name)]=sha(ROOT/name)
    for p in (OLD/'support.py',OLDM/'run_mechanism.py',OLDM/'PROTOCOL.json',OLDM/'FIXTURES.json',OLDM/'OBSERVATIONS.json',OLDM/'EVALUATION.json'):
        source_hashes[str(p)]=sha(p)
    for i,meta in enumerate(metas):
        fid=f'f{i:03d}';oldfile=OLDM/'fixtures'/meta['arrays']
        assert sha(oldfile)==meta['sha256'];source_hashes[str(oldfile)]=sha(oldfile)
        with np.load(oldfile) as data:images=data['images'].copy()
        imagefile=ROOT/'observed'/f'{fid}.npz'
        assert not imagefile.exists();np.savez_compressed(imagefile,images=images)
        fields=ownership_fields(meta)
        oraclefile=ROOT/'oracle_inputs'/f'{fid}.npz'
        assert not oraclefile.exists();np.savez_compressed(oraclefile,**fields)
        inputs.append(dict(id=fid,cameras=meta['score_cameras'],image_file=str(imagefile),sha256=sha(imagefile)))
        truth.append(dict(id=fid,historical_id=meta['id'],name=meta['name'],seed=meta['seed'],scene=meta['scene'],actual_cameras=meta['actual_cameras'],oracle_file=str(oraclefile),oracle_sha256=sha(oraclefile)))
    dump(ROOT/'OBS_INPUTS.json',inputs)
    dump(ROOT/'TRUTH.json',truth)
    dump(ROOT/'LOCK.json',dict(created=now(),host=socket.gethostname(),source_hashes=source_hashes,observation_manifest_sha256=sha(ROOT/'OBS_INPUTS.json'),truth_manifest_sha256=sha(ROOT/'TRUTH.json')))
    print(json.dumps(dict(stage='prepare',fixtures=len(inputs))))


def extract(which):
    lock=verify_source_lock(observation_only=True)
    method=load(ROOT/'METHOD.json');inputs=load(ROOT/'OBS_INPUTS.json')
    grid=np.arange(method['grid_start'],method['grid_stop']+method['grid_step']/2,method['grid_step'])
    oracle=(which=='oracle')
    truth=load(ROOT/'TRUTH.json') if oracle else None
    arms=[a for a in method['arms'] if (a.startswith('oracle_'))==oracle]
    allowed=[ROOT/'METHOD.json',ROOT/'OBS_INPUTS.json']+[Path(i['image_file']) for i in inputs]
    if oracle:allowed += [ROOT/'TRUTH.json']+[Path(t['oracle_file']) for t in truth]
    # Warm numpy archive/compression modules before the restricted read phase.
    with np.load(inputs[0]['image_file']) as d:assert d['images'].shape==(3,128,128)
    out_dir=ROOT/'curves'/which;out_dir.mkdir(exist_ok=True)
    stage_dir=ROOT/('oracle_stage' if oracle else 'observed_stage');stage_dir.mkdir(exist_ok=True)
    read_log=guard(allowed,[out_dir,stage_dir])
    blocked=[]
    for p,mode in [(ROOT/'CANDIDATES.json','r'),(ROOT/'CANDIDATES.json','r+'),(ROOT/'TRUTH.json','r')]:
        if oracle and p.name=='TRUTH.json':continue
        try:
            with p.open(mode):pass
        except PermissionError:blocked.append([str(p),mode])
        else:raise AssertionError('read guard did not reject '+str(p))
    rows=[];curve_hashes={}
    for index,inp in enumerate(inputs):
        assert sha(inp['image_file'])==inp['sha256']
        with np.load(inp['image_file']) as d:images=d['images'].copy()
        if oracle:
            t=truth[index];assert t['id']==inp['id'];assert sha(t['oracle_file'])==t['oracle_sha256']
            with np.load(t['oracle_file']) as d:fields={k:d[k].copy() for k in ('center','area','numerator')}
            metadata=None
        else:
            fields=estimate_fields(images,xy=method['xy']);metadata=fields['metadata']
        cams=cameras(inp['cameras'])
        for arm in arms:
            curves,intervals=score(images,cams,grid,method,arm,fields)
            file=out_dir/f"{inp['id']}_{arm}.npz"
            # In-memory bytes permit a hash without reopening write-only output.
            import io
            buffer=io.BytesIO();np.savez_compressed(buffer,**curves)
            payload=buffer.getvalue()
            with file.open('xb') as f:f.write(payload)
            digest=hashlib.sha256(payload).hexdigest();curve_hashes[str(file)]=digest
            rows.append(dict(id=inp['id'],arm=arm,intervals=intervals,curve_file=str(file),curve_sha256=digest,
                  accepted_grid_count=int(curves['accepted'].sum()),qualified_grid_count=int(curves['qualified'].sum()),estimator_metadata=metadata))
    output=dict(created=now(),rows=rows,read_log=sorted(set(read_log)),blocked_reads=blocked,
                interface='images/cameras only' if not oracle else 'truth diagnostic',source_lock=lock['observation_manifest_sha256'])
    payload=json.dumps(output,indent=2,allow_nan=False).encode()
    with (stage_dir/'OBSERVATIONS.json').open('xb') as f:f.write(payload)
    dump(stage_dir/'SEAL.json',dict(created=now(),observations_sha256=hashlib.sha256(payload).hexdigest(),curves=curve_hashes))
    print(json.dumps(dict(stage=which,arms=arms,observations=len(rows),sealed_before_candidates=True)))


def observations():
    out=[]
    for d in ('observed_stage','oracle_stage'):
        seal=load(ROOT/d/'SEAL.json');assert sha(ROOT/d/'OBSERVATIONS.json')==seal['observations_sha256']
        for p,h in seal['curves'].items():assert sha(p)==h
        out.extend(load(ROOT/d/'OBSERVATIONS.json')['rows'])
    return out


def infer():
    lock=verify_source_lock(observation_only=True);obs=observations()
    allowed=[ROOT/'CANDIDATES.json',ROOT/'decisions/DECISIONS.json']+[ROOT/d/'SEAL.json' for d in ('observed_stage','oracle_stage')]
    guard(allowed,[ROOT/'decisions'])
    assert sha(ROOT/'CANDIDATES.json')==lock['source_hashes'][str(ROOT/'CANDIDATES.json')]
    config=load(ROOT/'CANDIDATES.json')
    rows=[dict(id=o['id'],arm=o['arm'],**select(o['intervals'],config)) for o in obs]
    dump(ROOT/'decisions/DECISIONS.json',rows)
    dump(ROOT/'decisions/SEAL.json',dict(created=now(),decisions_sha256=sha(ROOT/'decisions/DECISIONS.json'),observation_seals={d:sha(ROOT/d/'SEAL.json') for d in ('observed_stage','oracle_stage')}))
    print(json.dumps(dict(stage='infer',decisions=len(rows))))


def interval_distance(a,b):
    la=sum(y-x for x,y in a);lb=sum(y-x for x,y in b)
    intersection=sum(max(0.,min(y,v)-max(x,u)) for x,y in a for u,v in b)
    return la+lb-2*intersection


def evaluate():
    verify_source_lock();obs=observations();ds=load(ROOT/'decisions/SEAL.json')
    assert sha(ROOT/'decisions/DECISIONS.json')==ds['decisions_sha256']
    for d,h in ds['observation_seals'].items():assert sha(ROOT/d/'SEAL.json')==h
    decisions={(d['id'],d['arm']):d for d in load(ROOT/'decisions/DECISIONS.json')}
    truth={t['id']:t for t in load(ROOT/'TRUTH.json')};method=load(ROOT/'METHOD.json')
    inputs={r['id']:r for r in load(ROOT/'OBS_INPUTS.json')}
    from oracle import _renderer
    renderer=_renderer()
    rows=[];mapping={(o['id'],o['arm']):o for o in obs}
    for o in obs:
        t=truth[o['id']];cam=renderer.deserialize_cam(t['actual_cameras'][0])
        z=float(renderer.intersect(t['scene'],cam,np.asarray(method['xy']))['depth'])
        d=decisions[o['id'],o['arm']]
        with np.load(o['curve_file']) as curve:
            ti=int(np.argmin(abs(curve['grid']-z)))
            true_mass=float(curve['mass'][ti]);true_ess=float(curve['ess'][ti]);true_count=int(curve['count'][ti])
        err=abs(d['selected_depth']-z)
        rows.append(dict(**o,**d,name=t['name'],seed=t['seed'],true_depth=z,absolute_error=err,squared_error=err**2,
            true_in_support=any(a<=z<=b for a,b in o['intervals']),exact=err<1e-9,empty=not bool(o['intervals']),
            true_mass=true_mass,true_ess=true_ess,true_count=true_count))
    summary=[]
    for group,predicate in [('all',lambda r:True),('ring',lambda r:r['name'].startswith('ring')),('other',lambda r:not r['name'].startswith('ring'))]:
        base={r['id']:r for r in rows if r['arm']=='full9' and predicate(r)}
        for arm in method['arms']:
            sub=[r for r in rows if r['arm']==arm and predicate(r)]
            dif=[r['absolute_error']-base[r['id']]['absolute_error'] for r in sub]
            summary.append(dict(group=group,arm=arm,n=len(sub),mae=float(np.mean([r['absolute_error'] for r in sub])),mse=float(np.mean([r['squared_error'] for r in sub])),
                exact=sum(r['exact'] for r in sub),truth_in_support=sum(r['true_in_support'] for r in sub),empty=sum(r['empty'] for r in sub),
                improved_vs_full9=sum(x < -1e-9 for x in dif),worsened_vs_full9=sum(x > 1e-9 for x in dif),unchanged_vs_full9=sum(abs(x)<=1e-9 for x in dif),
                retained_full9_exact=sum(r['exact'] and base[r['id']]['exact'] for r in sub),baseline_exact=sum(r['exact'] for r in base.values()),
                improved_vs_incumbent=sum(r['absolute_error']<abs(900-r['true_depth'])-1e-9 for r in sub),worsened_vs_incumbent=sum(r['absolute_error']>abs(900-r['true_depth'])+1e-9 for r in sub)))
    pairs=[]
    for size in ('ring9','ring25'):
        for seed in (11,29,47):
            f=next(t for t in truth.values() if t['name']==size+'_flat' and t['seed']==seed)
            b=next(t for t in truth.values() if t['name']==size+'_textured' and t['seed']==seed)
            with np.load(inputs[f['id']]['image_file']) as a,np.load(inputs[b['id']]['image_file']) as c:
                center_diff=float(np.max(abs(a['images'][0,63:66,63:66]-c['images'][0,63:66,63:66])))
            assert center_diff==0
            for arm in method['arms']:
                oa=mapping[f['id'],arm];ob=mapping[b['id'],arm]
                with np.load(oa['curve_file']) as a,np.load(ob['curve_file']) as bcurve:
                    xor=int(np.sum(a['accepted']!=bcurve['accepted']))
                    wd=float(np.max(abs(a['weights']-bcurve['weights'])))
                    both=np.isfinite(a['scores'])&np.isfinite(bcurve['scores'])
                    score_delta=float(np.max(abs(a['scores'][both]-bcurve['scores'][both]))) if both.any() else None
                    values_same=bool(np.allclose(a['reference'],bcurve['reference'],atol=1e-10) and np.allclose(a['source'],bcurve['source'],atol=1e-10))
                pairs.append(dict(pair=size,seed=seed,arm=arm,reference_center_difference=center_diff,accepted_xor=xor,
                  support_symmetric_difference=interval_distance(oa['intervals'],ob['intervals']),weight_max_difference=wd,finite_score_max_difference=score_delta,
                  both_empty=not oa['intervals'] and not ob['intervals'],both_nonempty=bool(oa['intervals']) and bool(ob['intervals']),
                  selected_depth_difference=abs(decisions[f['id'],arm]['selected_depth']-decisions[b['id'],arm]['selected_depth']),
                  component_arrays_same=values_same))
    old={(r['fixture_id'],r['arm']):r for r in load(OLDM/'EVALUATION.json')}
    checks=[]
    for r in rows:
        if r['arm'] not in ('full9','connected9'):continue
        archived=old[truth[r['id']]['historical_id'],'plane_'+r['arm']]
        assert r['intervals']==archived['intervals']
        assert r['selected_depth']==archived['selected_depth']
        checks.append([r['id'],r['arm']])
    # Acceptance is fixed in the plan; no thresholds selected using this result.
    primary=next(s for s in summary if s['group']=='ring' and s['arm']==method['primary_arm'])
    base=next(s for s in summary if s['group']=='ring' and s['arm']=='full9')
    flips={a:sum(p['selected_depth_difference']>1e-9 for p in pairs if p['arm']==a) for a in method['arms']}
    success=primary['mae']<base['mae'] and primary['retained_full9_exact']==base['exact'] and flips[method['primary_arm']]<flips['full9']
    dump(ROOT/'evaluation/ROWS.json',rows);dump(ROOT/'evaluation/PAIRS.json',pairs)
    dump(ROOT/'evaluation/RESULTS.json',dict(created=now(),summary=summary,baseline_reproductions=len(checks),primary_success=success,selection_flips=flips))
    print(json.dumps(dict(stage='evaluate',summary=summary,selection_flips=flips,primary_success=success),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','score','oracle','infer','evaluate'])
    stage=parser.parse_args().stage
    if stage in ('score','oracle'):extract(stage)
    else:{'prepare':prepare,'infer':infer,'evaluate':evaluate}[stage]()

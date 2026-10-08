"""CPU development replay. Separate lock, observation, prediction, evaluation."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import json
import math
import socket
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from support import camera, score_support, support_intervals, intersection

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent/'selector-attribution-20261007T180539Z'
TRACK = ROOT.parent/'track-discrimination-20261007T160716Z'
ANCESTOR = ROOT.parent/'surface-evidence-20261001T140058Z/evaluation'
PYTHON = '/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check(test, message):
    if not test:
        raise RuntimeError(message)


def key(row):
    return row['roi'], int(row['query'])


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def verify(files):
    for name, expected in files.items():
        check(sha(name) == expected, 'hash mismatch: '+name)


def lock():
    check(socket.gethostname() == 'liekkas', 'wrong host')
    verify(load(HISTORY/'RUN_LOCK.json')['files'])
    req = load(TRACK/'REQUESTS.json')
    for row in req['rows']:
        check(set(row) == {'scene','roi','query','pixel_xy'}, 'request schema')
    dump(ROOT/'REQUESTS.json', req)
    dump(ROOT/'INPUTS.json', load(HISTORY/'INPUTS.json'))
    files = [ROOT/name for name in ('AGENTS.md','PROTOCOL.json','support.py','test_support.py',
                                    'run_experiment.py','REQUESTS.json','INPUTS.json',
                                    'refine-logs/EXPERIMENT_PLAN.md')]
    files += [HISTORY/'policies.py', TRACK/'theory/kernel.py', TRACK/'observation/TRACKS.json',
              HISTORY/'RUN_LOCK.json']
    photos = {c['image_path']:c['image_sha256'] for s in req['scenes'].values() for c in s['cameras'].values()}
    verify(photos)
    dump(ROOT/'RUN_LOCK.json', dict(created_at=now(), host=socket.gethostname(),
         files={str(p):sha(p) for p in files}, photos=photos, historical_git_scope='read-only',
         tuning='fixed before raw extraction; development history already known'))
    print('Locked', len(files), 'files and', len(photos), 'images')


def observation_guard(allowed):
    reads = []
    allowed = {str(Path(p).resolve()) for p in allowed}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str,bytes,os.PathLike)):
            path = os.path.realpath(os.fsdecode(args[0]))
            mode, flags = args[1:3]
            writing = (isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing:
                if not path.startswith(str(ROOT)+'/'):
                    raise PermissionError('write outside new run')
            elif path not in allowed:
                raise PermissionError('observation read outside image/request contract: '+path)
            else:
                reads.append(path)
        elif event in ('socket.connect','subprocess.Popen','os.system'):
            raise PermissionError('observation external operation forbidden')
    sys.addaudithook(audit)
    return reads


def extract():
    start = time.monotonic()
    lockdata = load(ROOT/'RUN_LOCK.json')
    # Do not even hash-read candidate INPUTS in this process.
    core = {p:h for p,h in lockdata['files'].items() if Path(p).parent == ROOT and Path(p).name != 'INPUTS.json'}
    verify(core)
    req = load(ROOT/'REQUESTS.json')
    protocol = load(ROOT/'PROTOCOL.json')
    Image.init()
    readlog = observation_guard(list(core)+list(lockdata['photos']))
    rejected = []
    for forbidden in (ROOT/'INPUTS.json', ANCESTOR/'CANDIDATE_DISTANCES.csv'):
        try:
            forbidden.read_bytes()
        except PermissionError:
            rejected.append(str(forbidden))
        else:
            raise RuntimeError('observation guard failed')
    images, cams = {}, {}
    for sn, scene in req['scenes'].items():
        images[sn], cams[sn] = {}, {}
        for name, data in scene['cameras'].items():
            check(sha(data['image_path']) == data['image_sha256'], 'photo changed')
            with Image.open(data['image_path']) as im:
                images[sn][name] = np.asarray(im.convert('L').resize((data['width'],data['height']),
                      resample=Image.Resampling.BILINEAR),float)
            cams[sn][name] = camera(data)
    rows = []
    for request in req['rows']:
        sn = str(request['scene']); scene = req['scenes'][sn]
        refname = scene['reference']; lo, hi = scene['depth_range_mm']
        grid = np.arange(lo, hi+0.01, 1.)
        if grid[-1] < hi:
            grid = np.append(grid, hi)
        out = dict(request, depth_grid_mm=grid.tolist(), evidence={})
        for warp in ('translation','plane'):
            for mask in ('full9','center3','connected9'):
                name = warp+'_'+mask
                scans = [score_support(images[sn][refname], images[sn][src], cams[sn][refname],
                         cams[sn][src], request['pixel_xy'], grid, warp, mask) for src in scene['sources']]
                accepted = [np.isfinite(s['scores']) & (s['scores'] >= .6) for s in scans]
                joint = accepted[0] & accepted[1]
                intervals = support_intervals(grid, joint)
                diagnostics = []
                for src, s, good in zip(scene['sources'],scans,accepted):
                    diagnostics.append(dict(source=src, mask=s['mask'].astype(int).tolist(),
                        mask_count=s['mask_count'], anchor_std=s['anchor_std'], valid_count=s['valid_count'],
                        accepted_count=int(good.sum()), scores=[float(v) if np.isfinite(v) else None for v in s['scores']]))
                out['evidence'][name] = dict(intervals=intervals, joint_sample_indices=np.flatnonzero(joint).tolist(),
                      support_count=len(intervals), scans=diagnostics,
                      empty_reason=None if intervals else ('insufficient_reference_support' if any(s['mask_count']<9 or s['anchor_std']<3 for s in scans) else 'no_shared_depth'))
                if name == 'translation_full9':
                    legacy = intersection(support_intervals(grid,accepted[0]),support_intervals(grid,accepted[1]))
                    out['evidence']['legacy_full9'] = dict(intervals=legacy, support_count=len(legacy),
                         representation='independently padded source intervals; no same-grid requirement')
        rows.append(out)
        print(request['roi'],request['query'], {a:len(out['evidence'][a]['intervals']) for a in protocol['arms']},flush=True)
    dump(ROOT/'OBSERVATIONS.json', dict(created_at=now(), rows=rows, read_log=sorted(set(readlog)),
          blocked_negative_reads=rejected, wallclock_seconds=time.monotonic()-start,
          candidate_access=False, truth_access=False, interface='Python read guard, not OS security sandbox'))
    # Current process is forbidden from re-reading its output, hash from canonical serialized bytes.
    path = ROOT/'OBSERVATIONS.json'
    print('Observation sealed by exclusive file creation:', str(path))


def policy_module():
    spec = importlib.util.spec_from_file_location('_frozen_P', HISTORY/'policies.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer():
    verify(load(ROOT/'RUN_LOCK.json')['files'])
    def no_truth(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path = os.fsdecode(args[0]).lower()
            if '/evaluation/' in path or path.endswith('.ply') or 'candidate_distances' in path:
                raise PermissionError('prediction cannot read evaluation')
    sys.addaudithook(no_truth)
    policy = policy_module()
    inputs = load(ROOT/'INPUTS.json')['rows']
    observations = {key(r):r for r in load(ROOT/'OBSERVATIONS.json')['rows']}
    protocol = load(ROOT/'PROTOCOL.json')
    previous = {key(r):r for r in load(TRACK/'observation/TRACKS.json')['rows']}
    rows = []
    for row in inputs:
        obs = observations[key(row)]
        check(obs['evidence']['legacy_full9']['intervals'] == previous[key(row)]['star_full_intervals'],
              'legacy full9 support failed exact reproduction')
        decisions = {}
        for arm in protocol['arms']:
            x = dict(row, intervals={'star_full':obs['evidence'][arm]['intervals']})
            decisions[arm] = policy.new_policy(x, 'star_full', 'P')
        rows.append(dict(scene=row['scene'],roi=row['roi'],query=row['query'],current_id=row['current_id'],
                         primary=row['primary'],decisions=decisions))
    dump(ROOT/'DECISIONS.json', dict(created_at=now(),rows=rows,old_support_reproduced=21,
                                    rule='frozen P; same old candidate coordinates and current output'))
    paths = [ROOT/name for name in ('RUN_LOCK.json','OBSERVATIONS.json','DECISIONS.json')]
    dump(ROOT/'PREDICTIONS_SEALED.json',dict(created_at=now(),files={str(p):sha(p) for p in paths}))
    print('Sealed',len(rows),'requests x',len(protocol['arms']),'arms before evaluation')


def metric(rows, photo):
    primary = [r for r in rows if r['primary']]
    check(len(primary)==484 and all(r['distance_mm'] is not None for r in primary), 'primary denominator')
    per_roi = {}
    for roi in sorted({r['roi'] for r in primary}):
        ds = [r['distance_mm'] for r in primary if r['roi']==roi]
        per_roi[roi] = dict(n=len(ds),mse_mm2=math.fsum(d*d for d in ds)/len(ds),mae_mm=math.fsum(ds)/len(ds))
    differences = [r['distance_mm']-photo[key(r)]['distance_mm'] for r in primary]
    return dict(arm=rows[0]['arm'],n=484,mse_mm2=math.fsum(v['mse_mm2'] for v in per_roi.values())/4,
          mae_mm=math.fsum(v['mae_mm'] for v in per_roi.values())/4,roi=per_roi,
          improved=sum(d < -1e-9 for d in differences),worsened=sum(d > 1e-9 for d in differences),
          unchanged=sum(abs(d)<=1e-9 for d in differences),
          new_harm=sum(photo[key(r)]['distance_mm']<=1 and r['distance_mm']>1 for r in primary),
          severe=sum(r['distance_mm']>5 for r in primary),finite=sum(r['distance_mm'] is not None for r in rows),
          missing_finite=sum(not r['primary'] and r['distance_mm'] is not None for r in rows))


def evaluate():
    verify(load(ROOT/'RUN_LOCK.json')['files'])
    verify(load(ROOT/'PREDICTIONS_SEALED.json')['files'])
    archived = read_csv(TRACK/'evaluation/POINT_METRICS.csv')
    originals = {key(r):r for r in read_csv(ANCESTOR/'POINT_METRICS.csv') if r['arm']=='incumbent'}
    candidates = {(*key(r),int(r['candidate_id'])):r for r in read_csv(ANCESTOR/'CANDIDATE_DISTANCES.csv')}
    inputs = load(ROOT/'INPUTS.json')['rows']
    count = 0
    for row in inputs:
        for obj in row['objects']:
            cid = obj['candidate_id']
            old = originals[key(row)] if cid==-1 else candidates[(*key(row),cid)]
            check(obj['xyz_mm'] == [float(old[k]) for k in ('x_mm','y_mm','z_mm')], 'candidate identity')
            count += 1
    photo = {key(p):dict(arm='photo_U11',roi=p['roi'],query=int(p['query']),primary=p['primary']=='True',
           selected=int(p['selected']) if p['selected'] else None, reason=p['reason'],
           distance_mm=float(p['distance_mm']) if p['distance_mm'] else None) for p in archived if p['arm']=='photo_U11'}
    check(len(photo)==512,'full denominator')
    decisions = {key(r):r for r in load(ROOT/'DECISIONS.json')['rows']}
    observations = load(ROOT/'OBSERVATIONS.json')['rows']
    arms = load(ROOT/'PROTOCOL.json')['arms']
    allrows, transitions = list(photo.values()), []
    for arm in arms:
        for k,p in photo.items():
            q = dict(p,arm=arm)
            if k in decisions:
                d = decisions[k]['decisions'][arm]; cid = d['selected_candidate_id']
                q.update(selected=cid,reason=d['reason'])
                if cid is None:
                    q['distance_mm'] = None
                else:
                    target = originals[k] if cid==-1 else candidates[(*k,cid)]
                    q['distance_mm'] = float(target['distance_mm']) if target['distance_mm'] else None
                if cid != decisions[k]['current_id']:
                    transitions.append(dict(arm=arm,roi=k[0],query=k[1],photo_mm=p['distance_mm'],
                       output_mm=q['distance_mm'],photo_id=decisions[k]['current_id'],selected=cid))
            allrows.append(q)
    metrics = [metric([r for r in allrows if r['arm']==a], photo) for a in ['photo_U11']+arms]
    baseline = metrics[0]['mse_mm2']
    for m in metrics:
        m['mse_reduction_pct_vs_photo'] = 100*(baseline-m['mse_mm2'])/baseline
    legacy = next(m for m in metrics if m['arm']=='legacy_full9')
    historical = next(m for m in load(HISTORY/'evaluation/RESULTS.json')['metrics'] if m['arm']=='star_full_P')
    check(math.isclose(legacy['mse_mm2'],historical['mse_mm2'],abs_tol=1e-12), 'legacy metric mismatch')
    support_summary = {arm:dict(nonempty=sum(bool(r['evidence'][arm]['intervals']) for r in observations),
                      components=sum(len(r['evidence'][arm]['intervals']) for r in observations)) for arm in arms}
    diag = []
    for row in observations:
        for arm,ev in row['evidence'].items():
            if arm=='legacy_full9':
                continue
            s = ev['scans'][0]
            diag.append(dict(roi=row['roi'],query=row['query'],arm=arm,mask_count=s['mask_count'],
                 anchor_std=s['anchor_std'],joint_samples=len(ev['joint_sample_indices']),components=len(ev['intervals']),
                 empty_reason=ev['empty_reason']))
    primary = next(m for m in metrics if m['arm']=='plane_connected9')
    good_repairs = [('scan118_upper_fold',5),('scan118_upper_fold',95)]
    # Use actual ROI names from shared locked input, no hard-coded candidate choices.
    repaired_keys = [k for k in decisions if 'upper_fold' in k[0] and k[1] in (5,95)]
    retain = all(decisions[k]['decisions']['plane_connected9']['selected_candidate_id'] ==
                 decisions[k]['decisions']['legacy_full9']['selected_candidate_id'] for k in repaired_keys)
    success = primary['mse_mm2']<legacy['mse_mm2'] and primary['new_harm']<legacy['new_harm'] and retain and len(repaired_keys)==2
    dump(ROOT/'evaluation/RESULTS.json',dict(created_at=now(),metrics=metrics,support_summary=support_summary,
          candidate_coordinate_checks=count,legacy_replay=True,primary_success=success,
          primary_retains_two_large_repairs=retain,scope='two exposed scenes, nearest laser vertex metric, not physical layer labels',
          evaluation_inputs={str(p):sha(p) for p in [TRACK/'evaluation/POINT_METRICS.csv',ANCESTOR/'POINT_METRICS.csv',ANCESTOR/'CANDIDATE_DISTANCES.csv']}))
    write_csv(ROOT/'evaluation/POINT_METRICS.csv',allrows,list(allrows[0]))
    write_csv(ROOT/'evaluation/TRANSITIONS.csv',transitions,['arm','roi','query','photo_mm','output_mm','photo_id','selected'])
    write_csv(ROOT/'evaluation/SUPPORT_DIAGNOSTICS.csv',diag,list(diag[0]))
    print(json.dumps(dict(metrics=metrics,support_summary=support_summary,primary_success=success),indent=2))


if __name__=='__main__':
    check(__debug__, 'run without -O')
    {'lock':lock,'extract':extract,'infer':infer,'evaluate':evaluate}[sys.argv[1]]()

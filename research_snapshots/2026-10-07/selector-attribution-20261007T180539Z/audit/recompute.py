"""Independent read-only replay and distance verifier; outputs only into audit/.

No author selector, gain-kernel, evaluator or PLY-reader imports. Author synthetic
tests run separately in a child process with -B. The gain calculation uses exact
squared Euclidean distances at every endpoint, rather than author affine code.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import io
import json
import math
import socket
import struct
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z')
RUNS = ROOT.parent
PREV = RUNS / 'track-discrimination-20261007T160716Z'
SURFACE = RUNS / 'surface-evidence-20261001T140058Z'
OFFICIAL = RUNS / 'official-mechanism-20261001T183124Z'
TRANSFER = RUNS / 'colmap-transfer-20260930T180000Z'
PYTHON = '/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python'
HASHES = {}
CHECKS = Counter()

def raw(path):
    path = Path(path)
    data = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return data

def js(path): return json.loads(raw(path))
def table(path): return list(csv.DictReader(io.StringIO(raw(path).decode())))
def key(row): return row['roi'], int(row['query'])
def unique(rows, fields):
    indexed = {tuple(r[f] for f in fields): r for r in rows}
    assert len(indexed) == len(rows), (fields, 'duplicate keys')
    return indexed

def equal(actual, expected, name):
    assert actual == expected, (name, actual, expected)
    CHECKS[name] += 1

def near(actual, expected, name):
    assert math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-12), (name, actual, expected)
    CHECKS[name] += 1

def compare(a, b, name):
    if isinstance(b, dict):
        equal(set(a), set(b), name + ':keys')
        for k in b: compare(a[k], b[k], name + '.' + k)
    elif isinstance(b, float): near(a, b, name)
    else: equal(a, b, name)

def union(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
        if merged and lo <= merged[-1][1]: merged[-1][1] = max(hi, merged[-1][1])
        else: merged.append([lo, hi])
    return merged

def round_out(q, low):
    f = float(q)
    if (low and F(f) > q) or (not low and F(f) < q):
        f = math.nextafter(f, -math.inf if low else math.inf)
    return f

def gain(a, b, center, ray, intervals):
    """Independent exact subtraction of squared distances at all endpoints."""
    values = []
    for z in (z for pair in intervals for z in pair):
        target = [F(c) + F(z) * F(r) for c, r in zip(center, ray)]
        values.append(sum((F(x)-t)**2 - (F(y)-t)**2 for x,y,t in zip(a,b,target)))
    return round_out(min(values), True), round_out(max(values), False)

def new_choice(row, evidence, policy, saved):
    pool = {o['candidate_id']: o for o in row['objects']}
    current = row['current_id']
    support = union(row['intervals'][evidence])
    equal(saved['intervals'], support, 'merged_support')
    if current not in pool:
        equal(saved['reason'], 'NO_CURRENT_OUTPUT', 'fallback_reason')
        equal(saved['scores'], [], 'fallback_scores')
        return current
    if not support:
        equal(saved['reason'], 'EMPTY_UNKNOWN', 'fallback_reason')
        equal(saved['scores'], [], 'fallback_scores')
        return current
    target = None
    if policy in ('P','GP'):
        size = math.fsum(b-a for a,b in support)
        target = (math.fsum((b-a)*(a+(b-a)/2) for a,b in support)/size if size
                  else math.fsum(a for a,b in support)/len(support))
    elif policy in ('Q','QG'):
        tracks = row['tracks'][evidence]
        assert tracks
        winner = sorted(tracks, key=lambda t: (-math.fsum(t['ncc_reference_sources'])/2,t['id']))[0]
        projection = [max(a,min(b,winner['reference_depth_mm'])) for a,b in union(winner['intervals_mm'])]
        target = sorted(projection, key=lambda x: (abs(x-winner['reference_depth_mm']),x))[0]
        equal(saved['track_id'], winner['id'], 'track_id')
        equal(saved['track_intervals'], union(winner['intervals_mm']), 'track_support')
        equal(saved['track_reference_depth'], winner['reference_depth_mm'], 'track_depth')
        equal(saved['track_mean_ncc'], math.fsum(winner['ncc_reference_sources'])/2, 'track_ncc')
    equal(saved['target_depth'], target, 'target_depth')
    records = []
    for cid, obj in sorted(pool.items()):
        low, high = gain(pool[current]['xyz_mm'], obj['xyz_mm'], row['center'], row['ray'], support)
        plow, phigh = (None,None) if target is None else gain(pool[current]['xyz_mm'], obj['xyz_mm'], row['center'], row['ray'], [[target,target]])
        gate = cid != current and low > 0
        positive = cid != current and plow is not None and plow > 0
        eligible = gate if policy == 'R' else positive and (gate or policy in ('P','Q'))
        records.append(dict(candidate_id=cid,gain_low_mm2=low,gain_high_mm2=high,
            point_gain_low_mm2=plow,point_gain_high_mm2=phigh,ranking_score=low if policy=='R' else plow,
            gate_pass=gate,positive_target_gain=positive,eligible=eligible))
    equal(records, saved['scores'], 'exact_gain_score_rows')
    equal([r['candidate_id'] for r in records if r['gate_pass']], saved['gate_eligible_ids'], 'gate_eligibility')
    valid = [r for r in records if r['eligible']]
    equal([r['candidate_id'] for r in valid], saved['eligible_ids'], 'selection_eligibility')
    if not valid:
        equal(saved['reason'], 'NO_UNIFORM_MODEL_GAIN' if policy in ('GP','QG','R') else 'NO_POSITIVE_POINT_GAIN', 'keep_reason')
        return current
    winner = sorted(valid,key=lambda r:(-r['ranking_score'],r['candidate_id']))[0]
    equal(saved['winning_score'], winner['ranking_score'], 'winning_score')
    equal(saved['reason'], 'POSITIVE_GAIN_ON_ASSUMED_SET' if policy in ('GP','QG','R') else 'POSITIVE_POINT_GAIN', 'move_reason')
    return winner['candidate_id']

def old_choice(row, arm, saved):
    pool = {o['candidate_id']:o for o in row['objects']}
    proposals = sorted([o for o in pool.values() if o['candidate_id'] != -1 and o['old_score'] is not None and math.isfinite(o['old_score'])],key=lambda o:(-o['old_score'],o['candidate_id']))
    equal([o['candidate_id'] for o in proposals], saved['eligible_ids'], 'old_eligibility')
    if arm != 'old_replay' and row['current_id'] is None: return None
    if not proposals: return row['original_id']
    winner = proposals[0]
    score = pool.get(-1,{}).get('old_score')
    if score is not None and arm != 'old_raw' and winner['old_score']-score < .05-1e-12:
        return row['original_id']
    ambiguous = any(abs(o['depth_mm']-winner['depth_mm'])>=5 and winner['old_score']-o['old_score']<=.05+1e-12 for o in proposals[1:])
    if ambiguous and arm == 'old_replay': return row['original_id']
    return winner['candidate_id']

def convert(row):
    return dict(arm=row['arm'],roi=row['roi'],query=int(row['query']),primary=row['primary']=='True',
                selected=int(row['selected']) if row['selected'] else None,reason=row['reason'],
                distance_mm=float(row['distance_mm']) if row['distance_mm'] else None)

def read_vertices(path):
    """Read the dataset's simple binary little-endian PLY without old code."""
    data = raw(path)
    end = data.index(b'\n', data.index(b'end_header'))+1
    lines = data[:end].decode('ascii').splitlines()
    assert lines[0] == 'ply' and 'format binary_little_endian 1.0' in lines
    kinds = {'float':'<f4','double':'<f8','uchar':'u1','char':'i1','int':'<i4','uint':'<u4','short':'<i2','ushort':'<u2'}
    fields=[]; count=None; in_vertices=False
    for line in lines:
        words=line.split()
        if words[:2]==['element','vertex']:
            count=int(words[2]);in_vertices=True
        elif words[:1]==['element']:in_vertices=False
        elif words[:1]==['property'] and in_vertices:
            assert len(words)==3
            fields.append((words[2],kinds[words[1]]))
    arr=np.frombuffer(data,dtype=np.dtype(fields),count=count,offset=end)
    xyz=np.column_stack([arr[n].astype(np.float64) for n in ('x','y','z')])
    assert np.isfinite(xyz).all()
    return xyz

def main():
    assert socket.gethostname() == 'liekkas'
    assert Path(__file__).resolve().parent == ROOT/'audit'
    required = ['AGENTS.md','PROTOCOL.json','refine-logs/EXPERIMENT_PLAN_20261007T180539Z.md','policies.py','test_policies.py','run_ablation.py','INPUTS.json','REPRODUCTION_TARGETS.json','RUN_LOCK.json','DECISIONS.json','PREDICTIONS_SEALED.json','evaluation/RESULTS.json','evaluation/POINT_METRICS.csv','evaluation/TRANSITIONS.csv','evaluation/ATTRIBUTION.json','evaluation/ATTRIBUTION.csv','evaluation/SEALED.json','EXPERIMENT_TRACKER.md','REPORT_20261007T184027Z.md','COMMANDS.md']
    for name in required: raw(ROOT/name)
    raw(PREV/'theory/kernel.py');raw(SURFACE/'surface_kernel.py')
    proto=js(ROOT/'PROTOCOL.json'); inputs=js(ROOT/'INPUTS.json')['rows']; decisions=js(ROOT/'DECISIONS.json')['rows']
    old=js(PREV/'DECISIONS.json')['rows']; tracks=js(PREV/'observation/TRACKS.json')['rows']; evidence=js(OFFICIAL/'EVIDENCE.json')['rows']
    targets=js(ROOT/'REPRODUCTION_TARGETS.json')['rows']; results=js(ROOT/'evaluation/RESULTS.json')
    maps=[{key(r):r for r in rows} for rows in (inputs,decisions,old,tracks,evidence,targets)]
    for rows,idx in zip((inputs,decisions,old,tracks,evidence,targets),maps): equal(len(rows),len(idx),'unique_requests');equal(len(rows),21,'active_population')
    for idx in maps[1:]:equal(set(idx),set(maps[0]),'request_identity')
    arms=['photo_U11']+proto['new_arms']+proto['old_arms']; equal(len(set(arms)),17,'arms')
    inferred={}; coordinates=Counter()
    for row in inputs:
        k=key(row); d,o,t,e,rt=[m[k] for m in maps[1:]]
        equal(set(row),set('scene roi query primary current_id center ray intervals objects original_id tracks'.split()),'input_row_allowlist')
        for name in ('scene','roi','query','primary','current_id','center','ray','intervals'): equal(row[name],o[name],'inherited_'+name)
        for ev in ('star_full','star','cycle'): equal(row['intervals'][ev],t[ev+'_intervals'],'track_interval_ancestry')
        equal(set(row['tracks']),{'star','cycle'},'track_evidence_allowlist')
        for ev,ts in row['tracks'].items():
            expected=[dict(id=x['id'],reference_depth_mm=x['triangulation']['reference_depth_mm'],ncc_reference_sources=x['ncc_reference_sources'],intervals_mm=x['intervals_mm']) for x in t[ev+'_tracks']]
            equal(ts,expected,'track_allowlist_and_ancestry')
        epool={x['candidate_id']:x for x in e['candidates']}
        if e['incumbent'] is not None:epool[-1]=e['incumbent']
        equal({x['candidate_id'] for x in row['objects']},set(epool),'object_pool')
        for obj in row['objects']:
            equal(set(obj),{'candidate_id','xyz_mm','depth_mm','old_score'},'object_allowlist')
            ancestor=epool[obj['candidate_id']]
            equal(obj['xyz_mm'],ancestor['xyz_mm'],'coordinate_evidence_ancestry')
            equal(obj['depth_mm'],ancestor['depth_mm'],'depth_evidence_ancestry')
            equal(obj['old_score'],ancestor['arms']['U11_bestpair']['score'],'old_score_ancestry')
        original=-1 if -1 in epool else None
        photo=e['decisions']['U11_bestpair']['selected_candidate_id']
        photo=original if photo is None else photo
        equal(row['current_id'],photo,'photo_ancestry');equal(rt['photo_id'],photo,'reproduction_target_photo')
        equal(set(d['decisions']),set(arms[1:]),'decision_arm_completeness')
        for arm in arms[1:]:
            saved=d['decisions'][arm]
            if arm.startswith('old_'): choice=old_choice(row,arm,saved)
            else:
                ev,policy=arm.rsplit('_',1)
                choice=new_choice(row,ev,policy,saved)
            equal(choice,saved['selected_candidate_id'],'independent_selection')
            inferred[(arm,*k)]=choice
        equal(inferred[('old_replay',*k)],photo,'old_reproduction')
        for ev in ('star_full','star','cycle'):
            equal(inferred[(ev+'_R',*k)],o['decisions'][ev]['selected_candidate_id'],'R_reproduction')
            equal(rt['R'][ev],o['decisions'][ev]['selected_candidate_id'],'reproduction_target_R')
    originals={key(r):r for r in table(SURFACE/'evaluation/POINT_METRICS.csv') if r['arm']=='incumbent'}
    candrows=table(SURFACE/'evaluation/CANDIDATE_DISTANCES.csv')
    cands={(*key(r),int(r['candidate_id'])):r for r in candrows}
    equal(len(cands),len(candrows),'candidate_uniqueness');equal(len(cands),105,'proposal_population')
    for row in inputs:
        for obj in row['objects']:
            cid=obj['candidate_id']; origin=originals[key(row)] if cid==-1 else cands[(*key(row),cid)]
            equal(b''.join(struct.pack('>d',x) for x in obj['xyz_mm']),b''.join(struct.pack('>d',float(origin[n])) for n in ('x_mm','y_mm','z_mm')),'coordinate_binary_identity')
            coordinates['incumbents' if cid==-1 else 'proposals']+=1
    equal(dict(coordinates),dict(incumbents=18,proposals=105),'coordinate_counts')
    oldpoints=table(PREV/'evaluation/POINT_METRICS.csv')
    photo={key(r):convert(r) for r in oldpoints if r['arm']=='photo_U11'}
    equal(len(photo),512,'photo_population')
    equal({k for k in photo},set(originals),'full_ancestral_population')
    for k,p in photo.items():
        cid=maps[0][k]['current_id'] if k in maps[0] else -1
        origin=None if cid is None else originals[k] if cid==-1 else cands[(*k,cid)]
        distance=None if origin is None or not origin['distance_mm'] else float(origin['distance_mm'])
        equal(p['distance_mm'],distance,'full_photo_distance_ancestry')
        equal(p['primary'],originals[k]['primary']=='True','full_primary_ancestry')
    equal(results['support_counts'],{ev:sum(bool(r['intervals'][ev]) for r in inputs) for ev in ('star_full','star','cycle')},'support_counts')
    for arm in arms[1:]:
        expected=dict(moved=sum(inferred[(arm,*key(r))]!=r['current_id'] for r in inputs),
          eligible_alternatives=sum(len(d['decisions'][arm]['eligible_ids']) for d in decisions),
          reasons=dict(Counter(d['decisions'][arm]['reason'] for d in decisions)))
        equal(results['decision_counts'][arm],expected,'decision_counts')
    full=[]; transitions=[]
    for arm in arms:
        for k,p in photo.items():
            q=dict(p,arm=arm)
            if arm!='photo_U11' and k in maps[0]:
                cid=inferred[(arm,*k)];q['selected']=cid;q['reason']=maps[1][k]['decisions'][arm]['reason']
                origin=None if cid is None else originals[k] if cid==-1 else cands[(*k,cid)]
                q['distance_mm']=None if origin is None else float(origin['distance_mm'])
                if cid!=maps[0][k]['current_id']:
                    transitions.append(dict(arm=arm,roi=k[0],query=k[1],primary=q['primary'],photo_mm=p['distance_mm'],output_mm=q['distance_mm'],photo_id=maps[0][k]['current_id'],selected=cid,reason=q['reason']))
            full.append(q)
    actual=[convert(r) for r in table(ROOT/'evaluation/POINT_METRICS.csv')]
    equal(unique(full,('arm','roi','query')),unique(actual,('arm','roi','query')),'whole_point_table')
    trx=table(ROOT/'evaluation/TRANSITIONS.csv')
    for t in trx:
        for name in ('query','photo_id','selected'):t[name]=int(t[name]) if t[name] else None
        for name in ('photo_mm','output_mm'):t[name]=float(t[name]) if t[name] else None
        t['primary']=t['primary']=='True'
    equal(trx,transitions,'whole_transition_table')
    metrics=[]
    for arm in arms:
        ps=[p for p in full if p['arm']==arm]; primary=[p for p in ps if p['primary']]
        equal(len(primary),484,'primary_denominator')
        equal(sum(p['distance_mm'] is not None for p in ps),509,'finite_population')
        rois={}
        for roi in sorted({p['roi'] for p in primary}):
            ds=[p['distance_mm'] for p in primary if p['roi']==roi]
            rois[roi]=dict(n=len(ds),mse_mm2=math.fsum(x*x for x in ds)/len(ds),mae_mm=math.fsum(ds)/len(ds))
        deltas=[p['distance_mm']-photo[key(p)]['distance_mm'] for p in primary]
        m=dict(arm=arm,primary=484,mse_mm2=math.fsum(v['mse_mm2'] for v in rois.values())/4,
          mae_mm=math.fsum(v['mae_mm'] for v in rois.values())/4,roi=rois,
          scene_mse={str(s):math.fsum(v['mse_mm2'] for r,v in rois.items() if r.startswith('scan'+str(s)+'_'))/2 for s in (118,122)},
          severe_gt5=sum(p['distance_mm']>5 for p in primary), improved_vs_photo=sum(d<-1e-9 for d in deltas),
          worsened_vs_photo=sum(d>1e-9 for d in deltas),unchanged_vs_photo=sum(abs(d)<=1e-9 for d in deltas),
          new_harm_vs_photo=sum(photo[key(p)]['distance_mm']<=1<p['distance_mm'] for p in primary),
          finite=sum(p['distance_mm'] is not None for p in ps),missing_finite=sum(not p['primary'] and p['distance_mm'] is not None for p in ps),
          changed_active_ids=sum(t['arm']==arm for t in transitions))
        baseline=m['mse_mm2'] if not metrics else metrics[0]['mse_mm2']
        m['mse_reduction_pct_vs_photo']=100*(baseline-m['mse_mm2'])/baseline
        compare(m,next(x for x in results['metrics'] if x['arm']==arm),'metric_'+arm)
        metrics.append(m)
    by={m['arm']:m['mse_mm2'] for m in metrics}; contrasts=[]
    for ev in ('star_full','star','cycle'):
        for point,gate in ([('P','GP')] if ev=='star_full' else [('P','GP'),('Q','QG')]):
            p,g,r=(by[ev+'_'+x] for x in (point,gate,'R'))
            terms=[by['photo_U11']-p,p-g,g-r]
            near(math.fsum(terms),by['photo_U11']-r,'telescoping_identity')
            contrasts.append(dict(evidence=ev,point_policy=point,simple_gain_mm2=terms[0],gate_increment_mm2=terms[1],ranking_increment_mm2=terms[2],robust_extra_vs_simple_mm2=p-r,total_robust_gain_mm2=by['photo_U11']-r))
    equal(contrasts,js(ROOT/'evaluation/ATTRIBUTION.json')['rows'],'attribution_json')
    csvcontrasts=table(ROOT/'evaluation/ATTRIBUTION.csv')
    for c in csvcontrasts:
        for k in c:
            if k not in ('evidence','point_policy'):c[k]=float(c[k])
    equal(contrasts,csvcontrasts,'attribution_csv')
    for roi in sorted({k[0] for k in photo}):
        path=TRANSFER/'cpu'/(roi+'.npz'); blob=raw(path)
        with np.load(io.BytesIO(blob),allow_pickle=False) as z:
            equal(len(z['valid']),128,'original_cpu_roi_count')
            for query,valid in enumerate(z['valid']):equal(photo[(roi,query)]['primary'],bool(valid),'original_cpu_primary_flag')
    dataset=[]
    manifest=js(TRANSFER/'references/MANIFEST.json')
    for ref in manifest['files']:
        scene=ref['scene']; gt=read_vertices(ref['path']);equal(HASHES[ref['path']],ref['sha256'],'dataset_sha256')
        checkrows=[r for r in originals.values() if int(r['scene'])==scene and r['distance_mm']]+[r for r in candrows if int(r['scene'])==scene]
        coords=np.array([[float(r[n]) for n in ('x_mm','y_mm','z_mm')] for r in checkrows]); saved=np.array([float(r['distance_mm']) for r in checkrows])
        distances,_=cKDTree(gt).query(coords,k=1,workers=1)
        err=float(np.max(np.abs(distances-saved)));assert err<1e-10
        # A second, exhaustive algorithm verifies every active candidate and incumbent.
        active=[r for r in checkrows if key(r) in maps[0]]
        ac=np.array([[float(r[n]) for n in ('x_mm','y_mm','z_mm')] for r in active])
        brute=np.full(len(ac),np.inf)
        for start in range(0,len(gt),25000):brute=np.minimum(brute,cdist(ac,gt[start:start+25000]).min(axis=1))
        brute_err=float(np.max(np.abs(brute-np.array([float(r['distance_mm']) for r in active]))));assert brute_err<1e-10
        dataset.append(dict(scene=scene,reference_vertices=len(gt),checked_coordinate_rows=len(coords),max_cached_distance_error_mm=err,brute_force_active_coordinate_rows=len(active),max_brute_force_error_mm=brute_err))
    run=subprocess.run([PYTHON,'-B','-m','unittest','-v','test_policies.py'],cwd=ROOT,text=True,capture_output=True,check=False,env=os.environ.copy())
    assert run.returncode==0 and 'Ran 28 tests' in run.stderr and '\nOK\n' in run.stderr
    with (ROOT/'audit/SYNTHETIC_TESTS.txt').open('x') as f:f.write(run.stdout+run.stderr)
    output=dict(generated_at=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),python=sys.executable,
       selector_implementation='independent exact endpoint squared-distance arithmetic; no author selector/kernel/evaluator imports',
       checks=dict(CHECKS),independently_recomputed_choices=len(inferred),point_rows=len(full),transition_rows=len(transitions),
       metrics=metrics,attribution=contrasts,dataset_verification=dataset,synthetic_tests=dict(count=28,returncode=run.returncode),
       input_sha256=HASHES)
    with (ROOT/'audit/RECOMPUTED.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    for path,digest in HASHES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    print(json.dumps(dict(choices=len(inferred),point_rows=len(full),transitions=len(transitions),datasets=dataset,tests=28,all_checks_pass=True),indent=2))

if __name__=='__main__':main()

"""Independent read-only audit computations; writes only audit/RECOMPUTE.json."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import csv, json, hashlib, math, socket, ast
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parents[1]
assert socket.gethostname() == 'liekkas'
assert str(R) == '/srv/slam-research/grf/map-denoise/runs/track-discrimination-20261007T160716Z'
O = R.parent/'plane-support-20261007T084206Z'
A = R.parent/'surface-evidence-20261001T140058Z'
T = R.parent/'colmap-transfer-20260930T180000Z'
hashes = {}
def sha(p):
    p = Path(p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes[str(p)] = h
    return h
def load(p):
    sha(p)
    return json.loads(Path(p).read_text())
def csvrows(p):
    sha(p)
    with Path(p).open() as f:
        return list(csv.DictReader(f))
def key(p): return (p['roi'],int(p['query']))
def number(s): return float(s) if s != '' else None

seal_results = []
def checkseal(p, field='files', base=None):
    seal = load(p)
    failures = []
    for filename, expected in seal[field].items():
        target = (base/filename) if base is not None else Path(filename)
        if sha(target) != expected: failures.append(str(target))
    seal_results.append(dict(path=str(p),field=field,count=len(seal[field]),mismatches=failures))
    return seal
checkseal(R/'PREPARATION_SEAL.json')
checkseal(R/'PREPARATION_SEAL.json','source_files')
for p in [R/'RUN_LOCK.json',R/'PREDICTIONS_SEALED.json',R/'evaluation/SEALED.json',O/'evaluation/SEALED.json']:
    assert not checkseal(p)['files'] == {}
obsseal = checkseal(R/'observation/SEAL.json','sha256',R/'observation')
checkseal(R/'observation/SEAL.json','input_photo_sha256')
checkseal(R/'observation/LOCK.json','sha256',R)
checkseal(A/'evaluation/EVALUATION_COMPLETE.json','artifacts',A/'evaluation')
checkseal(O/'evaluation/RESULTS.json','evaluation_inputs')
assert sha(R/'REQUESTS.json') == obsseal['request_sha256']
assert sha(R/'PROTOCOL.json') == obsseal['protocol_sha256']
res = load(R/'evaluation/RESULTS.json')
checkseal(R/'evaluation/RESULTS.json','inputs')
points = csvrows(R/'evaluation/POINT_METRICS.csv')
assert len(points) == 7*512
photo = {key(p):p for p in points if p['arm']=='photo_U11'}
metrics = []
for expected in res['metrics']:
    arm = expected['arm']
    allp = [p for p in points if p['arm']==arm]
    assert len(allp)==len({key(p) for p in allp})==512
    pp = [p for p in allp if p['primary']=='True']
    assert len(pp)==484 and {key(p) for p in pp}=={key(p) for p in photo.values() if p['primary']=='True'}
    counts = Counter(p['roi'] for p in pp)
    roi_mse={r:sum(float(p['distance_mm'])**2 for p in pp if p['roi']==r)/n for r,n in counts.items()}
    mae=sum(sum(float(p['distance_mm']) for p in pp if p['roi']==r)/n for r,n in counts.items())/4
    d=[float(p['distance_mm']) for p in pp]
    b=[float(photo[key(p)]['distance_mm']) for p in pp]
    orig=[float(p['incumbent_distance_mm']) for p in pp]
    actual=dict(arm=arm,primary=484,mse_mm2=sum(roi_mse.values())/4,mae_mm=mae,roi_mse=roi_mse,
      scene_mse={s:sum(v for r,v in roi_mse.items() if r.startswith('scan'+s+'_'))/2 for s in ['118','122']},
      severe_gt5=sum(x>5 for x in d),improved_vs_photo=sum(x<y-1e-9 for x,y in zip(d,b)),
      worsened_vs_photo=sum(x>y+1e-9 for x,y in zip(d,b)),unchanged_vs_photo=sum(abs(x-y)<=1e-9 for x,y in zip(d,b)),
      good_le1=sum(x<=1 for x in d),new_harm_vs_photo=sum(y<=1 and x>1 for x,y in zip(d,b)),
      new_harm_vs_original=sum(y<=1 and x>1 for x,y in zip(d,orig)),
      finite=sum(p['distance_mm']!='' for p in allp),missing_finite=sum(p['primary']=='False' and p['distance_mm']!='' for p in allp))
    def same(x,y):
        if isinstance(x,dict): return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,float): return math.isclose(x,y,rel_tol=1e-13,abs_tol=1e-13)
        return x==y
    assert same(actual,expected),(arm,actual,expected)
    actual.update(roi_denominators=dict(counts),point_equal_mse_mm2=sum(x*x for x in d)/484,
      reduction_vs_photo_percent=100*(res['metrics'][1]['mse_mm2']-actual['mse_mm2'])/res['metrics'][1]['mse_mm2'])
    metrics.append(actual)

decisions=load(R/'DECISIONS.json')['rows']
tracks={key(p):p for p in load(R/'observation/TRACKS.json')['rows']}
requests=load(R/'REQUESTS.json')
reqmap={key(p):p for p in requests['rows']}
olddec={key(p):p for p in load(O/'DECISIONS.json')['rows']}
original={key(p):p for p in csvrows(A/'evaluation/POINT_METRICS.csv') if p['arm']=='incumbent'}
candidates={(*key(p),int(p['candidate_id'])):p for p in csvrows(A/'evaluation/CANDIDATE_DISTANCES.csv')}
bound_checks=0
coordinate_checks=Counter()
for row in decisions:
    k=key(row); old=olddec[k]; t=tracks[k]
    assert old['pixel_xy']==reqmap[k]['pixel_xy']==t['pixel_xy']
    scene=requests['scenes'][str(row['scene'])]; cam=scene['cameras'][scene['reference']]
    ray=np.array(cam['R']).T@np.linalg.solve(np.array(cam['K_half']),[*reqmap[k]['pixel_xy'],1.])
    np.testing.assert_allclose(ray,row['ray'],rtol=0,atol=0)
    assert row['center']==cam['center']
    objs={o['candidate_id']:o for o in row['objects']}
    assert objs=={o['candidate_id']:{'candidate_id':o['candidate_id'],'xyz_mm':o['xyz_mm']} for o in old['objects']}
    for cid,o in objs.items():
        ref=original[k] if cid==-1 else candidates[(*k,cid)]
        assert o['xyz_mm']==[float(ref[x]) for x in ['x_mm','y_mm','z_mm']]
        coordinate_checks['incumbent' if cid==-1 else 'candidate']+=1
    current=old['baseline_id'] if old['baseline_id'] is not None else (-1 if -1 in objs else None)
    assert current==row['current_id']
    assert number(photo[k]['distance_mm'])==(None if current is None else float((original[k] if current==-1 else candidates[(*k,current)])['distance_mm']))
    for arm,d in row['decisions'].items():
        intervals=row['intervals'][arm]
        assert intervals==t[arm+'_intervals']
        if current is None:
            assert d==dict(selected_candidate_id=None,reason='NO_CURRENT_OUTPUT',scores=[])
            continue
        if not intervals:
            assert d==dict(selected_candidate_id=current,reason='EMPTY_UNKNOWN',scores=[])
            continue
        a=list(map(F,objs[current]['xyz_mm'])); C=list(map(F,row['center'])); r=list(map(F,row['ray']))
        scoremap={s['candidate_id']:s for s in d['scores']}
        assert scoremap.keys()==objs.keys()
        lows={}
        for cid,o in objs.items():
            b=list(map(F,o['xyz_mm']))
            gains=[sum((ai-ci-F(z)*ri)**2-(bi-ci-F(z)*ri)**2 for ai,bi,ci,ri in zip(a,b,C,r)) for interval in intervals for z in interval]
            lo,hi=min(gains),max(gains); stored=scoremap[cid]
            assert F(stored['gain_low_mm2'])<=lo<=F(math.nextafter(stored['gain_low_mm2'],math.inf))
            assert F(math.nextafter(stored['gain_high_mm2'],-math.inf))<=hi<=F(stored['gain_high_mm2'])
            lows[cid]=stored['gain_low_mm2']; bound_checks+=1
        eligible=[cid for cid in objs if cid!=current and lows[cid]>0]
        winner=min(eligible,key=lambda cid:(-lows[cid],cid)) if eligible else current
        assert winner==d['selected_candidate_id']
        assert d['reason']==('POSITIVE_GAIN_ON_ASSUMED_SET' if eligible else 'NO_UNIFORM_MODEL_GAIN')

transitions=[]
for p in points:
    if p['arm'] not in ['star_full','star','cycle']: continue
    k=key(p); b=photo[k]
    expected=number(b['distance_mm'])
    if k in olddec:
        row=next(r for r in decisions if key(r)==k)
        idx=row['decisions'][p['arm']]['selected_candidate_id']
        if idx is not None: expected=float((original[k] if idx==-1 else candidates[(*k,idx)])['distance_mm'])
    assert number(p['distance_mm'])==expected
    if p['distance_mm']!=b['distance_mm']:
        transitions.append(dict(arm=p['arm'],roi=p['roi'],query=int(p['query']),primary=p['primary']=='True',
          photo_mm=number(b['distance_mm']),output_mm=number(p['distance_mm']),selected=int(p['selected']),reason=p['reason']))
assert transitions==res['transitions']
csvtrans=csvrows(R/'evaluation/TRANSITIONS.csv')
assert len(csvtrans)==len(transitions)==12

# Recompute all frozen candidate and finite historical incumbent distances from original laser files.
reader_path=T/'evaluate_transfer.py'; sha(reader_path)
node=next(n for n in ast.parse(reader_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='read_ply_xyz')
ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(reader_path),'exec'),ns)
gt_results=[]
for ref in load(T/'references/MANIFEST.json')['files']:
    assert sha(ref['path'])==ref['sha256']
    gt=ns['read_ply_xyz'](ref['path']);tree=cKDTree(gt)
    records=[p for p in [*original.values(),*candidates.values()] if int(p['scene'])==ref['scene'] and p['distance_mm']!='']
    coords=np.array([[float(p[k]) for k in ['x_mm','y_mm','z_mm']] for p in records])
    d=tree.query(coords,workers=1)[0]
    errors=np.abs(d-np.array([float(p['distance_mm']) for p in records]))
    assert max(errors)<1e-12
    gt_results.append(dict(scene=ref['scene'],vertices=len(gt),coordinates=len(records),max_error_mm=float(max(errors))))

for p in R.rglob('*'):
    if p.is_file() and 'audit' not in p.relative_to(R).parts and '__pycache__' not in p.parts:
        sha(p)
report=dict(host=socket.gethostname(),root=str(R),seal_checks=seal_results,metrics=metrics,
  candidate_coordinate_checks=dict(coordinate_checks),gain_bounds_checked=bound_checks,decisions_checked=63,
  transitions_checked=12,gt_recomputation=gt_results,audited_input_hashes=hashes)
out=R/'audit/RECOMPUTE.json'
with out.open('x') as f: json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='audited_input_hashes'},indent=2))

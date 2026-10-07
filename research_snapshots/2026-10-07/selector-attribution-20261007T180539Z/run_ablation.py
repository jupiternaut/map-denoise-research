"""Prepare GT-free frozen inputs, seal fixed selectors, then evaluate cached GT."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import math
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from policies import new_policy, old_policy, KERNEL_PATH

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'track-discrimination-20261007T160716Z'
EVIDENCE = ROOT.parent / 'official-mechanism-20261001T183124Z/EVIDENCE.json'
ANCESTOR = ROOT.parent / 'surface-evidence-20261001T140058Z/evaluation'


def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(row): return row['roi'], int(row['query'])


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def csvread(path):
    with Path(path).open(newline='') as stream: return list(csv.DictReader(stream))


def csvwrite(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(path):
    for name, digest in load(path)['files'].items():
        assert sha(name) == digest, name


def forbid_evaluation():
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]).lower()
            if any(tag in name for tag in ('/evaluation/', 'candidate_distances', 'point_metrics', '.ply')):
                raise PermissionError('Prediction cannot read evaluation: ' + name)
    sys.addaudithook(guard)


def prepare():
    forbid_evaluation()
    verify(PREV / 'RUN_LOCK.json')
    verify(PREV / 'PREDICTIONS_SEALED.json')
    archived = load(PREV / 'DECISIONS.json')['rows']
    evidence = {key(row): row for row in load(EVIDENCE)['rows']}
    tracks = {key(row): row for row in load(PREV / 'observation/TRACKS.json')['rows']}
    rows, targets = [], []
    for old in archived:
        ev, tr = evidence[key(old)], tracks[key(old)]
        objects = {item['candidate_id']: item for item in ev['candidates']}
        if ev['incumbent'] is not None:
            objects[-1] = ev['incumbent']
        assert set(objects) == {item['candidate_id'] for item in old['objects']}
        row = {field: old[field] for field in ('scene', 'roi', 'query', 'primary', 'current_id', 'center', 'ray', 'intervals')}
        row['objects'] = []
        for item in old['objects']:
            obj = objects[item['candidate_id']]
            assert item['xyz_mm'] == obj['xyz_mm']
            row['objects'].append(dict(candidate_id=item['candidate_id'], xyz_mm=item['xyz_mm'],
                                       depth_mm=obj['depth_mm'], old_score=obj['arms']['U11_bestpair']['score']))
        row['original_id'] = -1 if -1 in objects else None
        saved_id = ev['decisions']['U11_bestpair']['selected_candidate_id']
        saved_photo = saved_id if saved_id is not None else row['original_id']
        assert saved_photo == old['current_id']
        row['tracks'] = {
            arm: [dict(id=t['id'], reference_depth_mm=t['triangulation']['reference_depth_mm'],
                       ncc_reference_sources=t['ncc_reference_sources'], intervals_mm=t['intervals_mm'])
                  for t in tr[arm + '_tracks']]
            for arm in ('star', 'cycle')}
        for arm in ('star_full', 'star', 'cycle'):
            assert row['intervals'][arm] == tr[arm + '_intervals']
        rows.append(row)
        targets.append(dict(roi=old['roi'], query=old['query'], photo_id=saved_photo,
                            R={arm: old['decisions'][arm]['selected_candidate_id'] for arm in ('star_full', 'star', 'cycle')}))
    assert len(rows) == 21
    assert sum(obj['candidate_id'] != -1 for row in rows for obj in row['objects']) == 105
    inputs = [PREV / 'DECISIONS.json', PREV / 'observation/TRACKS.json', EVIDENCE,
              PREV / 'RUN_LOCK.json', PREV / 'PREDICTIONS_SEALED.json']
    dump(ROOT / 'INPUTS.json', dict(created_at=now(), schema='explicit allowlist; no evaluation fields',
                                    sources={str(path): sha(path) for path in inputs}, rows=rows))
    dump(ROOT / 'REPRODUCTION_TARGETS.json', dict(rows=targets))
    print('Prepared', len(rows), 'GT-free requests')


def freeze():
    forbid_evaluation()
    paths = [ROOT / name for name in ('AGENTS.md', 'PROTOCOL.json', 'policies.py', 'test_policies.py',
             'run_ablation.py', 'INPUTS.json', 'REPRODUCTION_TARGETS.json',
             'refine-logs/EXPERIMENT_PLAN_20261007T180539Z.md')]
    paths.append(KERNEL_PATH)
    for path, digest in load(ROOT / 'INPUTS.json')['sources'].items():
        assert sha(path) == digest
        paths.append(Path(path))
    dump(ROOT / 'RUN_LOCK.json', dict(created_at=now(), host=socket.gethostname(),
                                     files={str(path): sha(path) for path in paths}))
    print('Locked', len(paths), 'files')


def infer():
    start = time.monotonic()
    forbid_evaluation()
    verify(ROOT / 'RUN_LOCK.json')
    protocol = load(ROOT / 'PROTOCOL.json')
    rows = load(ROOT / 'INPUTS.json')['rows']
    expected = {key(row): row for row in load(ROOT / 'REPRODUCTION_TARGETS.json')['rows']}
    output = []
    for row in rows:
        decisions = {}
        for arm in protocol['new_arms']:
            evidence, policy = arm.rsplit('_', 1)
            decisions[arm] = new_policy(row, evidence, policy)
        decisions['old_replay'] = old_policy(row)
        decisions['old_no_ambiguity'] = old_policy(row, remove_ambiguity=True)
        decisions['old_raw'] = old_policy(row, remove_ambiguity=True, remove_margin=True)
        assert decisions['old_replay']['selected_candidate_id'] == expected[key(row)]['photo_id']
        for evidence, idx in expected[key(row)]['R'].items():
            assert decisions[evidence + '_R']['selected_candidate_id'] == idx
        output.append(dict(scene=row['scene'], roi=row['roi'], query=row['query'], primary=row['primary'],
                           current_id=row['current_id'], decisions=decisions))
    arms = protocol['new_arms'] + protocol['old_arms']
    assert len(arms) == 16
    dump(ROOT / 'DECISIONS.json', dict(created_at=now(), rows=output, evaluation_open_blocked=True,
        reproduction=dict(old_photo=21, star_full_R=21, star_R=21, cycle_R=21),
        wallclock_seconds=time.monotonic() - start))
    files = [ROOT / 'DECISIONS.json', ROOT / 'RUN_LOCK.json']
    dump(ROOT / 'PREDICTIONS_SEALED.json', dict(created_at=now(), files={str(path): sha(path) for path in files}))
    print(json.dumps({arm: dict(Counter(row['decisions'][arm]['reason'] for row in output)) for arm in arms}, indent=2))


def mean(values): return math.fsum(values) / len(values)


def metrics(points, photo):
    primary = [p for p in points if p['primary']]
    assert len(primary) == 484 and all(p['distance_mm'] is not None for p in primary)
    rois = sorted({p['roi'] for p in primary})
    per_roi = {}
    for roi in rois:
        ds = [p['distance_mm'] for p in primary if p['roi'] == roi]
        per_roi[roi] = dict(n=len(ds), mse_mm2=mean([d*d for d in ds]), mae_mm=mean(ds))
    deltas = [p['distance_mm'] - photo[key(p)]['distance_mm'] for p in primary]
    return dict(arm=points[0]['arm'], primary=484, mse_mm2=mean([v['mse_mm2'] for v in per_roi.values()]),
        mae_mm=mean([v['mae_mm'] for v in per_roi.values()]), roi=per_roi,
        scene_mse={str(s): mean([v['mse_mm2'] for r, v in per_roi.items() if r.startswith('scan'+str(s)+'_')]) for s in (118,122)},
        severe_gt5=sum(p['distance_mm'] > 5 for p in primary),
        improved_vs_photo=sum(d < -1e-9 for d in deltas), worsened_vs_photo=sum(d > 1e-9 for d in deltas),
        unchanged_vs_photo=sum(abs(d) <= 1e-9 for d in deltas),
        new_harm_vs_photo=sum(photo[key(p)]['distance_mm'] <= 1 and p['distance_mm'] > 1 for p in primary),
        finite=sum(p['distance_mm'] is not None for p in points),
        missing_finite=sum(not p['primary'] and p['distance_mm'] is not None for p in points))


def evaluate():
    verify(ROOT / 'RUN_LOCK.json')
    verify(ROOT / 'PREDICTIONS_SEALED.json')
    verify(PREV / 'evaluation/SEALED.json')
    archived = csvread(PREV / 'evaluation/POINT_METRICS.csv')
    original = {key(r): r for r in csvread(ANCESTOR / 'POINT_METRICS.csv') if r['arm'] == 'incumbent'}
    candidates = {(*key(r), int(r['candidate_id'])): r for r in csvread(ANCESTOR / 'CANDIDATE_DISTANCES.csv')}
    rows = load(ROOT / 'INPUTS.json')['rows']
    decisions = {key(r): r for r in load(ROOT / 'DECISIONS.json')['rows']}
    checked = Counter()
    for row in rows:
        for obj in row['objects']:
            cid = obj['candidate_id']
            target = original[key(row)] if cid == -1 else candidates[(*key(row), cid)]
            assert obj['xyz_mm'] == [float(target[k]) for k in ('x_mm','y_mm','z_mm')]
            checked['incumbents' if cid == -1 else 'proposals'] += 1
    assert checked == dict(incumbents=18, proposals=105)
    photo = {key(p): dict(arm='photo_U11', roi=p['roi'], query=int(p['query']), primary=p['primary']=='True',
             selected=int(p['selected']) if p['selected'] else None, reason=p['reason'],
             distance_mm=float(p['distance_mm']) if p['distance_mm'] else None)
             for p in archived if p['arm']=='photo_U11'}
    assert len(photo) == 512 and sum(p['primary'] for p in photo.values()) == 484
    protocol = load(ROOT / 'PROTOCOL.json')
    arms = ['photo_U11'] + protocol['new_arms'] + protocol['old_arms']
    allrows, transitions = list(photo.values()), []
    for arm in arms[1:]:
        for k, p in photo.items():
            q = dict(p, arm=arm)
            if k in decisions:
                decision = decisions[k]['decisions'][arm]
                cid = decision['selected_candidate_id']
                q.update(selected=cid, reason=decision['reason'])
                if cid is None:
                    q['distance_mm'] = None
                else:
                    target = original[k] if cid == -1 else candidates[(*k,cid)]
                    q['distance_mm'] = float(target['distance_mm']) if target['distance_mm'] else None
                if cid != decisions[k]['current_id']:
                    transitions.append(dict(arm=arm, roi=k[0], query=k[1], primary=p['primary'],
                        photo_mm=p['distance_mm'], output_mm=q['distance_mm'],
                        photo_id=decisions[k]['current_id'], selected=cid, reason=q['reason']))
            allrows.append(q)
    result = [metrics([p for p in allrows if p['arm']==arm], photo) for arm in arms]
    by_arm = {m['arm']:m for m in result}
    baseline = by_arm['photo_U11']['mse_mm2']
    for m in result:
        m['mse_reduction_pct_vs_photo'] = 100*(baseline-m['mse_mm2'])/baseline
        m['changed_active_ids'] = sum(t['arm']==m['arm'] for t in transitions)
    attribution = []
    for evidence in ('star_full','star','cycle'):
        for point, gate in [('P','GP')] + ([('Q','QG')] if evidence != 'star_full' else []):
            p, gp, robust = [by_arm[evidence+'_'+policy]['mse_mm2'] for policy in (point,gate,'R')]
            terms = [baseline-p, p-gp, gp-robust]
            assert math.isclose(math.fsum(terms), baseline-robust, rel_tol=1e-12, abs_tol=1e-12)
            attribution.append(dict(evidence=evidence, point_policy=point, simple_gain_mm2=terms[0],
                gate_increment_mm2=terms[1], ranking_increment_mm2=terms[2],
                robust_extra_vs_simple_mm2=p-robust, total_robust_gain_mm2=baseline-robust))
    source_paths = [PREV/'evaluation/POINT_METRICS.csv', PREV/'evaluation/SEALED.json',
                    ANCESTOR/'POINT_METRICS.csv', ANCESTOR/'CANDIDATE_DISTANCES.csv']
    availability = {e: sum(bool(r['intervals'][e]) for r in rows) for e in ('star_full','star','cycle')}
    counts = {arm: dict(moved=sum(d['decisions'][arm]['selected_candidate_id'] != d['current_id'] for d in decisions.values()),
                        eligible_alternatives=sum(len(d['decisions'][arm].get('eligible_ids',[])) for d in decisions.values()),
                        reasons=dict(Counter(d['decisions'][arm]['reason'] for d in decisions.values()))) for arm in arms[1:]}
    dump(ROOT/'evaluation/RESULTS.json', dict(created_at=now(), metrics=result, coordinate_checks=dict(checked),
        support_counts=availability, decision_counts=counts,
        scope='Exposed scans118/122, four ROI; 512 requests, 484 original-CPU-valid primary. No cross-scene confirmation.',
        evaluation_type='real_gt_reused_exact_coordinates',
        metric='nearest original DTU laser vertex distance, not official full-cloud benchmark or physical layer identity',
        inputs={str(p):sha(p) for p in source_paths}))
    dump(ROOT/'evaluation/ATTRIBUTION.json', dict(interpretation='conditional arithmetic contrasts, not causal component percentage shares', rows=attribution))
    csvwrite(ROOT/'evaluation/ATTRIBUTION.csv', attribution)
    csvwrite(ROOT/'evaluation/POINT_METRICS.csv', allrows)
    csvwrite(ROOT/'evaluation/TRANSITIONS.csv', transitions,
             ['arm','roi','query','primary','photo_mm','output_mm','photo_id','selected','reason'])
    files = list((ROOT/'evaluation').iterdir())
    dump(ROOT/'evaluation/SEALED.json', dict(created_at=now(), files={str(p):sha(p) for p in files if p.is_file()}))
    print(json.dumps(dict(metrics=result, attribution=attribution),indent=2))


if __name__ == '__main__':
    assert socket.gethostname() == 'liekkas'
    {'prepare':prepare, 'freeze':freeze, 'infer':infer, 'evaluate':evaluate}[sys.argv[1]]()

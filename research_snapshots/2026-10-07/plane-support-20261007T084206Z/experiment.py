"""One fixed CPU replay: freeze -> GT-free inference -> seal -> real evaluation."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import copy
import csv
import hashlib
import json
import math
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from selector import decisions, support
from theory.kernel import (optical_ray, pixel_ray_error, anchor_error_from_depth,
                           plane_depth_interval, certain_possible_support)

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/plane-support-20261007T084206Z')
OLD = ROOT.parent / 'official-mechanism-20261001T183124Z'
EVAL = ROOT.parent / 'surface-evidence-20261001T140058Z/evaluation'
PLAN = ROOT.parent / 'rescue-local-20260930T201603Z/PLAN.json'
PYTHON = '/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python'
INPUTS = [OLD / 'EVIDENCE.json', OLD / 'neighborhood/NEIGHBORS.jsonl', PLAN]


def now():
    return datetime.now(timezone.utc).isoformat()


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def dump(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(clean(v), f, indent=2, allow_nan=False)
        f.write('\n')


def verify(p):
    for filename, digest in load(p)['files'].items():
        assert sha(filename) == digest, filename


def check():
    assert socket.gethostname() == 'liekkas'
    assert Path(__file__).resolve().parent == ROOT
    assert sys.executable == PYTHON


def deny_reference():
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        s = os.fsdecode(args[0]).lower()
        if any(token in s for token in ('/evaluation/', '/gt/', 'stl118', 'stl122',
                 'point_metrics.csv', 'candidate_distances.csv', 'evaluation_truth')):
            raise PermissionError('Reference access prohibited during inference: ' + s)
    sys.addaudithook(audit)


def freeze():
    check()
    deny_reference()
    assert sha(INPUTS[0]) == load(OLD / 'EVIDENCE_SEALED.json')['files'][str(INPUTS[0])]
    assert sha(INPUTS[1]) == load(OLD / 'neighborhood/SEALED.json')['files'][str(INPUTS[1])]
    files = INPUTS + [ROOT / p for p in ('experiment.py', 'selector.py', 'theory/kernel.py',
          'PROTOCOL.json', 'AGENTS.md', 'refine-logs/EXPERIMENT_PLAN_20261007T084206Z.md')]
    dump(ROOT / 'RUN_LOCK.json', dict(created_at=now(), files={str(p): sha(p) for p in files}))
    print('FROZEN', len(files))


def sanitize(evidence):
    rows = []
    for r in evidence['rows']:
        objects = []
        for obj in r['candidates'] + ([] if r['incumbent'] is None else [r['incumbent']]):
            objects.append(dict(candidate_id=obj['candidate_id'], xyz_mm=obj['xyz_mm'],
                hypotheses=[dict(scores={'U11': {'ncc': h['scores']['U11']['ncc']}})
                            for h in obj['hypotheses']]))
        rows.append(dict(scene=r['scene'], roi=r['roi'], query=r['query'], pixel_xy=r['pixel_xy'],
            primary=r['primary'], objects=objects,
            baseline_id=r['decisions']['U11_bestpair']['selected_candidate_id']))
    return rows


def read_neighbors():
    groups = {}
    counts = Counter()
    with INPUTS[1].open() as f:
        for line in f:
            row = json.loads(line)
            if row['map'] != 'wide7photo':
                continue
            key = (row['roi'], row['query'])
            counts[row['status']] += 1
            groups.setdefault(key, [])
            if row['status'] != 'ok':
                continue
            groups[key].append({k: row[k] for k in ('neighbor_xy', 'depth_mm', 'normal_camera_unit',
                                                   'predicted_target_optical_z_mm')})
    return groups, counts


def make_intervals(row, neighbors, plan, protocol):
    scene = plan['scenes'][str(row['scene'])]
    cam = scene['cameras'][scene['reference']]
    K, R, C = np.array(cam['K_half']), np.array(cam['R']), np.array(cam['center'])
    ray = optical_ray(K, row['pixel_xy'])
    er = pixel_ray_error(K, protocol['pixel_error_euclidean_px'])
    for o in row['objects']:
        o['optical_depth_mm'] = float(R[2] @ (np.array(o['xyz_mm']) - C))
        assert o['optical_depth_mm'] > 0
    records = []
    for n in neighbors:
        rayj = optical_ray(K, n['neighbor_xy'])
        anchor = n['depth_mm'] * rayj
        ez = anchor_error_from_depth(n['depth_mm'], rayj, protocol['neighbor_depth_error_mm'], er)
        bound = plane_depth_interval(anchor, n['normal_camera_unit'], ray,
            normal_angle_rad=math.radians(protocol['normal_angle_error_deg']),
            anchor_error=ez, ray_error=er, residual_error=protocol['local_plane_residual_mm'])
        center = bound['center']
        if center is not None:
            assert abs(center - n['predicted_target_optical_z_mm']) < 1e-7
        informative = (bound['status'] == 'ok' and center > 0
                       and bound['radius'] <= protocol['maximum_informative_radius_mm'])
        records.append(dict(neighbor_xy=n['neighbor_xy'], center_mm=center, radius_mm=bound['radius'],
            lo_mm=bound['low'], hi_mm=bound['high'], informative=informative, bound=bound))
    return records


def inference():
    check()
    deny_reference()
    verify(ROOT / 'RUN_LOCK.json')
    start = time.monotonic()
    raw = load(INPUTS[0])
    rows = sanitize(raw)
    tampered = copy.deepcopy(raw)
    for r in tampered['rows']:
        r['official_diagnostic'] = {'arbitrary_forbidden': 1e30}
        for o in r['candidates'] + ([] if r['incumbent'] is None else [r['incumbent']]):
            o['official_normal_diagnostic'] = {'arbitrary_forbidden': -1e30}
    assert sanitize(tampered) == rows
    plan, protocol = load(PLAN), load(ROOT / 'PROTOCOL.json')
    neighbors, status_counts = read_neighbors()
    assert len(rows) == 21 and sum(o['candidate_id'] != -1 for r in rows for o in r['objects']) == 105
    results = []
    bound_checks = 0
    for row in rows:
        intervals = make_intervals(row, neighbors[(row['roi'], row['query'])], plan, protocol)
        # Independently implemented vote arithmetic must agree with theory kernel.
        if intervals:
            comparison = certain_possible_support([o['optical_depth_mm'] for o in row['objects']],
                [i['bound'] for i in intervals], tolerance=protocol['support_tolerance_mm'],
                max_radius=protocol['maximum_informative_radius_mm'])
            local = support(row['objects'], intervals, protocol)
            assert [s['certain_count'] for s in local] == comparison['certain_counts']
            assert [s['possible_count'] for s in local] == comparison['possible_counts']
            bound_checks += 1
        action = decisions(row['objects'], intervals, protocol, row['baseline_id'])
        # Deliberate changes to forbidden metadata cannot change inputs/decisions.
        row['intervals'] = intervals
        row['decisions'] = action
        results.append(row)
    summary = dict(created_at=now(), seconds=time.monotonic() - start, rows=len(results),
        candidates=105, gt_accessed=False, official_fields_invariance=True, vote_cross_checks=bound_checks,
        neighbor_status_counts=status_counts, decisions={a:dict(
            selected=sum(r['decisions'][a]['selected_candidate_id'] is not None for r in results),
            reasons=Counter(r['decisions'][a]['reason'] for r in results))
            for a in results[0]['decisions']})
    dump(ROOT / 'DECISIONS.json', dict(created_at=now(), rows=results, gt_accessed=False))
    dump(ROOT / 'INFERENCE_SUMMARY.json', summary)
    verify(ROOT / 'RUN_LOCK.json')
    dump(ROOT / 'PREDICTIONS_SEALED.json', dict(created_at=now(), files={str(ROOT / p): sha(ROOT / p)
         for p in ('DECISIONS.json', 'INFERENCE_SUMMARY.json')}))
    print(json.dumps(clean(summary), indent=2))


def readcsv(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def writecsv(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate():
    check()
    verify(ROOT / 'RUN_LOCK.json')
    verify(ROOT / 'PREDICTIONS_SEALED.json')
    # No GT reads before both inference/code seals have been verified.
    seal = load(EVAL / 'EVALUATION_COMPLETE.json')
    oldfiles = seal.get('files', {str(EVAL / p): h for p,h in seal.get('artifacts', {}).items()})
    assert oldfiles
    for filename, digest in oldfiles.items():
        assert sha(filename) == digest, filename
    oldpoints = readcsv(EVAL / 'POINT_METRICS.csv')
    oldcandidates = readcsv(EVAL / 'CANDIDATE_DISTANCES.csv')
    base = {(p['roi'], int(p['query'])): p for p in oldpoints if p['arm'] == 'incumbent'}
    official = {(p['roi'], int(p['query'])): p for p in oldpoints if p['arm'] == 'official_double'}
    cd = {(p['roi'], int(p['query']), int(p['candidate_id'])): p for p in oldcandidates}
    events = {(r['roi'], r['query']): r for r in load(ROOT / 'DECISIONS.json')['rows']}
    assert len(base) == 512 and len(events) == 21
    protocol = load(ROOT / 'PROTOCOL.json')
    allrows, metrics, diagnostics = [], [], []
    coordinate_checks = set()
    for arm in protocol['report_arms']:
        ds, paired, points = {}, [], []
        for key, p in base.items():
            primary = p['primary'] == 'True'
            d0 = float(p['distance_mm']) if p['distance_mm'] else None
            d, selected, reason = d0, None, 'UNCHANGED'
            if arm == 'official_double':
                d = float(official[key]['distance_mm']) if official[key]['distance_mm'] else None
                reason = 'FROZEN_OFFICIAL'
            elif arm != 'incumbent' and key in events:
                r = events[key]
                decision = r['decisions'][arm]
                selected, reason = decision['selected_candidate_id'], decision['reason']
                # Verify ALL candidate positions, not just the selected ones.
                for o in r['objects']:
                    if o['candidate_id'] == -1:
                        continue
                    lookup = cd[(*key, o['candidate_id'])]
                    np.testing.assert_array_equal(o['xyz_mm'], [float(lookup[k]) for k in ('x_mm','y_mm','z_mm')])
                    coordinate_checks.add((*key, o['candidate_id']))
                if selected is not None:
                    d = float(cd[(*key, selected)]['distance_mm'])
                if primary and d0 > 5:
                    candidate_distances = [float(cd[(*key, o['candidate_id'])]['distance_mm'])
                                          for o in r['objects'] if o['candidate_id'] != -1]
                    diagnostics.append(dict(arm=arm, roi=key[0], query=key[1],
                        repairable=any(x <= 5 for x in candidate_distances),
                        incumbent_distance_mm=d0, selected=selected, final_distance_mm=d,
                        repaired=d <= 5, reason=reason))
            assert d0 is None or d is not None
            point = dict(arm=arm,roi=key[0],query=key[1],primary=primary,
                         selected=selected,reason=reason,distance_mm=d,incumbent_distance_mm=d0)
            points.append(point)
            if primary:
                assert d is not None and d0 is not None
                ds.setdefault(key[0], []).append(d)
                paired.append((d0, d))
        pair = np.array(paired)
        assert len(pair) == 484
        mse_roi = {r: float(np.mean(np.square(values))) for r, values in ds.items()}
        metrics.append(dict(arm=arm,primary=484,mse_mm2=float(np.mean(list(mse_roi.values()))),
            mae_mm=float(np.mean([np.mean(values) for values in ds.values()])),roi_mse=mse_roi,
            scene_mse={s: float(np.mean([v for k,v in mse_roi.items() if k.startswith('scan'+s+'_')])) for s in ('118','122')},
            improved=int(np.sum(pair[:,1]<pair[:,0]-1e-9)),worsened=int(np.sum(pair[:,1]>pair[:,0]+1e-9)),
            unchanged=int(np.sum(np.abs(pair[:,1]-pair[:,0])<=1e-9)),
            severe_gt5=int(np.sum(pair[:,1]>5)),good_le1=int(np.sum(pair[:,1]<=1)),
            new_1mm_harm_vs_incumbent=int(np.sum((pair[:,0]<=1)&(pair[:,1]>1))),
            finite=sum(p['distance_mm'] is not None for p in points),
            original_missing_finite=sum(not p['primary'] and p['distance_mm'] is not None for p in points)))
        allrows.extend(points)
    assert len(coordinate_checks) == 105
    byarm = {a: {(p['roi'],p['query']):p for p in allrows if p['arm']==a} for a in protocol['report_arms']}
    transitions = []
    for arm in ('neighbor_point','neighbor_interval','joint_point','joint_interval'):
        for key, p in byarm[arm].items():
            photo = byarm['photo_U11'][key]
            if p['distance_mm'] != photo['distance_mm'] or p['selected'] != photo['selected']:
                transitions.append(dict(arm=arm,roi=key[0],query=key[1],primary=p['primary'],
                    incumbent_mm=p['incumbent_distance_mm'],photo_mm=photo['distance_mm'],output_mm=p['distance_mm'],
                    photo_selected=photo['selected'],selected=p['selected'],reason=p['reason']))
    mm = {m['arm']:m for m in metrics}
    harmful = lambda p: p['primary'] and p['incumbent_distance_mm']<=1 and p['distance_mm']>1
    additional_harms = [key for key,p in byarm['joint_interval'].items() if harmful(p) and not harmful(byarm['photo_U11'][key])]
    success = (mm['joint_interval']['mse_mm2'] < mm['photo_U11']['mse_mm2'] and not additional_harms)
    out = ROOT / 'evaluation'
    dump(out/'RESULTS.json',dict(created_at=now(),metrics=metrics,primary_success=success,
        additional_harms_vs_photo=additional_harms,coordinate_checks=105,diagnostics=diagnostics,
        evaluation_type='real_gt_reused_same_coordinates',scope='2 old scenes; 4 ROIs; fixed 21-query residual subset',
        reference='DTU official laser nearest vertex; no new alignment; not official full-cloud benchmark',
        evaluation_inputs={str(EVAL/p):sha(EVAL/p) for p in ('POINT_METRICS.csv','CANDIDATE_DISTANCES.csv','EVALUATION_COMPLETE.json')}))
    writecsv(out/'POINT_METRICS.csv',allrows)
    writecsv(out/'TRANSITIONS_VS_PHOTO.csv',transitions)
    writecsv(out/'SEVERE_DIAGNOSTICS.csv',diagnostics)
    dump(out/'SEALED.json',dict(created_at=now(),files={str(p):sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(metrics=metrics,primary_success=success,transitions=transitions),indent=2))


if __name__ == '__main__':
    {'freeze':freeze,'infer':inference,'evaluate':evaluate}[sys.argv[1]]()

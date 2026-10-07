"""Read-only frozen real replay verification; never rewrites experiment artifacts."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
import csv
import json
import hashlib
import math
from pathlib import Path
from collections import defaultdict
from fractions import Fraction as F

ROOT = Path(__file__).resolve().parent

def load(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csvread(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    checked = 0
    for name in ['RUN_LOCK.json', 'PREDICTIONS_SEALED.json', 'evaluation/SEALED.json']:
        for path, h in load(ROOT/name)['files'].items():
            assert sha(path) == h, path
            checked += 1
    seal = load(ROOT/'observation/SEAL.json')
    for path, h in seal['sha256'].items():
        assert sha(ROOT/'observation'/path) == h, path
        checked += 1
    for path, h in seal['input_photo_sha256'].items():
        assert sha(path) == h, path
        checked += 1
    assert seal['request_sha256'] == sha(ROOT/'REQUESTS.json')
    assert seal['protocol_sha256'] == sha(ROOT/'PROTOCOL.json')
    results = load(ROOT/'evaluation/RESULTS.json')
    for path, h in results['inputs'].items():
        assert sha(path) == h, path
        checked += 1
    points = csvread(ROOT/'evaluation/POINT_METRICS.csv')
    assert len(points) == 3584
    photo = {(p['roi'], int(p['query'])): p for p in points if p['arm'] == 'photo_U11'}
    for m in results['metrics']:
        armrows = [p for p in points if p['arm'] == m['arm']]
        assert len(armrows) == 512
        rows = [p for p in armrows if p['primary'] == 'True']
        assert len(rows) == 484
        values = defaultdict(list)
        counts = dict(improved_vs_photo=0, worsened_vs_photo=0, unchanged_vs_photo=0,
                      new_harm_vs_photo=0, new_harm_vs_original=0, severe_gt5=0)
        for p in rows:
            x = float(p['distance_mm'])
            b = float(photo[p['roi'], int(p['query'])]['distance_mm'])
            orig = float(p['incumbent_distance_mm'])
            values[p['roi']].append(x*x)
            counts['improved_vs_photo'] += x-b < -1e-9
            counts['worsened_vs_photo'] += x-b > 1e-9
            counts['unchanged_vs_photo'] += abs(x-b) <= 1e-9
            counts['new_harm_vs_photo'] += b <= 1 < x
            counts['new_harm_vs_original'] += orig <= 1 < x
            counts['severe_gt5'] += x > 5
        mse = math.fsum(math.fsum(v)/len(v) for v in values.values())/len(values)
        assert math.isclose(mse, m['mse_mm2'], rel_tol=1e-13, abs_tol=1e-13)
        for k, v in counts.items():
            assert v == m[k], (m['arm'], k, v, m[k])
        assert sum(bool(p['distance_mm']) for p in armrows) == m['finite']
        assert sum(p['primary'] != 'True' and bool(p['distance_mm']) for p in armrows) == m['missing_finite']

    # Independent exact rational calculation of squared distances at endpoints,
    # not the inference kernel's expanded affine formula.
    decisions = load(ROOT/'DECISIONS.json')['rows']
    decision_count = 0
    for row in decisions:
        objs = {o['candidate_id']: list(map(F, o['xyz_mm'])) for o in row['objects']}
        C, r = list(map(F, row['center'])), list(map(F, row['ray']))
        current = row['current_id']
        for arm, decision in row['decisions'].items():
            intervals = row['intervals'][arm]
            selected = current
            exact = {}
            if current in objs and intervals:
                endpoints = {F(x) for iv in intervals for x in iv}
                for idx, b in objs.items():
                    gains = []
                    for z in endpoints:
                        target = [c + z*q for c, q in zip(C, r)]
                        old = sum((a-t)**2 for a, t in zip(objs[current], target))
                        new = sum((a-t)**2 for a, t in zip(b, target))
                        gains.append(old-new)
                    exact[idx] = (min(gains), max(gains))
                for score in decision['scores']:
                    lo, hi = exact[score['candidate_id']]
                    assert F(score['gain_low_mm2']) <= lo
                    assert F(score['gain_high_mm2']) >= hi
                viable = [idx for idx, (lo, _) in exact.items() if idx != current and lo > 0]
                if viable:
                    selected = min(viable, key=lambda idx: (-exact[idx][0], idx))
            assert selected == decision['selected_candidate_id'], (row['roi'], row['query'], arm)
            decision_count += 1
    output = dict(status='PASS', checked_hash_entries=checked, point_rows=len(points),
                  arms=len(results['metrics']), primary_per_arm=484,
                  exact_rational_decisions=decision_count,
                  note='Deterministic read-only consistency check, not physical support validation')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()

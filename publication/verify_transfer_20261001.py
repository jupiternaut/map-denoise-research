"""Standard-library archive + table recomputation; not a new geometric run."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research_snapshots/2026-10-01/colmap-transfer-20260930T180000Z'
def near(a,b):
    if a is None or b is None:assert a is b,(a,b)
    else:assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9),(a,b)
def main():
    m=json.loads((ROOT/'publication/TRANSFER_20261001_MANIFEST.json').read_text())
    for r in m['records']:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'],p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p
    point=list(csv.DictReader((RUN/'evaluation/POINT_METRICS.csv').open()))
    rows=list(csv.DictReader((RUN/'evaluation/ROI_METRICS.csv').open()))
    for r in rows:
        q=[p for p in point if p['roi']==r['roi'] and p['method']==r['method'] and p['valid']=='True' and (r['scope']=='all_valid' or p['cpu_valid']=='True')]
        ds=[float(p['distance_mm']) for p in q];n=len(ds);near(n,r['n'])
        near(sum(x*x for x in ds)/n if n else None,float(r['mse_mm2']) if r['mse_mm2'] else None)
        near(sum(ds)/n if n else None,float(r['mae_mm']) if r['mae_mm'] else None)
        near(sum(x<=1 for x in ds),r['correct_1mm']);near(sum(x>5 for x in ds),r['wrong_5mm'])
        near(n/int(r['requested']),r['coverage'])
        near(sum(x<=1 for x in ds)/int(r['requested']),r['correct_per_requested'])
    report=json.loads((RUN/'evaluation/RESULTS.json').read_text())
    for s,summary in [('all',report['summary']),*report['scenes'].items()]:
        for v in summary:
            q=[r for r in rows if r['method']==v['method'] and r['scope']=='cpu_valid_common' and (s=='all' or r['scene']==s)]
            for a,b in [('roi_equal_mse_mm2','mse_mm2'),('roi_equal_mae_mm','mae_mm')]:near(v[a],sum(float(r[b]) for r in q)/len(q))
            for a,b in [('common_n','n'),('correct_1mm','correct_1mm'),('wrong_5mm','wrong_5mm')]:near(v[a],sum(int(r[b]) for r in q))
    print(f'PASS: {len(m["records"])} source artifacts; {len(point)} point rows; {len(rows)} scoped ROI rows; all summary tables.')
    print('This verifies published bytes and table arithmetic, not GT distances, fresh GPU inference, or universal transfer.')
if __name__=='__main__':main()

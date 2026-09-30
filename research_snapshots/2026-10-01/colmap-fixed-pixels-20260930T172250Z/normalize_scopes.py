"""Post-audit CSV schema correction only; do not change predictions/MSE/GT reads."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parent
path=ROOT/'evaluation/ROI_METRICS.csv'
rows=list(csv.DictReader(path.open()));pts=list(csv.DictReader((ROOT/'evaluation/POINT_METRICS.csv').open()))
for r in rows:
    group=[p for p in pts if p['roi']==r['roi'] and p['method']==r['method'] and p['valid']=='True'
        and (r['scope']=='all_valid' or p['cpu_valid']=='True')]
    assert len(group)==int(r['n'])
    r['method_all_valid_coverage']=r.pop('coverage')
    r['method_all_valid_correct_per_requested']=r.pop('correct_per_requested')
    r['scope_coverage']=len(group)/int(r['requested'])
    r['scope_correct_per_requested']=sum(float(p['distance_mm'])<=1 for p in group)/int(r['requested'])
dest=ROOT/'evaluation/ROI_METRICS_SCOPED.csv'
with dest.open('x') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (ROOT/'evaluation/SCOPE_CORRECTION.json').open('x') as f:
    json.dump(dict(original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        corrected_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        reason='common rows auxiliary coverage labels used entire method support; rename and add scope-specific values',
        changes_to_predictions=False,changes_to_mse=False,changes_to_results_json=False),f,indent=2)

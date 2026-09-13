import csv,json,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT/'exploration_v6')]
from run_v9 import RUNS,read,save,sha
from audit_metrics import synthetic_scores,rms_mm
from local_filter import corrected_world

def main():
    root=Path(sys.argv[1]);records=json.loads((root/'SEALED_BEFORE_GT.json').read_text())
    rows={(r['case'],r['method']):r for r in csv.DictReader((root/'RESULTS.csv').open())};worst=0.;count=0
    for r in records:
        if r['status']!='OK':continue
        assert sha(r['output'])==r['output_sha256'];assert sha(r['input'])==r['input_sha256']
        out=read(r['output'])['xyz_world'];state=read(r['state']);_,mask=corrected_world(state)
        np.testing.assert_array_equal(out[~mask],state['world'][~mask]);row=rows[(r['case'],r['method'])]
        if r['domain']=='synthetic':
            ev=read(r['evaluation']);ev.update(json.loads(ev.pop('json').tobytes().decode()))
            scores=synthetic_scores(out,ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],ev['gt_layer'],ev['true_gap_mm'])
            for k in ('surface_accuracy_mean_mm','matched_point_rms_mm','fitted_gap_at_same_xy_error_mm'):
                if k in scores:worst=max(worst,abs(scores[k]-float(row[k])))
        else:worst=max(worst,abs(rms_mm(out-read(r['original'])['xyz_world'])-float(row['measured_reference_rms_mm'])))
        count+=1
    assert worst<1e-8
    history=json.loads((root/'HISTORY_BEFORE.json').read_text());assert all(sha(p)==h for p,h in history.items())
    for p,h in json.loads((root/'SOURCES.json').read_text()).items():assert sha(root/'source'/Path(p).name)==h
    library=Path('/srv/slam-research/grf/map-denoise/envs/spatial-v12-AXeJsv/lib/python3.12/site-packages/pymeshlab')
    binaries={str(p):sha(p) for p in library.rglob('*.so') if 'filter_mls' in p.name or 'pmeshlab.' in p.name}
    dest=Path(tempfile.mkdtemp(prefix='local-v12-audit-',dir=RUNS));save(dest/'AUDIT.json',dict(run=str(root),outputs_checked=count,
        max_metric_difference_mm=worst,unsupported_rows_unchanged=True,historical_files_unchanged=len(history),
        source_snapshots_unchanged=True,external_binaries_sha256=binaries,external_version='2025.7.post1',numpy=np.__version__))
    print(dest);print('outputs',count,'max error',worst)
if __name__=='__main__':main()

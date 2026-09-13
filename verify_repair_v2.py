"""Independent saved-output checks and test log; never rewrites a run."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from hashutil import sha256_file, dump_json
from schema import read_patch, read_evaluation, validate_pose_consistency
from repair_pilot_v2 import old_manifest, visible_perturbation
from adapt import build_adapter, adapter_payload
from paths import PROJECT, PATCHES, PILOT, require_liekkas


def verify(dest):
    require_liekkas()
    dest = Path(dest).resolve()
    if (dest/'AUDIT_V2.json').exists(): raise FileExistsError('audit already exists')
    with (dest/'RESULTS_V2.csv').open() as f: rows = list(csv.DictReader(f))
    before = json.loads((dest/'OLD_MANIFEST_BEFORE.json').read_text())
    assert before == old_manifest(), 'old files changed'
    errors = []; checked = 0; max_diff = 0.; max_adapter_diff = 0.
    for row in rows:
        if row['ok'] != 'True':
            errors.append({'case': row['case'], 'error': row.get('error')}); continue
        path = Path(row['source_output'])
        assert sha256_file(path) == row['source_output_sha256']
        with np.load(path) as a: out = a['xyz_world']
        assert np.isfinite(out).all()
        if row['split'] == 'synthetic':
            group = 'ambiguous' if row['family']=='ambiguous_two_explanations' else 'identifiable'
            file = dest/'synthetics'/group/(row['case']+'.json')
            p,m = read_patch(file); validate_pose_consistency(p,m)
            ev = read_evaluation(file)
            if not ev['identifiable']: continue
            # Independent broadcasting calculation of finite-rectangle distance.
            rects = np.asarray(ev['surface_rectangles_mm'])
            closest = np.repeat(out[:,None,:]*1000, len(rects), axis=1)
            closest[:,:,0] = np.minimum(np.maximum(closest[:,:,0],rects[:,1]),rects[:,2])
            closest[:,:,1] = np.minimum(np.maximum(closest[:,:,1],rects[:,3]),rects[:,4])
            closest[:,:,2] = rects[:,0]
            d = np.linalg.norm(out[:,None,:]*1000-closest,axis=2).min(axis=1)
            truth = ev['gt_clean_xyz_world']
            # Brute-force reference sample coverage, independent of cKDTree.
            nearest = np.concatenate([np.linalg.norm(block[:,None,:]-out[None,:,:],axis=2).min(axis=1)*1000
                                      for block in np.array_split(truth,8)])
            expected = {'surface_accuracy_mean_mm':d.mean(),
                        'surface_accuracy_rms_mm':np.sqrt(np.mean(d*d)),
                        'reference_sample_coverage_1mm':np.mean(nearest<=1.+1e-9),
                        'reference_sample_coverage_2mm':np.mean(nearest<=2.+1e-9),
                        'matched_normal_mae_mm':np.mean(np.abs(out[:,2]-truth[:,2]))*1000}
        else:
            file = PATCHES/row['scene']/(row['patch_id']+'.json')
            p,m = read_patch(file); reference = p['xyz_world']
            if row['adapter_mode']=='current_input':
                visible = p if row['perturbation']=='unpert' else visible_perturbation(p,m,row['perturbation'],int(row['seed']))[0]
                expected_adapter = adapter_payload(build_adapter(visible['xyz_world']))
                actual_adapter = json.loads(path.with_suffix('.json').read_text())['adapter']
                for key in ('origin_m','basis','normal_world'):
                    diff=np.max(np.abs(np.asarray(actual_adapter[key])-np.asarray(expected_adapter[key])))
                    max_adapter_diff=max(max_adapter_diff,float(diff)); assert diff < 1e-12
                baseline_path=dest/'outputs'/f"{row['patch_id']}_unpert__sig{row['sigma_mm']}__{row['method']}.npz"
            else:
                baseline_path=PILOT/'outputs'/f"{row['patch_id']}_unpert__sig{row['sigma_mm']}__{row['method']}.npz"
            with np.load(baseline_path) as a: baseline=a['xyz_world']
            expected = {
                'recovery_vs_measured_reference_rms_mm':np.sqrt(np.sum((out-reference)**2)/len(out))*1000,
                'zero_injection_edit_rms_mm':np.sqrt(np.sum((baseline-reference)**2)/len(out))*1000,
                'response_vs_own_baseline_rms_mm':np.sqrt(np.sum((out-baseline)**2)/len(out))*1000}
        for key,value in expected.items():
            diff=abs(float(row[key])-float(value));max_diff=max(max_diff,diff)
            assert diff < 1e-8, (row['case'],key,diff)
        checked+=1
    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                        cwd=PROJECT,env=env,text=True,capture_output=True)
    with (dest/'TEST_LOG.txt').open('x') as f:f.write(proc.stdout+proc.stderr)
    report=dict(rows=len(rows),scored_outputs_checked=checked,ambiguous_outputs_not_scored=12,
                max_metric_abs_difference=max_diff,max_current_adapter_abs_difference=max_adapter_diff,
                old_files_unchanged=before==old_manifest(),test_exit_code=proc.returncode,errors=errors)
    dump_json(dest/'AUDIT_V2.json',report)
    if proc.returncode or errors or not report['old_files_unchanged']:raise RuntimeError(report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);verify(p.parse_args().run)

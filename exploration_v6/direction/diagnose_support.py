"""Read-only DA support diagnosis using saved PCA basis/bias and unchanged fits.

No truth or evaluator file is loaded. No candidate is re-estimated, accepted
gate is relaxed, or frozen output is changed. Each final local fit is replayed
once from the legal current XYZ and the already saved estimated scan bias.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode=True
import numpy as np
import direction_pooling as wrapper

HERE=Path(__file__).resolve().parent
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
FROZEN=RUNS/'direction-v6-edpxxdsb'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def run():
    import socket
    if socket.gethostname()!='liekkas':raise RuntimeError('exact host liekkas required')
    private=wrapper._load_private();pool=private._V4;graph=pool._V3
    files={p for p in FROZEN.rglob('*') if p.is_file()}
    files.update((Path(__file__).resolve(),Path(wrapper.__file__),Path(private.__file__),Path(pool.__file__),Path(graph.__file__)))
    before={str(p):digest(p) for p in sorted(files)}
    dest=Path(tempfile.mkdtemp(prefix='direction-support-diag-v6-',dir=RUNS));print(dest,flush=True)
    save(dest/'PROTECTED_BEFORE.json',before)
    shutil.copyfile(__file__,dest/'DIAGNOSTIC_SOURCE.py')
    rows=[];cases=[]
    for patch in ('da_junction','da_thin','da_wall'):
        for mode in ('zero','normal_translation'):
            name=patch+'__'+mode
            metadata=FROZEN/'outputs'/('real__'+name+'__pooled_pca.json')
            info=json.loads(metadata.read_text())['info'];saved=info['v4_info']['initial_info']
            with np.load(FROZEN/'real'/'inputs'/(name+'.npz'),allow_pickle=False) as a:
                world=a['xyz_world'];frames=a['scan_id']
            with np.load(metadata.with_suffix('.npz'),allow_pickle=False) as a:
                stored_support=a['support_mask'];stored_output=a['xyz_world']
            order=pool._canonical_order((world-world.mean(0))*1000.,frames)
            ordered=world[order];ids,scans=np.unique(frames[order],return_inverse=True)
            centered=(ordered-ordered.mean(0))*1000.
            basis=np.array(info['basis_calls'][0]['basis_world'])
            assert wrapper.fingerprint(centered)==info['basis_calls'][0]['point_coordinates_sha256']
            assert np.array_equal(ids,info['scan_ids'])
            bias=np.array(info['bias_mm']);sigma=float(info['sigma_mm'])
            local=centered@basis
            labels,members,geometry,grid=graph._cells(local)
            assert grid.tolist()==saved['grid_shape']
            predicted_support=np.zeros(len(world),bool)
            case_rows=[]
            for cell,take in enumerate(members):
                if len(take)<24 or len(np.unique(scans[take]))<2:continue
                xy,scale=geometry[cell]
                model=graph._local_model(local[take,2]-bias[scans[take]],xy,scale,sigma)
                residual=np.abs(model['residual']);slope=model['beta'][model['k']:]/scale
                median=float(np.median(residual));slope_norm=float(np.linalg.norm(slope))
                residual_cell_fail=median>2.5*sigma;slope_cell_fail=slope_norm>.25
                residual_point_pass=residual<=4.*sigma;confidence_pass=model['confidence']>=.8
                raw_point_pass=residual_point_pass&confidence_pass
                fit_ok=not(residual_cell_fail or slope_cell_fail)
                accepted=raw_point_pass if fit_ok else np.zeros(len(take),bool)
                predicted_support[take[accepted]]=True
                assert model['k']==saved['local_k'][str(cell)]
                assert int((~accepted).sum())==saved['rejected_points_by_cell'][str(cell)]
                row=dict(case=name,patch=patch,mode=mode,cell=cell,n_points=len(take),n_scans=len(np.unique(scans[take])),
                         selected_k=int(model['k']),sigma_mm=sigma,
                         residual_median_mm=median,residual_cell_threshold_mm=2.5*sigma,
                         residual_cell_fail=bool(residual_cell_fail),slope_norm=slope_norm,slope_threshold=.25,
                         slope_cell_fail=bool(slope_cell_fail),fit_ok=fit_ok,
                         abs_residual_min_mm=float(residual.min()),abs_residual_p95_mm=float(np.quantile(residual,.95)),
                         abs_residual_max_mm=float(residual.max()),confidence_min=float(model['confidence'].min()),
                         confidence_median=float(np.median(model['confidence'])),
                         point_residual_pass_count=int(residual_point_pass.sum()),
                         point_confidence_pass_count=int(confidence_pass.sum()),
                         raw_point_both_pass_count=int(raw_point_pass.sum()),accepted_count=int(accepted.sum()),
                         graph_rejection_count=int((~accepted).sum()),
                         reason=' and '.join((["cell median residual > 5 mm"] if residual_cell_fail else [])+
                                             (["cell slope > 0.25"] if slope_cell_fail else [])) or 'point-level gates only')
                rows.append(row);case_rows.append(row)
            support=np.zeros(len(world),bool);support[order]=predicted_support
            np.testing.assert_array_equal(support,stored_support)
            np.testing.assert_array_equal(stored_output,world)
            cases.append(dict(case=name,cells=len(case_rows),usable_points=sum(r['n_points'] for r in case_rows),
                 input_points=len(world),support_exact_replay=True,output_is_exact_identity=True,
                 residual_failed_cells=sum(r['residual_cell_fail'] for r in case_rows),
                 slope_failed_cells=sum(r['slope_cell_fail'] for r in case_rows),
                 raw_point_both_pass_count=sum(r['raw_point_both_pass_count'] for r in case_rows),
                 residual_median_range_mm=[min(r['residual_median_mm'] for r in case_rows),max(r['residual_median_mm'] for r in case_rows)],
                 slope_norm_range=[min(r['slope_norm'] for r in case_rows),max(r['slope_norm'] for r in case_rows)],
                 estimated_bias_mm=bias.tolist(),basis_world=basis.tolist(),input_direction='saved current-input PCA'))
    with (dest/'CELLS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(dest/'CASES.json',cases)
    after={p:digest(p) for p in before};assert before==after;save(dest/'PROTECTED_AFTER.json',after)
    summary=dict(cases=len(cases),local_fits_replayed=len(rows),all_saved_support_masks_reproduced=True,
                 all_six_outputs_exact_identity=True,residual_failed_cells=sum(r['residual_cell_fail'] for r in rows),
                 slope_failed_cells=sum(r['slope_cell_fail'] for r in rows),
                 both_failed_cells=sum(r['residual_cell_fail'] and r['slope_cell_fail'] for r in rows),
                 point_gate_only_cells=sum(r['fit_ok'] for r in rows),accepted_points=sum(r['accepted_count'] for r in rows),
                 preserved_files=len(before),protected_files_unchanged=True,
                 original_frozen_run=str(FROZEN),diagnostic_source_sha256=digest(__file__),
                 scope='replay final local fits only; no new candidate, gate relaxation, truth input or output edit')
    save(dest/'SUMMARY.json',summary)
    lines=['# DA support rejection diagnostic','',
           'Saved PCA basis and scan bias, unchanged current XYZ, unchanged local fit/gates. No evaluator truth loaded.','',
           '|Case|Cells|Median-residual range mm|Slope-norm range|Residual-failed cells|Slope-failed cells|',
           '|---|---:|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f"|{c['case']}|{c['cells']}|{c['residual_median_range_mm'][0]:.3f}–{c['residual_median_range_mm'][1]:.3f}|{c['slope_norm_range'][0]:.4f}–{c['slope_norm_range'][1]:.4f}|{c['residual_failed_cells']}|{c['slope_failed_cells']}|")
    lines+=['','All six actual support masks and identity outputs are reproduced. Thresholds remain median absolute residual <= 5 mm, slope norm <= 0.25, point absolute residual <= 8 mm and confidence >= 0.8.',
            'Raw point-pass counts in CELLS.csv are diagnostic, not newly accepted points. Failed cell-level criteria veto them.',
            'This identifies mismatch with the existing local model/gates at supplied sigma=2 mm. It does not distinguish true geometry, noise miscalibration, association error, or bad global direction by itself.',
            'No gate was relaxed; no parameter, existing run, estimator source, or point output was changed.']
    with (dest/'READOUT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()

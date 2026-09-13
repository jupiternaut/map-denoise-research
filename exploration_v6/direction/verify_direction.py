"""Read-only saved-output audit. No estimator is rerun."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
from hashutil import sha256_file,sha256_array
from paths import RUNS,require_liekkas


def verify(dest):
    dest=Path(dest)
    rows=list(csv.DictReader((dest/'RESULTS.csv').open()))
    assert len(rows)==96 and len({(r['split'],r['case'],r['method']) for r in rows})==96
    registry={(r['split'],r['case']):r for r in json.loads((dest/'CASES.json').read_text())}
    assert len(registry)==48
    errors=[];real_scores=0;synthetic_scores=0;old_replays=0;bounds=0
    output_lookup={(r['split'],r['case'],r['method']):r for r in rows}
    def compare(row,key,value):
        error=abs(float(row[key])-float(value));errors.append(error)
        assert error<1e-8,(row['case'],row['method'],key,error)
    for row in rows:
        assert row['ok']=='True',row
        entry=registry[row['split'],row['case']]
        with np.load(entry['input'],allow_pickle=False) as a:current=a['xyz_world'];scan=a['scan_id']
        provenance=entry['input'] if row['split']=='real' else entry['source_points']
        with np.load(provenance,allow_pickle=False) as a:source_ids=a['source_point_index']
        path=Path(row['output']);assert sha256_file(path)==row['output_sha256']
        with np.load(path,allow_pickle=False) as a:
            out=a['xyz_world'];support=a['support_mask'];moved=a['moved_mask']
            assert out.shape==current.shape and np.isfinite(out).all()
            np.testing.assert_array_equal(a['scan_id'],scan);np.testing.assert_array_equal(a['source_point_index'],source_ids)
            assert support.shape==(len(out),) and support.dtype==np.bool_
            np.testing.assert_array_equal(out[~support],current[~support])
            np.testing.assert_array_equal(moved,np.linalg.norm(out-current,axis=1)*1000.>1e-6)
            assert sha256_array(support)==row['support_mask_sha256']
            compare(row,'supported_fraction',support.mean())
        info=json.loads(path.with_suffix('.json').read_text())['info']
        report_mask=np.ones(len(out),bool);report_mask[np.asarray(info['unsupported_point_indices'],int)]=False
        np.testing.assert_array_equal(report_mask,support)
        axis=info.get('normal_world')
        n=np.asarray(axis,float) if axis is not None else None
        if n is not None:
            n/=np.linalg.norm(n)
            for call in info['basis_calls']:
                if call['basis_world'] is not None:np.testing.assert_allclose(np.asarray(call['basis_world'])[:,2],n,atol=1e-12,rtol=0.)
            edit=out-current;perp=edit-(edit@n)[:,None]*n
            assert np.sqrt(np.mean(np.sum(perp**2,axis=1)))*1000.<1e-8
        else:
            assert not support.any();np.testing.assert_array_equal(out,current)
        if row['method']=='difference':
            old_run=RUNS/('real-transfer-v5-zrtui_f5' if row['split']=='real' else 'exploration-v5-development-vdbkdouk')
            with np.load(old_run/'outputs'/(row['case']+'__pool_compatible.npz'),allow_pickle=False) as a:
                np.testing.assert_array_equal(out,a['xyz_world'])
            old_replays+=1
        if row['split']=='real':
            with np.load(entry['evaluation'],allow_pickle=False) as a:reference=a['reference_measured_xyz_world'];delta=a['injected_delta_world']
            np.testing.assert_array_equal(delta,current-reference)
            metadata=json.loads(Path(entry['evaluation_json']).read_text());m=np.array(metadata['construction_normal_world']);m/=np.linalg.norm(m)
            zero_row=output_lookup['real',row['patch']+'__zero',row['method']]
            with np.load(zero_row['output']) as a:zero=a['xyz_world']
            for prefix,difference in [('recovery',out-reference),('zero_edit',zero-reference),('self_response',out-zero),('input_edit',out-current),('injected',delta)]:
                normal=difference@m;tangent=difference-normal[:,None]*m
                for suffix,value in [('xyz_rms_mm',np.sqrt(np.mean(np.sum(difference**2,axis=1)))*1000.),
                                     ('normal_rms_mm',np.sqrt(np.mean(normal**2))*1000.),
                                     ('tangent_rms_mm',np.sqrt(np.mean(np.sum(tangent**2,axis=1)))*1000.)]:
                    compare(row,prefix+'_'+suffix,value);real_scores+=1
            perpendicular=delta if n is None else delta-(delta@n)[:,None]*n
            masked=perpendicular.copy();masked[~support]=delta[~support]
            for key,v in [('axis_pointwise_lower_bound_xyz_rms_mm',perpendicular),('support_axis_lower_bound_xyz_rms_mm',masked)]:
                bound=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))*1000.)
                compare(row,key,bound);bounds+=1
                assert float(row['recovery_xyz_rms_mm'])+1e-8>=bound
        else:
            evaluation={}
            with np.load(entry['evaluation'],allow_pickle=False) as a:
                for key in a.files:
                    if key=='json':evaluation.update(json.loads(bytes(a[key]).decode()))
                    else:evaluation[key]=a[key]
            p=out*1000.;distance=np.full(len(p),np.inf)
            for z,x0,x1,y0,y1 in evaluation['surface_rectangles_mm']:
                dx=np.maximum.reduce([x0-p[:,0],p[:,0]-x1,np.zeros(len(p))]);dy=np.maximum.reduce([y0-p[:,1],p[:,1]-y1,np.zeros(len(p))])
                distance=np.minimum(distance,np.sqrt(dx*dx+dy*dy+(p[:,2]-z)**2))
            for key,value in [('surface_accuracy_mean_mm',np.mean(distance)),('surface_accuracy_rms_mm',np.sqrt(np.mean(distance**2))),('surface_accuracy_p95_mm',np.quantile(distance,.95)),
                              ('matched_point_rms_mm',np.sqrt(np.mean(np.sum((out-evaluation['gt_clean_xyz_world'])**2,axis=1)))*1000.)]:
                compare(row,key,value);synthetic_scores+=1
            if 'fitted_gap_at_same_xy_error_mm' in row and row['fitted_gap_at_same_xy_error_mm']:
                levels=[];labels=evaluation['gt_layer']
                for label in (0,1):
                    q=p[labels==label];A=np.c_[q[:,:2],np.ones(len(q))];levels.append(np.linalg.lstsq(A,q[:,2],rcond=None)[0][2])
                compare(row,'fitted_gap_at_same_xy_error_mm',abs(levels[1]-levels[0]-evaluation['true_gap_mm']));synthetic_scores+=1
    agg=json.loads((dest/'AGGREGATES.json').read_text());aggregate_numbers=0
    for g in agg['groups']:
        selected=[r for r in rows if r['split']==g['split'] and r['case'] in g['cases'] and r['method']==g['method']]
        assert len(selected)==g['n_cases']
        for key,value in g.items():
            if not isinstance(value,dict):continue
            z=[float(r[key]) for r in selected if r.get(key) not in ('',None)]
            assert len(z)==value['n']
            for field,number in [('mean',np.mean(z)),('median',np.median(z))]:
                error=abs(number-value[field]);errors.append(error);assert error<1e-8;aggregate_numbers+=1
    for g in agg['paired']:
        z=[]
        for record in g['cases']:
            a=output_lookup[g['split'],record['case'],'difference'];b=output_lookup[g['split'],record['case'],'pooled_pca']
            gain=float(a[g['metric']])-float(b[g['metric']]);assert abs(gain-record['gain'])<1e-8;z.append(gain);aggregate_numbers+=1
        assert len(z)==g['n'];z=np.asarray(z)
        assert int(np.sum(z>1e-9))==g['improved'] and int(np.sum(z < -1e-9))==g['worse'] and int(np.sum(abs(z)<=1e-9))==g['tied']
        assert abs(z.mean()-g['mean_gain'])<1e-8 and abs(np.median(z)-g['median_gain'])<1e-8
    before=json.loads((dest/'PROTECTED_BEFORE.json').read_text());after=json.loads((dest/'PROTECTED_AFTER.json').read_text())
    assert before==after
    for path,digest in before.items():assert sha256_file(Path(path))==digest,path
    for item in json.loads((dest/'SOURCE_MANIFEST.json').read_text()).values():assert sha256_file(Path(item['snapshot']))==item['sha256']
    for item in json.loads((dest/'INPUT_MANIFEST.json').read_text()):
        assert sha256_file(Path(item['original']))==item['sha256']==sha256_file(Path(item['copy']))
    return dict(status='PASS',outputs=len(rows),real_scores_recomputed=real_scores,synthetic_scores_recomputed=synthetic_scores,
                axis_bounds_recomputed=bounds,aggregate_numbers_recomputed=aggregate_numbers,maximum_error_mm=max(errors),
                original_compatible_exact_replays=old_replays,point_order_masks_hashes_checked=len(rows),
                protected_files=len(before),historical_files_unchanged=True,estimator_rerun=False,
                interpretation='contract and arithmetic verification, not physical truth or independent validation')


if __name__=='__main__':
    require_liekkas();parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path)
    print(json.dumps(verify(parser.parse_args().run_dir),indent=2))

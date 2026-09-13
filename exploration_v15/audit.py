"""Independent arithmetic verification and frozen-table readout. Does not select new methods."""
from pathlib import Path
import sys,json,shutil
import numpy as np
from scipy.spatial import cKDTree
from run import v14, ROOT, HERE, apply, aggregate, choose


def manual_metrics(q, e):
    mm=q*1000; labels=e['gt_layer']; distances=[]; by_layer=[]
    for k, rect in enumerate(e['surface_rectangles_mm']):
        z,x0,x1,y0,y1=rect
        dx=np.maximum(np.maximum(x0-mm[:,0],mm[:,0]-x1),0)
        dy=np.maximum(np.maximum(y0-mm[:,1],mm[:,1]-y1),0)
        each=np.sqrt(dx*dx+dy*dy+(mm[:,2]-z)**2)
        distances.append(each)
        if np.any(labels==k):by_layer.append(np.mean(each[labels==k]))
    vals=dict(surface_accuracy_mean_mm=float(np.min(distances,axis=0).mean()),
        matched_point_rms_mm=float(np.linalg.norm(q-e['gt_clean_xyz_world'])/np.sqrt(len(q))*1000),
        layer_balanced_source_surface_mae_mm=float(np.mean(by_layer)),
        worst_layer_source_surface_mae_mm=float(max(by_layer)),
        reference_sample_coverage_1mm=float(np.mean(cKDTree(q).query(e['gt_clean_xyz_world'])[0]*1000<=1+1e-9)))
    if int(e['n_true_layers'])==2:
        m0=np.mean(mm[labels==0,2]);m1=np.mean(mm[labels==1,2]);gap=float(e['true_gap_mm'])
        midpoint=.5*(np.mean(e['gt_clean_xyz_world'][labels==0,2])*1000+
                      np.mean(e['gt_clean_xyz_world'][labels==1,2])*1000)
        vals.update(source_group_gap_error_mm=float(abs(m1-m0-gap)),
            gap_absolute_relative_error=float(abs((m1-m0)/gap-1)),
            severe_gap_contraction=float(m1-m0<gap/2),
            balanced_source_midplane_crossing=float(.5*(np.mean(mm[labels==0,2]>=midpoint)+np.mean(mm[labels==1,2]<=midpoint))))
    return vals


def paired(rows, a, b):
    ix={(r['case'],r['method']):r for r in rows};records=[]
    for r in rows:
        other=ix.get((r['case'],b))
        if r['method']!=a or r['status']=='FAILED' or not other or other['status']=='FAILED':continue
        d=r['surface_accuracy_mean_mm']-other['surface_accuracy_mean_mm']
        gd=r.get('gap_absolute_relative_error',np.nan)-other.get('gap_absolute_relative_error',np.nan)
        records.append(dict(case=r['case'],mae_delta_mm=d,gap_relative_error_delta=gd))
    return dict(n=len(records),better=sum(r['mae_delta_mm'] < -1e-8 for r in records),
        same=sum(abs(r['mae_delta_mm'])<=1e-8 for r in records),worse=sum(r['mae_delta_mm']>1e-8 for r in records),
        dual_joint_not_worse=sum(r['mae_delta_mm']<=1e-8 and r['gap_relative_error_delta']<=1e-8 for r in records),
        dual_n=sum(np.isfinite(r['gap_relative_error_delta']) for r in records),records=records)


def main(path):
    run=Path(path);assert (run/'SUMMARY.json').exists()
    dest=run/'audit';dest.mkdir()
    shutil.copyfile(HERE/'audit.py',dest/'audit.py')
    maximum=0.;counts={};verified_files={};all_rows={}
    for phase in ('development','exposed_recheck','confirmation'):
        records=json.loads((run/f'{phase}_SEALED_BEFORE_GT.json').read_text())
        rows=json.loads((run/f'{phase}_ROWS.json').read_text());all_rows[phase]=rows
        index={(r['case'],r['method']):r for r in rows};checked=0
        for r in records:
            for key in ('input','output'):
                if key in r and r[key] not in verified_files:
                    got=v14.sha(r[key]);assert got==r[key+'_sha256']
                    verified_files[r[key]]=got
            if r['status']=='FAILED':continue
            inp=v14.load(r['input']);out=v14.load(r['output']);q=out['xyz_world'];mask=out['support_mask']
            assert q.shape==inp['xyz_world'].shape and np.isfinite(q).all()
            np.testing.assert_array_equal(q[~mask],inp['xyz_world'][~mask])
            e=v14.load(r['evaluation']);e.update(json.loads(e.pop('json').tobytes().decode()))
            row=index[(r['case'],r['method'])]
            for metric,value in manual_metrics(q,e).items():
                delta=abs(value-row[metric]);maximum=max(maximum,delta)
                assert delta<1e-10,(phase,r['case'],r['method'],metric,delta)
            if r['method']=='constant_lbfgs64':
                assert r['info']['final_two_plane_nll'] <= r['info']['initial_two_plane_nll']+1e-8
            checked+=1
        counts[phase]=dict(records=len(records),checked=checked,failed=len(records)-checked,
                           reused=sum(r['reused'] for r in records))
    lock=json.loads((run/'SELECTION_LOCK.json').read_text())
    assert lock==choose(aggregate(all_rows['development']))
    sources=json.loads((run/'SOURCES_BEFORE.json').read_text())
    assert all(v14.sha(p)==h for p,h in sources.items())
    records=json.loads((run/'confirmation_SEALED_BEFORE_GT.json').read_text());replays=[]
    sample=[r for r in records if r['seed']==9151101 and r['gap']==2 and r['sigma']==2 and r['bias']==4 and r['retain']==.25]
    inp=v14.load(sample[0]['input']);state=v14.v7.freeze(inp['xyz_world'],inp['scan_id'],2.)
    for r in sample:
        if r['status']=='FAILED':continue
        q,_,_=apply(state,2.,r['method']);ref=v14.load(r['output'])['xyz_world']
        delta=float(np.max(abs(q-ref))*1000)
        tolerance=1e-3 if r['method'].startswith(('apss_','rimls_')) else 1e-8
        assert delta<tolerance,(r['method'],delta,tolerance)
        replays.append(dict(method=r['method'],max_delta_mm=delta,tolerance_mm=tolerance))
    v14.save(dest/'AUDIT.json',dict(counts=counts,max_metric_delta=maximum,unique_hashed_files=len(verified_files),
        unchanged_source_files=len(sources),selection_recomputed=True,replays=replays))
    v14.save(dest/'VERIFIED_FILES.json',verified_files)
    a=lock['primary']['spatial_'];b=lock['primary']['constant_'];c=lock['primary']['rimls_']
    comparisons={phase:{other:paired(rows,a,other) for other in dict.fromkeys([b,c,'old_original','apss_4.0'])}
                 for phase,rows in all_rows.items()}
    v14.save(dest/'PAIRED.json',comparisons)
    slices={field:{str(v):aggregate([r for r in all_rows['confirmation'] if r[field]==v])
            for v in sorted({r[field] for r in all_rows['confirmation']})} for field in ('seed','gap','sigma','bias','retain')}
    v14.save(dest/'SLICES.json',slices)
    lines=['# V15 自动核查读数','',f'运行：{run}','',f'配置选择：{json.dumps(lock,ensure_ascii=False)}','']
    for phase in ('development','exposed_recheck','confirmation'):
        table=aggregate(all_rows[phase]);lines += [f'## {phase}','',
            '| 方法 | MAE mm | RMS mm | 层距绝对相对误差 | <50%间距条件率 | 跨中面比例 | 1mm覆盖 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for method,r in table.items():
            lines.append('| '+method+' | '+' | '.join(f'{r[k]:.6f}' for k in
                ['surface_accuracy_mean_mm','matched_point_rms_mm','gap_absolute_relative_error',
                 'severe_gap_contraction','balanced_source_midplane_crossing','reference_sample_coverage_1mm'])+' |')
        lines += ['']
    lines += ['## 新确认逐因素 MAE','']
    for field,values in slices.items():
        lines += [f'### {field}','','| 值 | 空间选择 | 恒定选择 | RIMLS选择 |','|---|---:|---:|---:|']
        for value,table in values.items():
            lines.append(f'| {value} | '+' | '.join(f'{table[m]["surface_accuracy_mean_mm"]:.6f}' for m in (a,b,c))+' |')
        lines += ['']
    lines += ['## 范围','', '旧确认集仅为暴露后的补测；新确认仅三个新种子，仍为同族合成、给定sigma。',
              '来源层间距/跨中面是带固定GT来源的几何诊断，不能当作通用拓扑层数识别。',
              'RIMLS仍有未扫描的参数；64初值连续优化也不是全局最优证书。']
    with (dest/'READOUT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(counts=counts,max_delta=maximum,lock=lock),indent=2),flush=True)


if __name__=='__main__':main(sys.argv[1])

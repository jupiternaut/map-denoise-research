"""Independent arithmetic and artifact audit, plus descriptive effect readout."""
from pathlib import Path
import sys,json,tempfile,subprocess,shutil
import numpy as np
from scipy.spatial import cKDTree
from experiment import ROOT,RUNS,HERE,load,sha,save,aggregate,apply,v7

def main(path):
    run=Path(path);assert (run/'SUMMARY.json').exists()
    dest=Path(tempfile.mkdtemp(prefix='effect-v14-audit-',dir=RUNS));print(dest,flush=True)
    shutil.copyfile(HERE/'audit.py',dest/'audit.py')
    summary={};maximum=0.;replays=[]
    for phase in ('development','confirmation'):
        records=json.loads((run/f'{phase}_SEALED_BEFORE_GT.json').read_text())
        rows=json.loads((run/f'{phase}_ROWS.json').read_text());index={(r['case'],r['method']):r for r in rows}
        checked=0
        for r in records:
            if r['status']=='FAILED':continue
            assert sha(r['input'])==r['input_sha256'];assert sha(r['output'])==r['output_sha256']
            art=load(r['output']);q=art['xyz_world'];inp=load(r['input']);mask=art['support_mask']
            assert q.shape==inp['xyz_world'].shape and np.isfinite(q).all()
            np.testing.assert_array_equal(q[~mask],inp['xyz_world'][~mask])
            e=load(r['evaluation']);e.update(json.loads(e.pop('json').tobytes().decode()))
            d=[];layers=[]
            for k,(z,x0,x1,y0,y1) in enumerate(e['surface_rectangles_mm']):
                mm=q*1000
                dx=np.maximum(np.maximum(x0-mm[:,0],mm[:,0]-x1),0)
                dy=np.maximum(np.maximum(y0-mm[:,1],mm[:,1]-y1),0)
                each=np.sqrt(dx**2+dy**2+(mm[:,2]-z)**2);d.append(each)
                if np.any(e['gt_layer']==k):layers.append(each[e['gt_layer']==k].mean())
            values=dict(surface_accuracy_mean_mm=float(np.min(d,axis=0).mean()),
                matched_point_rms_mm=float(np.linalg.norm(q-e['gt_clean_xyz_world'])/np.sqrt(len(q))*1000),
                layer_balanced_source_surface_mae_mm=float(np.mean(layers)),worst_layer_source_surface_mae_mm=float(max(layers)),
                reference_sample_coverage_1mm=float(np.mean(cKDTree(q).query(e['gt_clean_xyz_world'])[0]*1000<=1.+1e-9)))
            row=index[(r['case'],r['method'])]
            for k,v in values.items():
                err=abs(v-row[k]);maximum=max(maximum,err);assert err<1e-10,(r['case'],r['method'],k,err)
            checked+=1
        summary[phase]=dict(records=len(records),checked=checked,failed=len(records)-checked)
    lock=json.loads((run/'SELECTION_LOCK.json').read_text());dev=json.loads((run/'development_AGGREGATES.json').read_text())
    for prefix,chosen in lock['chosen'].items():
        possible=[m for m in dev if m.startswith(prefix) and dev[m]['failed']==0]
        assert chosen==min(possible,key=lambda m:(round(dev[m]['surface_accuracy_mean_mm'],12),dev[m]['matched_point_rms_mm'],m))
    # Recompute a sparse high-noise double layer through full legal-input upstream.
    records=json.loads((run/'confirmation_SEALED_BEFORE_GT.json').read_text())
    sample=[r for r in records if r['seed']==9141101 and r['gap']==2 and r['sigma']==2 and r['bias']==4 and r['retain']==.25]
    inp=load(sample[0]['input']);state=v7.freeze(inp['xyz_world'],inp['scan_id'],2.)
    for r in sample:
        if r['status']=='FAILED':continue
        out,_,_=apply(state,2.,r['method']);ref=load(r['output'])['xyz_world'];error=float(np.max(abs(out-ref))*1000)
        # Official native kernels can exhibit tiny replay variation. Keep the original
        # scores/outputs; require <1 micrometre, separately from custom bit-level tests.
        tolerance=1e-3 if r['method'].startswith(('apss_','rimls_')) else 1e-8
        assert error<tolerance,(r['method'],error,tolerance)
        replays.append(dict(method=r['method'],max_delta_mm=error,tolerance_mm=tolerance))
    sources=json.loads((run/'SOURCES_BEFORE.json').read_text());assert all(sha(p)==h for p,h in sources.items())
    old=json.loads((RUNS/'atlas-v13-development-ff24px9y/HISTORY_BEFORE.json').read_text());assert all(sha(p)==h for p,h in old.items())
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    with (dest/'TESTS.log').open('x') as f:f.write(test.stdout+test.stderr)
    assert test.returncode==0
    save(dest/'AUDIT.json',dict(run=run,phases=summary,max_metric_delta_mm=maximum,replays=replays,
        current_sources_unchanged=len(sources),protected_historical_files_unchanged=len(old),selection_recomputed=True))
    rows=json.loads((run/'confirmation_ROWS.json').read_text());table=aggregate(rows)
    spatial=lock['chosen']['spatial_'];comparisons={}
    ix={(r['case'],r['method']):r for r in rows}
    for other in ('old_original',lock['chosen']['apss_'],lock['chosen']['rimls_']):
        deltas=[r['surface_accuracy_mean_mm']-ix[(r['case'],other)]['surface_accuracy_mean_mm'] for r in rows if r['method']==spatial and r['status']!='FAILED' and ix[(r['case'],other)]['status']!='FAILED']
        comparisons[other]=dict(better=sum(d < -1e-8 for d in deltas),same=sum(abs(d)<=1e-8 for d in deltas),worse=sum(d>1e-8 for d in deltas))
    save(dest/'PAIRED_COMPARISON.json',comparisons)
    lines=['# V14 效果实验自动读数','',f'运行：{run}',f'开发冻结配置：{lock["chosen"]}','',
        '| 方法 | 表面MAE mm | 对应RMS mm | 层等权距离 mm | 最差层距离 mm | 1mm覆盖 |',
        '|---|---:|---:|---:|---:|---:|']
    for m,a in table.items():lines.append(f'| {m} | {a["surface_accuracy_mean_mm"]:.6f} | {a["matched_point_rms_mm"]:.6f} | {a["layer_balanced_source_surface_mae_mm"]:.6f} | {a["worst_layer_source_surface_mae_mm"]:.6f} | {a["reference_sample_coverage_1mm"]:.6f} |')
    lines+=['','## 空间关联对照逐条件胜负',json.dumps(comparisons,ensure_ascii=False),'',
        '## 分因素表面MAE（mm）','这些切片是同一确认集，不是额外独立实验。']
    for field in ('sigma','retain','gap','bias','seed'):
        lines+=['',f'### {field}','| 值 | '+ ' | '.join(table)+' |','|---|'+'---:|'*len(table)]
        for v in sorted(set(r[field] for r in rows)):
            a=aggregate([r for r in rows if r[field]==v]);lines+=['| '+str(v)+' | '+' | '.join(f'{a[m]["surface_accuracy_mean_mm"]:.6f}' for m in table)+' |']
    lines+=['','## 证据范围','所有参数只按开发主指标选择；未按确认GT切换方法。',
        '只有五个种子（开发二、确认三），因素组合共享底层采样，不能当成240个独立场景。',
        '本轮供应正确sigma；层相关丢点是受控采样干预，不是新传感器模型。',
        'APSS/RIMLS扩展五档尺度，其他参数保持原实现；没有宣称穷尽现有方法。',
        '没有新增独立真实几何参考、没有证明3DGS收益、没有论文结题结论。']
    with (dest/'READOUT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(table,indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1])

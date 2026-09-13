"""Independent geometry audit and one preselected confirmation visualization."""
import csv,json,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'exploration_v9'),str(ROOT/'exploration_v6')]
from run_v9 import RUNS,PARENT,INPUTS,read,sha,save
from audit_metrics import synthetic_scores

def main():
    dest=Path(tempfile.mkdtemp(prefix='spatial-v11-audit-',dir=RUNS));summary=[]
    for stage,name in (('development','spatial-v11-development-zh54hd3y'),('confirmation','spatial-v11-confirmation-xjro7mgm')):
        root=RUNS/name;inputs=INPUTS if stage=='development' else root/'inputs'
        rows=list(csv.DictReader((root/'RESULTS.csv').open()));worst=0.
        for row in rows:
            assert sha(row['output'])==row['output_sha256'];out=read(row['output'])
            ev=read(inputs/'evaluation'/(row['case']+'.eval.npz'));ev.update(json.loads(ev.pop('json').tobytes().decode()))
            scores=synthetic_scores(out['xyz_world'],ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],ev['gt_layer'],float(row['gap_mm']))
            for k in ('surface_accuracy_mean_mm','matched_point_rms_mm','fitted_gap_at_same_xy_error_mm'):
                if k in scores:
                    d=abs(scores[k]-float(row[k]));assert d<1e-8;worst=max(worst,d)
            if not row['method'].startswith('old_'):
                sp=(PARENT if stage=='development' else root)/'states'/(row['case']+'.npz');state=read(sp)
                pred=out['prediction_world_order_mm'][state['order']];take=np.flatnonzero(state['support']);expected=state['world'].copy()
                expected[state['order'][take]]+=(pred[take]-state['local'][take,2])[:,None]*state['normal']/1000.
                np.testing.assert_array_equal(expected,out['xyz_world'])
        history=json.loads((root/'HISTORY_BEFORE.json').read_text());assert all(sha(p)==h for p,h in history.items())
        for p,h in json.loads((root/'SOURCES.json').read_text()).items():assert sha(root/'source'/Path(p).name)==h
        indexed={(r['case'],r['method']):r for r in rows};paired={}
        cases=sorted(set(r['case'] for r in rows))
        for baseline in ('old_original','old_multistart'):
            paired[baseline]={}
            for metric in ('surface_accuracy_mean_mm','matched_point_rms_mm'):
                dif=np.array([float(indexed[(c,'spatial_free_i36')][metric])-float(indexed[(c,baseline)][metric]) for c in cases])
                paired[baseline][metric]=dict(better=int(sum(dif < -1e-8)),tied=int(sum(abs(dif)<=1e-8)),worse=int(sum(dif>1e-8)),mean_difference_mm=float(dif.mean()))
        summary.append(dict(stage=stage,outputs_checked=len(rows),max_metric_difference_mm=worst,xyz_reconstructed_exactly=True,
            historical_files_unchanged=len(history),frozen_sources_unchanged=True,paired=paired))
    save(dest/'AUDIT.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=RUNS/'spatial-v11-confirmation-xjro7mgm';case='dual_g2_s9131011_b4'
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),sharex=True,sharey=True)
    for ax,label,p in zip(axes,('Measured input','Old original','Spatial free, 36 iterations'),
        (root/'inputs'/(case+'.npz'),root/'outputs'/(case+'__old_original.npz'),root/'outputs'/(case+'__spatial_free_i36.npz'))):
        xyz=read(p)['xyz_world']*1000;ax.scatter(xyz[:,0],xyz[:,2],s=3,alpha=.45)
        ax.plot([-60,15],[0,0],c='black',lw=2);ax.plot([-15,60],[2,2],c='black',lw=2)
        ax.set_title(label);ax.set_xlabel('X (mm)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Z (mm)');fig.suptitle('Preselected fresh case: 2 mm layers, 4 mm frame bias; black = GT surfaces')
    fig.tight_layout();fig.savefig(dest/'confirmation_example.png',dpi=170);plt.close(fig)
    print(dest);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

"""Identity readout and presentation QA; no fitting, rescoring selection or new data."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import v14
from models import METHODS
BLUE,GOLD,INK='#2864B4','#B78516','#303640'


def main(path,figures_only=False):
    run=Path(path);dest=run/'readout'
    jobs=json.loads((run/'JOBS_LOCK.json').read_text())
    identity=[]
    for j in jobs:
        inp,e=v14.load(j['input']),v14.load(j['evaluation'])
        q=inp['xyz_world']*1000.;ref=e['gt_clean_xyz_world']*1000.;label=e['gt_layer'];means=e['true_means_mm']
        r={k:j[k] for k in ('case','seed','gap','sigma','dependence','n')}
        r.update(method='identity',surface_mae_mm=float(np.min(abs(q[:,2,None]-means),axis=1).mean()),
                 matched_rms_mm=float(np.sqrt(np.mean(np.sum((q-ref)**2,axis=1)))),
                 balanced_source_mae_mm=float(np.mean([np.mean(abs(q[label==k,2]-means[k])) for k in np.unique(label)])),
                 coverage_1mm=float(np.mean(cKDTree(q).query(ref)[0]<=1.+1e-9)))
        if j['gap']:
            gap=q[label==1,2].mean()-q[label==0,2].mean()
            r.update(source_gap_mm=float(gap),source_gap_error_mm=float(abs(gap-j['gap'])))
        identity.append(r)
    if not figures_only:
        v14.save(dest/'IDENTITY_ROWS.json',identity);v14.csvsave(dest/'IDENTITY_RESULTS.csv',identity)
    summary=json.loads((dest/'SUMMARY.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.2),sharey=True,layout='constrained')
    for ax,(gap,sigma) in zip(axes.ravel(),[(g,s) for g in (2.,8.) for s in (1.,2.)]):
        for setting,color,marker,offset,label in [('free',BLUE,'o',-.07,'Free slope'),('oracle_slope',GOLD,'s',.07,'Known slope (diagnostic)')]:
            for lam in (0.,.5,1.):
                p=summary['slices'][f'g{gap:g}_n{sigma:g}'][str(lam)][setting]
                values=list(p['seed_values'].values())
                ax.scatter(lam+offset+np.linspace(-.018,.018,len(values)),values,color=color,marker=marker,s=14,alpha=.3)
                ax.errorbar(lam+offset,p['mean_gain'],yerr=[[p['mean_gain']-p['ci95_low']],[p['ci95_high']-p['mean_gain']]],
                     color=color,marker=marker,markersize=6,linewidth=1.4,capsize=4,label=label if lam==0 else None)
        ax.axhline(0.,color=INK,ls='--',linewidth=.8);ax.grid(axis='y',color='#E4E6E8',linewidth=.6)
        ax.set(title=f'Gap {gap:g} mm | noise sigma {sigma:g} mm',xlabel='Nominal dependence intervention (lambda)',
               ylabel='Constant MAE - spatial MAE (mm)',xticks=[0.,.5,1.],xlim=(-.2,1.2),ylim=(-1.08,1.52))
        ax.legend(fontsize=8,loc='lower left' if gap==8 and sigma==1 else 'upper left',frameon=False)
        if gap==8 and sigma==1:
            ax.text(.02,.91,'Differences are near zero\n(mean magnitude < 0.000021 mm)',transform=ax.transAxes,fontsize=9,color=INK)
    fig.suptitle('V16: paired surface-error differences (common vertical scale)\n10 seeds; points = seed pairs; bars = 95% t intervals; positive favors spatial',fontsize=12)
    fig.savefig(dest/'paired_mechanism_shared_scale.png',dpi=160);plt.close(fig)
    # Supplementary condition chosen AFTER aggregated results, never replacing the preregistered example.
    records=json.loads((run/'mechanism_SEALED_BEFORE_GT.json').read_text());rix={(r['case'],r['method']):r for r in records}
    rows=json.loads((run/'mechanism_ROWS.json').read_text());ix={(r['case'],r['method']):r for r in rows}
    case='s9161101_g8_n2_l1';rec=rix[case,'constant_free'];inp=v14.load(rec['input']);truth=v14.load(rec['evaluation'])
    labels=truth['gt_layer'];fig,axes=plt.subplots(1,5,figsize=(18,4.5),sharex=True,sharey=True,layout='constrained')
    for ax,method in zip(axes,('input',)+METHODS):
        q=(inp if method=='input' else v14.load(rix[case,method]['output']))['xyz_world']*1000.
        for k,color,marker in [(0,BLUE,'o'),(1,GOLD,'s')]:
            ax.scatter(q[labels==k,0],q[labels==k,2],s=7,color=color,marker=marker,alpha=.45)
        for h in (0.,8.):ax.axhline(h,color=INK,ls='--',linewidth=.7)
        title=method.replace('_oracle_slope','\nknown slope').replace('_free','\nfree slope')
        if method!='input':title+=f'\nMAE {ix[case,method]["surface_mae_mm"]:.3f} mm; K={ix[case,method]["model_k"]}'
        ax.set(title=title,xlabel='X (mm)',xlim=(-63,63),ylim=(-6.5,15.))
    axes[0].set_ylabel('Z (mm)')
    fig.suptitle('V16: supplementary mechanism example (gap 8 mm, sigma 2 mm, lambda 1)\nCondition selected after aggregate analysis; first seed, not best seed; GT colors evaluation-only; axes not equal-aspect',fontsize=12)
    fig.savefig(dest/'supplementary_slope_example.png',dpi=150);plt.close(fig)
    v14.save(dest/('PRESENTATION_QA_LEGEND.json' if figures_only else 'PRESENTATION_QA.json'),dict(primary_common_scale=True,old_auto_scaled_figure_retained=True,
                predefined_example='s9161101_g8_n1_l0 / l1',supplementary_example=case,
                supplementary_selection='condition selected after aggregate diagnostics, fixed first seed, descriptive only',
                identity_scored=len(identity),new_fits=0,legend_repositioned_to_avoid_points=figures_only))


if __name__=='__main__':main(sys.argv[1],'--figures-only' in sys.argv[2:])

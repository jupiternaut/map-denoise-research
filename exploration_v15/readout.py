"""Post-run descriptive diagnostics and figures; no parameter changes or selection."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import v14,aggregate


def main(path):
    run=Path(path);dest=run/'readout';dest.mkdir()
    dev=json.loads((run/'development_AGGREGATES.json').read_text())
    confirm=json.loads((run/'confirmation_AGGREGATES.json').read_text())
    rows=json.loads((run/'confirmation_ROWS.json').read_text())
    ix={(r['case'],r['method']):r for r in rows}
    records=json.loads((run/'confirmation_SEALED_BEFORE_GT.json').read_text())
    rix={(r['case'],r['method']):r for r in records}
    comparisons={}
    for other in ('constant_36','constant_lbfgs64','rimls_8.0','apss_4.0','old_original'):
        entries=[]
        for r in rows:
            if r['method']!='spatial_36':continue
            b=ix[(r['case'],other)]
            entries.append(dict(case=r['case'],gap=r['gap'],sigma=r['sigma'],retain=r['retain'],
                delta_mae=r['surface_accuracy_mean_mm']-b['surface_accuracy_mean_mm'],
                delta_gap=r.get('gap_absolute_relative_error',np.nan)-b.get('gap_absolute_relative_error',np.nan)))
        comparisons[other]=dict(surface_better=sum(r['delta_mae'] < -1e-8 for r in entries),
            surface_same=sum(abs(r['delta_mae'])<=1e-8 for r in entries),
            surface_worse=sum(r['delta_mae']>1e-8 for r in entries),
            dual_joint_better_or_equal=sum(r['gap']>0 and r['delta_mae']<=1e-8 and r['delta_gap']<=1e-8 for r in entries),
            dual_total=sum(r['gap']>0 for r in entries))
    solver=[]
    for r in records:
        if r['method']!='constant_lbfgs64':continue
        info=r['info'];case=r['case'];a=ix[(case,'constant_lbfgs64')];b=ix[(case,'constant_36')]
        solver.append(dict(case=case,nll_improvement=info['initial_two_plane_nll']-info['final_two_plane_nll'],
            versus_36_score_improvement=rix[(case,'constant_36')]['info']['score']-info['score'],
            mae_delta=a['surface_accuracy_mean_mm']-b['surface_accuracy_mean_mm'],
            successful_starts=sum(q['success'] for q in info['solver_diagnostics']),starts=info['starts']))
    v14.save(dest/'SOLVER_DIAGNOSTICS.json',solver)
    v14.save(dest/'PAIRED_SUMMARY.json',comparisons)
    budgets={m:{k:dev[m][k] for k in ['surface_accuracy_mean_mm','gap_absolute_relative_error']}
             for m in dev if m.startswith(('constant_','spatial_'))}
    v14.save(dest/'BUDGET_DIAGNOSTIC.json',budgets)
    # Two-objective development Pareto set is descriptive over tested configurations only.
    candidates=[m for m in dev if dev[m]['failed']==0]
    front=[]
    for m in candidates:
        x=dev[m]['surface_accuracy_mean_mm'];y=dev[m]['gap_absolute_relative_error']
        if not any(dev[n]['surface_accuracy_mean_mm']<=x and dev[n]['gap_absolute_relative_error']<=y and
                   (dev[n]['surface_accuracy_mean_mm']<x or dev[n]['gap_absolute_relative_error']<y) for n in candidates if n!=m):
            front.append(m)
    v14.save(dest/'DEVELOPMENT_FRONTIER.json',dict(methods=front,axes=['mean_surface_mae','mean_absolute_relative_source_gap_error'],
        scope='tested global configurations only; not all-method optimum; all-scene error and dual-only gap'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    for family,color in [('constant','#d97925'),('spatial','#187a6b')]:
        axes[0].plot([36,108,324],[dev[f'{family}_{i}']['surface_accuracy_mean_mm'] for i in (36,108,324)],
                     'o-',label=family,color=color)
    axes[0].axhline(dev['constant_lbfgs64']['surface_accuracy_mean_mm'],ls='--',color='#d97925',label='constant, 64-start L-BFGS')
    axes[0].set(xscale='log',xlabel='EM iterations (6 starts)',ylabel='Mean surface error (mm)',title='Development: solver budget')
    axes[0].legend(fontsize=8)
    scales=[.5,1,2,4,8,16,32,64,128]
    axes[1].plot(scales,[dev[f'rimls_{s:.1f}']['surface_accuracy_mean_mm'] for s in scales],'o-',label='RIMLS',color='#6950a1')
    axes[1].axhline(dev['spatial_36']['surface_accuracy_mean_mm'],color='#187a6b',ls='--',label='spatial 36')
    axes[1].set(xscale='log',xlabel='RIMLS filter scale',ylabel='Mean surface error (mm)',title='Development: expanded scale search')
    axes[1].legend(fontsize=8)
    for method,label,color,offset in [('spatial_36','Spatial','#187a6b',(7,4)),('rimls_8.0','RIMLS 8','#6950a1',(7,4)),
        ('apss_4.0','APSS 4','#4284b8',(7,4)),('constant_lbfgs64','Constant 64-start','#d97925',(-116,6)),
        ('old_original','Old original','#6a7075',(7,4))]:
        x=100*confirm[method]['gap_absolute_relative_error'];y=confirm[method]['surface_accuracy_mean_mm']
        axes[2].scatter([x],[y],s=65,color=color)
        axes[2].annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=9)
    axes[2].set(xlabel='Mean absolute source-gap error (%)',ylabel='Mean surface error (mm)',
        title='New seeds: accuracy / gap trade-off',xlim=(2,13),ylim=(.12,.55))
    fig.suptitle('V15: stronger baselines; three new seeds, same synthetic family, supplied noise scale',fontsize=12)
    fig.savefig(dest/'challenge_summary.png',dpi=160);plt.close(fig)
    # First fixed regular and difficult conditions, not the best/worst chosen by scores.
    for gap,sigma,retain in ((4,1.,1.),(2,2.,.25)):
        case=f's9151101_g{gap}_n{sigma}_b4_r{retain}'
        refrow=rix[(case,'spatial_36')];e=v14.load(refrow['evaluation']);e.update(json.loads(e.pop('json').tobytes().decode()))
        ref=e['gt_clean_xyz_world']*1000;labels=e['gt_layer']
        methods=['identity','constant_lbfgs64','rimls_8.0','spatial_36']
        allq={m:v14.load(rix[(case,m)]['output'])['xyz_world']*1000 for m in methods}
        limits=np.concatenate([q[:,2] for q in allq.values()])
        fig,axes=plt.subplots(1,4,figsize=(16,4),sharex=True,sharey=True,layout='constrained')
        for ax,m in zip(axes,methods):
            q=allq[m]
            ax.scatter(ref[:,0],ref[:,2],s=4,color='black',alpha=.25,label='clean reference')
            ax.scatter(q[:,0],q[:,2],c=labels,cmap='coolwarm',s=6,alpha=.55,vmin=0,vmax=1)
            ax.set(xlabel='world X (mm)',ylim=(limits.min()-.5,limits.max()+.5),
                title=f'{m}\nMAE={ix[(case,m)]["surface_accuracy_mean_mm"]:.3f} mm')
        axes[0].set_ylabel('world Z (mm)')
        fig.suptitle(case+'; colors are GT source labels for evaluation only',fontsize=12)
        fig.savefig(dest/f'{case}.png',dpi=150);plt.close(fig)
    stats=dict(comparisons=comparisons,development_frontier=front,
        solver_nll_improved=sum(r['nll_improvement']>1e-6 for r in solver),
        solver_score_better_but_geometry_worse=sum(r['versus_36_score_improvement']>1e-6 and r['mae_delta']>1e-8 for r in solver),
        solver_nonconverged_starts=sum(r['starts']-r['successful_starts'] for r in solver),
        solver_total_starts=sum(r['starts'] for r in solver))
    v14.save(dest/'SUMMARY.json',stats)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main(sys.argv[1])

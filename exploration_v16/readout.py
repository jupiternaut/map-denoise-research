"""Descriptive, seed-paired readout. Never changes models or chooses a new winner."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import v14
from models import METHODS

BLUE='#2864B4'
GOLD='#B78516'
INK='#303640'


def average(rows):
    result=dict(cases=len(rows))
    keys=sorted(set(k for r in rows for k,v in r.items() if isinstance(v,(float,int)) and k not in ('seed',)))
    for key in keys:
        vals=[r[key] for r in rows if isinstance(r.get(key),(float,int))]
        if vals:result[key]=float(np.mean(vals))
    for key in ('model_k','k'):
        if any(key in r for r in rows):
            result['selected_single_count']=sum(r.get(key)==1 for r in rows)
            result['selected_double_count']=sum(r.get(key)==2 for r in rows)
    return result


def paired(rows,known,metric='surface_mae_mm'):
    name='oracle_slope' if known else 'free'
    index={(r['case'],r['method']):r for r in rows}
    pairs=[]
    for r in rows:
        if r['method']!='constant_'+name:continue
        other=index[r['case'],'spatial_'+name]
        pairs.append(dict(case=r['case'],seed=r['seed'],gap=r['gap'],sigma=r['sigma'],dependence=r['dependence'],
                          constant=r[metric],spatial=other[metric],gain=r[metric]-other[metric],
                          constant_k=r['model_k'],spatial_k=other['model_k']))
    seeds=sorted(set(r['seed'] for r in pairs))
    values=np.array([np.mean([r['gain'] for r in pairs if r['seed']==seed]) for seed in seeds])
    mean=float(values.mean())
    half=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
    return dict(metric=metric,positive_means_spatial_better=True,seed_count=len(seeds),condition_pairs=len(pairs),
                mean_gain=mean,ci95_low=mean-half,ci95_high=mean+half,
                seed_values=dict(zip(map(str,seeds),map(float,values))),
                better=sum(r['gain']>1e-8 for r in pairs),same=sum(abs(r['gain'])<=1e-8 for r in pairs),
                worse=sum(r['gain'] < -1e-8 for r in pairs),pairs=pairs)


def main(path):
    run=Path(path);dest=run/'readout';dest.mkdir()
    rows=json.loads((run/'mechanism_ROWS.json').read_text())
    replay=json.loads((run/'replay_ROWS.json').read_text())
    assert all(r['status']=='OK' for r in rows+replay)
    dual=[r for r in rows if r['gap']>0]
    single=[r for r in rows if r['gap']==0]
    agg={str(lam):{m:average([r for r in dual if r['dependence']==lam and r['method']==m]) for m in METHODS}
         for lam in (0.,.5,1.)}
    neg={m:average([r for r in single if r['method']==m]) for m in METHODS}
    pair_summary={str(lam):{label:paired([r for r in dual if r['dependence']==lam],oracle)
                           for label,oracle in [('free',False),('oracle_slope',True)]} for lam in (0.,.5,1.)}
    slices={f'g{gap:g}_n{sigma:g}':{str(lam):{label:paired([r for r in dual if r['gap']==gap and r['sigma']==sigma and r['dependence']==lam],oracle)
                 for label,oracle in [('free',False),('oracle_slope',True)]} for lam in (0.,.5,1.)}
            for gap in (2.,8.) for sigma in (1.,2.)}
    ix={(r['case'],r['method']):r for r in rows}
    transitions=[]
    for r in dual:
        if not r['method'].endswith('_free'):continue
        other=ix[r['case'],r['method'].replace('_free','_oracle_slope')]
        transitions.append(dict(case=r['case'],seed=r['seed'],gap=r['gap'],sigma=r['sigma'],dependence=r['dependence'],
                     family=r['method'].split('_')[0],free_k=r['model_k'],known_k=other['model_k'],
                     oracle_mae_change=other['surface_mae_mm']-r['surface_mae_mm']))
    same_k=[]
    for r in dual:
        if r['method']!='constant_free':continue
        if all(ix[r['case'],m]['model_k']==2 for m in METHODS):
            same_k.extend(ix[r['case'],m] for m in METHODS)
    same_k_summary={str(lam):{label:paired([r for r in same_k if r['dependence']==lam],oracle)
                  for label,oracle in [('free',False),('oracle_slope',True)]} for lam in (0.,.5,1.)}
    replay_agg={m:average([r for r in replay if r['gap']>0 and r['method']==m]) for m in ('constant_free','spatial_free')}
    replay_k2={m:average([r for r in replay if r['gap']>0 and r['method']==m and r['k']==2]) for m in ('constant_free','spatial_free')}
    replay_k1={m:average([r for r in replay if r['gap']>0 and r['method']==m and r['k']==1]) for m in ('constant_free','spatial_free')}
    correlations={str(lam):dict(mean=float(np.mean([r['actual_x_layer_correlation'] for r in dual if r['method']=='constant_free' and r['dependence']==lam])),
                         min=float(min(r['actual_x_layer_correlation'] for r in dual if r['method']=='constant_free' and r['dependence']==lam)),
                         max=float(max(r['actual_x_layer_correlation'] for r in dual if r['method']=='constant_free' and r['dependence']==lam))) for lam in (0.,.5,1.)}
    starts=[]
    for r in json.loads((run/'mechanism_SEALED_BEFORE_GT.json').read_text()):
        ds=r['info'].get('solver_diagnostics',[])
        starts.extend(dict(case=r['case'],method=r['method'],**d) for d in ds)
    summary=dict(aggregates=agg,single_controls=neg,paired=pair_summary,slices=slices,
                 correlations=correlations,replay_all_dual=replay_agg,replay_k2_only=replay_k2,replay_k1_only=replay_k1,
                 all_four_k2_subset=same_k_summary,solver_starts=len(starts),solver_nonconverged=sum(not r['success'] for r in starts),
                 note='t intervals over ten seed-averaged paired differences, not across real scenes; same-K subset is descriptive and selected after fitting')
    v14.save(dest/'SUMMARY.json',summary)
    v14.save(dest/'K_TRANSITIONS.json',transitions)
    v14.csvsave(dest/'K_TRANSITIONS.csv',transitions)
    v14.save(dest/'SOLVER_STARTS.json',starts)
    pair_rows=[]
    for lam,block in pair_summary.items():
        for setting,p in block.items():pair_rows.extend(dict(**r,setting=setting) for r in p['pairs'])
    v14.csvsave(dest/'PAIRED_POINTS.csv',pair_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#7B8189',
                         'xtick.color':INK,'ytick.color':INK,'axes.titleweight':'medium'})
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.2),layout='constrained')
    for ax,(gap,sigma) in zip(axes.ravel(),[(g,s) for g in (2.,8.) for s in (1.,2.)]):
        for setting,color,marker,offset,label in [('free',BLUE,'o',-.07,'Free slope'),('oracle_slope',GOLD,'s',.07,'Known slope (diagnostic)')]:
            for lam in (0.,.5,1.):
                p=slices[f'g{gap:g}_n{sigma:g}'][str(lam)][setting]
                values=list(p['seed_values'].values())
                xx=lam+offset+np.linspace(-.018,.018,len(values))
                ax.scatter(xx,values,color=color,marker=marker,s=14,alpha=.3)
                ax.errorbar(lam+offset,p['mean_gain'],yerr=[[p['mean_gain']-p['ci95_low']],[p['ci95_high']-p['mean_gain']]],
                            color=color,marker=marker,markersize=6,linewidth=1.4,capsize=4,label=label if lam==0 else None)
        ax.axhline(0.,color=INK,ls='--',linewidth=.8)
        ax.grid(axis='y',color='#E4E6E8',linewidth=.6)
        ax.set(title=f'Gap {gap:g} mm | noise sigma {sigma:g} mm',xlabel='Nominal dependence intervention (lambda)',
               ylabel='Constant MAE - spatial MAE (mm)',xticks=[0.,.5,1.],xlim=(-.2,1.2))
        ax.legend(fontsize=8,loc='best',frameon=False)
    fig.suptitle('V16: paired surface-error differences\n10 seeds; dots = seed pairs; bars = 95% t intervals; positive favors spatial',fontsize=12)
    fig.savefig(dest/'paired_mechanism.png',dpi=160);plt.close(fig)
    records=json.loads((run/'mechanism_SEALED_BEFORE_GT.json').read_text())
    rix={(r['case'],r['method']):r for r in records}
    cases=['s9161101_g8_n1_l0','s9161101_g8_n1_l1']
    fig,axes=plt.subplots(2,5,figsize=(18,7),sharex=True,sharey=True,layout='constrained')
    for line,case in enumerate(cases):
        rec=rix[case,'constant_free'];inp=v14.load(rec['input']);truth=v14.load(rec['evaluation']);labels=truth['gt_layer']
        for col,method in enumerate(('input',)+METHODS):
            ax=axes[line,col]
            q=(inp if method=='input' else v14.load(rix[case,method]['output']))['xyz_world']*1000.
            for k,color,marker in [(0,BLUE,'o'),(1,GOLD,'s')]:
                ax.scatter(q[labels==k,0],q[labels==k,2],s=6,color=color,marker=marker,alpha=.45)
            for h in (0.,8.):ax.axhline(h,color=INK,ls='--',linewidth=.7)
            title=method.replace('_oracle_slope','\nknown slope').replace('_free','\nfree slope')
            if method!='input':title+=f'\nMAE {ix[case,method]["surface_mae_mm"]:.3f} mm; K={ix[case,method]["model_k"]}'
            ax.set(title=title,xlabel='X (mm)',ylim=(-4,12),xlim=(-63,63))
        axes[line,0].set_ylabel(f'lambda = {line}\nZ (mm)')
    fig.suptitle('V16: fixed first-seed examples, gap 8 mm, sigma 1 mm\nAll 768 points; color/marker = true source layer (evaluation only); axes not equal-aspect',fontsize=12)
    fig.savefig(dest/'fixed_examples.png',dpi=150);plt.close(fig)
    compact=dict(paired={lam:{s:{k:v for k,v in q.items() if k not in ('pairs','seed_values')} for s,q in b.items()} for lam,b in pair_summary.items()},
                 correlations=correlations,replay_all_dual=replay_agg,replay_k2_only=replay_k2,
                 single_controls=neg,solver_starts=len(starts),solver_nonconverged=summary['solver_nonconverged'])
    print(json.dumps(compact,indent=2),flush=True)


if __name__=='__main__':main(sys.argv[1])

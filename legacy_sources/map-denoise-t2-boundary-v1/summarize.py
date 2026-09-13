"""Summarize saved paired outputs, without selecting a winner per test case."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704')
DEVELOPMENT=['old-methods','icp-method','profile-methods','alternative-methods',
             'candidate-methods','proposal-methods-02','plane-method','normal-baseline']
METRICS=['normal_mae_mm','point_rmse_mm','nearest_surface_mae_mm','gap_abs_error_mm',
         'gap_retention','layer_confusion_fraction','min_layer_accuracy']
LABELS={'old:joint_forced':'Old joint', 'baseline:scalar_profile_hard':'Profile hard',
        'baseline:scalar_profile_lbfgs':'Profile soft', 'proposal:framewise_proposal_hard':'Framewise MAP',
        'plane:plane_profile_map':'Joint planes',
        'normal:within_frame_normal_then_profile':'Normal then profile'}


def aggregate(rows):
    out={'n':len(rows),'mean_ms':1000*np.mean([r['seconds'] for r in rows]),
         'median_ms':1000*np.median([r['seconds'] for r in rows]),
         'mean_points':np.mean([r['config']['points_per_frame']*r['config']['frames'] for r in rows])}
    for m in METRICS:
        values=[r['metrics'][m] for r in rows if r['metrics'].get(m) is not None]
        out[m]=float(np.mean(values)) if values else None
    return out


def summarize(directories,stage):
    rows=[];errors=[]
    for d in directories:
        p=RUN/d/'results.json'
        if not p.exists():continue
        doc=json.loads(p.read_text());rows.extend(doc['records'])
    errors=[r for r in rows if r['status']=='ERROR'];rows=[r for r in rows if r['status']!='ERROR']
    methods=list(dict.fromkeys(r['method'] for r in rows));groups=list(dict.fromkeys(r['group'] for r in rows))
    table={g:{m:aggregate(rr) for m in methods if (rr:=[r for r in rows if r['group']==g and r['method']==m])} for g in groups}
    core={str(gap):{m:aggregate(rr) for m in methods if (rr:=[r for r in rows if r['group']=='core' and r['config']['gap']==gap and r['method']==m])} for gap in sorted({r['config']['gap'] for r in rows if r['group']=='core'})}
    by_case={}
    for r in rows:
        key=str({k:v for k,v in r['config'].items() if k not in ('frames',)})
        by_case.setdefault(key,{}).setdefault(r['method'],[]).append(r)
    cases=[{'config':next(iter(methods_rows.values()))[0]['config'],
            'methods':{m:aggregate(rr) for m,rr in methods_rows.items()}} for methods_rows in by_case.values()]
    result={'stage':stage,'records':len(rows),'errors':errors,'groups':table,'core_gap':core,'cases':cases}
    with (ROOT/f'{stage}-summary.json').open('w') as f:json.dump(result,f,indent=2)
    lines=[f'# {stage}: saved-output summary','',f'{len(rows)} outputs; {len(errors)} errors. Means over the stated synthetic cases, not real-scene accuracy.','',
           '| Group | Method | n | Normal MAE (mm) | 3D RMSE (mm) | Gap error (mm) | Mean ms |',
           '|---|---|---:|---:|---:|---:|---:|']
    for g,mt in table.items():
        for m,v in mt.items():
            ge='—' if v['gap_abs_error_mm'] is None else f"{v['gap_abs_error_mm']:.4f}"
            lines.append(f"| {g} | {m} | {v['n']} | {v['normal_mae_mm']:.4f} | {v['point_rmse_mm']:.4f} | {ge} | {v['mean_ms']:.2f} |")
    (ROOT/f'{stage}-tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'stage':stage,'records':len(rows),'errors':len(errors)}))
    return result


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for m,label in LABELS.items():
        values=[(float(g),mset[m]['normal_mae_mm']) for g,mset in result['core_gap'].items() if m in mset]
        if values:axes[0].plot(*zip(*values),marker='o',label=label)
    axes[0].set(xlabel='Layer gap / noise sigma (sigma = 1 mm)',ylabel='Normal MAE (mm)',title='Fixed-normal specialization: 18 cases per gap')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.25)
    for m,label in LABELS.items():
        values=[(c['config']['normal_degrees'],c['methods'][m]['normal_mae_mm']) for c in result['cases'] if c['config']['normal_degrees'] and m in c['methods']]
        if values:
            values.sort();axes[1].plot(*zip(*values),marker='o',label=label)
    axes[1].set(xlabel='Initial normal error (degrees)',ylabel='Normal MAE (mm)',title='Normal estimation: 3 seeds per angle')
    axes[1].grid(alpha=.25)
    fig.savefig(out/'specialized-gains.png',dpi=170);plt.close(fig)
    # Preselected illustrative input: gap=6, 3-degree normal error, first seed.
    data=RUN/'development-data';manifest=json.loads((data/'manifest.json').read_text())
    entry=next(e for e in manifest['entries'] if e['group']=='normal_degrees' and e['config']['normal_degrees']==3 and e['seed']==31013)
    key=entry['key'];truth=np.load(data/f'{key}-EVAL.npz');rotation=truth['rotation'];raw=np.load(data/f'{key}.npz')['xyz_mm']@rotation
    examples=[('Input',raw),('Old fixed-normal joint',np.load(RUN/'old-methods/outputs'/f'{key}__joint_forced.npz')['xyz_mm']@rotation),
              ('Joint plane filter',np.load(RUN/'plane-method/outputs'/f'{key}__plane_profile_map.npz')['xyz_mm']@rotation)]
    fig,axes=plt.subplots(1,3,figsize=(11,3),sharex=True,sharey=True,constrained_layout=True)
    for ax,(title,points) in zip(axes,examples):
        ax.scatter(points[:,0],points[:,2],s=2,c=truth['labels'],cmap='coolwarm',vmin=0,vmax=1,alpha=.5)
        for z in (0,6):ax.axhline(z,color='black',lw=.8,ls='--')
        ax.set(title=title,xlabel='x (mm)',ylim=(-7,13));ax.grid(alpha=.2)
    axes[0].set_ylabel('z (mm); dashed lines = ground truth')
    fig.savefig(out/'normal-repair-example.png',dpi=170);plt.close(fig)
    # First fixed-normal replication configuration and first seed, not the best output.
    data=RUN/'replication-data';key='000-73013'
    if data.exists():
        truth=np.load(data/f'{key}-EVAL.npz');rotation=truth['rotation']
        methods=[('Input',np.load(data/f'{key}.npz')['xyz_mm']@rotation),
                 ('Old joint',np.load(RUN/'replication-all/outputs'/f'{key}__joint_forced.npz')['xyz_mm']@rotation),
                 ('Fast profile filter',np.load(RUN/'replication-all/outputs'/f'{key}__scalar_profile_hard.npz')['xyz_mm']@rotation)]
        fig,axes=plt.subplots(1,3,figsize=(11,3),sharex=True,sharey=True,constrained_layout=True)
        for ax,(title,points) in zip(axes,methods):
            ax.scatter(points[:,0],points[:,2],s=3,c=truth['labels'],cmap='coolwarm',vmin=0,vmax=1,alpha=.5)
            for z in (0,3.5):ax.axhline(z,color='black',lw=.8,ls='--')
            ax.set(title=title,xlabel='x (mm)',ylim=(-6,10));ax.grid(alpha=.2)
        axes[0].set_ylabel('z (mm); dashed lines = ground truth')
        fig.savefig(out/'fast-filter-example.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','replication'],default='development');p.add_argument('--figures',action='store_true');args=p.parse_args()
    dirs=DEVELOPMENT if args.stage=='development' else ['replication-all']
    result=summarize(dirs,args.stage)
    if args.figures:figures(result)

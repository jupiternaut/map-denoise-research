"""Read saved outputs/results only; no candidate selection or output mutation."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--extension',required=True);ap.add_argument('--scale');a=ap.parse_args()
    run,ext=Path(a.run),Path(a.extension);rows=json.loads((run/'results.json').read_text())['records']+json.loads((ext/'results.json').read_text())['records']
    scale=Path(a.scale) if a.scale else None
    if scale:rows+=json.loads((scale/'results.json').read_text())['records']
    good=[r for r in rows if r['status']=='ok'];groups=defaultdict(list)
    for r in good:groups[('all',r['arm'])].append(r);groups[(r['family'],r['arm'])].append(r)
    summary=[]
    for (family,arm),rr in groups.items():
        x={'family':family,'arm':arm,'n_cases':len(rr),'oracle':rr[0]['oracle']}
        for key in ['combined_mm','surface_mean_mm','coverage_mean_mm','min_component_recall','gap_absolute_error_mm','gap_intrusion_fraction','slot_intrusion_fraction','slot_edge_recall','rod_radius_error_mm','rod_recall']:
            values=[r['metrics'][key] for r in rr if r['metrics'].get(key) is not None]
            x[key]=float(np.mean(values)) if values else None
        x['seconds_median']=float(np.median([r['elapsed_s'] for r in rr]));summary.append(x)
    (run/'summary.json').write_text(json.dumps({'summary':summary,'extension':str(ext),'scale':str(scale) if scale else None,'errors':[r for r in rows if r['status']!='ok']},indent=2))
    with open(run/'summary.csv','w') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    for family in ['all','parallel_plates','slotted_sheet','concentric_shells','rod_plate']:
        print('\n'+family)
        for x in sorted([x for x in summary if x['family']==family],key=lambda x:x['combined_mm']):
            print(x['arm'],round(x['combined_mm'],5),round(x['surface_mean_mm'],5),round(x['coverage_mean_mm'],5),round(x['seconds_median'],4))
    arms=['identity','bilateral','old_A','pursuit_reference','pcl_mls_k8','pcl_mls_k12','pcl_mls_k16','pathnet_official']
    if scale:arms.insert(3,'old_A_k96_em16')
    families=['parallel_plates','slotted_sheet','concentric_shells','rod_plate']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ai,arm in enumerate(arms):
        data=[next(x for x in summary if x['family']==f and x['arm']==arm) for f in families]
        axes[0].plot(range(4),[x['surface_mean_mm'] for x in data],marker='o',label=arm)
        axes[1].plot(range(4),[x['min_component_recall'] for x in data],marker='o',label=arm)
    for ax in axes:ax.set_xticks(range(4),families,rotation=15,ha='right');ax.grid(alpha=.25)
    axes[0].set_ylabel('Mean distance to true surface (mm), lower better')
    axes[1].set_ylabel('Minimum component recall, higher better')
    axes[0].legend(fontsize=7);fig.suptitle('Development only: 4 procedural families, 3 conditions, 2 seeds')
    fig.tight_layout();fig.savefig(run/'family_metrics.png',dpi=150);plt.close(fig)
    # Explicitly representative resolved example and difficult curved example, not winner selection.
    shown=['identity','old_A','pursuit_reference','pathnet_official']
    if scale:shown.insert(2,'old_A_k96_em16')
    fig,axes=plt.subplots(2,len(shown),figsize=(3.25*len(shown),7))
    for ri,case in enumerate(['parallel_plates-resolved-s0','concentric_shells-resolved-s0']):
        with np.load(run/'outputs'/f'{case}-input.npz') as d:rot=d['rotation']
        for ci,arm in enumerate(shown):
            folder=ext if arm=='pathnet_official' else (scale if arm=='old_A_k96_em16' else run)
            path=folder/'outputs'/f'{case}-{arm}.npy'
            p=np.load(path)@rot;ax=axes[ri,ci]
            if ri==0:
                ax.scatter(p[:,0]*1000,p[:,2]*1000,s=1,alpha=.4);ax.set_ylim(-10,10);ax.set_xlim(-75,75);ax.set_ylabel('z (mm)')
                for h in (-4.5,4.5):ax.axhline(h,color='black',linestyle='--',linewidth=.6)
            else:
                ax.scatter(p[:,0]*1000,p[:,1]*1000,s=1,alpha=.4);ax.set_aspect('equal');ax.set_ylabel('y (mm)')
                angle=np.linspace(0,2*np.pi,300)
                for radius in (35.5,44.5):ax.plot(radius*np.cos(angle),radius*np.sin(angle),color='black',linestyle='--',linewidth=.6)
                ax.set_xlim(-50,50);ax.set_ylim(-50,50)
            ax.set_title(arm);ax.set_xlabel('x (mm)')
    fig.suptitle('Fixed illustrations: resolved double plane and curved shells, seed 0')
    fig.tight_layout();fig.savefig(run/'representative_outputs.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()

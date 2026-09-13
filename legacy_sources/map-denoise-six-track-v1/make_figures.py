"""Advisor figures from saved outputs; fixed example policy, no best-case search."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BASE=Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600')
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','axes.titleweight':'bold'})

# First post-freeze seed, selected by order, not improvement magnitude.
t2=BASE/'t1_t2/root-newseed-check-01'
key='crossed_imbalanced_s62011'
inp=np.load(t2/'inputs'/f'{key}.npz'); m=inp['roi']==0
frame=inp['frame'][m]; jitter=inp['xyz_m'][m,0]*5
fig, axes=plt.subplots(1,3,figsize=(12.4,3.7),sharey=True,constrained_layout=True)
for ax,method,title in zip(axes,['identity','frame_center_then_xyz','joint_guarded'],
                           ['Observed points','Frame means + mixture','Joint bias + layers']):
    p=np.load(t2/'outputs'/f'{key}__{method}.npz')['xyz_m'][m]*1000
    ax.scatter(frame+jitter,p[:,2],s=3,c=frame,cmap='viridis',alpha=.35,rasterized=True)
    for z in [0,8]:ax.axhline(z,color='#202833',ls='--',lw=1)
    ax.set(title=title,xlabel='Frame ID',ylim=(-7,15),xticks=[0,3,6,9])
axes[0].set_ylabel('Normal position (mm)')
fig.suptitle('Restricted model: fixed normal, scalar frame bias, unbalanced layer sampling',fontsize=12)
fig.savefig(FIG/'t2-crossed-frames.png',dpi=170);plt.close(fig)

t3=BASE/'t3_t6/development-0bx4br9p/outputs'
fig,axes=plt.subplots(2,4,figsize=(14.4,6.0),constrained_layout=True)
for row,family in enumerate(['parallel_plates','concentric_shells']):
    key=f'{family}-resolved-s0'; data=np.load(t3/f'{key}-input.npz');r=data['rotation']
    for col,method in enumerate(['input','old_A','old_A_k96_em16','pursuit_reference']):
        source = BASE/'t3_t6/scale-ablation-0fs7n0gm/outputs' if method=='old_A_k96_em16' else t3
        p=(data['xyz'] if method=='input' else np.load(source/f'{key}-{method}.npy'))@r*1000
        ax=axes[row,col]
        if family=='parallel_plates':
            ax.scatter(p[:,0],p[:,2],s=2,alpha=.35,color='#3977a8',rasterized=True)
            for z in [-4.5,4.5]:ax.axhline(z,color='#bd513c',ls='--',lw=1)
            ax.set(xlim=(-75,75),ylim=(-10,10),xlabel='x (mm)',ylabel='z (mm)')
        else:
            ax.scatter(p[:,0],p[:,1],s=2,alpha=.35,color='#3977a8',rasterized=True)
            a=np.linspace(0,2*np.pi,300)
            for radius in [35.5,44.5]:ax.plot(radius*np.cos(a),radius*np.sin(a),'--',color='#bd513c',lw=1)
            ax.set(xlim=(-52,52),ylim=(-52,52),xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
        if row==0:ax.set_title(['Input','Old mixture: k48','Simple reference: k96','New slow reference'][col])
fig.suptitle('Fixed examples: larger neighborhoods explain planar gain but can damage curvature',fontsize=12)
fig.savefig(FIG/'t3-planar-and-curved.png',dpi=170);plt.close(fig)

scale=json.loads((BASE/'t5/scale-results.json').read_text())
times=[float(np.median([x['seconds'] for x in row['timings']])) for row in scale[:2]]
fig,ax=plt.subplots(figsize=(6.0,3.5),constrained_layout=True)
bars=ax.barh(['Reference Python','Fused C++'],times,color=['#8c9baa','#2a8a74'])
for b,v in zip(bars,times):ax.text(v+.04,b.get_y()+b.get_height()/2,f'{v:.3f} s',va='center')
ax.set(xlim=(0,max(times)*1.25),xlabel='Median local runtime (seconds)')
ax.set_title('10,000 points; CPU only\nSame 32 EM iterations', fontsize=13)
ax.invert_yaxis()
fig.savefig(FIG/'t5-cost.png',dpi=170);plt.close(fig)
print(str(FIG))

"""Plot actual paired-query results only; no smoothing or chosen example ROI."""
from pathlib import Path
import importlib.util,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
template=Path('/home/grf/.codex/skills/kdense-matplotlib/scripts/plot_template.py')
spec=importlib.util.spec_from_file_location('plot_style',template)
style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
style.set_publication_style()
plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False})
data=json.loads((ROOT/'evaluation/RESULTS.json').read_text())
rows=[r for r in data['records'] if r['scope']=='paired_common']
summary=data['summary']['paired_common']
dest=ROOT/'figures';dest.mkdir(exist_ok=True)
rois=sorted({r['roi'] for r in rows});labels=[r.replace('scan24_','24 / ').replace('scan37_','37 / ').replace('_',' ') for r in rois]
colors=['#666666','#0072B2','#E69F00','#009E73']
fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
x=np.arange(4)
for k,(u,typ) in enumerate([('U0','KEEP'),('U0','oracle'),('U1','KEEP'),('U1','oracle')]):
    ys=[]
    for roi in rois:
        r=next(r for r in rows if r['upstream']==u and r['witness']=='W1' and r['roi']==roi and r['condition']=='native')
        ys.append(r['oracle_mse_mm2'] if typ=='oracle' else r['arms'][typ]['nearest_mse_mm2'])
    axes[0].bar(x+(k-1.5)*.19,ys,.19,label=u+' '+typ,color=colors[k],hatch='//' if typ=='oracle' else None)
axes[0].set(xticks=x,xticklabels=labels,ylabel='Nearest-reference MSE (mm², log scale)',yscale='log',title='Native input and candidate-pool error')
axes[0].tick_params(axis='x',rotation=18);axes[0].legend(ncol=2)
for k,(u,arm) in enumerate([('U0','construct_witness'),('U0','heldout_witness'),('U1','construct_witness'),('U1','heldout_witness')]):
    vals=[]
    for condition in ('native','minus3','plus3'):
        a=summary[u+'W0'][condition]['arms'][arm]['nearest_mse_mm2']
        b=summary[u+'W1'][condition]['arms'][arm]['nearest_mse_mm2']
        vals.append(a-b)
    axes[1].bar(np.arange(3)+(k-1.5)*.19,vals,.19,color=colors[k],label=u+' '+('C' if arm.startswith('construct') else 'H'))
axes[1].axhline(0,color='black',lw=.8);axes[1].set(xticks=np.arange(3),xticklabels=['native','−3 mm','+3 mm'],ylabel='W0 MSE − W1 MSE (mm²)',title='Witness-only benefit at fixed candidates')
axes[1].legend(ncol=2);axes[1].text(.02,.02,'Above zero: corrected witness is better',transform=axes[1].transAxes,fontsize=9)
fig.suptitle(f'Camera–photo pairing replay | common pixels {data["valid_u1_base_queries"]}/512 | 4 ROIs, 2 development scenes',fontsize=13)
for ext in ('png','pdf'):fig.savefig(dest/('camera_pairing_effects.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for k,(arm,label) in enumerate([('restore','Frozen restorer'),('construct_witness','C witness'),('heldout_witness','H witness')]):
    gain=[summary['U1W1'][c]['arms'][arm]['improvement_percent'] for c in ('native','minus3','plus3')]
    axes[0].bar(np.arange(3)+(k-1)*.24,gain,.24,label=label,color=colors[k+1])
axes[0].axhline(0,color='black',lw=.8);axes[0].set(xticks=np.arange(3),xticklabels=['native','−3 mm','+3 mm'],ylabel='Mean per-ROI MSE improvement vs own KEEP (%)',title='Complete correction: fixed output rules')
axes[0].legend()
names=[];bad=[];good=[]
for w in ('W0','W1'):
    for arm,label in [('construct_witness','C'),('heldout_witness','H')]:
        rec=[r for r in rows if r['upstream']=='U1' and r['witness']==w]
        b=sum(r['veto'][arm]['bad_blocked'] for r in rec);bn=sum(r['veto'][arm]['bad_original'] for r in rec)
        g=sum(r['veto'][arm]['good_lost'] for r in rec);gn=sum(r['veto'][arm]['good_original'] for r in rec)
        names.append(w+' / '+label);bad.append(100*b/bn if bn else 0);good.append(100*g/gn if gn else 0)
xx=np.arange(len(names));axes[1].bar(xx-.17,bad,.34,label='Bad moves blocked',color='#009E73');axes[1].bar(xx+.17,good,.34,label='Good moves lost',color='#D55E00',hatch='//')
axes[1].set(xticks=xx,xticklabels=names,ylabel='Fraction of original bad / good moves (%)',ylim=(0,100),title='Selection trade-off on corrected upstream')
axes[1].legend();fig.suptitle('Three conditions share query positions; move counts are not independent scenes',fontsize=12)
for ext in ('png','pdf'):fig.savefig(dest/('selection_tradeoff_v2.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
print(dest)

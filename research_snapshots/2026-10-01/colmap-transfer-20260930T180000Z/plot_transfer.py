"""Read-only plots from sealed queries and post-seal evaluation tables."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from mvs_transfer import ROOT,load

def save(fig,name):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plan=load(ROOT/'PLAN.json'); fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    for ax,(sid,s) in zip(axes,plan['scenes'].items()):
        ax.imshow(Image.open(s['cameras'][s['reference']]['image_path']).convert('RGB'))
        for k,r in enumerate(s['rois']):
            x0,y0,x1,y1=r['box_xyxy_original'];col=['#56B4E9','#E69F00'][k]
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=col,linewidth=2))
            ax.text(x0,y0-12,r['id'],color='black',bbox={'facecolor':col,'alpha':.8,'edgecolor':'none'})
            uv=np.array(r['pixel_xy'])*2+.5;ax.scatter(uv[:,0],uv[:,1],s=3,color=col,alpha=.7)
        ax.set_title(f'scan{sid}: photo-fixed ROIs, 256 queries');ax.axis('off')
    fig.suptitle('New same-source objects; queries fixed before reconstruction and GT');save(fig,'fixed_queries')
    if not (ROOT/'evaluation/RESULTS.json').exists():return
    report=load(ROOT/'evaluation/RESULTS.json')
    with (ROOT/'evaluation/ROI_METRICS.csv').open() as f:rows=list(csv.DictReader(f))
    names=[r['id'] for s in plan['scenes'].values() for r in s['rois']]
    colors=['#777777','#E69F00','#0072B2'];methods=['CPU','photo_fallback','geo_fallback']
    labels=['CPU input','COLMAP photo + fallback','COLMAP geo + fallback']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');x=np.arange(4)
    for j,(method,label,color) in enumerate(zip(methods,labels,colors)):
        rr=[next(r for r in rows if r['roi']==n and r['method']==method and r['scope']=='cpu_valid_common') for n in names]
        axes[0].bar(x+(j-1)*.24,[float(r['mse_mm2']) for r in rr],.24,label=label,color=color)
        axes[1].bar(x+(j-1)*.24,[int(r['correct_1mm'])/int(r['n'])*100 for r in rr],.24,color=color)
    for ax in axes:
        ax.set_xticks(x,[n.replace('scan','').replace('_','\n',1) for n in names]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Point-to-reference MSE (mm²)');axes[0].set_title('Identical CPU-valid support');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Within 1 mm (%)');axes[1].set_ylim(0,105);axes[1].set_title('Accuracy is not whole-scene completeness')
    fig.suptitle(f'Frozen transfer: 2 new DTU objects, {report["baseline_valid"]}/512 CPU-valid queries');save(fig,'transfer_results')
    geo=[r for r in report['tail'] if r['method']=='geo']; totals={k:sum(r[k] for r in geo) for k in ('improved','worsened','unchanged','old_good_1mm','old_good_harmed_beyond_1mm','old_wrong_5mm','rescued_1mm','rescued_5mm')}
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    vals=[totals[k] for k in ('improved','worsened','unchanged')]
    bars=axes[0].bar(['Improved','Worsened','Unchanged'],vals,color=['#009E73','#D55E00','#999999']);axes[0].bar_label(bars);axes[0].set_ylim(0,max(vals)*1.2);axes[0].set_ylabel('Queries');axes[0].set_title('Geo fallback vs CPU (strict distance comparison)')
    a,b=totals['old_good_1mm'],totals['old_wrong_5mm']
    vals=[totals['old_good_harmed_beyond_1mm'],totals['rescued_5mm'],totals['rescued_1mm']]
    bars=axes[1].bar([f'Good → >1mm\nbase n={a}',f'>5 → ≤5mm\nbase n={b}',f'>5 → ≤1mm\nbase n={b}'],vals,color=['#D55E00','#56B4E9','#0072B2']);axes[1].bar_label(bars);axes[1].set_ylim(0,max(vals+[1])*1.3);axes[1].set_title('Damage and repair are reported separately')
    save(fig,'repair_and_harm')
if __name__=='__main__':main()

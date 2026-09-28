"""Actual gain and attainable headroom, with a locked-arm tradeoff scatter."""
from common import *
import csv
os.environ['MPLCONFIGDIR']=str(OUT/'plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm

def main():
    verify_seal(OUT/'evaluation');out=OUT/'figures';out.mkdir(exist_ok=False)
    summary=json.loads((OUT/'evaluation/SUMMARY.json').read_text());data=summary['exposed_replay']
    lock=json.loads((OUT/'training/MODEL_LOCK.json').read_text())
    primary,recovery=lock['primary_arm'],lock['recovery_arm']
    arms=['prior__independent_absolute__balanced','prior__joint_support__balanced',primary,recovery,'oracle_AB']
    labels=['Previous independent','Previous shared + support',f'Locked balanced: {primary.split("__")[0]}',
            f'Locked recovery: {recovery.split("__")[0]}','Evaluator-only candidate oracle']
    rows=list(csv.DictReader((OUT/'evaluation/METRICS.csv').open()))
    checks=[]
    def gain(arm,cond):
        def mean(key):
            return np.mean([np.mean([float(r['source_MSE_mm2']) for r in rows
                if r['arm']==key and r['condition']==cond and int(r['scene'])==s]) for s in SCENES])
        value=float(100*(1-mean(arm)/mean('identity')))
        assert abs(value-data[cond][arm]['MSE_gain_percent'])<1e-9
        checks.append(dict(arm=arm,condition=cond,MSE_gain_percent=value))
        return value
    matrix=np.array([[gain(a,c) for c in CONDS] for a in arms])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.right':False,
        'axes.spines.top':False,'axes.linewidth':1.5,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(15,9.7),facecolor='white')
    grid=fig.add_gridspec(2,1,height_ratios=[1.05,1.],hspace=.56)
    ax=fig.add_subplot(grid[0]);lower=min(-1.,float(matrix.min()));upper=max(1.,float(matrix.max()))
    cmap=LinearSegmentedColormap.from_list('gain',['#B64342','#FFFFFF','#3775BA'])
    ax.imshow(matrix,cmap=cmap,norm=TwoSlopeNorm(vmin=lower,vcenter=0,vmax=upper),aspect='auto')
    for i in range(len(arms)):
        for j in range(5):
            v=matrix[i,j];ax.text(j,i,f'{v:+.2f}',ha='center',va='center',
                color='white' if v<lower*.65 or v>upper*.75 else '#272727',fontsize=14)
    ax.set_xticks(range(5),['Native','−1 mm','+1 mm','−3 mm','+3 mm'])
    ax.set_yticks(range(len(arms)),labels);ax.tick_params(length=0,pad=10)
    ax.axhline(3.5,color='#4D4D4D',lw=2,linestyle='--')
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title('A  MSE reduction versus KEEP (%) — positive is better',loc='left',pad=17)
    ax=fig.add_subplot(grid[1]);palette={'base':'#767676','geometry':'#42949E','interaction':'#3775BA'}
    scatter=[]
    for a in lock['expected_route_arms']:
        name,policy=a.split('__');family,capacity=name.rsplit('_',1)
        x=gain(a,'native');y=float(np.mean([gain(a,c) for c in CONDS[1:]]));scatter.append((a,x,y))
        ax.scatter(x,y,s=60 if capacity=='shallow' else 100,marker='o' if capacity=='shallow' else 's',
            facecolor=palette[family],alpha=.7,edgecolor='white',linewidth=.7)
    for a,marker,color,label in ((primary,'*','#0F4D92','Locked balanced'),(recovery,'D','#B64342','Locked recovery')):
        _,x,y=next(row for row in scatter if row[0]==a)
        ax.scatter(x,y,s=220,marker=marker,facecolor='none',edgecolor=color,linewidth=2,label=label,zorder=5)
    for family,color in palette.items():ax.scatter([],[],s=65,color=color,label=family)
    ax.axvline(0,color='#CFCECE',lw=1);ax.axhline(0,color='#CFCECE',lw=1)
    ax.set_xlabel('Native MSE reduction (%) — right is better')
    ax.set_ylabel('Mean reduction across\nfour injected conditions (%)')
    ax.set_title('B  All 24 locked working points — no replay-selected winner',loc='left',pad=15)
    ax.legend(loc='upper left',bbox_to_anchor=(1.015,1),fontsize=10)
    fig.suptitle('Visibility evidence: improvement, tradeoff, and remaining headroom',fontsize=20,fontweight='bold',y=.985)
    fig.text(.5,.936,'Three exposed replay scenes × four ROIs; equal ROI then equal scene weight',ha='center',fontsize=12,color='#4D4D4D')
    fig.text(.5,.013,'The oracle uses reference geometry only for evaluation; neither locked policy can use it.\nCircle / square: shallow / rich capacity. Native and recovery objectives are reported separately.',ha='center',fontsize=10,color='#4D4D4D')
    fig.subplots_adjust(left=.33,right=.83,top=.865,bottom=.13)
    for suffix in ('png','pdf','svg'):fig.savefig(out/f'visibility_gain.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    with (out/'PLOTTED_VALUES.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    save_json(out/'VALIDATION.json',dict(status='PASS',numeric_checks=len(checks),source='independent CSV aggregation'))
    seal(out,{str(p):sha(p) for p in (ROOT/'plot.py',OUT/'evaluation/SEALED.json')})
    print('FIGURE SEALED',len(checks))

if __name__=='__main__':main()

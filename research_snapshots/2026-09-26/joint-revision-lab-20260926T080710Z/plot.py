"""One result figure, with independently aggregated plotted values."""
from common import *
import csv
os.environ['MPLCONFIGDIR']=str(ROOT/'plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ARMS=('historical_post_AB','A_R__balanced','B_R__balanced','normalized_max__balanced',
      'independent_absolute__balanced','joint_common__balanced','joint_support__balanced')
LABELS=('Historical joint A/B','Previous A + reserved','Previous B + reserved',
        'Independent normalized max','Independent absolute utility',
        'Shared context','Shared context + support [primary]')

def main():
    check_host();verify_seal(ROOT/'evaluation')
    out=ROOT/'figures';out.mkdir(exist_ok=False)
    with (ROOT/'evaluation/METRICS.csv').open() as f:rows=list(csv.DictReader(f))
    summary=json.loads((ROOT/'evaluation/SUMMARY.json').read_text())['exposed_replay']
    data=[];gains=[];capture=[];checks=0
    for method in ARMS:
        grow=[];crow=[]
        for condition in CONDS:
            def mean(arm,key):
                return float(np.mean([np.mean([float(r[key]) for r in rows
                    if r['arm']==arm and r['condition']==condition and int(r['scene'])==sid]) for sid in SCENES]))
            mse=mean(method,'source_MSE_mm2');base=mean('identity','source_MSE_mm2')
            gain=100*(1-mse/base)
            positive=mean(method,'B_captured_incremental_MSE_mm2')
            total=mean(method,'B_incremental_MSE_mm2')
            fraction=positive/total
            for key,value in (('source_MSE_mm2',mse),('MSE_gain_percent',gain),
                              ('B_incremental_capture_fraction',fraction)):
                assert np.isclose(value,summary[condition][method][key],rtol=0,atol=1e-10)
                checks+=1
            grow.append(gain);crow.append(100*fraction)
            data.append(dict(method=method,condition=condition,MSE_gain_percent=gain,
                             B_incremental_gross_capture_percent=100*fraction))
        gains.append(grow);capture.append(crow)
    with (out/'PLOTTED_VALUES.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':1.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,(ax,bx)=plt.subplots(2,1,figsize=(12.8,9),gridspec_kw={'height_ratios':[7,3]})
    g=np.array(gains)
    cmap=LinearSegmentedColormap.from_list('gain',['#B64342','#FFFFFF','#3775BA'])
    norm=TwoSlopeNorm(vmin=min(-1.,g.min()),vcenter=0.,vmax=max(1.,g.max()))
    ax.imshow(g,cmap=cmap,norm=norm,aspect='auto')
    ax.set_yticks(range(len(ARMS)),LABELS)
    ax.set_xticks(range(5),('Native','−1 mm','+1 mm','−3 mm','+3 mm'))
    ax.set_title('A   Actual MSE reduction versus KEEP (%) — positive is better',loc='left',pad=14)
    for (i,j),v in np.ndenumerate(g):
        ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(norm(v)-.5)>.33 else '#272727')
    ax.add_patch(plt.Rectangle((-.49,len(ARMS)-1.49),4.98,.98,fill=False,lw=2,edgecolor='#0F4D92'))
    c=np.array(capture)[-3:]
    bx.imshow(c,cmap=LinearSegmentedColormap.from_list('capture',['#FFFFFF','#AADCA9','#42949E']),
              vmin=0,vmax=max(30.,c.max()),aspect='auto')
    bx.set_yticks(range(3),LABELS[-3:]);bx.set_xticks(range(5),('Native','−1 mm','+1 mm','−3 mm','+3 mm'))
    bx.set_title('B   B-only complementary benefit captured (%) — gross, not net gain',loc='left',pad=14)
    for (i,j),v in np.ndenumerate(c):bx.text(j,i,f'{v:.1f}',ha='center',va='center')
    for axis in (ax,bx):
        axis.tick_params(length=0,pad=9)
        for spine in axis.spines.values():spine.set_visible(False)
        axis.axvline(.5,color='#767676',lw=1.1)
    fig.suptitle('Joint KEEP / A / B selection',fontsize=22,fontweight='bold',y=.98)
    fig.text(.5,.928,'3 exposed replay scenes × 4 ROIs; ROI → scene equal weight; all five conditions retained',
             ha='center',fontsize=11,color='#4D4D4D')
    fig.text(.5,.025,'Panel B uses evaluator-only min(KEEP,A) as the reference for B complementarity.\n'
             'It does not subtract damage from incorrect choices and is not a deployable oracle result.',
             ha='center',fontsize=11,color='#4D4D4D')
    fig.subplots_adjust(left=.35,right=.975,top=.87,bottom=.11,hspace=.55)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'joint_revision.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    save_json(out/'VALIDATION.json',dict(status='PASS',numeric_checks=checks,
        source='independent CSV equal-ROI/equal-scene aggregation',tolerance=1e-10))
    seal(out,{str(p):sha(p) for p in (ROOT/'plot.py',ROOT/'PROTOCOL.md',ROOT/'evaluation/SEALED.json')})
    print('FIGURE SEALED',checks,flush=True)

if __name__=='__main__':main()

"""Measured oracle expansion versus actual step revision; CSV-checked figures."""
from common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    verify_seal(OUT/'evaluation')
    summary=json.loads((OUT/'evaluation/SUMMARY.json').read_text())['exposed_replay']
    with (OUT/'evaluation/METRICS.csv').open() as f: rows=list(csv.DictReader(f))
    checks=[]
    for c in CONDS:
        before=np.mean([float(r['source_MSE_mm2']) for r in rows if r['condition']==c and r['arm']=='identity'])
        for arm,record in summary[c].items():
            if arm=='decomposition':continue
            value=np.mean([float(r['source_MSE_mm2']) for r in rows if r['condition']==c and r['arm']==arm])
            gain=100*(1-value/before)
            if abs(gain-record['MSE_gain_percent'])>1e-10:raise AssertionError('CSV/figure mismatch')
            checks.append(dict(condition=c,arm=arm,gain=gain))
    folder=OUT/'figures';folder.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':1.5,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(15,5.8))
    palettes=['#767676','#42949E','#0F4D92','#B64342']
    groups=[(('oracle_discrete','oracle_selected_segment','oracle_segment_union'),
             ('Discrete KEEP/A/B oracle','Continuous, fixed chosen branch','Continuous, either A/B branch'),
             'A  Candidate capacity: evaluation reference used'),
            (('prior_recovery','constant_050','grid','quadratic'),
             ('Prior recovery, full step','Constant half step','Photo grid step (primary)','Photo + quadratic refinement'),
             'B  Actual methods: no evaluation reference')]
    labels=['Native','-1 mm','+1 mm','-3 mm','+3 mm']
    for ax,(arms,names,title) in zip(axes,groups):
        width=.8/len(arms);x=np.arange(5)
        allvalues=[]
        for j,(arm,name) in enumerate(zip(arms,names)):
            values=[summary[c][arm]['MSE_gain_percent'] for c in CONDS]
            allvalues.extend(values)
            bars=ax.bar(x+(j-(len(arms)-1)/2)*width,values,width,label=name,color=palettes[j],
                        edgecolor='black',linewidth=.55,hatch='//' if j==len(arms)-1 else None)
            for rect,value in zip(bars,values):
                ax.text(rect.get_x()+rect.get_width()/2,value+(1 if value>=0 else -1),f'{value:.1f}',
                        ha='center',va='bottom' if value>=0 else 'top',fontsize=8.5,rotation=90)
        ax.axhline(0,color='black',lw=.9)
        ax.set_xticks(x,labels);ax.set_title(title,loc='left',fontsize=13)
        ax.set_ylabel('Source MSE reduction vs unchanged input (%)')
        ax.set_ylim(min(-5,min(allvalues)-12),max(allvalues)+16)
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),fontsize=10)
    fig.suptitle('Continuous coordinates expand the candidate set; photographs must still select the step',fontsize=15,y=.98)
    fig.text(.5,.015,'Exposed replay: 3 scenes × 4 ROI × 5 input conditions. Equal ROI/scene weights; not independent confirmation.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.15,1,.93),pad=1.3)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'continuous_step.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    save_json(folder/'VALIDATION.json',dict(status='PASS',numeric_checks=len(checks),checks=checks))
    seal(folder,{str(ROOT/'plot.py'):sha(ROOT/'plot.py'),str(OUT/'evaluation/SUMMARY.json'):sha(OUT/'evaluation/SUMMARY.json'),
                 str(OUT/'evaluation/METRICS.csv'):sha(OUT/'evaluation/METRICS.csv')})
    print('FIGURES SEALED',len(checks),'checks')

if __name__=='__main__':main()

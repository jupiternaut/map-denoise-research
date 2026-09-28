"""Post-hoc candidate-capacity and selection gaps from sealed measurements."""
from pathlib import Path
import os,json,csv

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'figures_capacity'
os.environ.setdefault('MPLCONFIGDIR',str(DEST/'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import OUT,sha,verify_seal,seal

CONDITIONS=('native','minus3','plus3')
DISPLAY=('Native','−3 mm','+3 mm')
SERIES=('oracle_proposals','oracle_K2_only','multi_field')
LABELS=('9-position pool oracle','KEEP/K2-only oracle','Actual K2 output')
COLORS=('#767676','#0F4D92','#B64342')


def main():
    if (DEST/'SEALED.json').exists():
        raise FileExistsError('reviewed figure already sealed')
    DEST.mkdir(exist_ok=True)
    for folder in (OUT/'evaluation',OUT/'posthoc_field_capacity'):
        verify_seal(folder)
    main_path=OUT/'evaluation/SUMMARY.json'
    cap_path=OUT/'posthoc_field_capacity/SUMMARY.json'
    main=json.loads(main_path.read_text())['exposed_replay']
    cap=json.loads(cap_path.read_text())
    assert cap['post_hoc'] and cap['registered_primary_unchanged']
    rows=[];gaps=[];values=[]
    for cond in CONDITIONS:
        c=cap['by_condition'][cond]['oracle_K2_only']
        initial=main[cond]['identity']['source_MSE_mm2']
        assert abs(initial-c['identity_MSE_mm2'])<1e-12
        assert abs(main[cond]['multi_field']['source_MSE_mm2']-c['actual_MSE_mm2'])<1e-12
        errors=(main[cond]['oracle_proposals']['source_MSE_mm2'],
                c['source_MSE_mm2'],main[cond]['multi_field']['source_MSE_mm2'])
        gains=[100*(1-v/initial) for v in errors]
        assert np.allclose(gains,[main[cond]['oracle_proposals']['MSE_gain_percent'],
                                 c['MSE_gain_percent'],main[cond]['multi_field']['MSE_gain_percent']],atol=1e-10,rtol=0)
        values.append(gains)
        for arm,label,error,gain in zip(SERIES,LABELS,errors,gains):
            rows.append(dict(condition=cond,arm=arm,label=label,source_MSE_mm2=error,
                identity_MSE_mm2=initial,MSE_gain_percent=gain,post_hoc=True,
                uses_GT_selector=arm!='multi_field',aggregation='ROI then scene equally'))
        gaps.append(dict(condition=cond,pool_to_field_gap_mm2=errors[1]-errors[0],
            selection_gap_mm2=errors[2]-errors[1],
            pool_to_field_gap_percentage_points=gains[0]-gains[1],
            selection_gap_percentage_points=gains[1]-gains[2]))
    values=np.asarray(values)
    plt.rcParams.update({'font.family':['DejaVu Sans'],'font.size':12,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.5,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(11.5,7.2))
    x=np.arange(3);w=.22
    for j,(label,color) in enumerate(zip(LABELS,COLORS)):
        bars=ax.bar(x+(j-1)*w,values[:,j],width=.20,color=color,edgecolor='#333333',
                    linewidth=.7,hatch=('//','..','')[j],label=label,zorder=3)
        for bar,value in zip(bars,values[:,j]):
            ax.text(bar.get_x()+bar.get_width()/2,value+(2 if value>=0 else -3),
                    f'{value:.2f}%',ha='center',va='bottom' if value>=0 else 'top',fontsize=10.5)
    ax.axhline(0,color='#333333',linewidth=1.3,zorder=2)
    ax.grid(axis='y',color='#E5E5E5',linewidth=.6,zorder=0)
    ax.set_ylim(min(0,values.min())-18,max(0,values.max())+18)
    ax.set_xticks(x,DISPLAY)
    ax.set_ylabel('Source MSE reduction vs identity (%)')
    ax.set_xlabel('Input condition')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=3,fontsize=10.5)
    fig.suptitle('K2 capacity versus selecting its useful corrections',fontsize=18,y=.975)
    fig.text(.5,.915,'POST-HOC DIAGNOSTIC · frozen outputs · 3 exposed scenes × 4 ROI',
             color='#B64342',fontsize=12,ha='center')
    text=[['Pool → K2 capacity gap (pp)']+[f"{g['pool_to_field_gap_percentage_points']:.2f}" for g in gaps],
          ['K2 oracle → actual selection gap (pp)']+[f"{g['selection_gap_percentage_points']:.2f}" for g in gaps]]
    table=ax.table(cellText=text,colLabels=['Arithmetic MSE-gain differences',*DISPLAY],
                   colWidths=[.55,.15,.15,.15],cellLoc='center',bbox=[0,-.40,1,.22])
    table.auto_set_font_size(False);table.set_fontsize(10.5)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD');cell.set_linewidth(.5)
        if row==0:cell.set_facecolor('#F0F3F5')
        if col==0:cell.set_text_props(ha='left')
    fig.text(.5,.023,
        'Both oracle series use evaluation-reference selection; actual K2 does not. Zero marks identity; negative means damage.\n'
        'Pool and K2 candidate sets are not nested: their gap is descriptive, not an isolated causal compression estimate.',
        ha='center',fontsize=9.5,color='#444444')
    fig.subplots_adjust(left=.10,right=.98,bottom=.29,top=.83)
    for suffix in ('png','pdf','svg'):
        fig.savefig(DEST/f'field_capacity.{suffix}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    for filename,records in (('FIGURE_DATA.csv',rows),('GAP_DATA.csv',gaps)):
        with (DEST/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    sources={str(path):sha(path) for path in (Path(__file__),main_path,cap_path,
        main_path.parent/'SEALED.json',cap_path.parent/'SEALED.json')}
    with (DEST/'VALIDATION.json').open('w') as stream:
        json.dump(dict(status='PASS',post_hoc=True,plot_values=9,gap_values=6,
            source_sha256=sources,source_MSE_and_gain_checks=True,primary_unchanged=True),
            stream,indent=2,allow_nan=False)
    print('CAPACITY FIGURE READY',DEST,flush=True)


if __name__=='__main__':
    main()

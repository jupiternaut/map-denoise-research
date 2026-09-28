"""Publication figures from sealed measured outcomes, never hand-entered values."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT/'figures'
os.environ.setdefault('MPLCONFIGDIR', str(FIGURES/'mplconfig'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from common import OUT, SCENES, CONDS, sha, verify_seal, save_json, seal

ACTUAL = ('prior_recovery','point_wta','single_field','multi_field',
          'multi_field_visibility','multi_field_graph')
ORACLES = ('oracle_discrete','oracle_continuous','oracle_proposals','oracle_all_layers','oracle_expanded')
LABELS = {
    'prior_recovery':'Prior recovery',
    'point_wta':'Pointwise photo WTA',
    'single_field':'Single surface field',
    'multi_field':'Multi-surface field (primary)',
    'multi_field_visibility':'+ visibility proxy',
    'multi_field_graph':'+ graph optimization',
    'oracle_discrete':'Old discrete KEEP/A/B',
    'oracle_continuous':'Old continuous segments',
    'oracle_proposals':'New 9-position search pool',
    'oracle_all_layers':'Search pool + fitted fields',
    'oracle_expanded':'+ old continuous oracle',
}
COLORS = {
    'prior_recovery':'#767676','point_wta':'#B64342',
    'single_field':'#3775BA','multi_field':'#0F4D92',
    'multi_field_visibility':'#42949E','multi_field_graph':'#9A4D8E',
    'oracle_discrete':'#767676','oracle_continuous':'#B64342',
    'oracle_proposals':'#42949E','oracle_all_layers':'#3775BA','oracle_expanded':'#0F4D92',
}
HATCHES = ('','//','..','','\\\\','xx')
CONDITION_LABELS = ('Native','−1 mm','+1 mm','−3 mm','+3 mm')


def validated_rows():
    folder = OUT/'evaluation'
    verify_seal(folder)
    summary_file, metrics_file = folder/'SUMMARY.json', folder/'METRICS.csv'
    summary = json.loads(summary_file.read_text())
    data = summary['exposed_replay']
    with metrics_file.open() as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != 720:
        raise AssertionError('expected 60 cases x 12 methods')
    plotted, checks = [], []
    for condition in CONDS:
        selected = [r for r in raw if r['condition']==condition]
        values = {}
        for arm in ('identity',)+ACTUAL+ORACLES:
            rr = [r for r in selected if r['arm']==arm]
            if len(rr)!=12 or any(sum(int(r['scene'])==s for r in rr)!=4 for s in SCENES):
                raise AssertionError('fixed 3 scenes x 4 ROI aggregation violated')
            values[arm] = float(np.mean([
                np.mean([float(r['source_MSE_mm2']) for r in rr if int(r['scene'])==sid])
                for sid in SCENES]))
        for arm in ACTUAL+ORACLES:
            gain = 100.*(1.-values[arm]/values['identity'])
            cell = data[condition][arm]
            delta = abs(gain-cell['MSE_gain_percent'])
            mse_delta = abs(values[arm]-cell['source_MSE_mm2'])
            if delta>1e-9 or mse_delta>1e-12:
                raise AssertionError((condition,arm,delta,mse_delta))
            checks.append(dict(condition=condition,arm=arm,
                               gain_difference=delta,MSE_difference=mse_delta))
            plotted.append(dict(panel='A_actual' if arm in ACTUAL else 'B_GT_diagnostic',
                condition=condition,arm=arm,label=LABELS[arm],
                source_MSE_mm2=values[arm],identity_MSE_mm2=values['identity'],
                MSE_gain_percent=gain,ROI_count=12,scene_count=3,
                data_role='exposed_replay',GT_selector=arm in ORACLES))
    return plotted, checks, {str(path):sha(path) for path in (
        Path(__file__),ROOT/'PROTOCOL.md',summary_file,metrics_file,folder/'SEALED.json')}


def apply_style():
    plt.rcParams.update({
        'font.family':['DejaVu Sans'],
        'font.size':13,'axes.titlesize':15,'axes.labelsize':13,
        'axes.linewidth':1.8,'axes.spines.right':False,'axes.spines.top':False,
        'legend.frameon':False,'legend.fontsize':10.5,
        'xtick.labelsize':12,'ytick.labelsize':12,
        'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white',
        'savefig.facecolor':'white','hatch.linewidth':.6,
    })


def draw_panel(ax,arms,table,title,letter):
    x = np.arange(len(CONDS))
    width = .78/len(arms)
    values = []
    for j,arm in enumerate(arms):
        y = np.asarray([table[(condition,arm)] for condition in CONDS])
        values.extend(y)
        ax.bar(x+(j-(len(arms)-1)/2)*width,y,width=width*.93,
               color=COLORS[arm],edgecolor='#303030',linewidth=.6,
               hatch=HATCHES[j],label=LABELS[arm],zorder=3)
    lo,hi = min(0.,min(values)),max(0.,max(values))
    span = max(hi-lo,10.)
    ax.set_ylim(lo-.08*span,hi+.11*span)
    ax.axhline(0,color='#333333',linewidth=1.2,zorder=2)
    ax.set_xticks(x,CONDITION_LABELS)
    ax.set_xlim(-.6,len(CONDS)-.4)
    ax.set_xlabel('Input condition')
    ax.set_ylabel('Source MSE reduction vs identity (%)')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis='y',color='#E5E5E5',linewidth=.6,zorder=0)
    ax.set_title(title,loc='left',pad=18)
    ax.text(-.095,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=17)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,
              columnspacing=1.4,handlelength=1.6,handletextpad=.5)


def main():
    if (FIGURES/'SEALED.json').exists():
        raise FileExistsError('figure stage sealed; do not overwrite reviewed figures')
    FIGURES.mkdir(parents=True,exist_ok=True)
    rows,checks,sources = validated_rows()
    table = {(r['condition'],r['arm']):r['MSE_gain_percent'] for r in rows}
    apply_style()
    fig,axes = plt.subplots(1,2,figsize=(19,7.8),gridspec_kw={'width_ratios':[1.15,1.]})
    draw_panel(axes[0],ACTUAL,table,'Actual observation-only outputs','A')
    draw_panel(axes[1],ORACLES,table,'Oracle diagnostics — not deployable','B')
    fig.suptitle('Multi-surface fields: realized geometry vs candidate-pool potential',
                 fontsize=19,y=.98)
    fig.text(.5,.925,'Exposed replay: 3 scenes × 4 fixed ROI per condition · higher is better',
             ha='center',fontsize=12,color='#444444')
    fig.text(.5,.025,
        'ROI-then-scene equal weighting. Negative values mean damage. Separate y-axis scales; zero baselines shown.\n'
        'The 9-position search pool broadens ray offsets; the next oracle increment adds fitted fields. Oracle selection uses reference geometry.',
        ha='center',va='bottom',fontsize=10.5,color='#444444')
    fig.tight_layout(rect=(.015,.095,.99,.9),pad=1.4,w_pad=3.)
    for extension in ('png','pdf','svg'):
        fig.savefig(FIGURES/f'multisurface_gain.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    with (FIGURES/'FIGURE_DATA.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    validation=dict(status='PASS',n_plot_values=len(rows),
        source_sha256=sources,checks=checks,GT_and_actual_separate=True,
        aggregation='mean of four ROI per scene, mean of three scenes; then relative MSE',
        uncertainty='no inferential error bars: exposed finite replay, not independent scene confirmation')
    with (FIGURES/'VALIDATION.json').open('w') as stream:
        json.dump(validation,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('FIGURES READY',len(rows),'verified values',FIGURES,flush=True)


if __name__=='__main__':
    main()

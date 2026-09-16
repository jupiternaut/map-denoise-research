"""Static research-figure exports from frozen E0-E results."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_support import ROOT

BLUE='#2864AC'
ORANGE='#CA792C'
INK='#25313D'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#78838D',
    'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titlepad':12,'savefig.facecolor':'white'})


def save(fig, name):
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    data=json.loads((ROOT/'results/RESULTS.json').read_text())
    arms=['original_bh_projection','no_bh_projection','test_projection',
          'test_veto_projection','test_veto_argmin','test_margin_projection']
    names=['Original BH + projection','No BH + projection','TEST + projection',
           'TEST + VETO + projection','TEST + VETO + argmin','TEST + margin + projection']
    fig,axes=plt.subplots(2,2,figsize=(14.2,8.2),sharex=True,layout='constrained')
    for row,phase in enumerate(['development','locked_same_family']):
        for col,fbig in enumerate([.01,.10]):
            ax=axes[row,col]
            s=next(s for s in data['summary']['groups'] if s['phase']==phase and s['fbig']==fbig
                   and s['calibration']=='exch' and s['alpha']==.2)
            values=[s['arms'][a]['delta_mae'] for a in arms]
            y=np.arange(len(arms))
            ax.barh(y,values,color=BLUE,alpha=.72,height=.55,edgecolor=BLUE,lw=.6)
            for i,arm in enumerate(arms):
                ax.scatter(s['arms'][arm]['seed_delta_mae'],i+np.array([-.12,0,.12]),
                           s=15,color=INK,zorder=4)
                ax.text(values[i]+(.006 if values[i]>=0 else -.006),i,f'{values[i]:+.4f}',
                        ha='left' if values[i]>=0 else 'right',va='center',fontsize=9)
            ax.axvline(0,color=INK,lw=1)
            ax.set_yticks(y,names);ax.invert_yaxis()
            ax.set_xlim(-.17,.27)
            ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
            cohort='Development 101-103' if row==0 else 'Locked same-family 201-203'
            ax.set_title(f'{cohort} | large-error fraction {fbig:.0%}',loc='left',fontsize=11)
            ax.set_xlabel('MAE change from identity (mm); negative is better')
    fig.suptitle('E0-E: geometric error by admission and output rule\n'
                 'exch calibration, alpha_tgt=0.2; 20,000 points/population; bars: seed mean, dots: seeds',
                 fontsize=14)
    save(fig,'mae_comparison')

    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharey=True,layout='constrained')
    groups=['big','base','twosheet_back','twosheet_front','misfit']
    labels=['Large errors','Ordinary points','True back layer','True front layer','Misfit']
    for ax,fb in zip(axes,[.01,.10]):
        rs=[r for r in data['runs'] if r['phase']=='locked_same_family' and r['fbig']==fb
            and r['calibration']=='exch' and r['alpha']==.2]
        for arm,offset,color,label in [('test_projection',-.17,ORANGE,'TEST'),
                                      ('test_veto_projection',.17,BLUE,'TEST + VETO')]:
            vals=[np.mean([r['metrics'][arm]['groups'][g]['weighted_delta'] for r in rs]) for g in groups]
            ax.barh(np.arange(5)+offset,vals,height=.3,color=color,edgecolor=color,label=label)
        ax.set_yticks(range(5),labels);ax.invert_yaxis()
        ax.axvline(0,color=INK,lw=1);ax.set_xlim(-.3,.21)
        ax.set_title(f'Large-error fraction {fb:.0%}',loc='left')
        ax.set_xlabel('Contribution to total MAE change (mm)')
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left',frameon=False)
    fig.suptitle('E0-E: subgroup loss accounting\n'
                 'Locked same-family seeds 201-203; exch, alpha_tgt=0.2; actual subgroup n/N weights',fontsize=13)
    save(fig,'veto_accounting')

    fig,axes=plt.subplots(1,3,figsize=(14.2,4.7),layout='constrained')
    amps=[0,.15,.30,.60]
    for world,offset,color,label in [('real_back',-.18,BLUE,'True back layer'),
                                    ('ghost',.18,ORANGE,'Ghost needing correction')]:
        vals=[np.mean([r['worlds'][world]['test_veto_projection']['groups']['ALL']['mae']
                       for r in data['crossed'] if r['amplitude']==a and r['noise']==.03]) for a in amps]
        axes[0].bar(np.arange(4)+offset,vals,width=.34,color=color,label=label)
    axes[0].set_xticks(range(4),['0','.15','.30','.60'])
    axes[0].set_xlabel('Injected secondary/main valley amplitude')
    axes[0].set_ylabel('Final coordinate MAE (mm)')
    axes[0].set_title('Same outputs, different physical targets',fontsize=10)
    axes[0].legend(fontsize=8,frameon=False,loc='upper center')
    axes[0].set_ylim(0,7.8)
    for ax,a in zip(axes[1:],[0,.3]):
        p=ROOT/f'results/crossed/seed501_weak{a:.2f}_noise0.03/outputs.npz'
        with np.load(p) as z:
            t,c=z['grid'],z['curves'][0]
            ax.plot(t,c,color=INK,lw=1.3)
            ax.axvline(0,color=BLUE,ls='--',label='Incumbent / real-back truth')
            ax.axvline(6,color=ORANGE,ls=':',label='Ghost truth')
            selected=z['out_test_veto_projection'][0]
            ax.plot(selected,np.interp(selected,t,c),'o',color=BLUE,ms=7,label='Selected output')
            ax.set_title(f'Fixed example row 0 | amplitude {a:.2f}',fontsize=10)
            ax.set_xlabel('Candidate correction (mm)');ax.set_ylabel('Observed matching cost')
    axes[1].legend(fontsize=7,frameon=False,loc='lower left')
    fig.suptitle('E0-E: evidence-shape counterexamples\n'
                 'Matched observable inputs; evaluation truth differs; noise=0.03; two seeds x 512 points',fontsize=13)
    save(fig,'paired_evidence')


if __name__=='__main__':main()

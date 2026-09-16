"""Visualize selected proxy actions; no parameter selection or fitting."""
from pathlib import Path
import sys,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    dest=Path(sys.argv[1]);read=lambda p:json.loads(Path(p).read_text());bs=read(dest/'SEALED_BEFORE_GT.json');rows=read(dest/'RESULTS.json');look={(r['case'],r['method']):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,scene in zip(axes,(24,37)):
        for b in [b for b in bs if b['scene']==scene]:
            case=b['job']['case'];choice=b['choices']['photo_selector'];base=look[case,'identity'];out=look[case,'photo_selector'];loss=b['photo_loss']
            x=100*(1-loss[choice]/loss['identity']) if loss['identity'] else 0;y=out['accuracy_mm']-base['accuracy_mm']
            ax.scatter(x,y,c='#c7473a' if y>0 else '#007b73',s=48,alpha=.8)
            if y>.2:ax.annotate(case.split('_')[-1],(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.axhline(0,color='#334155',lw=1);ax.set(xlabel='Photometric loss improvement (%)',ylabel='Reference MAE change (mm)',title=f'scan{scene}: above zero = geometry worsened')
        ax.grid(alpha=.15)
    fig.suptitle('Lower image loss does not necessarily mean better geometry',fontsize=14)
    fig.tight_layout();p=dest/'figures/selected_actions.png';assert not p.exists();fig.savefig(p,dpi=160);plt.close(fig)
if __name__=='__main__':main()

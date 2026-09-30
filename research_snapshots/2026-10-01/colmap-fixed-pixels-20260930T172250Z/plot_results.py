"""Figures from sealed evaluation only, fixed denominator and explicit oracle."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'evaluation/ROI_METRICS_SCOPED.csv').open()))
pts=list(csv.DictReader((ROOT/'evaluation/POINT_METRICS.csv').open()))
result=json.loads((ROOT/'evaluation/RESULTS.json').read_text())
rois=list(dict.fromkeys(r['roi'] for r in rows));short=['Window\nscan24','Gable\nscan24','Scissor\nscan37','Clamp\nscan37']
methods=['CPU','photo_fallback','geo_fallback'];labels=['Current CPU','COLMAP photo + fallback','COLMAP geometric + fallback']
colors=['#6b7280','#d55e00','#0072b2']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
folder=ROOT/'figures';folder.mkdir(exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for j,(method,label,color) in enumerate(zip(methods,labels,colors)):
    yy=[float(next(r['mse_mm2'] for r in rows if r['roi']==roi and r['method']==method and r['scope']=='all_valid')) for roi in rois]
    bars=ax[0].bar(np.arange(4)+(j-1)*.25,yy,.24,label=label,color=color)
    for b,y in zip(bars,yy):ax[0].text(b.get_x()+b.get_width()/2,y*1.1,f'{y:.1f}',ha='center',va='bottom',fontsize=8)
ax[0].set_yscale('log');ax[0].set_ylim(.3,1500);ax[0].set_xticks(np.arange(4),short)
ax[0].set_ylabel('Point-to-reference MSE (mm², log scale)')
ax[0].set_title('Geometric consistency improves all four ROIs')
ax[0].legend(fontsize=8,loc='upper left');ax[0].grid(axis='y',alpha=.15)
p=[r for r in pts if r['method']=='geo_fallback' and r['valid']=='True']
x=np.array([float(r['cpu_distance_mm']) for r in p]);y=np.array([float(r['distance_mm']) for r in p])
for sign,color,label in [(y<x-1e-9,'#009e73','Improved'),(y>x+1e-9,'#d55e00','Worsened'),(np.isclose(x,y,atol=1e-9,rtol=0),'#777777','Unchanged / fallback')]:
    ax[1].scatter(np.maximum(x[sign],.01),np.maximum(y[sign],.01),s=13,alpha=.65,c=color,label=f'{label}: {sign.sum()}/497')
ax[1].plot([.01,150],[.01,150],'k--',lw=1);ax[1].set_xscale('log');ax[1].set_yscale('log')
ax[1].set_xlim(.01,150);ax[1].set_ylim(.01,150)
ax[1].set_xlabel('Current CPU distance (mm)');ax[1].set_ylabel('Geometric + fallback distance (mm)')
ax[1].set_title('Lower average error does not mean every point improves')
ax[1].legend(fontsize=8,loc='upper left');ax[1].grid(alpha=.15)
fig.suptitle('Same corrected cameras, same photos, same 497 original valid queries',fontsize=13)
for ext in ['png','pdf']:fig.savefig(folder/f'mvs_comparison.{ext}',dpi=220)
plt.close(fig)

plan=json.loads((ROOT.parent/'camera-pairing-replay-20260930T162130Z/PLAN_CORRECTED.json').read_text())
fig,axes=plt.subplots(4,2,figsize=(10,12),layout='constrained')
norm=LogNorm(.05,50);cmap=plt.get_cmap('viridis')
for i,roi in enumerate(rois):
    s=plan['scenes']['24' if roi.startswith('scan24') else '37'];spec=next(r for r in s['rois'] if r['id']==roi)
    photo=np.array(Image.open(s['cameras'][s['reference']]['image_path']).resize((777,581)))
    for j,method in enumerate(['CPU','geo_fallback']):
        ax=axes[i,j];ax.imshow(photo)
        p=[r for r in pts if r['roi']==roi and r['method']==method]
        valid=[r for r in p if r['valid']=='True'];missing=[r for r in p if r['valid']!='True']
        sc=ax.scatter([float(r['u']) for r in valid],[float(r['v']) for r in valid],c=[float(r['distance_mm']) for r in valid],cmap=cmap,norm=norm,s=24,edgecolors='white',linewidths=.3)
        ax.scatter([float(r['u']) for r in missing],[float(r['v']) for r in missing],marker='x',c='magenta',s=20)
        x0,y0,x1,y1=np.array(spec['box_xyxy_original'])/2
        ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(roi+' — '+('CPU' if j==0 else 'geometric + fallback'),fontsize=9)
fig.colorbar(sc,ax=axes.ravel().tolist(),label='Point-to-reference distance (mm, log color)',shrink=.5)
fig.suptitle('Fixed pixels; magenta crosses = original CPU missing (not replaced)',fontsize=12)
for ext in ['png','pdf']:fig.savefig(folder/f'fixed_pixel_errors.{ext}',dpi=200)
plt.close(fig)
print('4 figure files generated')

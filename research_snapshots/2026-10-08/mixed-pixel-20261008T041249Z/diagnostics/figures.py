"""Deterministic data plots from sealed result files; no image editing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run import ROOT,load,dump,sha
OUT=ROOT/'figures'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
COLORS=['#0072B2','#D55E00','#009E73','#777777']

def text(d,xy,value,size=20,color='#222222',bold=False):
    d.text(xy,str(value),font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color)
def plot(d,box,xlim,ylim,series,title,xlabel,ylabel):
    x0,y0,x1,y1=box
    px=lambda x:x0+(x-xlim[0])/(xlim[1]-xlim[0])*(x1-x0)
    py=lambda y:y1-(y-ylim[0])/(ylim[1]-ylim[0])*(y1-y0)
    text(d,(x0,y0-48),title,23,bold=True)
    for y in np.linspace(*ylim,5):
        d.line([(x0,py(y)),(x1,py(y))],fill='#dddddd',width=1)
        text(d,(x0-72,py(y)-10),f'{y:.1f}',15)
    for x in np.linspace(*xlim,6):text(d,(px(x)-16,y1+12),f'{x:.0f}',15)
    d.line([(x0,y0),(x0,y1),(x1,y1)],fill='#222222',width=2)
    for j,(label,xs,ys) in enumerate(series):
        pts=[(px(x),py(y)) for x,y in zip(xs,ys) if np.isfinite(y)]
        if len(pts)>1:d.line(pts,fill=COLORS[j],width=3)
        text(d,(x0+10,y0+10+25*j),label,17,COLORS[j])
    text(d,(x0+(x1-x0)/2-60,y1+43),xlabel,18)
    text(d,(x0,y0-23),ylabel,16)
    return px,py

def main():
    OUT.mkdir(exist_ok=True)
    im=Image.new('RGB',(1480,600),'white');d=ImageDraw.Draw(im)
    text(d,(65,20),'Mixed pixels: measured mechanism, not yet a successful repair pipeline',28,bold=True)
    row=load(ROOT/'e0/RESULTS.json')['mixed_color']
    z=np.asarray(row['candidates']);dynamic=np.asarray(row['dynamic_loss'])
    fixed=np.full(len(z),dynamic[1])
    plot(d,(95,145,685,475),(450,900),(0,9000),[('Dynamic coverage',z,dynamic),('Frozen coverage',z,fixed)],
         'A. Analytic two-color edge','Depth (mm)','Raw sensor MSE (gray squared)')
    arr=np.asarray(row['dynamic_coverage']);variance=np.var(arr,axis=0);i=int(np.argmax(variance))
    plot(d,(840,145,1400,475),(450,900),(0,1),[('Foreground area fraction',z,arr[:,i])],
         'B. One boundary pixel','Depth (mm)','Alpha (0 to 1)')
    text(d,(65,562),'E0: analytic area integration; equal-color negative control has zero loss at every candidate.',17)
    im.save(OUT/'01_boundary_mechanism.png')

    im=Image.new('RGB',(1480,620),'white');d=ImageDraw.Draw(im)
    text(d,(65,20),'Correct evidence can be lost when converted into a uniform support interval',27,bold=True)
    with np.load(ROOT/'ordinary_stage/w010_ED_600.npz') as f:g=f['grid'];ld=f['loss']
    with np.load(ROOT/'ordinary_stage/w010_EF_540.npz') as f:lf=f['loss']
    top=float(np.ceil(max(ld.max(),lf.max())/1000)*1000)
    plot(d,(100,150,690,480),(300,1000),(0,top),[('Estimated / dynamic',g,ld),('Estimated / frozen at 540',g,lf)],
         'A. Flat contrast, w010','Depth (mm)','Raw sensor MSE (gray squared)')
    with np.load(ROOT/'ordinary_stage/w002_ED_600.npz') as f:g=f['grid'];l=f['loss']
    px,py=plot(d,(850,150,1410,480),(300,1000),(0,400),[('Estimated dynamic loss',g,l)],
              'B. Textured single plane, w002','Depth (mm)','Raw sensor MSE (gray squared)')
    d.line([(px(486),py(50)),(px(782),py(50))],fill='#999999',width=7)
    for x,c,label,yy in [(600,COLORS[2],'true / best candidate: 600',220),(634,'#777777','support mean: 634',248),(660,COLORS[1],'P selects: 660',276)]:
        d.line([(px(x),py(0)),(px(x),py(48))],fill=c,width=2)
        text(d,(870,yy),label,16,c)
    text(d,(65,566),'Examples selected after evaluation for explanation only. Gray segment = accepted interval [486,782].',17)
    text(d,(65,590),'All curves were sealed before geometric evaluation. No plot-driven method changes.',16)
    im.save(OUT/'02_evidence_vs_decision.png')

    s=load(ROOT/'evaluation/SUMMARY.json');im=Image.new('RGB',(1350,670),'white');d=ImageDraw.Draw(im)
    text(d,(65,25),'E1 endpoint: the new decision pipeline does not beat full9',29,bold=True)
    x0,y0,x1,y1=120,135,1250,495
    for value in (0,15,30,45,60):
        y=y1-value/60*(y1-y0);d.line([(x0,y),(x1,y)],fill='#dddddd');text(d,(65,y-10),value,17)
    text(d,(65,90),'Depth MAE (mm), lower is better',20)
    for j,inc in enumerate(('540','600','660')):
        center=300+j*370
        for k,a in enumerate(('K','N','ED')):
            v=s['summary'][inc][a]['mae'];x=center+(k-1)*72;y=y1-v/60*(y1-y0)
            if v:d.rectangle((x-24,y,x+24,y1),fill=COLORS[k])
            else:d.line([(x-24,y1),(x+24,y1)],fill=COLORS[k],width=4)
            text(d,(x-28,y-26),f'{v:.2f}',17)
        text(d,(center-120,y1+22),{'540':'-60 mm input offset','600':'Correct input','660':'+60 mm input offset'}[inc],19)
    for k,label in enumerate(('KEEP','full9 + P','Estimated dynamic + P')):
        d.rectangle((190+k*335,567,211+k*335,588),fill=COLORS[k]);text(d,(221+k*335,565),label,20)
    text(d,(65,615),'36 image worlds / 24 design groups; 30 distinct image tensors. Paired and duplicate cases are not independent.',16)
    text(d,(65,642),'Simulation development only. ED: 2/36 repairs per offset; 4/36 damaged correct inputs (2 distinct scenes).',16)
    im.save(OUT/'03_endpoint_mae.png')
    dump(OUT/'MANIFEST.json',dict(script_sha256=sha(__file__),files={p.name:sha(p) for p in OUT.glob('*.png')},
        sources=['e0/RESULTS.json','ordinary_stage/w010_ED_600.npz','ordinary_stage/w010_EF_540.npz','ordinary_stage/w002_ED_600.npz','evaluation/SUMMARY.json'],
        scope='descriptive development figures; no inferential uncertainty claimed'))
if __name__=='__main__':main()

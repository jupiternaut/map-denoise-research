"""Small specialized multi-frame layer probes; estimators never receive Truth."""
from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
import sys
import numpy as np

OLD = Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py')
spec = importlib.util.spec_from_file_location('frozen_t2', OLD)
old = importlib.util.module_from_spec(spec); sys.modules[spec.name] = old; spec.loader.exec_module(old)
Input = old.Input


@dataclass(frozen=True)
class Config:
    family: str = 'dual'
    gap: float = 6.
    sigma: float = 1.
    imbalance: float = .9
    points_per_frame: int = 160
    frames: int = 8
    bias_amplitude: float = 4.
    crossed_fraction: float = 1.
    noise_factor: float = 1.
    normal_degrees: float = 0.
    slope: float = 0.
    outlier_fraction: float = 0.


def ray_hits(rng, frame, frames, gap, count, proportion):
    """Visible hits on two finite parallel opaque plates, not random overlap labels.

    Camera at z=220 mm. Rear at z=0, x in [-60,15]; front z=gap,
    x in [-15,60]. Select among physically visible hits to vary scan coverage.
    """
    camera = np.array([-35.+70.*frame/max(frames-1,1), -80., 220.])
    q = np.c_[rng.uniform(-70,70, max(count*30,6000)),
              rng.uniform(-45,45,max(count*30,6000)), np.zeros(max(count*30,6000))]
    direction = q-camera
    hits=[]; valid=[]
    for z, low, high in [(0.,-60.,15.),(gap,-15.,60.)]:
        t=(z-camera[2])/direction[:,2]
        hit=camera+t[:,None]*direction
        mask=(hit[:,0]>=low)&(hit[:,0]<=high)&(abs(hit[:,1])<=50)&(t>0)
        hits.append(hit); valid.append(mask)
    labels=np.where(valid[1],1,0)  # front intersection is always nearer
    usable=valid[0]|valid[1]
    physical=np.where(labels[:,None]==1,hits[1],hits[0])
    n1=int(round(count*proportion)); indices=[]
    for label,n in [(0,count-n1),(1,n1)]:
        choices=np.flatnonzero(usable&(labels==label))
        if n and len(choices)==0: raise RuntimeError('Ray generator has no requested visible surface')
        if n: indices.extend(rng.choice(choices,n,replace=len(choices)<n).tolist())
    indices=np.array(indices); rng.shuffle(indices)
    p=physical[indices]
    return p,labels[indices],np.repeat(camera[None],count,axis=0)


def generate(config, seed):
    c=config; rng=np.random.default_rng(seed); nf=c.frames; n=c.points_per_frame
    b=c.bias_amplitude*np.sin(np.arange(nf)*1.37+.13*seed); b-=b.mean()
    if c.family=='ambiguous_single': b=np.r_[np.full(nf//2,-c.gap/2),np.full(nf-nf//2,c.gap/2)]
    if c.family=='ambiguous_dual': b=np.zeros(nf)
    p_all=[]; gt_all=[]; labels_all=[]; frames=[]; origins=[]; ratios=[]
    for f in range(nf):
        prob=.5+c.imbalance*(f/max(nf-1,1)-.5)
        if f>=int(round(nf*c.crossed_fraction)): prob=float(f>=nf/2)
        if c.family.startswith('ambiguous'): prob=float(f>=nf/2)
        if c.family=='single': prob=0.
        if c.family=='raycast':
            clean,lab,origin=ray_hits(rng,f,nf,c.gap,n,prob)
        else:
            xy=rng.uniform(-40.,40.,(n,2))
            # Exact deterministic counts: avoids accidental loss of a rare layer.
            lab=np.r_[np.zeros(n-int(round(n*prob)),int),np.ones(int(round(n*prob)),int)]
            rng.shuffle(lab)
            clean=np.c_[xy,c.gap*lab]
            origin=np.full((n,3),np.nan)
            if c.family=='ambiguous_single': clean[:,2]=c.gap/2;lab[:]=0
        obs=clean.copy(); eps=rng.normal(0,c.sigma,n)
        obs[:,2]+=b[f]+eps
        if c.family.startswith('ambiguous'):
            obs[:,2]=c.gap*float(f>=nf/2)+eps
        obs[:,2]+=c.slope*np.sin(.9*f)*clean[:,0]
        if c.outlier_fraction:
            take=rng.choice(n,int(n*c.outlier_fraction),replace=False)
            obs[take,2]+=rng.choice([-1.,1.],len(take))*rng.uniform(7*c.sigma,14*c.sigma,len(take))
        p_all.append(obs);gt_all.append(clean);labels_all.append(lab);origins.append(origin)
        frames.append(np.full(n,f));ratios.append(float(lab.mean()))
    obs=np.concatenate(p_all);truth=np.concatenate(gt_all);fid=np.concatenate(frames)
    angle=np.deg2rad(c.normal_degrees)
    R=np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
    inp=Input(obs@R.T,fid,np.zeros(len(fid),int),c.sigma*c.noise_factor,
              max(8.,2*c.bias_amplitude))
    gt={'xyz_mm':truth,'labels':np.concatenate(labels_all),'bias_mm':b,
        'rotation':R,'origins_mm':np.concatenate(origins),'proportions':np.array(ratios),
        'gap':None if c.family in ('single','ambiguous_single') else c.gap}
    return inp,gt


def evaluate(output_mm, bias_mm, truth, inp):
    output=output_mm@truth['rotation']; gt=truth['xyz_mm'];err=output-gt
    labels=truth['labels'];frame=inp.frame
    clean_levels=np.unique(gt[:,2]);z=output[:,2]
    nearest=np.min(abs(z[:,None]-clean_levels),axis=1)
    frame_mean_error=np.array([err[frame==f,2].mean() for f in np.unique(frame)])
    result={'normal_mae_mm':float(abs(err[:,2]).mean()),
            'point_rmse_mm':float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),
            'nearest_surface_mae_mm':float(nearest.mean()),
            'frame_residual_std_mm':float(np.std(frame_mean_error)),
            'output_z_std_mm':float(np.std(z)),
            'mean_displacement_mm':float(np.mean(np.linalg.norm(output_mm-inp.xyz_mm,axis=1))),
            'bias_rmse_mm':float(np.sqrt(np.mean((np.asarray(bias_mm)-truth['bias_mm'])**2)))}
    if truth['gap'] is not None:
        gap=float(z[labels==1].mean()-z[labels==0].mean())
        band=min(1.,truth['gap']/4)
        result.update(gap_hat_mm=gap,gap_abs_error_mm=abs(gap-truth['gap']),
                      gap_retention=gap/truth['gap'],
                      layer_confusion_fraction=float(np.mean(abs(err[:,2])>truth['gap']/2)),
                      min_layer_accuracy=float(min(np.mean(abs(err[labels==l,2])<=band) for l in (0,1))))
    else:
        result.update(gap_hat_mm=None,gap_abs_error_mm=None,gap_retention=None,
                      layer_confusion_fraction=None,min_layer_accuracy=None)
    return result


def suite(stage):
    items=[]
    if stage=='development':
        for gap in [2.,3.,4.,6.,8.,12.]:
            for imbalance in [0.,.7,.9]:
                for n in [40,160]:
                    items.append(('core',Config(gap=gap,imbalance=imbalance,points_per_frame=n)))
        base=asdict(Config())
        for field,values in [('noise_factor',[.5,1.5,2.]),('normal_degrees',[1.,3.,6.]),
                             ('slope',[.02,.05]),('outlier_fraction',[.03,.1]),
                             ('crossed_fraction',[.25,.5,.75]),('bias_amplitude',[0.,2.,8.,16.])]:
            for value in values: items.append((field,Config(**{**base,field:value})))
        for family in ['raycast','single','ambiguous_single','ambiguous_dual']:
            items.append((family,Config(**{**base,'family':family,'gap':8.})))
    else:
        # Chosen before development output; new seeds and off-grid parameters.
        for family in ['dual','raycast']:
            for gap in [3.5,5.,7.,10.]:
                items.append(('replication',Config(family=family,gap=gap,imbalance=.85,points_per_frame=96)))
        items.extend([('replication',Config(family='single',gap=8.,points_per_frame=96)),
                      ('replication',Config(gap=7.,normal_degrees=2.,points_per_frame=96)),
                      ('replication',Config(gap=7.,slope=.03,points_per_frame=96))])
    return items

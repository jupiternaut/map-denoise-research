"""Moderate-SNR/unequal-occupancy development probe. No GT enters estimate."""
import sys,json,tempfile,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2')
from experiment import Input,evaluate,estimate as old_estimate
from joint_framewise import METHODS,estimate

def make(gap,sigma,extreme,n,seed):
    rng=np.random.default_rng(seed);nf=10;b=3*np.sin(np.arange(nf)*1.37+seed*.123);b-=b.mean()
    points=[];clean=[];lab=[];frames=[]
    for f in range(nf):
        p=.5 if not extreme else .02+.96*f/(nf-1)
        label=(rng.random(n)<p).astype(int);xy=rng.uniform(-30,30,(n,2));z=gap*label
        points.append(np.c_[xy,z+b[f]+rng.normal(0,sigma,n)]);clean.append(np.c_[xy,z]);lab.append(label);frames.append(np.full(n,f))
    xyz=np.concatenate(points);inp=Input(xyz,np.concatenate(frames),np.zeros(len(xyz),int),sigma,8.)
    truth={'clean_xyz_mm':np.concatenate(clean),'layer':np.concatenate(lab),'bias_mm':b,'gap_mm':gap}
    return inp,truth

def main():
    root=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/candidate');root.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='moderate-snr-',dir=root));(run/'outputs').mkdir();rows=[];print(run,flush=True)
    for gap in (4.,5.,6.):
      for extreme in (False,True):
       for seed in (411,413):
        inp,truth=make(gap,1.2,extreme,100,seed);case=f'gap{gap}-extreme{int(extreme)}-s{seed}'
        for method in ('joint_forced','joint_guarded')+METHODS:
            st=time.perf_counter();out,b,info=(old_estimate(inp,method) if method.startswith('joint_') else estimate(inp,method));duration=time.perf_counter()-st
            path=run/'outputs'/f'{case}-{method}.npz';np.savez_compressed(path,xyz_mm=out,bias_mm=b)
            with np.load(path) as d:m=evaluate(inp,truth,d['xyz_mm'],d['bias_mm'])
            r={'case':case,'gap':gap,'extreme':extreme,'seed':seed,'method':method,'status':info['status'],'elapsed_s':duration,'metrics':m,'info':info}
            rows.append(r)
            with open(run/'records.jsonl','a') as fp:fp.write(json.dumps(r)+'\n')
        print(case,'done',flush=True)
    (run/'results.json').write_text(json.dumps({'scope':'development', 'records':rows},indent=2));print('COMPLETED',run,flush=True)
if __name__=='__main__':main()

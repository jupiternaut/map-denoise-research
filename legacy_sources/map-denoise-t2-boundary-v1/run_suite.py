"""Freeze legal observations, run named operators, save outputs before evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
from common import Input, old, Config, generate, evaluate, suite, OLD

ROOT=Path(__file__).resolve().parent
RUN=Path('/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704')
MODULES={'candidate':ROOT/'candidate/joint_framewise.py',
         'alternative':ROOT/'alternative/split_consensus.py'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def write(path,value):
    with Path(path).open('x') as out:json.dump(plain(value),out,ensure_ascii=False,indent=2)


def dataset(path,stage,seeds):
    path.mkdir(parents=True,exist_ok=False);entries=[]
    for number,(group,c) in enumerate(suite(stage)):
        for seed in seeds:
            inp,truth=generate(c,seed);key=f'{number:03d}-{seed}'
            np.savez_compressed(path/f'{key}.npz',xyz_mm=inp.xyz_mm,frame=inp.frame,roi=inp.roi,
                                sigma_mm=inp.sigma_mm,bias_bound_mm=inp.bias_bound_mm)
            np.savez_compressed(path/f'{key}-EVAL.npz',**{k:v for k,v in truth.items() if k!='gap'},
                                gap=np.nan if truth['gap'] is None else truth['gap'])
            entries.append({'key':key,'group':group,'config':asdict(c),'seed':seed,
                            'input_sha256':sha(path/f'{key}.npz')})
    write(path/'manifest.json',{'stage':stage,'seeds':seeds,'entries':entries,
          'common_source_sha256':sha(ROOT/'common.py'),
          'scope':'specialized synthetic probes; raycast is analytic finite-plate visibility, not a real scan'})
    print('DATASET',path,len(entries),flush=True)


def load_module(prefix,path):
    spec=importlib.util.spec_from_file_location(f'integrated_{prefix}',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def run(data_path,destination,methods,groups=None,extra=None):
    destination.mkdir(parents=True,exist_ok=False);(destination/'outputs').mkdir()
    manifest=json.loads((data_path/'manifest.json').read_text())
    modules={'old':old};files={'old':OLD}
    if extra:
        for entry in extra:
            prefix,path=entry.split('=',1);MODULES[prefix]=Path(path)
    for method in methods:
        prefix,_=method.split(':',1)
        if prefix not in modules:
            modules[prefix]=load_module(prefix,MODULES[prefix]);files[prefix]=MODULES[prefix]
    frozen={str(p):sha(p) for p in [ROOT/'common.py',ROOT/'run_suite.py',*files.values()]}
    write(destination/'manifest.json',{'dataset':str(data_path),'methods':methods,'groups':groups,
          'source_hashes':frozen,'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}})
    rows=[];started=time.perf_counter()
    for index,entry in enumerate(manifest['entries']):
        if groups and entry['group'] not in groups:continue
        key=entry['key'];path=data_path/f'{key}.npz';assert sha(path)==entry['input_sha256']
        with np.load(path) as f:
            inp=Input(f['xyz_mm'],f['frame'],f['roi'],float(f['sigma_mm']),float(f['bias_bound_mm']))
        with np.load(data_path/f'{key}-EVAL.npz') as f: truth={k:f[k] for k in f.files}
        truth['gap']=None if np.isnan(truth['gap']) else float(truth['gap'])
        before=inp.xyz_mm.copy()
        for method in methods:
            prefix,name=method.split(':',1);start=time.perf_counter()
            try:
                output,bias,info=modules[prefix].estimate(inp,name);elapsed=time.perf_counter()-start
                assert np.array_equal(inp.xyz_mm,before),'Estimator mutated input'
                assert output.shape==before.shape and np.isfinite(output).all()
                outpath=destination/'outputs'/f'{key}__{name}.npz'
                np.savez_compressed(outpath,xyz_mm=output,bias_mm=bias)
                write(outpath.with_suffix('.json'),info)
                with np.load(outpath) as f:metrics=evaluate(f['xyz_mm'],f['bias_mm'],truth,inp)
                rows.append({**entry,'method':method,'seconds':elapsed,'status':info.get('status','UNKNOWN'),
                             'metrics':metrics})
            except Exception as error:
                rows.append({**entry,'method':method,'seconds':time.perf_counter()-start,
                             'status':'ERROR','error':repr(error)})
        if index%12==0:print(f'{destination.name}: case {index+1}/{len(manifest["entries"])}',flush=True)
    assert all(sha(path)==digest for path,digest in frozen.items()),'Source changed during run'
    write(destination/'results.json',{'records':rows,'elapsed_s':time.perf_counter()-started,
          'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'source_hashes':frozen,'errors':sum(r['status']=='ERROR' for r in rows)})
    print(json.dumps({'output':str(destination),'records':len(rows),'errors':sum(r['status']=='ERROR' for r in rows),
                      'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','replication'],default='development')
    p.add_argument('--build',action='store_true');p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--seeds',default='31013,31019,31033');p.add_argument('--output',type=Path)
    p.add_argument('--methods');p.add_argument('--groups');p.add_argument('--module',action='append')
    a=p.parse_args()
    if a.build:dataset(a.dataset,a.stage,[int(s) for s in a.seeds.split(',')])
    if a.methods:run(a.dataset,a.output,a.methods.split(','),a.groups.split(',') if a.groups else None,a.module)

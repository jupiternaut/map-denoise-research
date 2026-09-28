from pathlib import Path
import os,sys,json,hashlib,socket,importlib.util
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent
PREV=Path('/home/grf/Documents/Codex/2026-09-26/reserved-evidence-lab-20260926T011717Z')
BASE=Path('/home/grf/Documents/Codex/2026-09-26/relative-gain-lab-20260926T004220Z')
OLD=Path('/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z')
CLOSEOUT=Path('/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z')
DATA=Path('/srv/slam-research/grf/map-denoise/datasets')
SCENES=(55,65,69)
CONDS=('native','minus1','plus1','minus3','plus3')
SEED=20260926
sys.path.extend([str(CLOSEOUT),str(CLOSEOUT/'package'),str(PREV),str(BASE)])
spec=importlib.util.spec_from_file_location('base_io_cross',BASE/'common.py')
prior_io=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior_io)
sha,save_json,save_npz=prior_io.sha,prior_io.save_json,prior_io.save_npz
read_points,write_points=prior_io.read_points,prior_io.write_points
seal,verify_seal,check_host=prior_io.seal,prior_io.verify_seal,prior_io.check_host
in_box,observed,voxel=prior_io.in_box,prior_io.observed,prior_io.voxel

def cases_for_scene(scene):
    raw=OLD/'real_results' if scene in (24,37) else CLOSEOUT/'confirmation'/f'scan{scene}'
    return sorted(p for p in raw.glob(f'scan{scene}_*__*') if p.is_dir())

def cached_features(path,candidate='B'):
    filename='features.npz' if path.parent.name=='real_results' else 'FEATURES.npz'
    with np.load(path/filename,allow_pickle=False) as z:return z[candidate+'_all_post']

def apply_threshold(score,spec):
    if spec['action']=='keep':return np.zeros(len(score),bool)
    if spec['action']=='all':return np.ones(len(score),bool)
    return score>spec['threshold']

"""Reuse frozen data, not prior writable outputs. Exact host: liekkas."""
from pathlib import Path
import os, sys, json, importlib.util
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=Path('/srv/slam-research/grf/map-denoise/runs/visibility-revision-20260926T085529Z')
JOINT=Path('/home/grf/Documents/Codex/2026-09-26/joint-revision-lab-20260926T080710Z')
def module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj
prior=module('frozen_joint_data',JOINT/'common.py')
for key in ('CROSS','RESERVED','BASE','OLD','CLOSEOUT','DATA','SCENES','CONDS','SEED',
            'sha','save_json','save_npz','read_points','write_points','seal','verify_seal',
            'check_host','cases_for_scene','development_metadata','load_observations','io'):
    globals()[key]=getattr(prior,key)
old_router=module('frozen_joint_router',JOINT/'router.py')
routes=old_router.routes
materialize=old_router.materialize
case_weights=old_router.case_weights
def new_evidence(case,ids=None):
    sid=int(case.name.split('_')[0][4:])
    with np.load(OUT/'evidence'/f'scan{sid}'/case.name/'VISIBILITY.npz',allow_pickle=False) as z:
        expected=np.arange(len(z['row_ids'])) if ids is None else ids
        assert np.array_equal(z['row_ids'],expected)
        return z['features']

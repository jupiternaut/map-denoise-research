"""Frozen observation adapter; actual methods do not open evaluation references."""
from pathlib import Path
import os,sys,json,importlib.util
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=Path('/srv/slam-research/grf/map-denoise/runs/multisurface-field-20260926T102842Z')
CONT=Path('/home/grf/Documents/Codex/2026-09-26/continuous-step-lab-20260926T100617Z')
CONT_OUT=Path('/srv/slam-research/grf/map-denoise/runs/continuous-step-20260926T100617Z')
def module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
prior=module('frozen_continuous_common',CONT/'common.py')
for k in ('JOINT','VIS','VIS_CODE','RECOVERY','CROSS','RESERVED','BASE','OLD','CLOSEOUT','DATA','SCENES','CONDS',
          'sha','save_json','save_npz','read_points','write_points','seal','verify_seal','check_host',
          'cases_for_scene','io','frozen_file','geometry_and_route','selected_endpoint'):
    globals()[k]=getattr(prior,k)
ACTUAL_ARMS=('identity','prior_recovery','point_wta','single_field','multi_field','multi_field_visibility','multi_field_graph')

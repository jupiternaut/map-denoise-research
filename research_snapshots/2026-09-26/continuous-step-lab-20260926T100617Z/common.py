"""Frozen observation/data utilities; references are opened only by evaluate.py."""
from pathlib import Path
import os, sys, json, importlib.util, hashlib, socket
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parent
OUT = Path('/srv/slam-research/grf/map-denoise/runs/continuous-step-20260926T100617Z')
JOINT = Path('/home/grf/Documents/Codex/2026-09-26/joint-revision-lab-20260926T080710Z')
VIS = Path('/srv/slam-research/grf/map-denoise/runs/visibility-revision-20260926T085529Z')
VIS_CODE = Path('/home/grf/Documents/Codex/2026-09-26/visibility-revision-lab-20260926T085529Z')
RECOVERY = 'base_shallow__recovery'
def module(name, path):
    if name in sys.modules: return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj
prior = module('continuous_frozen_data', JOINT/'common.py')
for key in ('CROSS','RESERVED','BASE','OLD','CLOSEOUT','DATA','SCENES','CONDS',
            'sha','save_json','save_npz','read_points','write_points','seal',
            'verify_seal','check_host','cases_for_scene','io'):
    globals()[key] = getattr(prior, key)

def frozen_file(path, folder, sources):
    path, folder = Path(path), Path(folder)
    seal_path = folder/'SEALED.json'
    record = json.loads(seal_path.read_text())
    digest = sha(path)
    if record['files'][str(path.relative_to(folder))] != digest:
        raise AssertionError('frozen input changed: '+str(path))
    sources[str(path)] = digest
    sources[str(seal_path)] = sha(seal_path)

def geometry_and_route(case, sources):
    points = []
    for name in ('identity','A_all','B_all'):
        path = case/(name+'.ply')
        frozen_file(path, case.parent, sources)
        points.append(read_points(path))
    path = VIS/'inference'/case.parent.name/case.name/'routes.npz'
    frozen_file(path, path.parent.parent, sources)
    with np.load(path, allow_pickle=False) as z:
        route = z[RECOVERY]
    if route.dtype != np.uint8 or route.shape != (len(points[0]),) or np.any(route>2):
        raise AssertionError('invalid frozen route')
    if any(x.shape != points[0].shape for x in points):
        raise AssertionError('row mismatch')
    return np.stack(points, axis=1), route

def selected_endpoint(geometry, route):
    return geometry[np.arange(len(route)), route]

def materialize_step(geometry, route, step):
    p = geometry[:,0]
    return p + np.asarray(step)[:,None]*(selected_endpoint(geometry,route)-p)

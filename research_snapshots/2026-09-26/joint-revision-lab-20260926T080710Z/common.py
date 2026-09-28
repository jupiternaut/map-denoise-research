"""Thin adapter to frozen observations; evaluation references live outside this API."""
from pathlib import Path
import os, sys, json, importlib.util
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parent
CROSS = Path('/home/grf/Documents/Codex/2026-09-26/boundary-cross-lab-20260926T014111Z')
RESERVED = Path('/home/grf/Documents/Codex/2026-09-26/reserved-evidence-lab-20260926T011717Z')
BASE = Path('/home/grf/Documents/Codex/2026-09-26/relative-gain-lab-20260926T004220Z')
OLD = Path('/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z')
CLOSEOUT = Path('/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z')
DATA = Path('/srv/slam-research/grf/map-denoise/datasets')
SEED = 20260926
SCENES = (55, 65, 69)
CONDS = ('native', 'minus1', 'plus1', 'minus3', 'plus3')
sys.path.extend([str(CLOSEOUT), str(CLOSEOUT/'package'), str(RESERVED), str(BASE)])
spec = importlib.util.spec_from_file_location('joint_frozen_io', BASE/'common.py')
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
sha, save_json, save_npz = io.sha, io.save_json, io.save_npz
read_points, write_points = io.read_points, io.write_points
seal, verify_seal, check_host = io.seal, io.verify_seal, io.check_host

def cases_for_scene(scene):
    base = OLD/'real_results' if scene in (24, 37) else CLOSEOUT/'confirmation'/f'scan{scene}'
    return sorted(p for p in base.glob(f'scan{scene}_*__*') if p.is_dir())

def load_observations(path, rows=None, support=False):
    """Two matching [N,D] feature matrices; row geometry is [N,3,3] in mm."""
    scene = int(path.name.split('_')[0][4:])
    key = 'features.npz' if scene in (24,37) else 'FEATURES.npz'
    with np.load(path/key, allow_pickle=False) as z:
        xa, xb = z['A_all_post'], z['B_all_post']
    ids = np.arange(len(xa)) if rows is None else rows
    geometry = np.stack([read_points(path/(name+'.ply'))[ids]
                         for name in ('identity','A_all','B_all')], axis=1)
    arrays = []
    for candidate, source, x in (('A', RESERVED, xa), ('B', CROSS, xb)):
        with np.load(source/'evidence'/f'scan{scene}'/path.name/'PAIRED.npz') as z:
            assert np.array_equal(z['row_ids'], ids), 'evidence row identity'
            arrays.append(np.column_stack([x[ids], z['R']]).astype(np.float32))
    if support:
        with np.load(ROOT/'evidence'/f'scan{scene}'/path.name/'SUPPORT.npz') as z:
            assert np.array_equal(z['row_ids'], ids), 'support row identity'
            arrays = [np.column_stack([arrays[0], z['A']]),
                      np.column_stack([arrays[1], z['B']])]
    return np.stack(arrays,axis=1), geometry

def development_metadata():
    with np.load(BASE/'data/train.npz',allow_pickle=False) as z:
        d = {k:z[k] for k in ('scene','condition','case_id','row_id','calibration_support','e0','e1')}
    with np.load(CROSS/'data/train.npz',allow_pickle=False) as z:
        for key in ('scene','condition','case_id','row_id','calibration_support','e0'):
            assert np.array_equal(d[key],z[key]), key
        d['losses'] = np.column_stack([d['e0'],d.pop('e1'),z['e1']])
    assert d['losses'].shape==(79594,3) and np.isfinite(d['losses']).all()
    d['records'] = json.loads((BASE/'data/TRAIN_MANIFEST.json').read_text())['records']
    return d

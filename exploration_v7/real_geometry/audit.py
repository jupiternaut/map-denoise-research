"""Independently score saved plane equations on the saved held-out point IDs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'exploration_v6'/'direction'))
import direction_pooling as wrapper

FROZEN=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/direction-v6-edpxxdsb')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path);args=parser.parse_args()
    if socket.gethostname()!='liekkas':raise RuntimeError('exact host required')
    dest=args.run_dir.resolve()
    expected=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
    assert dest.parent==expected and dest.name.startswith('real-geometry-v7-')
    models=json.loads((dest/'TRAINING_MODELS.json').read_text())
    model_lookup={(r['patch'],r['cell'],r['scale'],r['k'],r['sharing'],r['scan_key']):r for r in models}
    rows=list(csv.DictReader((dest/'HOLDOUT.csv').open()))
    pool=wrapper._load_private()._V4
    local_data={}
    for patch in sorted({r['patch'] for r in rows}):
        with np.load(FROZEN/'real'/'inputs'/f'{patch}__zero.npz',allow_pickle=False) as a:
            world,frames=a['xyz_world'],a['scan_id']
        info=json.loads((FROZEN/'outputs'/f'real__{patch}__zero__pooled_pca.json').read_text())['info']
        order=pool._canonical_order((world-world.mean(0))*1000.,frames)
        ids,scans=np.unique(frames[order],return_inverse=True)
        basis=np.asarray(info['basis_calls'][0]['basis_world'])
        local=((world[order]-world[order].mean(0))*1000.)@basis
        local_data[patch]=(local,ids,scans)
    comparisons=0;scored=0;splitchecks=0
    for row in rows:
        patch,cell=row['patch'],int(row['cell']);local,ids,scans=local_data[patch]
        sid=int(np.flatnonzero(ids==int(row['evaluation_scan_id']))[0])
        with np.load(dest/'splits'/f'{patch}__c{cell}.npz',allow_pickle=False) as a:
            train=a[row['scale']+'_train_sorted_indices'];test=a['common_test_sorted_indices']
        assert not np.intersect1d(train,test).size;splitchecks+=1
        test=test[scans[test]==sid]
        assert len(test)==int(row['test_points'])
        if row['status']!='SCORED':continue
        key=-1 if row['sharing']=='joint' else sid
        model=model_lookup[(patch,cell,row['scale'],int(row['per_fit_capacity']),row['sharing'],key)]
        centers=np.asarray(model['centers']);normals=np.asarray(model['normals'])
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-12)
        # Plane offsets and matrix multiplication avoid production broadcasting.
        offsets=np.sum(centers*normals,axis=1)
        distances=abs(local[test]@normals.T-offsets)
        selected=np.argmin(distances,axis=1)
        orth=distances[np.arange(len(test)),selected]
        cosine=abs(normals[selected,2])
        axis=np.divide(orth,cosine,out=np.full_like(orth,np.inf),where=cosine>1e-12)
        values={'orthogonal_median_mm':np.median(orth),'orthogonal_rms_mm':np.sqrt(np.mean(orth**2)),
                'axis_cosine_median':np.median(cosine),'axis_cosine_min':np.min(cosine)}
        if np.isfinite(axis).any():values['axis_distance_median_mm']=np.median(axis[np.isfinite(axis)])
        for name,value in values.items():
            np.testing.assert_allclose(float(row[name]),value,rtol=1e-10,atol=1e-8);comparisons+=1
        assert int(row['coverage_count'])==len(test)
        scored+=1
    before=json.loads((dest/'PROTECTED_BEFORE.json').read_text())
    for path,digest in before.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    result=dict(scored_rows_independently_recomputed=scored,numeric_comparisons=comparisons,
        disjoint_split_checks=splitchecks,protected_files_rechecked=len(before),passed=True,
        scope='independent scoring of saved surfaces/splits; does not establish correct physics or denoising improvement')
    with (dest/'AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

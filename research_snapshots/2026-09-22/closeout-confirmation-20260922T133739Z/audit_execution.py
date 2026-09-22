"""Independent row and artifact audit; does not read evaluator references."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scene_adapter import sha,save
ROOT=Path(__file__).resolve().parent
def read(p):return np.asarray(o3d.io.read_point_cloud(str(p)).points).copy()
def main():
    prep=json.loads((ROOT/'PREPARATION_AUDIT.json').read_text())
    for n,h in prep['sha256'].items():
        if sha(ROOT/n)!=h:raise AssertionError('preparation changed '+n)
    records=[];sources=[];max_bias_error=0
    for sid in (40,55,65,69):
        folder=ROOT/('adaptation' if sid==40 else 'confirmation')/f'scan{sid}'
        seal=json.loads((folder/'SEALED.json').read_text());sources.append(seal['source'])
        for n,h in seal['files'].items():
            if sha(folder/n)!=h:raise AssertionError('sealed artifact changed '+n)
        rois=json.loads((folder/'ROIS.json').read_text())
        for roi in rois:
            if roi['status']!='READY':raise AssertionError('missing ROI')
            native=read(folder/(roi['id']+'__native')/'identity.ply')
            for c,bias in [('native',0.),('minus1',-1.),('plus1',1.),('minus3',-3.),('plus3',3.)]:
                case=folder/(roi['id']+'__'+c)
                p=read(case/'identity.ply');a=read(case/'A_all.ply');b=read(case/'B_all.ply')
                err=float(np.max(abs(np.linalg.norm(p-native,axis=1)-abs(bias))))
                max_bias_error=max(max_bias_error,err)
                if err>1e-10:raise AssertionError('injection magnitude')
                with np.load(case/'DECISIONS.npz') as f:
                    decisions={k:f[k] for k in f.files}
                if not np.array_equal(decisions['post_A_keep'],(decisions['pred_post_A']>0).astype(np.int8)):
                    raise AssertionError('main threshold mismatch')
                expected_ab=np.argmax(np.column_stack([np.zeros(len(p)),decisions['pred_post_A'],decisions['pred_post_B']]),axis=1)
                if not np.array_equal(expected_ab,decisions['post_AB_keep']):raise AssertionError('secondary choice mismatch')
                if np.sum(decisions['random_A_keep']!=0)!=np.sum(decisions['post_A_keep']!=0):raise AssertionError('count control mismatch')
                for arm in ('post_A_keep','post_AB_keep','random_A_keep'):
                    mask=decisions[arm];expected=np.where((mask==1)[:,None],a,np.where((mask==2)[:,None],b,p))
                    if not np.array_equal(read(case/(arm+'.ply')),expected):raise AssertionError('output routing mismatch '+arm)
                records.append(dict(scene=sid,case=case.name,rows=len(p),routing_bitwise=True))
        print('AUDIT_PASS',sid,flush=True)
    if not all(s==sources[0] for s in sources):raise AssertionError('constructors differ across scenes')
    if len(records)!=80:raise AssertionError('case count')
    save(ROOT/'EXECUTION_AUDIT.json',dict(cases=80,records=records,original_preparation_unchanged=True,
        constructor_hashes_identical_across_scenes=True,reference_access=False,
        max_injected_magnitude_error_mm=max_bias_error,sealed_files_verified=True))
    print('AUDIT_ALL_PASS',80,flush=True)
if __name__=='__main__':main()

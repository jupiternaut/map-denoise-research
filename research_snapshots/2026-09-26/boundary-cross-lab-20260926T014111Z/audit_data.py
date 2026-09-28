"""Independent B label/feature identity audit; development labels only."""
from common import *


def main():
    check_host()
    verify_seal(ROOT/'data')
    verify_seal(BASE/'data')
    with np.load(ROOT/'data/train.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    with np.load(BASE/'data/train.npz',allow_pickle=False) as z:
        previous={k:z[k] for k in z.files}
    assert len(data['X'])==79594 and int(data['calibration_support'].sum())==78598
    assert set(np.unique(data['scene']))=={24,37}
    for key in ('e0','scene','condition','case_id','row_id','calibration_support'):
        assert np.array_equal(data[key],previous[key]),key
    manifest=json.loads((BASE/'data/TRAIN_MANIFEST.json').read_text())
    labels_seal=json.loads((OLD/'selector_results/SEALED.json').read_text())['files']
    raw_seal=json.loads((OLD/'real_results/SEALED.json').read_text())['files']
    checks=[]
    for rec in manifest['records']:
        sid=rec['scene'];other=37 if sid==24 else 24
        mask=data['case_id']==rec['case_id'];ids=data['row_id'][mask]
        label=OLD/'selector_results'/f'train{sid}_test{other}'/(rec['case']+'_training_labels.npz')
        feature=OLD/'real_results'/rec['case']/'features.npz'
        assert sha(label)==labels_seal[str(label.relative_to(OLD/'selector_results'))]
        assert sha(feature)==raw_seal[str(feature.relative_to(OLD/'real_results'))]
        with np.load(label,allow_pickle=False) as z:
            assert np.array_equal(z['row_ids'],ids)
            assert np.array_equal(z['B_gain'],data['gain'][mask])
            gain_difference_rows=int(np.sum(z['A_gain']!=z['B_gain']))
        with np.load(feature,allow_pickle=False) as z:
            assert np.array_equal(z['B_all_post'][ids],data['X'][mask])
            feature_difference_rows=int(np.sum(np.any(z['A_all_post'][ids]!=z['B_all_post'][ids],axis=1)))
        np.testing.assert_array_equal(data['e1'][mask],np.maximum(data['e0'][mask]-data['gain'][mask],0))
        assert np.isfinite(data['X'][mask]).all() and np.isfinite(data['e1'][mask]).all()
        assert np.min(data['e1'][mask])>=0
        checks.append(dict(case=rec['case'],rows=int(mask.sum()),B_gain_differs_from_A_rows=gain_difference_rows,
                           B_features_differ_from_A_rows=feature_difference_rows))
    assert len(checks)==24
    assert sum(r['B_gain_differs_from_A_rows'] for r in checks)>0
    assert sum(r['B_features_differ_from_A_rows'] for r in checks)>0
    reproduction=json.loads((ROOT/'data/B_LABEL_REPRODUCTION.json').read_text())
    assert reproduction['status']=='PASS' and reproduction['maximum_absolute_difference']<=1e-8
    save_json(ROOT/'AUDIT_DATA.json',dict(status='PASS',checks=checks,rows=len(data['X']),
        calibration_rows=int(data['calibration_support'].sum()),replay_labels_read=False,
        independently_verified_archived_B_labels=True,metadata_identical_to_A=True,
        constructor_features_are_B=True,target_definition='B_gain/(identity_error2+B_error2+0.01)',
        development_geometry_recomputation_report=reproduction['maximum_absolute_difference'],
        source_sha256={str(ROOT/'audit_data.py'):sha(ROOT/'audit_data.py'),str(ROOT/'data/SEALED.json'):sha(ROOT/'data/SEALED.json')}))
    print('AUDIT DATA PASS',len(checks),'cases',flush=True)


if __name__=='__main__':main()

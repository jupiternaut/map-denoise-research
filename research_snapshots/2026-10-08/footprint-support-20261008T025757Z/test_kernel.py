import json
from pathlib import Path
import unittest
import numpy as np
from kernel import weighted_ncc,score,support,select

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'METHOD.json').read_text())

class KernelTests(unittest.TestCase):
    def test_full_weights_match_old_scorer(self):
        yy,xx=np.mgrid[:128,:128]
        img=120+30*np.sin(xx*.43)+20*np.cos(yy*.29)
        images=np.stack([img,img,img])
        camera=dict(K=np.array([[160.,0,64],[0,160,64],[0,0,1]]),R=np.eye(3),C=np.zeros(3))
        cams=[camera,camera,camera];grid=np.array([500.,600.,700.])
        for arm in ('full9','connected9'):
            out,intervals=score(images,cams,grid,P,arm)
            old=support.score_support(img,img,camera,camera,[64,64],grid,'plane',arm)
            np.testing.assert_allclose(out['scores'][0],old['scores'],atol=1e-14,equal_nan=True)
            self.assertEqual(intervals,support.support_intervals(grid,np.isfinite(old['scores'])&(old['scores']>=.6)))

    def test_mass_gate_not_just_ess(self):
        a=np.arange(81,dtype=float)[None,:];b=np.stack([a,a]);valid=np.ones_like(b,bool)
        normal=weighted_ncc(a,b,np.ones_like(a),valid,P)
        tiny=weighted_ncc(a,b,np.ones_like(a)*.001,valid,P)
        self.assertTrue(np.isfinite(normal['scores']).all());self.assertTrue(np.isnan(tiny['scores']).all())
        np.testing.assert_allclose(normal['ess'],tiny['ess'])

    def test_component_uses_identical_majority_mask(self):
        yy,xx=np.mgrid[:128,:128];img=120+30*np.sin(xx*.43)+20*np.cos(yy*.29)
        images=np.stack([img,img,img]);area=np.ones_like(images)*.7
        camera=dict(K=np.array([[160.,0,64],[0,160,64],[0,0,1]]),R=np.eye(3),C=np.zeros(3))
        fields=dict(area=area,center=np.ones_like(images),numerator=images*area)
        results=[score(images,[camera]*3,np.array([500.,600.,700.]),P,a,fields)[0] for a in ('oracle_majority','oracle_component')]
        np.testing.assert_equal(results[0]['weights'],results[1]['weights'])
        np.testing.assert_allclose(results[0]['scores'],results[1]['scores'],atol=1e-14)

    def test_empty_keeps_and_multimodal_mean_is_explicit(self):
        config={'incumbent_mm':900,'depths_mm':[450,540,600,660,900]}
        self.assertEqual(select([],config)['selected_depth'],900)
        out=select([[530,550],[650,670]],config)
        self.assertEqual(out['support_mean'],600);self.assertEqual(out['selected_depth'],600)

if __name__=='__main__':unittest.main()

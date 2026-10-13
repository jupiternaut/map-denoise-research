import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import search_method as m
from experiment import verify_one

DEV=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z/data/dev/public.json')
OBS=json.loads(DEV.read_text())['observations'][0]['observation']

class AdapterTest(unittest.TestCase):
    def test_base_unchanged(self):
        m.warm_load('base')
        old=m.OLD.prepare(OBS,6.,'certified');new=m.prepare(OBS,6.,'base')
        self.assertEqual(old['cells'],new['cells'])
        self.assertTrue(verify_one(OBS,new)['valid'])

    def test_joint_cover_and_zero(self):
        m.warm_load('joint')
        zero=m.prepare(OBS,0.,'joint')
        self.assertEqual((zero['lower'],zero['upper']),(540.,660.))
        self.assertTrue(verify_one(OBS,zero)['valid'])
        s=m.prepare(OBS,6.,'joint')
        self.assertTrue(verify_one(OBS,s)['valid'])

    def test_refusal_keeps(self):
        m.warm_load('joint')
        class Refused:
            def __init__(self,*args): self.tiles=()
            def check_cell(self,*args):return {'verified':False,'certificate':None,'reason':'forced_refusal'}
        with patch.object(m.OLD.CERT,'ExactContext',Refused),patch.object(m.JOINT,'JointContext',Refused):
            s=m.prepare(OBS,6.,'joint')
        self.assertFalse(s['exclusions'])
        self.assertEqual((s['lower'],s['upper']),(540.,660.))

    def test_tolerance_restored(self):
        m.warm_load('fine');before=m.FREE.CELL_TOLERANCE
        s=m.prepare(OBS,6.,'fine')
        self.assertEqual(m.FREE.CELL_TOLERANCE,before)
        self.assertTrue(verify_one(OBS,s)['valid'])

if __name__=='__main__':unittest.main()

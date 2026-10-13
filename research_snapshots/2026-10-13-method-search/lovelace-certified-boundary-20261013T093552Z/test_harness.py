import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import search_method as m

DEV=json.loads((m.PARENT/'task_bundle/dev.json').read_text())

class HarnessTest(unittest.TestCase):
    def test_numeric_same_as_frozen_search(self):
        obs=DEV['observations'][0]['observation']
        original=m.FREE.prepare(obs,6.)
        recorded=m.prepare(obs,6.,'numeric')
        self.assertEqual(original['lower'],recorded['lower'])
        self.assertEqual(original['upper'],recorded['upper'])
        self.assertEqual(original['cells'],recorded['cells'])
        self.assertEqual(original['stats']['infeasible_cells'],len(recorded['exclusions']))

    def test_zero_budget_keeps(self):
        s=m.prepare(DEV['observations'][0]['observation'],0.,'numeric')
        self.assertEqual((s['lower'],s['upper']),(540.,660.))
        self.assertEqual(len(s['cells']),3)
        self.assertEqual(m.decode(s,601.4,'numeric'),601.4)

    def test_refused_certificate_cannot_prune(self):
        class Context:
            def __init__(self,observation): pass
            def check_cell(self,*args):
                return {'verified':False,'certificate':None,'reason':'deliberate_test_refusal'}
        with patch.object(m,'CERT',SimpleNamespace(ExactContext=Context),create=True):
            s=m.prepare(DEV['observations'][0]['observation'],6.,'certified')
        self.assertEqual(s['exclusions'],[])
        self.assertEqual((s['lower'],s['upper']),(540.,660.))
        self.assertTrue(s['certificate_attempts'])

if __name__=='__main__':unittest.main()

import unittest
from baseline import choose

class TestP(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(choose([], [1,2,3],2)['selected_depth'],2)
    def test_mean_and_tie(self):
        r=choose([[0,1],[3,4]],[1,3,4],4)
        self.assertEqual(r['support_mean'],2)
        self.assertEqual(r['selected_depth'],1)
    def test_correct_current(self):
        self.assertEqual(choose([[1,3]],[1,2,3],2)['selected_depth'],2)
    def test_weighted(self):
        self.assertAlmostEqual(choose([[0,2],[4,5]],[0,1,2],0)['support_mean'],6.5/3)

if __name__=='__main__':unittest.main()

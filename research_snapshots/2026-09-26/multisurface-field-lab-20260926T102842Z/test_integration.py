import unittest
import numpy as np
from infer import choose
from graph import couple

class ChoiceTests(unittest.TestCase):
    def test_unknown_always_keep(self):
        label,_=choose(np.array([[1.,0.,0.]]),np.array([[0.,3.,-3.]]),np.array([False]))
        self.assertEqual(label[0],0)
    def test_tie_keep_and_displacement_units(self):
        label,cost=choose(np.array([[.1,.1,.1],[.2,.1,.12]]),
                          np.array([[0.,0.,0.],[0.,3.,-3.]]),np.array([True,True]))
        np.testing.assert_array_equal(label,[0,1])
        self.assertAlmostEqual(cost[1,1],.145)
    def test_graph_does_not_merge_strongly_supported_two_layers(self):
        uv=np.column_stack((np.arange(12),np.zeros(12)))
        labels=np.where(np.arange(12)<6,1,2)
        unary=np.ones((12,3));unary[np.arange(12),labels]=0.
        result,_=couple(uv,np.zeros(12),np.zeros(12,int),unary,np.ones(12,bool))
        np.testing.assert_array_equal(result,labels)

if __name__=='__main__':unittest.main()

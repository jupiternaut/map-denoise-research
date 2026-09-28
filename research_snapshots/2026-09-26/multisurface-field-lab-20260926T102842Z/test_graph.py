import unittest
import numpy as np
from graph import couple,edges_for_patches

class GraphTests(unittest.TestCase):
    def test_duplicate_uv_never_self_edges(self):
        uv=np.array([[0.,0],[0,0],[1,0]])
        edge,_=edges_for_patches(uv,np.zeros(3),np.zeros(3,int))
        self.assertTrue(np.all(edge[:,0]!=edge[:,1]))
        np.testing.assert_array_equal(edge,[[0,1],[0,2],[1,2]])
    def test_disjoint(self):
        uv=np.array([[0.,0],[1,0],[2,0]])
        edge,_=edges_for_patches(uv,np.zeros(3),np.arange(3))
        self.assertEqual(len(edge),0)
    def test_no_regression_energy_and_missing(self):
        uv=np.column_stack((np.arange(12),np.zeros(12)))
        u=np.tile([.1,0.,1.],(12,1));u[5]=[0,.03,1]
        support=np.ones(12,bool);support[7]=False
        labels,m=couple(uv,np.zeros(12),np.zeros(12,int),u,support)
        self.assertEqual(labels[7],0)
        self.assertLessEqual(m['final_energy'],m['initial_energy'])
        self.assertEqual(labels[5],1)
    def test_image_edge(self):
        uv=np.array([[0.,0],[1,0]])
        _,w0=edges_for_patches(uv,np.zeros(2),np.zeros(2,int))
        _,w1=edges_for_patches(uv,np.array([0.,1]),np.zeros(2,int))
        self.assertLess(w1[0],w0[0]*1e-10)

if __name__=='__main__':unittest.main()

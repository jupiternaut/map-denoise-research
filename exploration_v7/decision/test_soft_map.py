import inspect
import unittest
import numpy as np
from soft_map import project_soft_map


class DecisionTests(unittest.TestCase):
    def fixture(self):
        world=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
        order=np.array([2,0,3,1]);height=np.array([100.,200.,300.,400.])
        design=np.c_[np.ones(4),np.arange(4.),np.zeros(4)]
        return dict(xyz_world_m=world,order=order,design_sorted=design,local_height_sorted_mm=height,
                    active_sorted=np.array([0,1,3]),support_sorted=np.array([True,False,False,True]),
                    normal_world=np.array([0.,0.,1.]),group_ids_world=np.array([1,1,0,-1]),
                    coefficients_mm=np.array([[101.,2.,0.],[410.,3.,0.]]))

    def test_coordinate_order_and_support(self):
        a=self.fixture();out,support=project_soft_map(**a)
        expected=a['xyz_world_m'].copy()
        expected[2,2]+=.001
        expected[1,2]+=.019
        np.testing.assert_allclose(out,expected,atol=1e-15)
        np.testing.assert_array_equal(support,[False,True,True,False])
        np.testing.assert_array_equal(out[~support],a['xyz_world_m'][~support])

    def test_input_immutable(self):
        a=self.fixture();before={k:v.copy() for k,v in a.items()}
        project_soft_map(**a)
        for k,v in a.items():np.testing.assert_array_equal(v,before[k])

    def test_no_truth_or_file_argument(self):
        names=inspect.signature(project_soft_map).parameters
        self.assertEqual(set(names),set(self.fixture()))
        with self.assertRaises(TypeError):project_soft_map(**self.fixture(),gt_layer=np.zeros(4))

    def test_reject_inactive_support(self):
        a=self.fixture();a['support_sorted'][2]=True
        with self.assertRaises(ValueError):project_soft_map(**a)

    def test_row_permutation_equivariance(self):
        a=self.fixture();out,support=project_soft_map(**a)
        perm=np.array([3,2,0,1]);inverse=np.argsort(perm)
        b={k:v.copy() for k,v in a.items()}
        b['xyz_world_m']=a['xyz_world_m'][perm]
        b['group_ids_world']=a['group_ids_world'][perm]
        b['order']=inverse[a['order']]
        other,mask=project_soft_map(**b)
        np.testing.assert_array_equal(other,out[perm]);np.testing.assert_array_equal(mask,support[perm])


if __name__=='__main__':unittest.main()

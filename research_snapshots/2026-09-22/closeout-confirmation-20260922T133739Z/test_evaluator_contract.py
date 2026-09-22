"""Small counterexamples for the new fixed-native-row evaluator contract."""
import numpy as np
from evaluate_closeout import metrics,dist,observed,in_box
from scene_adapter import save
from pathlib import Path

def main():
    checks=[]
    ref=np.array([[0.,0,0],[10,0,0]])
    m=metrics(ref,ref)
    assert m['E_sym_mm']==0 and m['fscore_1mm']==1
    checks.append('identity equals zero and P/R/F equal one')
    q=ref+np.array([0,0,3.]);d=dist(q,ref)
    assert np.array_equal(d,np.array([3.,3.])) and np.mean(d*d)==9
    checks.append('known orthogonal displacement gives 9 mm2 MSE')
    native=np.array([[0.,0,0],[1.,0,0]]);lo=np.array([-1.,-1,-1]);hi=np.array([2.,1,1])
    support=in_box(native,lo,hi);shift=native+np.array([0,0,3.])
    assert support.sum()==2 and in_box(shift,lo,hi).sum()==0
    assert np.mean(dist(shift,native)[support]**2)==9
    checks.append('moving all points out of AABB cannot erase source MSE')
    empty=metrics(np.zeros((0,3)),ref)
    assert empty['E_sym_mm']==1000 and empty['recall_1mm']==0
    checks.append('empty output is not a zero-error success')
    obs=dict(BB=np.array([[0,0,0],[2,2,2]]),Res=np.array([1.]),ObsMask=np.ones((3,3,3),bool))
    assert observed(np.array([[0.,0,0],[-2,0,0],[4,0,0]]),obs).tolist()==[True,False,False]
    checks.append('observation grid bounds are enforced')
    save(Path(__file__).resolve().parent/'EVALUATOR_TESTS.json',dict(passed=len(checks),checks=checks,new_reference_access=False))
    print('EVALUATOR_CONTRACT_PASS',len(checks))

if __name__=='__main__':main()

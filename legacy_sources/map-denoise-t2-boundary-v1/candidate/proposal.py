"""Offline unconstrained-output adapter to the frozen joint estimator.

The fitted bias remains constrained exactly as in joint_framewise.py; only the
post-fit support WAIT is removed. This is geometric-potential evaluation, not a
production safety or observability claim. No GT input is accepted.
"""
import time
import numpy as np
from joint_framewise import _fit, CONFIG

METHODS=('framewise_proposal_hard','framewise_proposal_soft','framewise_proposal_global_pi')

def estimate(inp,method):
    if method not in METHODS:raise KeyError(method)
    start=time.perf_counter();fit=_fit(inp,per_frame=method!='framewise_proposal_global_pi')
    hard=method=='framewise_proposal_hard';out=np.asarray(inp.xyz_mm,dtype=float).copy()
    target=np.asarray(inp.roi)==0
    out[target,2]=fit['mu'][np.argmax(fit['r'],axis=1)] if hard else fit['r']@fit['mu']
    if (~target).any():out[~target,2]=fit['anchor_mu']
    b=np.zeros(int(np.max(inp.frame))+1);b[fit['frame_ids'].astype(int)]=fit['b']
    info={'status':'APPLY','output_policy':'offline_proposal_no_support_gate','k':fit['k'],
          'mu_mm':fit['mu'].tolist(),'gap_mm':float(np.ptp(fit['mu'])),'score':fit['score'],
          'frame_pi':fit['pi'].tolist(),'frame_ids':fit['frame_ids'].tolist(),
          'observable':fit['observable'],'crossed_frames':fit['crossed_frames'],
          'would_wait_in_guarded_adapter':not fit['observable'],'start':fit['start'],
          'iterations':fit['iterations'],'candidate_scores':fit['candidate_scores'],
          'projection':'MAP' if hard else 'posterior_mean','config':CONFIG.copy(),
          'requires_ground_truth':False,'elapsed_s':time.perf_counter()-start}
    return out,b,info

"""Independent numeric audit helpers; no estimator or old evaluator imports."""
from __future__ import annotations
import numpy as np


def rms_mm(delta):
    return float(1000*np.sqrt(np.mean(np.einsum('ij,ij->i',delta,delta))))


def decompose(delta, axis):
    axis=np.asarray(axis,float)
    axis=axis/np.linalg.norm(axis)
    height=np.asarray(delta,float)@axis
    tangent=delta-height[:,None]*axis
    return dict(xyz_rms_mm=rms_mm(delta),
                normal_rms_mm=float(1000*np.sqrt(np.mean(height**2))),
                tangent_rms_mm=rms_mm(tangent))


def real_scores(output, current, reference, zero_output, axis):
    result={}
    for label,delta in (
        ('recovery',output-reference),('zero_edit',zero_output-reference),
        ('self_response',output-zero_output),('input_edit',output-current),
        ('injected',current-reference)):
        result.update({label+'_'+k:v for k,v in decompose(delta,axis).items()})
    return result


def axis_bounds(output, current, reference, axis, supported):
    """Pointwise allowed-axis floor, tighter with fixed unsupported rows.

    This is an intervention-restoration bound to the measured reference, not
    physical geometry truth. It allows an arbitrary per-point displacement.
    """
    n=np.asarray(axis,float);n=n/np.linalg.norm(n)
    delta=current-reference
    orth=delta-(delta@n)[:,None]*n
    limited=orth.copy()
    limited[~supported]=delta[~supported]
    edit=output-current
    off_axis=edit-(edit@n)[:,None]*n
    if not np.array_equal(output[~supported],current[~supported]):
        raise AssertionError('unsupported points changed')
    if rms_mm(off_axis)>1e-8:
        raise AssertionError('output not in advertised one-axis space')
    if rms_mm(output-reference)+1e-8<rms_mm(limited):
        raise AssertionError('reported output violates geometric movement floor')
    return dict(axis_only_floor_mm=rms_mm(orth),
                axis_and_support_floor_mm=rms_mm(limited),
                off_axis_edit_rms_mm=rms_mm(off_axis),
                actual_recovery_xyz_rms_mm=rms_mm(output-reference))


def synthetic_scores(output, reference, rectangles_mm, source_layers, true_gap_mm=None):
    """Finite rectangle distance + point correspondence + labelled affine gap."""
    q=np.asarray(output,float)*1000
    nearest=np.full(len(q),np.inf)
    for z,xmin,xmax,ymin,ymax in rectangles_mm:
        closest=np.column_stack((np.clip(q[:,0],xmin,xmax),
                                 np.clip(q[:,1],ymin,ymax),np.full(len(q),z)))
        nearest=np.minimum(nearest,np.linalg.norm(q-closest,axis=1))
    scores=dict(surface_accuracy_mean_mm=float(nearest.mean()),
                surface_accuracy_rms_mm=float(np.sqrt(np.mean(nearest**2))),
                surface_accuracy_p95_mm=float(np.quantile(nearest,.95)),
                matched_point_rms_mm=rms_mm(output-reference))
    fits={}
    for label in np.unique(source_layers):
        take=q[source_layers==label]
        design=np.column_stack((take[:,:2],np.ones(len(take))))
        if len(take)>=6 and np.linalg.matrix_rank(design)==3:
            fits[int(label)]=np.linalg.lstsq(design,take[:,2],rcond=None)[0]
    if true_gap_mm is not None and 0 in fits and 1 in fits:
        gap=float(fits[1][2]-fits[0][2])
        scores['fitted_gap_at_same_xy_mm']=gap
        scores['fitted_gap_at_same_xy_error_mm']=abs(gap-float(true_gap_mm))
    return scores

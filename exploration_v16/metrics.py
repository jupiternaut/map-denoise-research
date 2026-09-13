"""Evaluation only. Truth never imported by model fitting."""
import numpy as np
from scipy.spatial import cKDTree


def error_decomposition(out, counterfactual, reference, support=None):
    geometry = (counterfactual-reference)*1000.
    assignment = (out-counterfactual)*1000.
    total = (out-reference)*1000.
    a = float(np.mean(np.sum(geometry**2, axis=1)))
    b = float(np.mean(np.sum(assignment**2, axis=1)))
    cross = float(2*np.mean(np.sum(geometry*assignment, axis=1)))
    mse = float(np.mean(np.sum(total**2, axis=1)))
    result = dict(counterfactual_geometry_mse_mm2=a, assignment_delta_mse_mm2=b,
                  geometry_assignment_cross_mm2=cross, total_mse_mm2=mse,
                  decomposition_error_mm2=abs(mse-a-b-cross))
    if support is not None and np.any(support):
        result.update({f'supported_{k}':v for k,v in error_decomposition(out[support], counterfactual[support], reference[support]).items()})
    return result


def dependence_diagnostics(inp, truth):
    x = inp['design']
    labels = truth['gt_layer'].astype(float)
    one = np.linalg.lstsq(x, inp['height_mm'], rcond=1e-12)[0]
    kreg = np.linalg.lstsq(x, labels, rcond=1e-12)[0]
    ereg = np.linalg.lstsq(x, truth['eps_mm'], rcond=1e-12)[0]
    expected = truth['true_slope']+float(truth['true_gap_mm'])*kreg[1:]+ereg[1:]
    centered = x[:, 1:]-x[:, 1:].mean(0)
    cov = centered.T@(labels-labels.mean())/len(labels)
    corr = float(np.corrcoef(x[:, 1], labels)[0,1]) if labels.std() else 0.
    return dict(actual_x_layer_correlation=corr, cov_u_layer=cov.tolist(),
                ols_slope=one[1:].tolist(), label_ols_slope=kreg[1:].tolist(),
                noise_ols_slope=ereg[1:].tolist(), ols_predicted_slope=expected.tolist(),
                ols_identity_error=float(np.max(abs(one[1:]-expected))))


def score_new(inp, truth, art):
    q = art['xyz_world']*1000.
    ref = truth['gt_clean_xyz_world']*1000.
    labels = truth['gt_layer']
    gap = float(truth['true_gap_mm'])
    levels = truth['true_means_mm']
    # All XY stay in the finite rectangles and are unchanged by this module.
    np.testing.assert_array_equal(art['xyz_world'][:, :2], inp['xyz_world'][:, :2])
    d = np.min(abs(q[:, 2, None]-levels), axis=1)
    per_source = [float(np.mean(abs(q[labels==k, 2]-levels[k]))) for k in np.unique(labels)]
    result = dict(surface_mae_mm=float(d.mean()), matched_rms_mm=float(np.sqrt(np.mean(np.sum((q-ref)**2, axis=1)))),
                  balanced_source_mae_mm=float(np.mean(per_source)), worst_source_mae_mm=max(per_source),
                  coverage_1mm=float(np.mean(cKDTree(q).query(ref)[0] <= 1.+1e-9)),
                  model_k=int(art['k']), slope_error=float(np.linalg.norm(art['slope']-truth['true_slope'])),
                  slope_angle_deg=float(np.degrees(np.arctan(np.linalg.norm(art['slope'])/50.))))
    means, slope = art['means'], art['slope']
    groups = art['groups']
    cf = q.copy()
    if gap and len(means) == 2:
        cf[:, 2] = means[labels]+inp['design'][:, 1:]@slope
        alpha = float(np.mean(groups[labels==0] == 1))
        beta = float(np.mean(groups[labels==1] == 0))
        intercept_gap = float(means[1]-means[0])
        slope_part = float((inp['design'][labels==1, 1:].mean(0)-inp['design'][labels==0, 1:].mean(0))@slope)
        identity_gap = intercept_gap*(1-alpha-beta)+slope_part
        result.update(alpha=alpha, beta=beta, assignment_error=.5*(alpha+beta),
                      fitted_gap_mm=intercept_gap, fitted_perpendicular_gap_mm=intercept_gap/np.sqrt(1+np.sum((slope/50.)**2)),
                      source_gap_from_assignment_mm=intercept_gap*(1-alpha-beta), source_gap_from_slope_mm=slope_part)
    elif gap:
        cf[:, 2] = means[0]+inp['design'][:, 1:]@slope
        identity_gap = float((inp['design'][labels==1, 1:].mean(0)-inp['design'][labels==0, 1:].mean(0))@slope)
        result.update(alpha=None, beta=None, assignment_error=None, fitted_gap_mm=0., fitted_perpendicular_gap_mm=0.,
                      source_gap_from_assignment_mm=0., source_gap_from_slope_mm=identity_gap)
    else:
        # No true two-component identity for a split monolayer: do not invent an oracle association.
        result['single_false_split'] = int(len(means) == 2)
    if gap:
        actual_gap = float(q[labels==1, 2].mean()-q[labels==0, 2].mean())
        result.update(fitted_gap_error_mm=abs(result['fitted_gap_mm']-gap), source_gap_mm=actual_gap,
                      source_gap_error_mm=abs(actual_gap-gap), source_gap_relative_error=abs(actual_gap/gap-1),
                      gap_identity_error_mm=abs(actual_gap-identity_gap),
                      balanced_midplane_crossing=.5*(np.mean(q[labels==0,2]>=gap/2)+np.mean(q[labels==1,2]<=gap/2)),
                      counterfactual_source_gap_mm=float(cf[labels==1,2].mean()-cf[labels==0,2].mean()))
        result.update(error_decomposition(q/1000., cf/1000., ref/1000.))
    return result


def score_replay(inp, truth, state, model, output):
    labels = truth['gt_layer']
    reference = truth['gt_clean_xyz_world']
    means, slope = model['means'], model['slope']
    basis = state['basis']
    plane_vector = basis[:,2]-basis[:,:2]@(slope/state['common_scale'])
    vz = float(plane_vector[2])
    result = dict(k=int(model['k']), fitted_normal_angle_deg=float(np.degrees(np.arccos(np.clip(abs(vz)/np.linalg.norm(plane_vector),0,1)))),
                  support=float(state['support'].mean()), plane_vector_world=plane_vector.tolist(),
                  source_geometry_scope='V15 same frozen basis/bias/support; no independent real geometry')
    if len(np.unique(labels)) != 2 or abs(vz) < 1e-6:
        return result, {}
    world_order = np.argsort(means/vz)
    inverse = np.argsort(world_order)
    group_world = np.empty(len(labels), int)
    group_world[state['order']] = inverse[model['groups']]
    ordered_labels = labels[state['order']]
    pred_truth = (means[world_order[ordered_labels]] if len(means) == 2 else means[0])+state['design'][:,1:]@slope
    counterfactual = inp['xyz_world'].copy()
    take = np.flatnonzero(state['support'])
    rows = state['order'][take]
    counterfactual[rows] += ((pred_truth[take]-state['local'][take,2])/1000.)[:,None]*state['normal']
    mask = np.zeros(len(labels), bool)
    mask[rows] = True
    gap = float(truth['true_gap_mm'])
    fitted_gap = float(abs(np.diff(means)[0]/vz)) if len(means) == 2 else 0.
    # Ground-truth points to their globally assigned fitted plane, not nearest plane.
    gt_local_dot = ((reference-state['center'])*1000.)@plane_vector
    expected_intercept = means[world_order[labels]] if len(means) == 2 else means[0]
    plane_error = abs(gt_local_dot-expected_intercept)/np.linalg.norm(plane_vector)
    actual_gap = float((output[labels==1,2].mean()-output[labels==0,2].mean())*1000.)
    cf_gap = float((counterfactual[labels==1,2].mean()-counterfactual[labels==0,2].mean())*1000.)
    result.update(fitted_vertical_gap_mm=fitted_gap, fitted_gap_error_mm=abs(fitted_gap-gap),
                  fitted_perpendicular_gap_mm=float(np.ptp(means)/np.linalg.norm(plane_vector)),
                  source_gap_mm=actual_gap, source_gap_error_mm=abs(actual_gap-gap), counterfactual_source_gap_mm=cf_gap,
                  counterfactual_source_gap_error_mm=abs(cf_gap-gap),
                  fitted_source_plane_mae_mm=float(plane_error.mean()),
                  assignment_error=float(.5*sum(np.mean(group_world[labels==k]!=k) for k in (0,1))) if len(means)==2 else None,
                  supported_assignment_error=float(.5*sum(np.mean(group_world[(labels==k)&mask]!=k) for k in (0,1))) if len(means)==2 and all(np.any((labels==k)&mask) for k in (0,1)) else None)
    result.update(error_decomposition(output, counterfactual, reference, mask))
    return result, dict(counterfactual_xyz_world=counterfactual, world_sorted_group_ids=group_world,
                        source_to_fitted_plane_error_mm=plane_error)

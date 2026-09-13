"""Read-only attribution of saved search versus final hard-refit outputs.

No estimator import, solver, hyperparameter selection or new dataset run.
Truth is read only for explicitly labelled post-hoc geometric diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_summary(world, truth, evaluation):
    q = world * 1000.
    distances = []
    for z, xmin, xmax, ymin, ymax in evaluation['surface_rectangles_mm']:
        dx = q[:, 0] - np.clip(q[:, 0], xmin, xmax)
        dy = q[:, 1] - np.clip(q[:, 1], ymin, ymax)
        distances.append(np.sqrt(dx**2 + dy**2 + (q[:, 2]-z)**2))
    return {'surface_accuracy_mean_mm': float(np.min(distances, axis=0).mean()),
            'matched_point_rms_mm': float(1000*np.sqrt(np.mean(np.sum((world-truth)**2, axis=1))))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records, inputs_used = [], {}
    manifest_path = args.run/'INPUT_MANIFEST.json'
    inputs_used[str(manifest_path)] = fingerprint(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for item in manifest:
        case = item['case']
        if not case.startswith(('dual_g4_', 'dual_g8_')):
            continue
        state_path = args.run/'states'/(case+'.npz')
        truth_path = Path(item['evaluation'])
        inputs_used[str(state_path)] = fingerprint(state_path)
        inputs_used[str(truth_path)] = fingerprint(truth_path)
        state, evaluation = arrays(state_path), arrays(truth_path)
        if 'json' in evaluation:
            evaluation.update(json.loads(evaluation['json'].tobytes().decode()))
        order, active, support = state['order'], state['active'], state['support']
        x = state['design'][active]
        truth = evaluation['gt_clean_xyz_world']
        truth_local = ((truth[order]-state['center'])*1000.) @ state['basis']
        truth_z = truth_local[active, 2]
        labels = evaluation['gt_layer'][order][active]
        eps = evaluation['gt_eps_mm'][order][active]
        selected = support[active]
        weights = state['weights'][active]
        for budget in (0, 1, 3, 6):
            method = 'original_b0_independent' if budget == 0 else f'reassociate_b{budget}_independent'
            path = args.run/'outputs'/f'{case}__{method}.npz'
            meta_path = path.with_suffix('.json')
            inputs_used[str(path)] = fingerprint(path)
            inputs_used[str(meta_path)] = fingerprint(meta_path)
            saved, metadata = arrays(path), json.loads(meta_path.read_text())
            info, row = metadata['info'], metadata['row']
            groups = saved['group_ids'][order][active]
            candidate = saved['candidate_mask'][order][active]
            pred_all = x @ saved['coefficients'].T
            pred_map = pred_all[np.arange(len(active)), groups]
            # Reconstruct only a pre-existing search stage's geometry: saved
            # soft-fit planes + saved hard MAP labels, without another fit.
            pred = state['corrected'].copy()
            pred[active] = pred_map
            stage_ordered = state['ordered_world'].copy()
            stage_ordered[support] += ((pred[support]-state['local'][support, 2])/1000.)[:, None]*state['normal']
            stage_world = state['world'].copy()
            stage_world[order] = stage_ordered
            final_world = saved['xyz_world']
            stage_scores = geometry_summary(stage_world, truth, evaluation)
            final_scores = geometry_summary(final_world, truth, evaluation)
            for key, number in final_scores.items():
                if not np.isclose(number, row[key], atol=1e-11, rtol=0):
                    raise AssertionError('saved-output metric replay differs: '+key)
            if budget == 0 and not np.allclose(stage_world, final_world, atol=1e-14, rtol=0):
                raise AssertionError('original stage arithmetic does not replay output')
            final_local = ((final_world[order]-state['center'])*1000.) @ state['basis']
            final_coeff = np.c_[info['intercepts_mm_at_common_origin'], info['group_slopes_common_xy']]
            ids = np.asarray(info['fitted_dictionary_ids'])
            fitted_id = np.searchsorted(ids, groups)
            final_pred = (x @ final_coeff.T)[np.arange(len(active)), fitted_id]
            group_records = []
            for group in np.unique(groups):
                all_group = groups == group
                take = all_group & selected
                group_weight = weights[all_group]
                raw_error = state['corrected'][active][all_group]-truth_z[all_group]
                fit_error = final_pred[all_group]-truth_z[all_group]
                record = {'group': int(group), 'active_count': int(all_group.sum()),
                          'supported_count': int(take.sum()),
                          'evaluation_only_active_layer_counts': {str(int(k)): int(np.sum(labels[all_group] == k)) for k in np.unique(labels)},
                          'evaluation_only_weighted_raw_error_mean_mm': float(np.average(raw_error, weights=group_weight)),
                          'evaluation_only_weighted_final_fit_error_mean_mm': float(np.average(fit_error, weights=group_weight)),
                          'weighted_intercept_normal_equation_residual_mm': float(np.average(raw_error-fit_error, weights=group_weight))}
                if take.any():
                    record.update(evaluation_only_supported_eps_mean_mm=float(eps[take].mean()),
                                  evaluation_only_supported_raw_error_mean_mm=float((state['corrected'][active][take]-truth_z[take]).mean()),
                                  evaluation_only_supported_pre_refit_error_mean_mm=float((pred_map[take]-truth_z[take]).mean()),
                                  evaluation_only_supported_final_error_mean_mm=float((final_local[active, 2][take]-truth_z[take]).mean()))
                group_records.append(record)
            original_metadata = json.loads((args.run/'outputs'/f'{case}__original_b0_independent.json').read_text())
            origin_cells = {d['group']: d['origin_cells'] for d in original_metadata['info']['dictionary']}
            cells = state['point_cell'][active]
            outside = np.array([int(c) not in origin_cells[int(g)] for c, g in zip(cells, groups)])
            best_candidate = np.min(np.where(candidate, abs(pred_all-truth_z[:, None]), np.inf), axis=1)
            records.append({'case': case, 'method': method, 'budget': budget,
                            'scope': 'post-hoc read-only stage arithmetic; oracle values are evaluation only',
                            'pre_hard_refit_saved_MAP_geometry': stage_scores,
                            'final_saved_geometry': final_scores,
                            'supported_pre_refit_true_normal_mae_mm': float(abs(pred_map-truth_z)[selected].mean()),
                            'supported_final_true_normal_mae_mm': float(abs(final_local[active, 2]-truth_z)[selected].mean()),
                            'evaluation_only_best_reachable_candidate_true_normal_mae_mm': float(best_candidate[selected].mean()),
                            'supported_selected_outside_original_component_cells_fraction': float(outside[selected].mean()),
                            'final_measurement_rms_mm': info['measurement_residual_rms_mm'],
                            'objective_trace': saved['objective_trace'].tolist(),
                            'entropy_mean': info['entropy_mean'],
                            'weightless_group_impurity': row['evaluation_only_weightless_group_impurity'],
                            'groups': group_records})
    after = {path: fingerprint(Path(path)) for path in inputs_used}
    if after != inputs_used:
        raise AssertionError('a source artifact changed during read-only diagnostic')
    result = {'scope': 'saved g4/g8 primary-case stage diagnosis, no estimator or fitting run',
              'source_run': str(args.run), 'source_artifacts_sha256': inputs_used,
              'sources_unchanged': True, 'records': records,
              'ideal_sign_split_gaussian': {'sigma_mm': 1.,
                  'conditional_mean_magnitude_mm': float(np.sqrt(2./np.pi)),
                  'false_plane_gap_mm': float(2.*np.sqrt(2./np.pi)),
                  'interpretation': 'analytical illustrative model, not a fit to these data'}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output), 'records': len(records),
                      'sources_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()

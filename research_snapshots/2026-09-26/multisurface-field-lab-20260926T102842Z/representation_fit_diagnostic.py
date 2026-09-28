"""Fixed-case, observation-only audit of raw field fit vs per-layer clipping."""

import json
import time

import numpy as np

import common as c
from representation_audit import cameras_metadata_only


CASE = 'scan55_roi0__native'


def stats(value):
    value = np.asarray(value, dtype=float).ravel()
    if not len(value):
        return dict(n=0)
    return dict(n=len(value), minimum=float(value.min()), median=float(np.median(value)),
        mean=float(value.mean()), p90=float(np.quantile(value,.9)),
        p95=float(np.quantile(value,.95)), maximum=float(value.max()),
        greater_3_fraction=float(np.mean(value>3.)),
        greater_6_fraction=float(np.mean(value>6.)))


def basis_for(uv, centre):
    u,v=((uv-np.asarray(centre))/32.).T
    return np.column_stack([np.ones(len(uv)),u,v,u*u,u*v,v*v])


def main():
    c.check_host()
    # This is only a resource-slot marker. Do not read evaluation results.
    if not (c.OUT/'evaluation/scan55/SEALED.json').is_file():
        raise RuntimeError('wait for scan55 evaluator to release its CPU slot')
    started=time.monotonic()
    scene=c.OUT/'inference/scan55'
    seal=c.verify_seal(scene)
    case=scene/CASE
    files=[case/x for x in ('POOL.npz','EVIDENCE.npz','FIELDS.json','META.json')]
    sources={str(p):c.sha(p) for p in files}
    for p in files:
        if sources[str(p)]!=seal['files'][str(p.relative_to(scene))]:
            raise AssertionError('sealed inference differs')
    views_path=c.RESERVED/'evidence/scan55/VIEWS.json'
    views=json.loads(views_path.read_text())
    cameras,camera_files=cameras_metadata_only(55,views)
    sources.update({str(p):c.sha(p) for p in [views_path,*camera_files]})
    for p in [views_path,*camera_files]:
        if sources[str(p)]!=seal['source'][str(p)]:
            raise AssertionError('observation camera source changed')
    camera=cameras[views['scan55_roi0']['reference']]
    with np.load(case/'POOL.npz',allow_pickle=False) as z:
        points=z['points'][:,0]
    keys=('uv','depth','pool_offsets','pool_cost','pool_common_count',
          'k1_support','k2_support','patch_index','field_offsets')
    with np.load(case/'EVIDENCE.npz',allow_pickle=False) as z:
        e={k:z[k] for k in keys}
    fields=json.loads((case/'FIELDS.json').read_text())
    rays=points-camera.center
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    zrate=rays@camera.matrix[2]
    active=e['k1_support'] & e['k2_support']
    ids=np.flatnonzero(active)
    raw_offset=np.full((len(points),3),np.nan)
    rho_input_residual=np.full((len(points),3),np.nan)
    photo_best_rho_residual=np.full((len(points),3),np.nan)
    any_candidate_rho_residual=np.full((len(points),3),np.nan)
    photo_weighted_rho_residual=np.full((len(points),3),np.nan)
    patch_records=[]
    for family,columns in (('K1',[0]),('K2',[1,2])):
        for patch in fields[family]:
            core=np.flatnonzero(active & (e['patch_index']==patch['patch']))
            if patch['status']!='fit' or not len(core):
                continue
            basis=basis_for(e['uv'][core],patch['centre_uv'])
            beta=np.asarray(patch['coefficients'])
            predicted_rho=basis@beta.T
            zc=patch['z_center_mm']
            predicted_depth=1./(1./zc+predicted_rho/zc**2)
            displacement=(predicted_depth-e['depth'][core,None])/zrate[core,None]
            raw_offset[np.ix_(core,columns)]=displacement
            input_rho=zc**2*(1./e['depth'][core]-1./zc)
            rho_input_residual[np.ix_(core,columns)]=np.abs(input_rho[:,None]-predicted_rho)
            candidate_depth=e['depth'][core,None]+e['pool_offsets'][core]*zrate[core,None]
            candidate_rho=zc**2*(1./candidate_depth-1./zc)
            residual=np.abs(candidate_rho[:,:,None]-predicted_rho[:,None,:])
            photo_best=e['pool_cost'][core].argmin(axis=1)
            photo_best_rho_residual[np.ix_(core,columns)]=residual[np.arange(len(core)),photo_best]
            any_candidate_rho_residual[np.ix_(core,columns)]=residual.min(axis=1)
            weight=np.exp(-(e['pool_cost'][core]-e['pool_cost'][core].min(axis=1,keepdims=True))/.15)
            weight/=weight.sum(axis=1,keepdims=True)
            photo_weighted_rho_residual[np.ix_(core,columns)]=(residual*weight[:,:,None]).sum(axis=1)
            if family=='K2':
                halo=(np.all(np.abs(e['uv']-np.asarray(patch['centre_uv']))<=32.,axis=1)
                      & (e['pool_common_count']>=2))
                halo_z=e['depth'][halo]
                halo_rho=zc**2*(1./halo_z-1./zc)
                core_z=e['depth'][core]
                core_rho=input_rho
                patch_records.append(dict(patch=patch['patch'],n_core=len(core),n_halo=int(halo.sum()),
                    input_core_depth_span_mm=float(np.ptp(core_z)),
                    input_core_depth_p95_minus_p05_mm=float(np.quantile(core_z,.95)-np.quantile(core_z,.05)),
                    input_halo_depth_span_mm=float(np.ptp(halo_z)),
                    input_halo_depth_p95_minus_p05_mm=float(np.quantile(halo_z,.95)-np.quantile(halo_z,.05)),
                    input_core_rho_span_mm=float(np.ptp(core_rho)),
                    input_halo_rho_span_mm=float(np.ptp(halo_rho)),
                    nearest_input_to_K2_ray_mm=stats(np.min(np.abs(displacement),axis=1)),
                    nearest_input_to_K2_rho_mm=stats(np.min(np.abs(input_rho[:,None]-predicted_rho),axis=1)),
                    nearest_candidate_to_K2_rho_mm=stats(residual.min(axis=(1,2))),
                    best_photo_candidate_to_K2_rho_mm=stats(residual[np.arange(len(core)),photo_best].min(axis=1)),
                    layer_prior=patch['layer_prior']))
    if np.any(~np.isfinite(raw_offset[active])):
        raise AssertionError('shared supported row missing field coefficients')
    expected=np.clip(raw_offset[active],-6.,6.)
    error=float(np.max(np.abs(expected-e['field_offsets'][active,1:])))
    if error>1e-8:
        raise AssertionError('independent raw field reconstruction mismatch')
    motion=raw_offset[active]
    clipped=np.abs(motion[:,1:])>6.
    best_index=e['pool_cost'][active].argmin(axis=1)
    photo_best_offsets=e['pool_offsets'][active,np.arange(9)[best_index]]
    best_to_field=np.abs(motion-photo_best_offsets[:,None])
    candidate_to_field=np.abs(motion[:,None,:]-e['pool_offsets'][active,:,None])
    k2_counts=clipped.sum(axis=1)
    measures={
        'K1_input_to_field_ray_mm':stats(np.abs(motion[:,0])),
        'K2_input_to_nearest_field_ray_mm':stats(np.min(np.abs(motion[:,1:]),axis=1)),
        'K2_individual_layer_input_distance_ray_mm':stats(np.abs(motion[:,1:])),
        'K1_input_to_field_rho_mm':stats(rho_input_residual[active,0]),
        'K2_input_to_nearest_field_rho_mm':stats(rho_input_residual[active,1:].min(axis=1)),
        'K1_best_photo_candidate_to_field_ray_mm':stats(best_to_field[:,0]),
        'K2_best_photo_candidate_to_nearest_field_ray_mm':stats(best_to_field[:,1:].min(axis=1)),
        'K2_any_candidate_to_any_field_ray_mm':stats(candidate_to_field[:,:,1:].min(axis=(1,2))),
        'K1_best_photo_candidate_to_field_rho_mm':stats(photo_best_rho_residual[active,0]),
        'K2_best_photo_candidate_to_nearest_field_rho_mm':stats(photo_best_rho_residual[active,1:].min(axis=1)),
        'K2_any_candidate_to_any_field_rho_mm':stats(any_candidate_rho_residual[active,1:].min(axis=1)),
        'K1_photo_weighted_candidate_absolute_rho_residual_mm':stats(photo_weighted_rho_residual[active,0]),
        'K2_best_fixed_layer_photo_weighted_absolute_rho_residual_mm':stats(photo_weighted_rho_residual[active,1:].min(axis=1)),
    }
    span_keys=('input_core_depth_span_mm','input_core_depth_p95_minus_p05_mm',
               'input_halo_depth_span_mm','input_halo_depth_p95_minus_p05_mm',
               'input_core_rho_span_mm','input_halo_rho_span_mm')
    spans={key:stats([p[key] for p in patch_records]) for key in span_keys}
    result=dict(status='PASS',case=CASE,fixed_case_not_selected_by_quality=True,
        reference_access=False,evaluation_results_read=False,rows=len(points),supported_rows=len(ids),
        raw_fields_before_clipping=True,coordinate_difference_mm=error,
        both_K2_layers_beyond_6mm_fraction=float(np.mean(k2_counts==2)),
        exactly_one_K2_layer_beyond_6mm_fraction=float(np.mean(k2_counts==1)),
        neither_K2_layer_beyond_6mm_fraction=float(np.mean(k2_counts==0)),
        K2_individual_layer_clip_fraction=float(clipped.mean()),
        measurements=measures,patch_spans_unweighted=spans,patches=patch_records,
        seconds=time.monotonic()-started,source_sha256=sources,
        units='ray distance is physical Euclidean displacement; rho mm is centered/scaled inverse depth and only approximately negative depth residual',
        interpretation='per-layer far predictions do not imply every layer is far; nearest-layer residual uses no true surface identity and is not geometry accuracy',
        candidate_residual='candidate positions are shared observation proposals; best-photo uses minimum observed photo cost, not GT; any-candidate minimum is a fit diagnostic only')
    c.save_json(c.ROOT/'REPRESENTATION_FIT_DIAGNOSTIC.json',result)
    lines=['# Fixed-case field-fit diagnostic','',f'Case: `{CASE}` (fixed before looking at geometry quality).',
        'No GT, evaluation results or native-parent geometry were read. All fields below are reconstructed before ±6 mm clipping.', '',
        '| Quantity | Median (mm) | p90 (mm) | >3 mm | >6 mm |','|---|---:|---:|---:|---:|']
    for key in ('K1_input_to_field_ray_mm','K2_individual_layer_input_distance_ray_mm',
                'K2_input_to_nearest_field_ray_mm','K2_best_photo_candidate_to_nearest_field_ray_mm',
                'K2_any_candidate_to_any_field_ray_mm','K2_input_to_nearest_field_rho_mm'):
        s=measures[key]
        lines.append(f"| {key} | {s['median']:.4f} | {s['p90']:.4f} | {100*s['greater_3_fraction']:.2f}% | {100*s['greater_6_fraction']:.2f}% |")
    lines += ['',f"K2 clipping by row: neither layer {100*result['neither_K2_layer_beyond_6mm_fraction']:.2f}%; "
        f"exactly one {100*result['exactly_one_K2_layer_beyond_6mm_fraction']:.2f}%; both {100*result['both_K2_layers_beyond_6mm_fraction']:.2f}%.",
        '', 'The percentage of clipped individual layers is not the percentage of points with no nearby layer.',
        'Any-candidate proximity is more permissive than matching the best photometric candidate, and neither establishes true geometry.',
        'Patch depth/rho spans and all per-patch residuals are preserved in the JSON; patch-span summaries weight patches equally.',
        f"Median core depth span: {spans['input_core_depth_span_mm']['median']:.4f} mm; "
        f"median halo depth span: {spans['input_halo_depth_span_mm']['median']:.4f} mm.",
        f"Independent coefficient reconstruction vs saved clipped offsets: maximum difference {error:.3g} mm.",
        'This diagnostic can expose coarse patch/model mismatch or an unfavorable local fit, but does not distinguish those causes or certify a global optimum.', '']
    with (c.ROOT/'REPRESENTATION_FIT_DIAGNOSTIC.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(status='PASS',seconds=result['seconds'],case=CASE,
        individual_layer_clip=result['K2_individual_layer_clip_fraction'],
        both_layers_clip=result['both_K2_layers_beyond_6mm_fraction'],
        nearest_layer=measures['K2_input_to_nearest_field_ray_mm']),indent=2))


if __name__=='__main__':
    main()

"""Read-only audit of sealed V8 outputs; does not import the new estimator.

Reconstructs WLS as a single block system, integrates selection moments with
adaptive quadrature, and scores output geometry with a separate evaluator.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import sys
import numpy as np
from scipy.integrate import quad

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'exploration_v6'))
from audit_metrics import synthetic_scores

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    with np.load(p, allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}

def write(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve()
    assert socket.gethostname()=='liekkas'
    records=json.loads((run/'OUTPUTS_SEALED_BEFORE_GT.json').read_text())
    rows={r['output']:r for r in csv.DictReader((run/'RESULTS.csv').open())}
    sources=json.loads((run/'SOURCE_MANIFEST.json').read_text())
    protected=json.loads((run/'PROTECTED_BEFORE.json').read_text())
    for p,h in {**sources,**protected}.items():assert digest(p)==h,p
    maximum=dict(projection_mm=0.,metric_mm=0.,quadrature_mm=0.,quadrature_mass=0.)
    integration_count=0; audited=[]; map_violations=0; all_supported=[]
    for record in records:
        target=Path(record['output']);assert digest(target)==record['output_sha256']
        out=read(target);state=read(record['state']);old=read(record['original_artifacts'])
        inputs=read(record['input']);world=inputs['xyz_world'];xyz=out['xyz_world']
        active=state['active'];original=state['order'][active]
        np.testing.assert_array_equal(out['group_ids'],old['group_ids'])
        np.testing.assert_array_equal(out['support_mask'],old['support_mask'])
        np.testing.assert_array_equal(out['scan_id'],inputs['scan_id'])
        np.testing.assert_array_equal(out['source_point_index'],inputs['source_point_index'])
        np.testing.assert_array_equal(xyz[~out['support_mask']],world[~out['support_mask']])
        assert np.isfinite(xyz).all()
        all_supported.append(float(out['support_mask'].mean()))
        group=out['group_ids'][original];ids=np.unique(group);x=state['design'][active]
        # Unlike the estimator's separate fits, one block-diagonal design.
        design=np.hstack([x*(group==g)[:,None] for g in ids])
        weight=np.sqrt(state['weights'][active])
        corrected=state['corrected'][active]-out['subtracted_noise_mm'][original]
        coefficient=np.linalg.lstsq(design*weight[:,None],corrected*weight,rcond=1e-12)[0]
        height=state['local'][:,2].copy();height[active]=design@coefficient
        take=np.flatnonzero(state['support']);original_take=state['order'][take]
        expected=world.copy()
        expected[original_take]+=(height[take]-state['local'][take,2])[:,None]*state['normal']/1000.
        error=float(np.max(abs(expected-xyz))*1000)
        maximum['projection_mm']=max(maximum['projection_mm'],error);assert error<1e-8
        if record['mode']=='conditional':
            # Observe whether the saved assignment really obeys its interval.
            y=state['corrected'][active]
            lower,upper=out['selection_lower_mm'],out['selection_upper_mm']
            map_violations+=int(np.sum((y<lower-1e-8)|(y>upper+1e-8)))
            # Deterministic first/middle/last rows, not chosen by score or size.
            for j in np.unique([0,len(active)//2,len(active)-1]):
                moment=mass=0.
                for mean,prior in zip(out['candidate_mean_mm'][j],out['candidate_prior'][j]):
                    if prior==0:continue
                    density=lambda e: np.exp(-e*e/2)/np.sqrt(2*np.pi)
                    lo,hi=lower[j]-mean,upper[j]-mean # sigma = 1 mm this run
                    mass+=prior*quad(density,lo,hi,epsabs=2e-12)[0]
                    moment+=prior*quad(lambda e:e*density(e),lo,hi,epsabs=2e-12)[0]
                expected_noise=moment/mass
                delta=abs(expected_noise-out['subtracted_noise_mm'][original[j]])
                mass_delta=abs(mass-out['modeled_selection_probability'][j])
                maximum['quadrature_mm']=max(maximum['quadrature_mm'],float(delta))
                maximum['quadrature_mass']=max(maximum['quadrature_mass'],float(mass_delta))
                assert delta<1e-8 and mass_delta<1e-9
                integration_count+=1
        # Evaluator only: read truth after every estimator output was sealed.
        ev=read(Path(record['input']).parent/'evaluation'/(record['case']+'.eval.npz'))
        ev.update(json.loads(ev.pop('json').tobytes().decode()))
        score=synthetic_scores(xyz,ev['gt_clean_xyz_world'],ev['surface_rectangles_mm'],
                               ev['gt_layer'],ev.get('true_gap_mm'))
        row=rows[str(target)]
        for k,v in score.items():
            if row.get(k) not in ('',None):
                delta=abs(v-float(row[k]));assert delta<1e-8,(k,delta)
                maximum['metric_mm']=max(maximum['metric_mm'],delta)
        # A non-extrapolated source-mean separation supplements fitted gap.
        z=xyz[:,2]*1000;labels=ev['gt_layer']
        if set(np.unique(labels))=={0,1}:
            gap=float(z[labels==1].mean()-z[labels==0].mean())
            np.testing.assert_allclose(gap,float(row['source_group_gap_mm']),atol=1e-10)
        audited.append(dict(case=record['case'],budget=record['budget'],mode=record['mode']))
    assert map_violations==0,map_violations
    summary=dict(outputs=len(audited),metric_recomputations=len(audited),
        block_wls_reconstructions=len(audited),independent_quadrature_rows=integration_count,
        maximum_errors=maximum,map_interval_violations=map_violations,
        mean_supported_fraction=float(np.mean(all_supported)),
        outputs_sources_and_120_inputs_unchanged=True,
        audit_source=str(Path(__file__).resolve()),audit_sha256=digest(__file__),
        scope='numeric implementation and evaluator audit, not physical-model validation')
    write(run/'AUDIT.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

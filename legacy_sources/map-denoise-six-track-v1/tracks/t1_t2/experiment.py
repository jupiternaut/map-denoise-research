"""Restricted scalar-normal identifiability and joint bias/layer experiments.

All estimator inputs are in Input; truth is carried separately to the evaluator.
Units inside model: mm. XYZ exports: metres. Provided patch IDs are coarse ROIs,
not target surface labels. ROI 1, if present, is declared a stable single plane.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


@dataclass
class Input:
    xyz_mm: np.ndarray
    frame: np.ndarray
    roi: np.ndarray
    sigma_mm: float
    bias_bound_mm: float = 8.0


def identifiability_probe():
    # Exactly the same legal inputs admit two incompatible truths.
    y = np.array([0., 0., 8., 8.])
    f = np.array([0, 0, 1, 1])
    world_layers = np.array([0., 0., 8., 8.])
    world_ghost = np.full(4, 4.)
    b_ghost = np.array([-4., 4.])
    assert np.array_equal(y, world_layers)
    assert np.array_equal(y, world_ghost + b_ghost[f])
    # Design matrix has one +1 in a frame column and one in a surface column.
    def design(edges, nf=2, ns=2):
        a = np.zeros((len(edges), nf + ns))
        for row, (frame, surf) in enumerate(edges):
            a[row, frame] = a[row, nf + surf] = 1.
        return a
    disconnected = design([(0, 0), (1, 1)])
    crossed = design([(0, 0), (0, 1), (1, 0), (1, 1)])
    # An independently declared single-plane reference also connects sources.
    anchor = design([(0, 0), (1, 1), (0, 2), (1, 2)], ns=3)
    out = {
        "same_input_y_mm": y.tolist(), "same_input_frame": f.tolist(),
        "input_sha256": hashlib.sha256(y.tobytes() + f.tobytes()).hexdigest(),
        "truth_A_z_mm": world_layers.tolist(), "truth_A_bias_mm": [0., 0.],
        "truth_B_z_mm": world_ghost.tolist(), "truth_B_bias_mm": b_ghost.tolist(),
        "indistinguishable_exactly": True,
        "known_association_disconnected_rank": int(np.linalg.matrix_rank(disconnected)),
        "known_association_disconnected_nullity": 4-int(np.linalg.matrix_rank(disconnected)),
        "known_association_crossed_rank": int(np.linalg.matrix_rank(crossed)),
        "known_association_crossed_nullity": 4-int(np.linalg.matrix_rank(crossed)),
        "known_association_anchor_rank": int(np.linalg.matrix_rank(anchor)),
        "known_association_anchor_nullity": 5-int(np.linalg.matrix_rank(anchor)),
        "sigma_bound_mm": 1., "single_plane_max_within_frame_span_mm": 2.,
        "crossed_example_span_mm": 8.,
        "rank_claim_scope": "fixed associations; unknown association adds ambiguities",
    }
    assert out["known_association_crossed_nullity"] == 1
    assert out["known_association_disconnected_nullity"] == 2
    assert out["known_association_anchor_nullity"] == 1
    return out


CASES = (
    "crossed_balanced", "crossed_imbalanced", "single_ghost_no_anchor",
    "thin_segregated_no_anchor", "single_ghost_anchor", "thin_segregated_anchor",
    "crossed_low_snr", "spatial_bias_mismatch", "outlier_mismatch",
)


def generate(case, seed):
    rng = np.random.default_rng(seed)
    nf, n = 10, 160
    sigma = 1.2 if case != "crossed_low_snr" else 2.5
    gap = 8. if case != "crossed_low_snr" else 4.
    anchor = case.endswith("_anchor") and not case.endswith("no_anchor")
    single = case.startswith("single")
    segregated = "segregated" in case or case == "single_ghost_no_anchor"
    b = 3.5 * np.sin(np.arange(nf) * 1.7 + seed * .19)
    b -= b.mean()
    # This pair is bit-identical in legal input, including noise and XY.
    if case in ("single_ghost_no_anchor", "thin_segregated_no_anchor"):
        b = np.r_[np.full(nf//2, -4.), np.full(nf//2, 4.)] if single else np.zeros(nf)
    xyz, f_all, roi_all, clean, label_all = [], [], [], [], []
    for f in range(nf):
        xy = rng.uniform(-50., 50., size=(n, 2))
        if single:
            lab = np.zeros(n, dtype=int)
            z = np.full(n, 4.)
        else:
            prob = .5 if case != "crossed_imbalanced" else .15 + .7*f/(nf-1)
            lab = np.full(n, int(f >= nf//2)) if segregated else (rng.random(n)<prob).astype(int)
            z = gap * lab.astype(float)
        eps = rng.normal(0., sigma, n)
        # Consume identical random numbers for both observationally identical worlds.
        if case == "single_ghost_no_anchor":
            pass
        obs_z = z + b[f] + eps
        if case == "spatial_bias_mismatch":
            obs_z += 5. * np.sin(f*.9) * xy[:, 0] / 50.
        if case == "outlier_mismatch":
            idx = rng.choice(n, int(n*.1), replace=False)
            obs_z[idx] += rng.choice([-1., 1.], len(idx))*rng.uniform(7., 14.,len(idx))
        xyz.append(np.c_[xy, obs_z]); clean.append(np.c_[xy, z])
        f_all.append(np.full(n, f)); roi_all.append(np.zeros(n, int)); label_all.append(lab)
        if anchor:
            na = 80
            xy_ref = rng.uniform(-50.,50.,size=(na,2)) + np.array([130.,0.])
            xyz.append(np.c_[xy_ref,20.+b[f]+rng.normal(0.,sigma,na)])
            clean.append(np.c_[xy_ref,np.full(na,20.)])
            f_all.append(np.full(na,f)); roi_all.append(np.ones(na,int)); label_all.append(np.full(na,2))
    inp = Input(np.concatenate(xyz), np.concatenate(f_all), np.concatenate(roi_all), sigma)
    truth = {"clean_xyz_mm": np.concatenate(clean), "layer": np.concatenate(label_all),
             "bias_mm": b, "gap_mm": None if single else gap, "case":case}
    return inp, truth


def frame_mean(y, f, nf, mask=None):
    if mask is not None:
        y, f = y[mask], f[mask]
    return np.bincount(f, weights=y, minlength=nf)/np.maximum(np.bincount(f,minlength=nf),1)


def fit_layer_mixture(y, sigma):
    """Own scalar fixed-noise mixture, BIC selection, no provenance or truth."""
    best = None
    for k in (1,2):
        for init in range(2 if k == 2 else 1):
            mu = np.quantile(y, [.25,.75]) if k == 2 else np.array([np.mean(y)])
            if k == 2 and init:
                mu = np.array([y.mean()-sigma,y.mean()+sigma])
            pi = np.full(k,1./k)
            for _ in range(60):
                lp = -.5*((y[:,None]-mu)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))+np.log(pi)
                r = np.exp(lp-logsumexp(lp,axis=1)[:,None])
                mass = r.sum(0)+1e-9
                new = (r*y[:,None]).sum(0)/mass
                pi = np.clip(mass/len(y),.01,.99);pi/=pi.sum()
                if np.max(abs(mu-new))<1e-6:
                    mu=new;break
                mu=new
            lp = -.5*((y[:,None]-mu)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))+np.log(pi)
            ll=logsumexp(lp,axis=1)
            r=np.exp(lp-ll[:,None])
            score=-2*ll.sum()+(2*k-1)*np.log(len(y))
            fit={"k":k,"mu":mu,"r":r,"score":score,"pred":r@mu}
            if best is None or score<best["score"]:best=fit
    return best


def filter_xyz(inp, xyz):
    out=xyz.copy(); details={}
    for roi in np.unique(inp.roi):
        m=inp.roi==roi
        fit=fit_layer_mixture(out[m,2],inp.sigma_mm) if roi==0 else {"pred":np.full(m.sum(),out[m,2].mean()),"k":1,"mu":np.array([out[m,2].mean()])}
        out[m,2]=fit["pred"]
        details[str(roi)]={"k":int(fit["k"]),"mu_mm":fit["mu"].tolist()}
    return out,details


def joint_fit(inp):
    """Restricted latent-layer/common-frame-bias EM; all available input declared."""
    y=inp.xyz_mm[:,2]; f=inp.frame; nf=int(f.max())+1; target=inp.roi==0
    reference=~target; has_anchor=bool(reference.any()); sigma=inp.sigma_mm
    count=np.bincount(f,minlength=nf)
    starts=[np.zeros(nf),frame_mean(y,f,nf,target)-y[target].mean()]
    if has_anchor:starts.append(frame_mean(y,f,nf,reference)-y[reference].mean())
    best=None
    for k in (1,2):
        for start in starts:
            b=np.clip(start.copy(),-inp.bias_bound_mm,inp.bias_bound_mm)
            b-=np.average(b,weights=count)
            mu=np.quantile((y-b[f])[target],[.25,.75]) if k==2 else np.array([(y-b[f])[target].mean()])
            a=(y-b[f])[reference].mean() if has_anchor else 0.
            pi=np.full(k,1./k)
            for it in range(100):
                lp=-.5*((y[target,None]-b[f[target],None]-mu)/sigma)**2+np.log(pi)
                r=np.exp(lp-logsumexp(lp,axis=1)[:,None])
                mass=r.sum(0)+1e-12
                mu=(r*(y-b[f])[target,None]).sum(0)/mass
                pi=np.clip(mass/target.sum(),.01,.99);pi/=pi.sum()
                if has_anchor:a=(y-b[f])[reference].mean()
                pred=np.empty(len(y));pred[target]=r@mu
                if has_anchor:pred[reference]=a
                newb=frame_mean(y-pred,f,nf)
                newb=np.clip(newb,-inp.bias_bound_mm,inp.bias_bound_mm)
                gauge=np.average(newb,weights=count);newb-=gauge;mu+=gauge;a+=gauge
                change=np.max(abs(newb-b));b=newb
                if change<1e-6:break
            lp=-.5*((y[target,None]-b[f[target],None]-mu)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))+np.log(pi)
            ll=logsumexp(lp,axis=1);r=np.exp(lp-ll[:,None])
            loglike=ll.sum()
            if has_anchor:loglike+=(-.5*((y[reference]-b[f[reference]]-a)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))).sum()
            bic=-2*loglike+(nf-1+2*k-1+int(has_anchor))*np.log(len(y))
            pred=np.empty(len(y));pred[target]=r@mu
            if has_anchor:pred[reference]=a
            candidate={"score":float(bic),"k":k,"mu":mu.copy(),"anchor_mu":float(a),"b":b.copy(),"pred":pred,"r":r,"iterations":it+1}
            if best is None or bic<best["score"]:best=candidate
    crossed_frames=0
    if best["k"]==2:
        for fid in range(nf):
            r=best["r"][f[target]==fid]
            if (r.sum(0)>=12).all() and ((r>.9).sum(0)>=8).all():crossed_frames+=1
    gap=float(np.ptp(best["mu"]))
    observable=has_anchor or (best["k"]==2 and crossed_frames>=2 and gap>3*sigma)
    best["observable"]=bool(observable);best["crossed_frames"]=crossed_frames
    return best


def estimate(inp, method):
    nf=int(inp.frame.max())+1;y=inp.xyz_mm[:,2];zero=np.zeros(nf)
    if method=="identity":return inp.xyz_mm.copy(),zero,{"status":"UNCHANGED"}
    if method=="xyz_mixture":
        out,info=filter_xyz(inp,inp.xyz_mm)
        return out,zero,{"status":"APPLY","fit":info}
    if method in ("frame_center_then_xyz","anchor_or_frame_center_then_xyz"):
        reference=(inp.roi==1) if method.startswith("anchor") and (inp.roi==1).any() else inp.roi==0
        b=frame_mean(y,inp.frame,nf,reference)-y[reference].mean()
        x=inp.xyz_mm.copy();x[:,2]-=b[inp.frame]
        out,info=filter_xyz(inp,x)
        return out,b,{"status":"APPLY","reference_roi":int(reference[inp.roi==1].all()) if (inp.roi==1).any() else 0,"fit":info}
    if method in ("joint_forced","joint_guarded"):
        fit=joint_fit(inp)
        info={key:fit[key] for key in ("score","k","anchor_mu","iterations","observable","crossed_frames")}
        info["mu_mm"]=fit["mu"].tolist()
        if method=="joint_guarded" and not fit["observable"]:
            info["status"]="WAIT";info["latent_bias_mm"]=fit["b"].tolist()
            return inp.xyz_mm.copy(),zero,info
        out=inp.xyz_mm.copy();out[:,2]=fit["pred"];info["status"]="APPLY"
        return out,fit["b"],info
    if method=="open3d_icp_then_xyz":
        import open3d as o3d
        x=inp.xyz_mm/1000.;out=x.copy();transforms=[]
        chosen_roi=1 if (inp.roi==1).any() else 0
        ref=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x[(inp.frame==0)&(inp.roi==chosen_roi)]))
        for fid in range(nf):
            m=(inp.frame==fid)&(inp.roi==chosen_roi)
            src=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x[m]))
            reg=o3d.pipelines.registration.registration_icp(src,ref,.025,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            T=reg.transformation;allm=inp.frame==fid
            out[allm]=x[allm]@T[:3,:3].T+T[:3,3]
            transforms.append(T.tolist())
        # Remove arbitrary common translation; rotations remain as ICP estimated.
        out+=(x.mean(0)-out.mean(0));out*=1000.
        b=frame_mean(inp.xyz_mm[:,2]-out[:,2],inp.frame,nf)
        out,info=filter_xyz(inp,out)
        return out,b,{"status":"APPLY","reference_roi":chosen_roi,"transforms":transforms,"fit":info,"scope":"Open3D 0.19 ICP plus our shared scalar post-filter; not joint BA"}
    raise KeyError(method)


METHODS=("identity","xyz_mixture","frame_center_then_xyz","anchor_or_frame_center_then_xyz","joint_forced","joint_guarded","open3d_icp_then_xyz")


def evaluate(inp, truth, output, b):
    target=inp.roi==0;gt=truth["clean_xyz_mm"]
    e=output[target]-gt[target];z=output[target,2];labels=truth["layer"][target]
    gap_hat=float(z[labels==1].mean()-z[labels==0].mean()) if truth["gap_mm"] is not None else None
    surface=np.unique(gt[target,2]);nearest=np.min(abs(z[:,None]-surface),axis=1)
    return {"normal_mae_mm":float(abs(e[:,2]).mean()),"normal_rmse_mm":float(np.sqrt(np.mean(e[:,2]**2))),
            "point_rmse_mm":float(np.sqrt(np.mean(np.sum(e**2,axis=1)))),
            "nearest_surface_mae_mm":float(nearest.mean()),
            "layer_gap_hat_mm":gap_hat,
            "layer_gap_abs_error_mm":abs(gap_hat-truth["gap_mm"]) if gap_hat is not None else None,
            "output_normal_span_q95_q05_mm":float(np.quantile(z,.95)-np.quantile(z,.05)),
            "bias_rmse_mm":float(np.sqrt(np.mean((b-truth["bias_mm"])**2))),
            "target_points":int(target.sum())}


def run(root, seeds):
    wall_start=time.perf_counter()
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    (root/"outputs").mkdir();(root/"inputs").mkdir()
    (root/"identifiability.json").write_text(json.dumps(identifiability_probe(),indent=2))
    rows=[]
    for case in CASES:
        for seed in seeds:
            inp,truth=generate(case,seed);key=f"{case}_s{seed:02d}"
            np.savez_compressed(root/"inputs"/f"{key}.npz",xyz_m=inp.xyz_mm/1000.,frame=inp.frame,roi=inp.roi,sigma_mm=inp.sigma_mm,bias_bound_mm=inp.bias_bound_mm)
            # Evaluator truth is a separate file and never passed to estimate.
            np.savez_compressed(root/"inputs"/f"{key}_EVAL_TRUTH.npz",clean_xyz_m=truth["clean_xyz_mm"]/1000.,layer=truth["layer"],bias_mm=truth["bias_mm"])
            for method in METHODS+("oracle_bias_then_xyz",):
                begin=time.perf_counter()
                if method.startswith("oracle"):
                    b=truth["bias_mm"].copy();x=inp.xyz_mm.copy();x[:,2]-=b[inp.frame]
                    output,details=filter_xyz(inp,x);info={"status":"ORACLE_DIAGNOSTIC","fit":details}
                else:output,b,info=estimate(inp,method)
                elapsed=time.perf_counter()-begin
                outpath=root/"outputs"/f"{key}__{method}.npz"
                # Save first, then evaluate that serialized geometry.
                np.savez_compressed(outpath,xyz_m=output/1000.,bias_mm=b)
                (outpath.with_suffix(".json")).write_text(json.dumps(info,indent=2))
                saved=np.load(outpath)
                row={"case":case,"seed":seed,"method":method,"status":info["status"],"elapsed_s":elapsed,**evaluate(inp,truth,saved["xyz_m"]*1000.,saved["bias_mm"])}
                rows.append(row)
            print(case,seed,"done",flush=True)
    (root/"raw_results.json").write_text(json.dumps(rows,indent=2))
    with (root/"raw_results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregate=[]
    for case in CASES:
        for method in METHODS+("oracle_bias_then_xyz",):
            selected=[r for r in rows if r["case"]==case and r["method"]==method]
            a={"case":case,"method":method,"n":len(selected),"wait":sum(r["status"]=="WAIT" for r in selected)}
            for key in ("normal_mae_mm","normal_rmse_mm","point_rmse_mm","nearest_surface_mae_mm","layer_gap_abs_error_mm","bias_rmse_mm","elapsed_s"):
                v=[r[key] for r in selected if r[key] is not None]
                a[key]=float(np.mean(v)) if v else None
            aggregate.append(a)
    (root/"aggregate.json").write_text(json.dumps(aggregate,indent=2))
    metadata={"seeds":seeds,"cases":CASES,"methods":METHODS,"numpy":np.__version__,"threads":{k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},"restriction":"known common normal, supplied noise sigma and bias bound, optional declared planar reference ROI; local scalar model; all exposed development cases", "wall_s":time.perf_counter()-wall_start,"peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"resource_scope":"one whole sequential process; not a per-method memory comparison; first Open3D call includes import; no GPU used", "guard_scope":"support heuristic, not a calibrated identifiability or error guarantee"}
    shutil.copy2(__file__,root/"experiment.py")
    metadata["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print("RESULT",root,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--seeds",default="0,1,2,3,4,5")
    args=p.parse_args();run(args.output,[int(x) for x in args.seeds.split(",")])

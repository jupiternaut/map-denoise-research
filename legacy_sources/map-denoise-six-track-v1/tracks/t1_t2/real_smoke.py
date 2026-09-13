"""Execution-only Oxford smoke; no reference truth and no accuracy claims."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from experiment import Input, estimate


def run(source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);records=[]
    for p in sorted(Path(source).glob("*-sample.npz")):
        data=np.load(p);xyz=np.asarray(data["points"]);normal=np.asarray(data["normal"]).reshape(3)
        normal=normal/np.linalg.norm(normal)
        e=np.array([1.,0.,0.]) if abs(normal[0])<.8 else np.array([0.,1.,0.])
        e=e-normal*np.dot(e,normal);e=e/np.linalg.norm(e);basis=np.stack([e,np.cross(normal,e),normal],axis=1)
        origin=xyz.mean(0);local=(xyz-origin)@basis*1000.
        origframe=np.asarray(data["frame_index"]);_,frame=np.unique(origframe,return_inverse=True)
        inp=Input(local,frame,np.zeros(len(local),int),sigma_mm=10.,bias_bound_mm=50.)
        for method in ("identity","xyz_mixture","joint_forced","joint_guarded"):
            t=time.perf_counter();out,b,details=estimate(inp,method);elapsed=time.perf_counter()-t
            world=out/1000.@basis.T+origin
            target=output/f"{p.stem}__{method}.npz"
            np.savez_compressed(target,points=world,frame_index=origframe,bias_mm=b)
            displacement=np.linalg.norm(world-xyz,axis=1)*1000.
            record={"input":str(p),"method":method,"n_points":len(xyz),"n_frames":len(np.unique(frame)),"elapsed_s":elapsed,"status":details["status"],"mean_displacement_mm":float(displacement.mean()),"max_displacement_mm":float(displacement.max()),"all_finite":bool(np.isfinite(world).all()),"details":details,"no_ground_truth":True,"sigma_mm_assumption_not_calibrated":10.,"bias_bound_mm_assumption_not_calibrated":50.}
            records.append(record)
    (output/"smoke.json").write_text(json.dumps(records,indent=2))
    print(json.dumps([{k:r[k] for k in ("input","method","status","n_points","n_frames","mean_displacement_mm")} for r in records],indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.source,a.output)

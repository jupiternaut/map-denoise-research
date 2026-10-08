"""Frozen stage-2 continuous-depth confirmation; Python standard library only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from .interval_model import Model
from .interval_solver import solve_outer, decide, encode, decode

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
EPSILON = Q(1,100)
RADII = (Q(0),Q(1,5),Q(1))
TRUTH_DEPTHS = (Q(3921,7),Q(4205,7),Q(4477,7))
SEEDS = tuple(range(2000,2004))
KINDS = ("lovelace_triangle","rectangle")
TOLERANCE = Q(15,32)
MAX_BOXES = 511


def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def source_hashes():
    # The independent checker is versioned by its own receipt at verification;
    # it cannot influence prediction/search/selection or change this run.
    files = [p for p in ROOT.glob("*.py") if p.name != "verify_continuous.py"]
    files += [ROOT/"EXPERIMENT_PLAN.md",ROOT/"THEORY.md"]
    files += list((PROJECT/"finite_world").glob("*.py"))
    files += list((PROJECT/"finite_world"/"assets").glob("*.json"))
    return {p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def linf(left,right):
    if len(left)!=len(right) or not left:
        raise ValueError("fixed nonempty image dimension required")
    return max(abs(a-b) for a,b in zip(left,right))


def nominal_grid(observation,model,incumbent,forecasts):
    scored = [(linf(pred,observation),z) for z,pred in forecasts]
    scored.append((linf(model.point_prediction(incumbent),observation),incumbent))
    best = min(loss for loss,_ in scored)
    tied = {z for loss,z in scored if loss==best}
    if incumbent in tied:
        return incumbent
    return min(tied,key=lambda z:(abs(z-incumbent),z))


def generate_observation(kind,radius,truth,seed):
    # The generator knows this actual world; solve_outer does not receive it.
    model = Model(kind,radius)
    rng = random.Random(f"stage2:{kind}:{radius}:{truth}:{seed}")
    weights = (Q(-1),Q(-1,2),Q(0),Q(1,2),Q(1))
    offsets = (radius*rng.choice(weights),Q(0),radius*rng.choice(weights))
    prediction = model.point_prediction(truth,offsets)
    observation = tuple(value+EPSILON*rng.choice(weights) for value in prediction)
    return model,offsets,prediction,observation


def aggregate(scenes):
    groups = defaultdict(list)
    for scene in scenes:
        truth = decode(scene["truth_depth_mm"])
        hull = scene["outer"]["hull"]
        width = None if hull is None else decode(hull[1])-decode(hull[0])
        for row in scene["decisions"]:
            a = decode(row["incumbent"])
            outputs = {"certificate":decode(row["certificate"]["output"]),
                       **{k:decode(v) for k,v in row["baselines"].items()}}
            for method,b in outputs.items():
                groups[(scene["kind"],str(decode(scene["camera_radius_mm"])),row["state"],method)].append(
                    (abs(a-truth),abs(b-truth),b!=a,width))
    result=[]
    for (kind,radius,state,method),rows in sorted(groups.items()):
        n=len(rows)
        result.append({"kind":kind,"camera_radius_mm":radius,"state":state,"method":method,
                       "count":n,"improved":sum(b<a for a,b,_,_ in rows),
                       "worse":sum(b>a for a,b,_,_ in rows),"same":sum(b==a for a,b,_,_ in rows),
                       "moved":sum(moved for _,_,moved,_ in rows),
                       "mae_mm":encode(sum(b for _,b,_,_ in rows)/n),
                       "mse_mm2":encode(sum(b*b for _,b,_,_ in rows)/n),
                       "mean_hull_width_mm":None if any(w is None for _,_,_,w in rows) else encode(sum(w for _,_,_,w in rows)/n)})
    return result


def control_cases():
    model=Model("lovelace_triangle",Q(0))
    truth=TRUTH_DEPTHS[0]
    # Declared tiny budget cannot cover deliberately substituted wrong-world photo.
    observation=model.point_prediction(Q(645))
    actual_prediction=model.point_prediction(truth)
    outer=solve_outer(observation,model.enclosure,epsilon=EPSILON,tolerance=TOLERANCE,max_boxes=MAX_BOXES)
    d=decide(outer,truth)
    b=decode(d["output"])
    return [{"name":"wrong_world_photo_exceeds_noise_budget","contract":"violated_photometric_budget",
             "kind":model.kind,"camera_radius_mm":encode(Q(0)),"truth_depth_mm":encode(truth),
             "actual_offsets_mm":[encode(Q(0))]*3,"observation":[encode(x) for x in observation],
             "epsilon":encode(EPSILON),"actual_noise_linf":encode(linf(observation,actual_prediction)),
             "outer":outer,"incumbent":encode(truth),"decision":d,
             "actual_gain_mm2":encode(-(b-truth)**2),
             "explanation":"positive lower gain over wrong outer set cannot protect excluded truth"},
            {"name":"zero_search_budget_retains_root","contract":"in_contract_termination_control",
             "outer":solve_outer(actual_prediction,model.enclosure,epsilon=EPSILON,max_boxes=0),
             "decision":decide(solve_outer(actual_prediction,model.enclosure,epsilon=EPSILON,max_boxes=0),truth)}]


def run(output):
    if output.exists():
        raise FileExistsError("preserve existing experiment; choose a fresh --output directory")
    output.mkdir(parents=True)
    started=time.perf_counter()
    hashes=source_hashes()
    write_json(output/"SOURCE_LOCK.json",{"schema":1,"utc":datetime.now(timezone.utc).isoformat(),
               "python":sys.version,"hashes":hashes})
    for relative in hashes:
        copy=output/"source"/relative
        copy.parent.mkdir(parents=True,exist_ok=True)
        copy.write_bytes((PROJECT/relative).read_bytes())
    protocol={"schema":1,"scope":"continuous_depth_and_bounded_lateral_camera_positions_only",
              "depth_domain_mm":[encode(Q(540)),encode(Q(660))],"epsilon":encode(EPSILON),
              "tolerance_mm":encode(TOLERANCE),"max_boxes":MAX_BOXES,
              "kinds":list(KINDS),"camera_radii_mm":[encode(x) for x in RADII],
              "truth_depths_mm":[encode(x) for x in TRUTH_DEPTHS],"seeds":list(SEEDS),
              "models":[Model(kind,radius).to_dict() for kind in KINDS for radius in RADII],
              "baseline":"nominal-camera integer grid 540..660 plus incumbent; KEEP ties",
              "decision_policy":"outer hull midpoint proposal, strict positive endpoint gain else KEEP",
              "evaluator_only":"truth_depth_mm and actual_offsets_mm never enter solve_outer or decide"}
    write_json(output/"PROTOCOL.json",protocol)
    # Compile nominal baseline forecasts BEFORE selecting any confirmation input.
    forecasts={kind:tuple((Q(z),Model(kind,0).point_prediction(Q(z))) for z in range(540,661)) for kind in KINDS}
    forecast_json={kind:[{"depth":encode(z),"prediction":[encode(x) for x in p]} for z,p in rows]
                   for kind,rows in forecasts.items()}
    write_json(output/"BASELINE_FORECASTS.json",forecast_json)
    write_json(output/"FORECAST_LOCK.json",{"sha256":hashlib.sha256((output/"BASELINE_FORECASTS.json").read_bytes()).hexdigest()})
    scenes=[]
    with (output/"scenes.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        for kind in KINDS:
            for radius in RADII:
                for truth in TRUTH_DEPTHS:
                    for seed in SEEDS:
                        model,offsets,prediction,observation=generate_observation(kind,radius,truth,seed)
                        outer=solve_outer(observation,model.enclosure,epsilon=EPSILON,
                                          tolerance=TOLERANCE,max_boxes=MAX_BOXES)
                        decisions=[]
                        for state,offset in (("correct",0),("minus60",-60),("plus60",60)):
                            a=truth+offset
                            d=decide(outer,a)
                            baseline=nominal_grid(observation,model,a,forecasts[kind])
                            decisions.append({"state":state,"incumbent":encode(a),"certificate":d,
                                              "baselines":{"keep":encode(a),"nominal_grid":encode(baseline)}})
                        scene={"scene_id":f"{kind}:{radius}:{truth}:{seed}","kind":kind,
                               "camera_radius_mm":encode(radius),"truth_depth_mm":encode(truth),"seed":seed,
                               "actual_offsets_mm":[encode(x) for x in offsets],"epsilon":encode(EPSILON),
                               "observation":[encode(x) for x in observation],"outer":outer,"decisions":decisions}
                        stream.write(json.dumps(scene,separators=(",",":"))+"\n")
                        scenes.append(scene)
                        if len(scenes)%12==0:
                            print(json.dumps({"scenes_complete":len(scenes),"kind":kind,"radius_mm":str(radius),
                                              "elapsed_seconds":f"{time.perf_counter()-started:.3f}"}),flush=True)
    write_json(output/"controls.json",control_cases())
    summary={"scope":protocol["scope"],"scene_configurations":len(scenes),"decisions":sum(len(s["decisions"]) for s in scenes),
             "rows":aggregate(scenes),"evaluated_boxes_total":sum(s["outer"]["evaluated_boxes"] for s in scenes),
             "unresolved_budget_leaves":sum(sum(n["status"]=="unresolved_budget" for n in s["outer"]["trace"]) for s in scenes),
             "elapsed_seconds":f"{time.perf_counter()-started:.3f}"}
    write_json(output/"summary.json",summary)
    if source_hashes()!=hashes:
        raise RuntimeError("sources changed while confirmation was running")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"confirmation-v1")
    args=parser.parse_args()
    run(args.output.resolve())

"""Public independent verification first; private exact coverage/evaluation last."""
import argparse
import csv
from fractions import Fraction as Q
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parent
P = ROOT.parent
A8 = P / "independent-anchors-1"
EXPECTED = Path("/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-anchors-1")
OUT = ROOT / "outputs/continuous-v1"
PYTHON = Path("/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python")
import verify_continuous as independent
exact = independent.exact
ARMS = ("none", "camera", "appearance", "both")
POLICIES = ("HULL", "UNION")
SCIENCE_NAMES = ("AGENTS.md", "PROTOCOL.md", "REGISTRATION.json", "continuous_solver.py",
                 "test_continuous_solver.py", "verify_continuous.py", "test_verify_continuous.py",
                 "run_continuous.py", "audit_continuous.py", "execute_continuous.py",
                 "test_continuous_pipeline.py")
def load(path):
    return json.loads(Path(path).read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path)
    require(path.resolve().is_relative_to(OUT),"auditorwriteoutsideA9output")
    path.parent.mkdir(parents=True,exist_ok=True)
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    try:
        with path.open("x") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
def sources():
    files=load(A8/"outputs/anchor-v1/SOURCE_LOCK.json")["files"].copy()
    for name in SCIENCE_NAMES:
        files[str(ROOT/name)]=sha(ROOT/name)
    return dict(sorted(files.items()))
def inputs():
    paths=(A8/"ARTIFACTS.json",A8/"FINAL_SEAL.json",A8/"outputs/anchor-v1/observations.json",
           A8/"outputs/anchor-v1/tasks.json",A8/"outputs/anchor-v1/STAGE_STATUS.json",
           A8/"outputs/anchor-v1/verification/VERIFY.json",A8/"data/anchors.json",A8/"data/private/measurements.json",
           P/"appearance-variation-1/data/private/truth.json",
           P/"joint-feasibility-1/outputs/polytope-v1/outers.json")
    return {str(path):sha(path) for path in paths}
def validate_old():
    for rel,digest in load(A8/"ARTIFACTS.json")["files"].items():
        require(sha(A8/rel)==digest,"oldA8artifactdrift")
    require(load(A8/"outputs/anchor-v1/STAGE_STATUS.json")["status"]=="COMPLETE" and
            load(A8/"outputs/anchor-v1/verification/VERIFY.json")["status"]=="PASS","oldA8acceptancestatus")
    for path,digest in load(A8/"outputs/anchor-v1/SOURCE_LOCK.json")["files"].items():
        require(sha(path)==digest,"oldtransitivesourcedrift")
def require(value, message):
    if not value:
        raise RuntimeError(message)
def pairs(intervals):
    return tuple(tuple(map(exact, pair)) for pair in intervals)
def contains(intervals, point):
    return any(lo <= point <= hi for lo, hi in pairs(intervals))
def index(records, fields):
    result = {}
    for record in records:
        key = tuple(record[field] for field in fields)
        require(key not in result, "duplicate record identity")
        result[key] = record
    return result
def covered_by(inner, outer):
    return all(any(ol <= il and ih <= oh for ol, oh in pairs(outer)) for il, ih in pairs(inner))
def group_names(fact, public, initial_error):
    groups = ["all", "correct" if initial_error == 0 else "shifted"]
    if initial_error:
        groups.append("negative" if fact["case"] == "equal_full_negative" else "informative")
        if fact["case"] != "equal_full_negative" and exact(fact["truth"]) == Q(9007, 14) and public["variation_index"] == 1:
            groups.append("primary")
    return groups
def summarize(rows):
    groups = {}
    for row in rows:
        for unit in ("ALL", row["unit_id"]):
            for group in row["groups"]:
                groups.setdefault((row["arm"], row["policy"], unit, group), []).append(row)
    result = []
    for (arm, policy, unit, group), cell in sorted(groups.items()):
        initial = [Q(r["initial_ae_mm"]) for r in cell]
        final = [Q(r["final_ae_mm"]) for r in cell]
        n = len(cell)
        imae, fmae = sum(initial, Q(0)) / n, sum(final, Q(0)) / n
        imse, fmse = sum((x*x for x in initial), Q(0)) / n, sum((x*x for x in final), Q(0)) / n
        result.append({"arm": arm, "policy": policy, "unit": unit, "group": group, "n": n,
            "initial_MAE_mm": str(imae), "final_MAE_mm": str(fmae),
            "initial_MSE_mm2": str(imse), "final_MSE_mm2": str(fmse),
            "MAE_gain_mm": str(imae-fmae), "MSE_gain_mm2": str(imse-fmse),
            "MSE_reduction_percent": None if imse == 0 else float(100*(1-fmse/imse)),
            "improved": sum(b<a for a,b in zip(initial, final)),
            "equal": sum(b==a for a,b in zip(initial, final)),
            "worsened": sum(b>a for a,b in zip(initial, final)),
            "moved": sum(r["moved"] for r in cell),
            "max_AE_increase_mm": str(max([Q(0)] + [b-a for a,b in zip(initial, final)])),
            "mean_displacement_mm": str(sum((Q(r["displacement_mm"]) for r in cell), Q(0))/n)})
    return result
def main(output, deadline):
    require(ROOT == EXPECTED and output == OUT and socket.gethostname() == "liekkas" and Path(sys.executable) == PYTHON,
            "wrong target/runtime")
    require(load(OUT / "RUN_STATUS.json")["status"] == "COMPLETE", "allpublicinference mustcomplete")
    frozen = load(OUT / "SOURCE_LOCK.json")["files"]
    require(sources() == frozen and all(sha(p) == h for p,h in frozen.items()), "source lock drift")
    validate_old()
    input_hashes = load(OUT / "INPUTS.json")["files"]
    require(inputs() == input_hashes, "input bytes drift")
    receipt = load(OUT / "PREDICTION_RECEIPT.json")
    require(receipt["private_parsed"] is False and receipt["sets_sha256"] == sha(OUT / "sets.json") and
            receipt["predictions_sha256"] == sha(OUT / "predictions.json"), "prediction seal mismatch")
    observations = index(load(OUT / "observations.json"), ("observation_id",))
    tasks = index(load(OUT / "tasks.json"), ("task_id",))
    anchors = {a["observation_id"]: a for a in load(A8 / "data/anchors.json")}
    oldarms = load(P / "joint-feasibility-1/outputs/polytope-v1/outers.json")
    sets = index(load(OUT / "sets.json"), ("observation_id", "arm"))
    predictions = index(load(OUT / "predictions.json"), ("task_id", "arm", "policy"))
    require(len(observations)==36 and len(tasks)==324 and len(sets)==144 and len(predictions)==2592, "matrix denominator")
    require(set(sets)=={(key[0], arm) for key in observations for arm in ARMS}, "outersetmatrix")
    require(set(predictions)=={(key[0],arm,policy) for key in tasks for arm in ARMS for policy in POLICIES}, "actionmatrix")
    public_stats = []
    def alarm(*args):
        raise TimeoutError("A9 shared deadline during audit")
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
    try:
        for (oid, arm), record in sets.items():
            public = observations[oid,]
            require([b["branch"] for b in record["branches"]] == ["none","partial","full"], "branch order/allthree")
            checked = []
            for branch_record in record["branches"]:
                checked.append(independent.verify_branch(public, oldarms[oid]["box"]["joint"], anchors[oid], arm,
                                                         branch_record["branch"], branch_record))
            # Independent closed union, not solver aggregate.
            aggregation = independent.aggregate_branches(checked, record)
            union = pairs(aggregation["outer_intervals"])
            require(pairs(record["outer_intervals"]) == union, "aggregate closed union mismatch")
            hull = None if not union else (union[0][0], union[-1][1])
            require((None if record["hull"] is None else tuple(map(exact,record["hull"]))) == hull, "aggregate hull mismatch")
            leaves = [r for b in record["branches"] for r in b["trace"] if r["status"].startswith("retained_")]
            public_stats.append({"observation_id":oid,"arm":arm,
                 "union_length_mm":str(sum((hi-lo for lo,hi in union),Q(0))),
                 "hull_width_mm":None if hull is None else str(hull[1]-hull[0]),
                 "components":len(union),"classified_nodes":sum(b["classified_nodes"] for b in record["branches"]),
                 "pruned_nodes":sum(b["pruned_nodes"] for b in record["branches"]),
                 "retained_leaves":len(leaves),"retained_deadline":sum(r["status"]=="retained_deadline" for r in leaves),
                 "retained_node_budget":sum(r["status"]=="retained_node_budget" for r in leaves),
                 "unknown_classifications":sum(b["classification_status_counts"].get("UNKNOWN",0) for b in checked),
                 "max_retained_width_mm":str(max([Q(0)]+[exact(r["hi"])-exact(r["lo"]) for r in leaves])),
                 "seconds":sum(b["seconds"] for b in record["branches"])})
            print(json.dumps({"stage":"A9-public-audit","sets_verified":len(public_stats),"arm":arm,"observation_id":oid}),flush=True)
        for (tid,arm,policy), record in predictions.items():
            task = tasks[tid,]
            require(record["observation_id"] == task["observation_id"] and exact(record["incumbent"])==exact(task["incumbent"]),
                    "actionidentity/incumbent")
            independent.verify_selection(task["incumbent"],sets[task["observation_id"],arm]["outer_intervals"],policy,record)
        save(OUT / "verification/PUBLIC_GATE.json", {"status":"PASS","sets":144,"trees":432,"predictions":2592,
             "completed_epoch":time.time(),"prediction_receipt_sha256":sha(OUT/"PREDICTION_RECEIPT.json"),
             "private_parsed":False,"assurance":"independentauthor, samefamilyprovisional"})
        # PRIVATE IS FIRST PARSED HERE, after every public prediction and proof is fixed and checked.
        facts = {f["observation_id"]:f for f in load(P/"appearance-variation-1/data/private/truth.json")}
        measurements = {m["observation_id"]:m for m in load(A8/"data/private/measurements.json")}
        require(set(facts)==set(anchors)==set(measurements)=={key[0] for key in observations}, "frozen trueworldidentity")
        coverage = []
        a8 = independent._a8
        for (oid,arm), record in sets.items():
            public,fact,measurement = observations[oid,],facts[oid],measurements[oid]
            truth,branch = exact(fact["truth"]),fact["rear_branch"]
            witness = a8.actual_witness(fact,measurement,arm)
            a8.verify_forward(public,oldarms[oid]["box"]["joint"],anchors[oid],truth,branch,arm,witness,
                              "A9 actualphoto+enabledindependentanchors")
            rows,bounds = a8.necessary_problem(public,oldarms[oid]["box"]["joint"],anchors[oid],truth,branch,arm)
            a8._proof.verify_witness(rows,bounds,witness["darkness"]+witness["t"]+measurement["appearance_t"],"A9 actualpointouter")
            generating = next(b for b in record["branches"] if b["branch"]==branch)
            require(contains(generating["outer_intervals"],truth) and contains(record["outer_intervals"],truth),
                    "actual generating world/depth excluded")
            # Check actual nuisance point in every retained generating cell containing truth.
            for leaf in generating["trace"]:
                if leaf["status"].startswith("retained_") and exact(leaf["lo"]) <= truth <= exact(leaf["hi"]):
                    cell_rows,cell_bounds=independent.necessary_problem(public,oldarms[oid]["box"]["joint"],
                          anchors[oid],arm,branch,exact(leaf["lo"]),exact(leaf["hi"]))
                    a8._proof.verify_witness(cell_rows,cell_bounds,witness["darkness"]+witness["t"]+measurement["appearance_t"],
                                             "A9 actualpoint inside retained whole-depth cell")
            coverage.append({"observation_id":oid,"arm":arm,"truth_mm":str(truth),"generating_branch":branch,
                             "branch_covered":True,"union_covered":True,"photo_and_anchors_verified":True})
        rows = []
        for (tid,arm,policy), record in predictions.items():
            task = tasks[tid,]; public = observations[task["observation_id"],]; fact=facts[task["observation_id"]]
            truth,old,new = exact(fact["truth"]),exact(task["incumbent"]),exact(record["selected"])
            initial,final=abs(old-truth),abs(new-truth)
            require(initial or new==old,"correctincumbentmustnotmove")
            if policy=="HULL":
                require(final<=initial,"HULLnonregression violated despite trueworldcoverage")
            rows.append({"task_id":tid,"observation_id":task["observation_id"],"arm":arm,"policy":policy,
                 "unit_id":public["unit_id"],"variation_index":public["variation_index"],"case":fact["case"],
                 "truth_mm":str(truth),"incumbent_mm":str(old),"selected_mm":str(new),
                 "initial_ae_mm":str(initial),"final_ae_mm":str(final),"ae_gain_mm":str(initial-final),
                 "displacement_mm":str(abs(new-old)),"moved":new!=old,"groups":group_names(fact,public,initial)})
        summaries=summarize(rows)
        for arm in ARMS:
            for policy in POLICIES:
                cells={r["group"]:r for r in summaries if r["arm"]==arm and r["policy"]==policy and r["unit"]=="ALL"}
                require({k:cells[k]["n"] for k in ("all","shifted","correct","primary","negative")}==
                        {"all":324,"shifted":288,"correct":36,"primary":40,"negative":48},"fixedgroupdenominators")
        nesting=[]
        for oid, in observations:
            none,both=sets[oid,"none"],sets[oid,"both"]
            nesting.append({"observation_id":oid,"both_subset_none":covered_by(both["outer_intervals"],none["outer_intervals"]),
                            "none_hull_mm":None if none["hull"] is None else str(exact(none["hull"][1])-exact(none["hull"][0])),
                            "both_hull_mm":None if both["hull"] is None else str(exact(both["hull"][1])-exact(both["hull"][0]))})
        per_observation=[]
        for oid, in observations:
            for arm in ARMS:
                for policy in POLICIES:
                    selected=[r for r in rows if r["observation_id"]==oid and r["arm"]==arm and r["policy"]==policy]
                    per_observation.append({"observation_id":oid,"arm":arm,"policy":policy,"n":9,
                       "mean_AE_gain_mm":str(sum((Q(r["ae_gain_mm"]) for r in selected),Q(0))/9)})
        save(OUT/"evaluation/ROWS.json",rows)
        save(OUT/"evaluation/SUMMARY.json",summaries)
        save(OUT/"evaluation/COVERAGE.json",coverage)
        save(OUT/"evaluation/OUTER_STATS.json",public_stats)
        save(OUT/"evaluation/EMPIRICAL_NESTING.json",nesting)
        save(OUT/"evaluation/PER_OBSERVATION.json",per_observation)
        csv_path=OUT/"evaluation/ROWS.csv"
        with csv_path.open("x") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        require(sources()==frozen and inputs()==input_hashes,"postevaluationfrozenbytes")
        save(OUT/"verification/VERIFY.json",{"status":"PASS","sets":144,"trees":432,"actions":2592,
             "true_world_checks":144,"true_world_coverage_per_arm":36,"epoch":time.time(),
             "public_gate_sha256":sha(OUT/"verification/PUBLIC_GATE.json"),
             "assurance":"same-family provisional; independent reconstruction, finite synthetic model only"})
        print(json.dumps({"stage":"A9-audit","status":"PASS","actions":2592,"coverage":144}),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT);parser.add_argument("--deadline",type=float,required=True)
    args=parser.parse_args();main(args.output.resolve(),args.deadline)

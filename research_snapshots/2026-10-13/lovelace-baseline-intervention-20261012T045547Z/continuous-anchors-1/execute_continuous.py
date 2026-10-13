"""One frozen A9 batch, shared execution and independent-check budget."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
EXPECTED=Path("/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-anchors-1")
PYTHON=Path("/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python")
OUT=ROOT/"outputs/continuous-v1"
def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")
def main():
    if socket.gethostname()!="liekkas" or ROOT!=EXPECTED or Path(sys.executable)!=PYTHON or OUT.exists():
        raise RuntimeError("wrong exact target/runtime or formal batch already exists")
    gate=load(ROOT/".aris/REVIEW_GATE.json")
    if gate["status"]!="PASS" or any(sha(path)!=digest for path,digest in gate["files"].items()):
        raise RuntimeError("freshsourceandtestgate mustpassatcurrentbytes")
    from run_continuous import sources,validate_old,inputs
    validate_old();frozen=sources();input_hashes=inputs()
    if any(sha(path)!=digest for path,digest in frozen.items()):raise RuntimeError("frozen ancestral source changed")
    registration=load(ROOT/"REGISTRATION.json")
    OUT.mkdir(parents=True)
    start=time.time();deadline=start+registration["whole_seconds"]
    save(OUT/"BUDGET.json",{"start_epoch":start,"deadline_epoch":deadline,
         "main_deadline_epoch":start+registration["main_seconds"],
         "limit_seconds":registration["whole_seconds"],"verification_reserve_seconds":registration["verification_reserve_seconds"],
         "scope":"onepubliccontinuousbatch+independentverification; noreset"})
    save(OUT/"SOURCE_LOCK.json",{"files":frozen,"explicit_local_allowlist":list(__import__("run_continuous").SCIENCE_NAMES)})
    save(OUT/"INPUT_PRELOCK.json",{"files":input_hashes})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    commands=[("main",[str(PYTHON),"-B",str(ROOT/"run_continuous.py"),"--output",str(OUT)]),
              ("audit",[str(PYTHON),"-B",str(ROOT/"audit_continuous.py"),"--output",str(OUT),"--deadline",str(deadline)])]
    receipts=[];status="INCOMPLETE";error=None
    try:
        for label,cmd in commands:
            remaining=deadline-time.time()
            if remaining<=0:raise TimeoutError("A9 shared whole deadline")
            step=time.time()
            with (OUT/(label+".log")).open("x") as stream:
                cp=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
            receipts.append({"step":label,"command":cmd,"seconds":time.time()-step,"returncode":cp.returncode})
            print(json.dumps(receipts[-1]),flush=True)
            if cp.returncode:raise RuntimeError(label+"failed; retainincompleteartifacts")
        if load(OUT/"verification/VERIFY.json")["status"]!="PASS":raise RuntimeError("independentverificationnotPASS")
        if sources()!=frozen or inputs()!=input_hashes or time.time()>deadline:raise RuntimeError("frozenbytes/deadline")
        status="COMPLETE"
    except Exception as exc:
        error=type(exc).__name__+": "+str(exc)
    save(OUT/"STAGE_STATUS.json",{"status":status,"error":error,"seconds":time.time()-start,
         "deadline_epoch":deadline,"steps":receipts})
    print(json.dumps({"stage":"A9","status":status,"error":error,"seconds":time.time()-start}),flush=True)
    if status!="COMPLETE":raise SystemExit(1)
if __name__=="__main__":main()


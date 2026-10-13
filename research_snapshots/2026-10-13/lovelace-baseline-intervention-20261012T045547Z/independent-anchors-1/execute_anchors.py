"""Single shared budget for generation, public inference, independent verification."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/independent-anchors-1')
PYTHON=Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python')
OUT=ROOT/'outputs/anchor-v1'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or Path(sys.executable)!=PYTHON or OUT.exists():
        raise RuntimeError('wrong target/runtime or existing one-shot formal output')
    gate=load(ROOT/'.aris/REVIEW_GATE.json')
    if gate['status']!='PASS' or any(sha(p)!=h for p,h in gate['files'].items()):raise RuntimeError('source review must pass')
    from run_anchors import sources,validate_old_seals
    validate_old_seals()
    frozen=sources()
    if any(sha(p)!=h for p,h in frozen.items()):raise RuntimeError('old source drift')
    OUT.mkdir(parents=True);start=time.time();deadline=start+3600
    save(OUT/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,'main_deadline_epoch':start+3000,
        'limit_seconds':3600,'scope':'A8 one generation/main/independent-verification window; no reset'})
    save(OUT/'SOURCE_LOCK.json',{'files':frozen,'explicit_local_allowlist':list(__import__('run_anchors').SCIENCE_NAMES)})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    commands=[('generation',[str(PYTHON),'-B',str(ROOT/'prepare_anchors.py'),'--deadline',str(deadline)]),
        ('main',[str(PYTHON),'-B',str(ROOT/'run_anchors.py'),'--output',str(OUT)]),
        ('verification',[str(PYTHON),'-B',str(ROOT/'verify_anchors.py'),'--output',str(OUT),'--deadline',str(deadline)])]
    receipts=[];status='INCOMPLETE';error=None
    try:
        for label,cmd in commands:
            remaining=deadline-time.time()
            if remaining<=0:raise TimeoutError('whole deadline')
            step=time.time()
            with (OUT/(label+'.log')).open('x') as stream:
                cp=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
            receipts.append({'step':label,'command':cmd,'seconds':time.time()-step,'returncode':cp.returncode})
            print(json.dumps(receipts[-1]),flush=True)
            if cp.returncode:raise RuntimeError(label+' failed')
        if load(OUT/'verification/VERIFY.json')['status']!='PASS':raise RuntimeError('verification not PASS')
        if sources()!=frozen or time.time()>deadline:raise RuntimeError('source drift or deadline')
        status='COMPLETE'
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    save(OUT/'STAGE_STATUS.json',{'status':status,'error':error,'seconds':time.time()-start,'deadline_epoch':deadline,'steps':receipts})
    print(json.dumps({'stage':'A8','status':status,'error':error,'seconds':time.time()-start}),flush=True)
    if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

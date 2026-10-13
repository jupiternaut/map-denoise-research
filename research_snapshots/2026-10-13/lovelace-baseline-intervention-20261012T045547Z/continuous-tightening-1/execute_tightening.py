"""Exactly one frozen formal batch with a shared 1800-second deadline."""
import subprocess
import time
from io_contract import *
def main():
    target();require(not OUT.exists(),'formal output already exists; no automatic rerun')
    gate=load(ROOT/'.aris/REVIEW_GATE.json')
    require(gate['status']=='PASS' and all(sha(p)==h for p,h in gate['files'].items()),'fresh review/current tests gate')
    validate_old();frozen=sources();locked=inputs();reg=load(ROOT/'REGISTRATION.json')
    require(all(sha(p)==h for p,h in frozen.items()),'frozen ancestral source')
    OUT.mkdir(parents=True);start=time.time();end=start+reg['whole_seconds']
    save(OUT/'BUDGET.json',{'start_epoch':start,'deadline_epoch':end,'main_deadline_epoch':start+reg['main_seconds'],
        'whole_seconds':reg['whole_seconds'],'verification_reserve_seconds':reg['verification_reserve_seconds']})
    save(OUT/'SOURCE_LOCK.json',{'files':frozen});save(OUT/'INPUT_PRELOCK.json',{'files':locked})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    receipts=[];status='INCOMPLETE';error=None
    try:
        for label,name in (('main','run_tightening.py'),('audit','audit_tightening.py')):
            step=time.time();remaining=min(end-step,(start+reg['main_seconds']-step) if label=='main' else end-step)
            require(remaining>0,'shared budget exhausted')
            cmd=[str(PYTHON),'-B',str(ROOT/name)]
            with (OUT/(label+'.log')).open('x') as f:
                cp=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
            receipts.append({'step':label,'seconds':time.time()-step,'returncode':cp.returncode,'command':cmd})
            print(json.dumps(receipts[-1]),flush=True);require(cp.returncode==0,label+' failed; preserve incomplete batch')
        require(load(OUT/'verification/VERIFY.json')['status']=='PASS','independent gate')
        require(time.time()<=end and sources()==frozen and inputs()==locked,'deadline/frozen bytes')
        status='COMPLETE'
    except Exception as exc: error=type(exc).__name__+': '+str(exc)
    save(OUT/'STAGE_STATUS.json',{'status':status,'error':error,'seconds':time.time()-start,'steps':receipts})
    print(json.dumps({'stage':'A10','status':status,'error':error,'seconds':time.time()-start}),flush=True)
    if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

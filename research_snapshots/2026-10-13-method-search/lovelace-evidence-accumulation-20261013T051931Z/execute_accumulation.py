"""Single immutable formal run, 1800 seconds total, same-family review separate."""
import subprocess
import time
from a11_io_contract import *
def main():
    target();require(not OUT.exists(),'existing formal run must not be overwritten')
    gate=load(ROOT/'TEST_GATE.json')
    require(gate['status']=='PASS' and all(sha(p)==h for p,h in gate['files'].items()),'current tests gate')
    validate_old();frozen=sources();locked=inputs();reg=load(ROOT/'REGISTRATION.json')
    require(all(sha(p)==h for p,h in frozen.items()),'frozen sources')
    OUT.mkdir(parents=True);start=time.time();end=start+reg['whole_seconds']
    save(OUT/'BUDGET.json',{'start_epoch':start,'deadline_epoch':end,
        'main_deadline_epoch':start+reg['main_seconds'],'whole_seconds':reg['whole_seconds']})
    save(OUT/'SOURCE_LOCK.json',{'files':frozen});save(OUT/'INPUT_PRELOCK.json',{'files':locked})
    receipts=[];status='INCOMPLETE';error=None
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        for label,name in (('main','run_accumulation.py'),('audit','audit_accumulation.py')):
            step=time.time();remaining=min(end-step,start+reg['main_seconds']-step) if label=='main' else end-step
            require(remaining>0,'shared budget exhausted')
            cmd=[str(PYTHON),'-B',str(ROOT/name)]
            with (OUT/(label+'.log')).open('x') as f:
                cp=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
            receipts.append({'step':label,'seconds':time.time()-step,'returncode':cp.returncode,'command':cmd})
            print(json.dumps(receipts[-1]),flush=True);require(cp.returncode==0,label+' failed')
        require(load(OUT/'verification/VERIFY.json')['status']=='PASS','independent verification')
        require(time.time()<=end and sources()==frozen and inputs()==locked,'final budget/frozen bytes')
        status='COMPLETE'
    except Exception as exc: error=type(exc).__name__+': '+str(exc)
    save(OUT/'STAGE_STATUS.json',{'stage':'A11','status':status,'error':error,'seconds':time.time()-start,'steps':receipts})
    print(json.dumps({'stage':'A11','status':status,'error':error,'seconds':time.time()-start}),flush=True)
    if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

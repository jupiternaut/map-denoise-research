"""Run all registered handwritten tests before creating an immutable gate."""
import subprocess
import time
from a11_io_contract import *
def main():
    target();require(not (ROOT/'TEST_GATE.json').exists(),'test gate already exists')
    before={str(ROOT/n):sha(ROOT/n) for n in SCIENCE_NAMES}
    command=[str(PYTHON),'-B','-m','unittest','-v','test_accumulation_solver','test_verify_accumulation','test_accumulation_pipeline']
    start=time.time();cp=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
    log=ROOT/('TEST_GATE_'+str(time.time_ns())+'.log')
    with log.open('x') as f:f.write(cp.stdout+cp.stderr)
    require(cp.returncode==0,'tests failed: '+str(log))
    require(before=={str(ROOT/n):sha(ROOT/n) for n in SCIENCE_NAMES},'sources changed during tests')
    gate=dict(status='PASS',files=before,command=command,returncode=cp.returncode,seconds=time.time()-start,
        log_path=str(log),log_sha256=sha(log))
    with (ROOT/'TEST_GATE.json').open('x') as f:json.dump(gate,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='files'}),flush=True)
if __name__=='__main__':main()

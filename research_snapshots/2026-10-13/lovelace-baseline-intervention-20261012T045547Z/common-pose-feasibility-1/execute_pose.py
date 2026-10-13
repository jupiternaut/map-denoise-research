"""One A7 fixed batch with a shared main and independent verification deadline."""
import argparse,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_pose as runner
def execute(output):
    output=Path(output).resolve()
    if socket.gethostname()!='liekkas' or ROOT!=runner.EXPECTED or Path(sys.executable)!=runner.PYTHON or output!=ROOT/'outputs/pose-v1' or output.exists():raise RuntimeError('exact new target required')
    gate=runner.load(ROOT/'.aris/REVIEW_GATE.json')
    if gate.get('status')!='PASS' or gate.get('files')!=runner.sources():raise RuntimeError('review gate missing/stale/incomplete')
    owned=False
    try:
        output.mkdir(parents=True);owned=True
        start=time.time();deadline=start+runner.LIMIT
        runner.save(output/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,'limit_seconds':runner.LIMIT,'scope':runner.BUDGET_SCOPE})
        runner.save(output/'SOURCE_LOCK.json',{'source_root':str(ROOT),'files':runner.sources()})
        runner.run(output)
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('no verify budget')
        subprocess.run([sys.executable,'-B',str(ROOT/'verify_pose.py'),'--output',str(output)],check=True,timeout=remaining)
        receipt=runner.load(output/'verification/VERIFY.json')
        if receipt['status']!='PASS' or receipt['output']!=str(output):raise RuntimeError('wrong verifier receipt')
        if time.time()>=deadline:raise TimeoutError('deadline')
        runner.save(output/'STAGE_STATUS.json',{'status':'COMPLETE','independently_verified':True,'elapsed_main_plus_verification_seconds':time.time()-start,'deadline_epoch':deadline})
    except BaseException as e:
        if owned:
            runner.save(output/'INCOMPLETE.json',{'error':type(e).__name__,'message':str(e)})
            old=runner.load(output/'RUN_STATUS.json') if (output/'RUN_STATUS.json').exists() else {}
            runner.save(output/'RUN_STATUS.json',{**old,'status':'INCOMPLETE'},replace=True)
            runner.save(output/'STAGE_STATUS.json',{'status':'INCOMPLETE','independently_verified':False})
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/pose-v1');execute(p.parse_args().output)

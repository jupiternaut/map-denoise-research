"""One A6 diagnostic batch with shared absolute main/verification budget."""
import argparse,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_point as runner
def execute(output):
    output=Path(output).resolve()
    if socket.gethostname()!='liekkas' or ROOT!=runner.EXPECTED or Path(sys.executable)!=runner.PYTHON or output.parent!=ROOT/'outputs' or output.exists():
        raise RuntimeError('requires exact new output/runtime')
    gate=runner.load(ROOT/'.aris/REVIEW_GATE.json')
    if gate.get('status')!='PASS' or any(runner.sha(p)!=h for p,h in gate['files'].items()):
        raise RuntimeError('missing/current review gate')
    owned=False
    try:
        output.mkdir(parents=True);owned=True
        start=time.time();deadline=start+runner.LIMIT
        runner.save(output/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,'limit_seconds':runner.LIMIT,'scope':runner.BUDGET_SCOPE})
        runner.save(output/'SOURCE_LOCK.json',{'source_root':str(ROOT),'files':runner.sources()})
        runner.run(output)
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('no verification budget')
        subprocess.run([sys.executable,'-B',str(ROOT/'verify_point.py'),'--output',str(output)],check=True,timeout=remaining)
        receipt=runner.load(output/'verification/VERIFY.json')
        if receipt.get('status')!='PASS' or receipt.get('output')!=str(output):raise RuntimeError('not exact-target PASS')
        if time.time()>=deadline:raise TimeoutError('shared budget at completion')
        runner.save(output/'STAGE_STATUS.json',{'status':'COMPLETE','independently_verified':True,
                    'elapsed_main_plus_verification_seconds':time.time()-start,'deadline_epoch':deadline})
    except BaseException as e:
        if owned:
            runner.save(output/'INCOMPLETE.json',{'error':type(e).__name__,'message':str(e),'stage':'main_or_verification'})
            old=runner.load(output/'RUN_STATUS.json') if (output/'RUN_STATUS.json').exists() else {}
            runner.save(output/'RUN_STATUS.json',{**old,'status':'INCOMPLETE'},replace=True)
            runner.save(output/'STAGE_STATUS.json',{'status':'INCOMPLETE','independently_verified':False})
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/point-v1');execute(p.parse_args().output)


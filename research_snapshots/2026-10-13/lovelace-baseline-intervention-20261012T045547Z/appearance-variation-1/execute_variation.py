"""Single shared deadline over one-shot prepare, batch, independent replay."""
import argparse, json, socket, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_variation as runner
def execute(output):
    output=Path(output).resolve()
    if socket.gethostname()!='liekkas' or ROOT!=runner.EXPECTED or output.parent!=ROOT/'outputs' or output.exists() or (ROOT/'data').exists():
        raise RuntimeError('requires exact target and new one-shot data/output')
    gate=runner.load(ROOT/'.aris/REVIEW_GATE.json')
    if gate.get('status')!='PASS' or any(runner.sha(path)!=digest for path,digest in gate['files'].items()):
        raise RuntimeError('review gate missing approval or current source bytes')
    ownership=False
    try:
        output.mkdir(parents=True);ownership=True
        start=time.time();deadline=start+runner.LIMIT
        runner.save(output/'BUDGET.json',{'start_epoch':start,'deadline_epoch':deadline,
            'limit_seconds':runner.LIMIT,'scope':runner.BUDGET_SCOPE})
        runner.save(output/'SOURCE_LOCK.json',{'source_root':str(ROOT),'files':runner.sources()})
        subprocess.run([sys.executable,'-B',str(ROOT/'prepare_variation.py'),'--deadline',str(deadline)],
                       check=True,timeout=max(0.001,deadline-time.time()))
        runner.run(output)
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('shared deadline before independent replay')
        subprocess.run([sys.executable,'-B',str(ROOT/'verify_variation.py'),'--output',str(output)],
                       check=True,timeout=remaining)
        receipt=runner.load(output/'verification/VERIFY.json')
        if receipt.get('status')!='PASS' or receipt.get('output')!=str(output):
            raise RuntimeError('exact-target independent verification not PASS')
        if time.time()>=deadline:raise TimeoutError('shared budget at completion')
        runner.save(output/'STAGE_STATUS.json',{'status':'COMPLETE','independently_verified':True,
            'deadline_epoch':deadline,'elapsed_prepare_main_plus_verification_seconds':time.time()-start})
    except BaseException as error:
        if ownership:
            if not (output/'INCOMPLETE.json').exists():
                runner.save(output/'INCOMPLETE.json',{'error':type(error).__name__,'message':str(error),'stage':'prepare_main_or_verification'})
            status=runner.load(output/'RUN_STATUS.json') if (output/'RUN_STATUS.json').exists() else {}
            runner.save(output/'RUN_STATUS.json',{**status,'status':'INCOMPLETE','failure_stage':'prepare_main_or_verification'},replace=True)
            if not (output/'STAGE_STATUS.json').exists():
                runner.save(output/'STAGE_STATUS.json',{'status':'INCOMPLETE','independently_verified':False})
        raise
    return str(output)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/variation-v1')
    print(execute(parser.parse_args().output))

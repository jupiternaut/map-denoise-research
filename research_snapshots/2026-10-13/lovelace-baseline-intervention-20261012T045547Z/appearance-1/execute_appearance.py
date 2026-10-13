"""One finite A4 CPU job: public batch and independent subprocess verification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_appearance as runner


def execute(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'outputs') or output.exists():
        raise RuntimeError('A4 requires a new exact output; existing results are read-only')
    ownership={}
    try:
        runner.run(output,ownership=ownership)
        budget=runner.load(output/'BUDGET.json')
        remaining=budget['deadline_epoch']-time.time()
        if remaining<=0:
            raise TimeoutError('shared budget exhausted before verification')
        subprocess.run([sys.executable,'-B',str(ROOT/'verify_appearance.py'),'--output',str(output)],
                       check=True,timeout=remaining)
        receipt=runner.load(output/'verification/VERIFY.json')
        if receipt.get('status')!='PASS' or receipt.get('output')!=str(output):
            raise RuntimeError('exact-target independent verification not PASS')
        if time.time()>=budget['deadline_epoch']:
            raise TimeoutError('shared budget exhausted at phase completion')
        runner.save(output/'STAGE_STATUS.json',{'status':'COMPLETE','independently_verified':True,
            'deadline_epoch':budget['deadline_epoch'],'elapsed_main_plus_verification_seconds':time.time()-budget['start_epoch']})
    except BaseException as error:
        # A verifier failure must not leave the main-program COMPLETE as phase completion.
        if ownership.get('created') is True and output.is_dir() and output.is_relative_to(ROOT/'outputs'):
            if not (output/'INCOMPLETE.json').exists():
                runner.save(output/'INCOMPLETE.json',{'error':type(error).__name__,'message':str(error),'stage':'main_or_verification'})
            status=runner.load(output/'RUN_STATUS.json') if (output/'RUN_STATUS.json').exists() else {}
            runner.save(output/'RUN_STATUS.json',{**status,'status':'INCOMPLETE','failure_stage':'main_or_verification'},replace=True)
            if not (output/'STAGE_STATUS.json').exists():
                runner.save(output/'STAGE_STATUS.json',{'status':'INCOMPLETE','independently_verified':False})
        raise
    return str(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/appearance-v1')
    print(execute(parser.parse_args().output))

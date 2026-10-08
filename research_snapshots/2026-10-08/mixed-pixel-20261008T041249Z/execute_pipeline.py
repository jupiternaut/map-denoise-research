"""Reproducible staged commands, exclusive logs, abort on execution error."""
import os
from pathlib import Path
import subprocess
import sys
from run import ROOT,dump,now

def main():
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    commands=[['-m','unittest','discover','-v'],['run.py','prepare'],['run.py','ordinary'],
              ['run.py','oracle'],['run.py','infer'],['evaluate.py']]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    records=[]
    for i,args in enumerate(commands):
        command=[sys.executable,'-B']+args;start=now()
        path=logs/f'{i:02d}.log'
        with path.open('xb') as out:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        record=dict(command=command,started=start,ended=now(),returncode=result.returncode,log=str(path))
        records.append(record);dump(logs/f'{i:02d}.json',record)
        print(__import__('json').dumps(record),flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    dump(logs/'COMMANDS.json',records)
if __name__=='__main__':main()

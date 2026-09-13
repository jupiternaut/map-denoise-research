"""Run latest regression suites without changing frozen experiment checkpoints."""
from pathlib import Path
import sys,os,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=[]
    for suite in ('exploration_v18','exploration_v17','exploration_v16','exploration_v15','exploration_v14'):
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s',suite,'-p','test_*.py','-v'],cwd=ROOT,env=env,text=True,capture_output=True)
        result.append(dict(suite=suite,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        print(suite,p.returncode,p.stderr[-150:],flush=True)
    with (ROOT/'evidence/PUBLICATION_TESTS.json').open('x') as f:json.dump(result,f,indent=2)
    if any(r['returncode'] for r in result):raise SystemExit(1)
if __name__=='__main__':main()

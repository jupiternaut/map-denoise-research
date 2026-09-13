"""Run regression suites in isolated Python processes to avoid historical module-name collisions."""
from pathlib import Path
import sys,subprocess,os,json


def main(path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    results=[]
    for suite in ('exploration_v16','exploration_v15','exploration_v14'):
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s',suite,'-p','test_*.py','-v'],
                         cwd=root,env=env,text=True,capture_output=True)
        results.append(dict(suite=suite,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        print(suite,p.returncode,p.stderr[-250:],flush=True)
    with (Path(path)/'TEST_RESULTS.json').open('x') as f:json.dump(results,f,indent=2)
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':main(sys.argv[1])

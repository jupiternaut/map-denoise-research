"""Run isolated suites; never overwrite historical PUBLICATION_TESTS.json."""
from pathlib import Path
import os,sys,json,subprocess,argparse
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--dest',required=True);args=p.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=[]
    for folder in ('exploration_v19','exploration_v19/track_a','exploration_v19/track_b',
                   'exploration_v18','exploration_v17','exploration_v16','exploration_v15','exploration_v14'):
        cp=subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-p','test_*.py','-v'],cwd=ROOT,env=env,text=True,capture_output=True)
        result.append(dict(suite=folder,returncode=cp.returncode,stdout=cp.stdout,stderr=cp.stderr))
        print(folder,cp.returncode,cp.stderr[-120:],flush=True)
    with (Path(args.dest)/'TEST_RESULTS.json').open('x') as f:json.dump(result,f,indent=2)
    assert all(r['returncode']==0 for r in result)
if __name__=='__main__':main()

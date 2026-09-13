from pathlib import Path
import sys,os,json,subprocess,shutil,hashlib,time

def main(dest):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    tests=[]
    for suite in ('exploration_v17','exploration_v16','exploration_v15','exploration_v14'):
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s',suite,'-p','test_*.py','-v'],cwd=root,env=env,text=True,capture_output=True)
        tests.append(dict(suite=suite,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        print(suite,p.returncode,p.stderr[-150:],flush=True)
    with (dest/'TEST_RESULTS.json').open('x') as f:json.dump(tests,f,indent=2)
    assert all(t['returncode']==0 for t in tests)
    delivery=dest/'final_delivery';delivery.mkdir()
    for p in (root/'exploration_v17').iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):shutil.copyfile(p,delivery/p.name)
    manifest={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.rglob('*') if p.is_file()}
    with (dest/'FINAL_MANIFEST.json').open('x') as f:json.dump(dict(created=time.time(),files=manifest,total_bytes=sum(v['bytes'] for v in manifest.values())),f,indent=2)
    print('SEALED',len(manifest),'files',sum(v['bytes'] for v in manifest.values()),'bytes',flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))

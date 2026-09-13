"""Protect V7 and preceding evidence by an explicit path/hash set."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import tempfile

PROJECT=Path(__file__).resolve().parents[1]
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
PREVIOUS=RUNS/'session-v7-t01y9kft'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',type=Path);a=p.parse_args()
    if socket.gethostname()!='liekkas':raise RuntimeError('wrong target')
    if a.verify:
        before=json.loads((a.verify/'BEFORE.json').read_text());after={p:digest(p) for p in before if Path(p).is_file()}
        result=dict(files=len(before),missing=sorted(set(before)-set(after)),changed=[p for p in after if before[p]!=after[p]])
        result['unchanged']=not(result['missing'] or result['changed']);save(a.verify/'AFTER.json',after);save(a.verify/'VERIFICATION.json',result)
        print(json.dumps(result,indent=2));assert result['unchanged'];return
    old=json.loads((PREVIOUS/'PROTECTED_BEFORE.json').read_text())
    for p,h in old.items():
        if digest(p)!=h:raise RuntimeError('earlier checkpoint mismatch '+p)
    paths=set(old)
    names=('reassociation-v7-primary-0frtnsm9','reassociation-v7-development-vbwoojgf','decision-v7-nk7sm4yf',
           'real-geometry-v7-x5ximvmn','real-geometry-v7-rlp6r6j5','v7-warm-cost-ifprdsy7','session-v7-t01y9kft')
    for root in [PROJECT/'exploration_v7',*[RUNS/n for n in names]]:
        if not root.is_dir():raise FileNotFoundError(root)
        paths.update(str(p) for p in root.rglob('*') if p.is_file())
    dest=Path(tempfile.mkdtemp(prefix='session-v8-',dir=RUNS));save(dest/'BEFORE.json',{p:digest(p) for p in sorted(paths)})
    print(str(dest));print('Protected',len(paths))
if __name__=='__main__':main()

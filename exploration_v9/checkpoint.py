"""Protect V8 and all earlier evidence, never modify it."""
import argparse,hashlib,json,socket,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',type=Path);a=p.parse_args()
    assert socket.gethostname()=='liekkas'
    if a.verify:
        before=json.loads((a.verify/'BEFORE.json').read_text())
        after={p:sha(p) for p in before if Path(p).is_file()}
        report=dict(files=len(before),missing=sorted(set(before)-set(after)),changed=[p for p in after if after[p]!=before[p]])
        report['unchanged']=not(report['missing'] or report['changed']);save(a.verify/'VERIFICATION.json',report)
        print(json.dumps(report,indent=2));assert report['unchanged'];return
    before=json.loads((RUNS/'session-v8-1c5rygxc/BEFORE.json').read_text())
    for p,h in before.items():assert sha(p)==h,p
    for root in (ROOT/'exploration_v8',RUNS/'selection-compensation-v8-ey291bta',RUNS/'session-v8-1c5rygxc'):
        for p in root.rglob('*'):
            if p.is_file():before[str(p)]=sha(p)
    dest=Path(tempfile.mkdtemp(prefix='session-v9-',dir=RUNS));save(dest/'BEFORE.json',before)
    print(dest);print('Protected',len(before))
if __name__=='__main__':main()

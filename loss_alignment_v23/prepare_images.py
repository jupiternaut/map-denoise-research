"""Bounded acquisition of original GeoSVR-linked DTU images, no model rendering."""
from pathlib import Path
import urllib.request, tarfile, hashlib, json, time
DEST=Path('/srv/slam-research/grf/map-denoise/datasets/loss-alignment-v23')
URL='https://drive.usercontent.google.com/download?id=1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ&export=download&confirm=t'
class Stream:
    def __init__(self,r):self.r=r;self.n=0;self.last=0
    def read(self,n):
        if self.n+n>3_000_000_000:raise RuntimeError('3GB transfer cap')
        b=self.r.read(n);self.n+=len(b)
        if self.n-self.last>200_000_000:print('stream MB',round(self.n/1e6),flush=True);self.last=self.n
        return b
def main():
    DEST.mkdir(exist_ok=True);records=[];start=time.time();s=None;error=None;counts={24:0,37:0}
    try:
        with urllib.request.urlopen(URL,timeout=45) as r:
            s=Stream(r)
            with tarfile.open(fileobj=s,mode='r|gz') as t:
                for e in t:
                    if not e.isfile():continue
                    parts=Path(e.name).parts
                    scene=next((x for x in counts if f'scan{x}' in parts),None)
                    if scene is None:continue
                    if '/images/' not in e.name or Path(e.name).suffix.lower() not in ('.png','.jpg','.jpeg'):continue
                    if e.size>30_000_000:raise ValueError('image member too large')
                    p=DEST/f'scan{scene}'/'image'/Path(e.name).name;p.parent.mkdir(parents=True,exist_ok=True)
                    b=t.extractfile(e).read()
                    with p.open('xb') as f:f.write(b)
                    records.append(dict(member=e.name,path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
                    counts[scene]+=1
                    if min(counts.values())>=49:break
    except Exception as e:error=repr(e)
    result=dict(url=URL,files=records,counts=counts,error=error,stream_bytes=s.n if s else 0,seconds=time.time()-start)
    manifest=DEST/'DOWNLOAD.json'
    if manifest.exists():manifest=DEST/f'DOWNLOAD_RETRY_{time.time_ns()}.json'
    with manifest.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    if min(counts.values())<49:raise SystemExit(1)
if __name__=='__main__':main()

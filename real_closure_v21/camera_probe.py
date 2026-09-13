"""Bounded read of the exact GeoSVR-linked archive; retain only scan24 metadata."""
import urllib.request, tarfile, json, hashlib, time
from pathlib import Path

DEST=Path('/srv/slam-research/grf/map-denoise/datasets/real-closure-v21')
URL='https://drive.usercontent.google.com/download?id=1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ&export=download&confirm=t'

class Limited:
    def __init__(self,r):self.r=r;self.n=0;self.last=0
    def read(self,n):
        if self.n+n>3_000_000_000:raise RuntimeError('3 GB transfer limit')
        b=self.r.read(n);self.n+=len(b)
        if self.n-self.last>100_000_000:
            print('read MB',round(self.n/1e6),flush=True);self.last=self.n
        return b

def main():
    DEST.mkdir(exist_ok=True)
    result=dict(url=URL,status='not_found',started=time.time())
    limited=None
    try:
        with urllib.request.urlopen(URL,timeout=40) as response:
            limited=Limited(response)
            with tarfile.open(fileobj=limited,mode='r|gz') as tar:
                for entry in tar:
                    if '/scan24/' in entry.name and entry.isfile() and entry.name.endswith('cameras.npz'):
                        data=tar.extractfile(entry).read()
                        p=DEST/'cameras_geosvr_linked.npz'
                        with p.open('xb') as f:f.write(data)
                        result.update(status='found',member=entry.name,path=str(p),sha256=hashlib.sha256(data).hexdigest())
                        break
    except Exception as e:result['error']=repr(e)
    result.update(bytes_read=0 if limited is None else limited.n,seconds=time.time()-result['started'])
    with (DEST/'CAMERA_PROBE.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

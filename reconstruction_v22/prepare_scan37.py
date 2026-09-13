"""Download one independent scene's published output and exact metadata, not train data."""
from pathlib import Path
import sys,json,hashlib,urllib.request,tarfile,time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'published_outputs_v2'))
import remote_zip
from remote_zip import extract
DEST=Path('/srv/slam-research/grf/map-denoise/datasets/reconstruction-v22-scan37')
ARCHIVE='https://drive.usercontent.google.com/download?id=1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ&export=download&confirm=t'
class RetryingRemote(remote_zip.Remote):
    def read(self,n=-1):
        position=self.pos
        for attempt in range(4):
            try:return super().read(n)
            except (TimeoutError,OSError) as error:
                self.pos=position
                if attempt==3:raise
                print('retry range',position,repr(error),flush=True)
remote_zip.Remote=RetryingRemote
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mesh():
    url='https://huggingface.co/Fictionary/GeoSVR/resolve/main/meshes_complete/DTU/scan37_mesh.ply'
    p=DEST/'scan37_mesh.ply'
    if not p.exists():
        part=DEST/'scan37_mesh.ply.part'
        with urllib.request.urlopen(url,timeout=60) as r,part.open('xb') as f:
            while b:=r.read(1<<20):f.write(b)
        assert part.stat().st_size==41042161;assert digest(part)=='40f4db9aaaaab4d07548ab88aeb33fc0428eee29b49b09f2bff74239e4049ad5'
        part.rename(p)
    return dict(kind='mesh',url=url,path=str(p),bytes=p.stat().st_size,sha256=digest(p))
class Stream:
    def __init__(self,r):self.r=r;self.n=0;self.last=0
    def read(self,n):
        if self.n+n>3_000_000_000:raise RuntimeError('3GB stream cap')
        b=self.r.read(n);self.n+=len(b)
        if self.n-self.last>300_000_000:print('camera stream MB',round(self.n/1e6),flush=True);self.last=self.n
        return b
def camera():
    result=dict(kind='camera',url=ARCHIVE,status='not_found');s=None;p=DEST/'cameras.npz'
    if p.exists():return dict(kind='camera',url=ARCHIVE,status='found',path=str(p),sha256=digest(p),cached=True)
    try:
        with urllib.request.urlopen(ARCHIVE,timeout=60) as response:
            s=Stream(response)
            with tarfile.open(fileobj=s,mode='r|gz') as t:
                for entry in t:
                    if entry.isfile() and '/scan37/' in entry.name and entry.name.endswith('cameras.npz'):
                        b=t.extractfile(entry).read()
                        with p.open('xb') as f:f.write(b)
                        result.update(status='found',member=entry.name,path=str(p),sha256=digest(p));break
    except Exception as e:result['error']=repr(e)
    result['stream_bytes']=0 if s is None else s.n
    return result
def main():
    DEST.mkdir(exist_ok=True);start=time.time();results=[]
    # Preserve any previous incomplete bytes before a from-start retry.
    for partial in DEST.glob('*.part'):
        partial.rename(partial.with_name(partial.name+f'.failed-{time.time_ns()}'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(mesh),pool.submit(camera),
            pool.submit(extract,'https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip','Points/stl/stl037_total.ply',DEST/'stl037_total.ply'),
            pool.submit(extract,'https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip','SampleSet/MVS Data/ObsMask/ObsMask37_10.mat',DEST/'ObsMask37_10.mat')]
        for fut in as_completed(futures):
            try:r=fut.result()
            except Exception as e:r=dict(status='FAILED',error=repr(e))
            results.append(r);print(json.dumps(r),flush=True)
    manifest=DEST/'DOWNLOAD_MANIFEST.json'
    if manifest.exists():manifest=DEST/f'DOWNLOAD_RETRY_{time.time_ns()}.json'
    with manifest.open('x') as f:json.dump(dict(files=results,seconds=time.time()-start),f,indent=2)
if __name__=='__main__':main()

"""Fetch actual undistorted-image COLMAP calibration, not normalized-screen metadata."""
import urllib.request,tarfile,time,json,hashlib
from pathlib import Path
from prepare_images import DEST,URL,Stream
def main():
    records=[];counts={24:set(),37:set()};error=None;s=None
    try:
        with urllib.request.urlopen(URL,timeout=45) as r:
            s=Stream(r)
            with tarfile.open(fileobj=s,mode='r|gz') as t:
                for e in t:
                    parts=Path(e.name).parts;scene=next((i for i in counts if f'scan{i}' in parts),None)
                    if scene is None or not e.isfile() or '/sparse/0/' not in e.name:continue
                    if Path(e.name).name not in ('cameras.bin','images.bin','points3D.bin'):continue
                    p=DEST/f'scan{scene}'/'sparse/0'/Path(e.name).name;p.parent.mkdir(parents=True,exist_ok=True)
                    b=t.extractfile(e).read()
                    with p.open('xb') as f:f.write(b)
                    records.append(dict(member=e.name,path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));counts[scene].add(p.name);print(e.name,len(b),flush=True)
                    if min(map(len,counts.values()))==3:break
    except Exception as e:error=repr(e)
    with (DEST/f'CALIBRATION_DOWNLOAD_{time.time_ns()}.json').open('x') as f:json.dump(dict(files=records,counts={str(k):sorted(v) for k,v in counts.items()},error=error,stream_bytes=s.n if s else 0),f,indent=2)
    if min(map(len,counts.values()))<3:raise SystemExit(1)
if __name__=='__main__':main()

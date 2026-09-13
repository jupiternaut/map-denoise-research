"""Bounded streaming probe of GeoSVR-linked DTU archive; extract cameras only."""
import urllib.request,tarfile,json,io,hashlib
from pathlib import Path
ROOT=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference')
URL='https://drive.usercontent.google.com/download?id=1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ&export=download&confirm=t'
class Limited:
    def __init__(self,r):self.r=r;self.n=0
    def read(self,n):
        if self.n+n>100_000_000:raise RuntimeError('100 MB camera probe cap reached')
        b=self.r.read(n);self.n+=len(b);return b
def main():
    report=dict(source=URL,source_provenance='GeoSVR README DTU link',status='not_found',bytes_read=0)
    with urllib.request.urlopen(URL,timeout=45) as response:
        limited=Limited(response)
        try:
            with tarfile.open(fileobj=limited,mode='r|gz') as tar:
                for i,entry in enumerate(tar):
                    if i<12:print(entry.name,entry.size,flush=True)
                    if entry.isfile() and entry.name.endswith('scan24/cameras.npz'):
                        b=tar.extractfile(entry).read();(ROOT/'cameras_geosvr_linked.npz').write_bytes(b)
                        report.update(status='found',member=entry.name,sha256=hashlib.sha256(b).hexdigest());break
        except Exception as e:report['error']=str(e)
        report['bytes_read']=limited.n
    (ROOT/'CAMERA_PROBE.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()

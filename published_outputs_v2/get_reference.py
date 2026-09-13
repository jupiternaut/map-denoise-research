import json,urllib.request,concurrent.futures,hashlib
from pathlib import Path
from remote_zip import extract
ROOT=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v2-reference')
def camera():
    url='https://huggingface.co/datasets/turandai/gaussian-surfels-dtu/resolve/main/scan24/cameras.npz'
    p=ROOT/'cameras_surfels.npz'
    if not p.exists():
        with urllib.request.urlopen(url,timeout=40) as r:p.write_bytes(r.read())
    return dict(url=url,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        note='Candidate normalization only; must cross-check GeoSVR evaluation mesh before use')
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        jobs=[ex.submit(extract,'https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip','Points/stl/stl024_total.ply',ROOT/'stl024_total.ply'),
              ex.submit(extract,'https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip','SampleSet/MVS Data/ObsMask/ObsMask24_10.mat',ROOT/'ObsMask24_10.mat'),
              ex.submit(extract,'https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip','SampleSet/MVS Data/ObsMask/Plane24.mat',ROOT/'Plane24.mat'),
              ex.submit(camera)]
        records=[j.result() for j in jobs]
    (ROOT/'MANIFEST.json').write_text(json.dumps(records,indent=2));print(records,flush=True)
if __name__=='__main__':main()

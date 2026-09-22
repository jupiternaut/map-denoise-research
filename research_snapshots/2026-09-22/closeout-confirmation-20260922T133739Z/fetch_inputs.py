"""Bounded, resumable retrieval of locked INPUTS ONLY; no reference retrieval."""
import concurrent.futures, hashlib, json, os, shutil, socket, time, urllib.request, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1')
SCENES=(40,55,65,69)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def fetch(url,path,size,expected=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=size:raise RuntimeError('existing size mismatch '+str(path))
        digest=sha(path)
        if expected and digest!=expected:raise RuntimeError('existing hash mismatch')
        return dict(path=str(path),url=url,bytes=size,sha256=digest,reused=True)
    part=path.with_suffix(path.suffix+'.part')
    for attempt in range(8):
        start=part.stat().st_size if part.exists() else 0
        if start==size:break
        if start>size:raise RuntimeError('oversize partial')
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{size-1}','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=45) as r:
                cr=r.headers.get('Content-Range')
                if r.status==206 and cr!=f'bytes {start}-{size-1}/{size}':raise RuntimeError('range mismatch '+str(cr))
                if r.status not in (200,206) or (r.status==200 and start):raise RuntimeError('resume not supported')
                with part.open('ab') as out:
                    total=start;last=start
                    while True:
                        b=r.read(min(1<<20,size-total+1))
                        if not b:break
                        if total+len(b)>size:raise RuntimeError('download exceeds manifest')
                        out.write(b);total+=len(b)
                        if total-last>=128*1024*1024:
                            print(path.name,total,'/',size,flush=True);last=total
            if part.stat().st_size!=size:raise RuntimeError('short transfer')
            break
        except Exception as e:
            print('RETRY',path.name,attempt+1,str(e),flush=True)
            if attempt==7:raise
    digest=sha(part)
    if expected and digest!=expected:raise RuntimeError('download hash mismatch '+str(path))
    part.rename(path)
    print('DOWNLOADED',path.name,size,flush=True)
    return dict(path=str(path),url=url,bytes=size,sha256=digest,reused=False)
def main():
    if socket.gethostname()!='liekkas':raise RuntimeError('wrong host')
    audit=json.loads((ROOT/'PREPARATION_AUDIT.json').read_text())
    for name in ('EXPERIMENT_PROTOCOL.md','DATA_READINESS.json','TRAINING_LOCK.json'):
        if sha(ROOT/name)!=audit['sha256'][name]:raise RuntimeError('protocol lock changed '+name)
    DATA.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(DATA).free<10*1024**3:raise RuntimeError('need 10GiB staging headroom')
    meta=json.loads((ROOT/'DATA_READINESS.json').read_text())['results'];jobs=[]
    image=meta['images']
    jobs.append((image['url']+'&closeout_input=20260922',DATA/'downloads/dtu.tar.gz',image['archive_bytes'],None))
    for sid in SCENES:
        v=meta['meshes']['scenes'][str(sid)][0]
        url='https://huggingface.co/Fictionary/GeoSVR/resolve/main/'+v['path']
        jobs.append((url,DATA/f'inputs/scan{sid}/mesh.ply',v['size'],v['lfs']['oid']))
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda args:fetch(*args),jobs):records.append(result)
    extracted=[];total=0
    with tarfile.open(DATA/'downloads/dtu.tar.gz','r:gz') as tar:
        for m in tar:
            parts=Path(m.name).parts
            scene=next((p for p in parts if p in {f'scan{x}' for x in SCENES}),None)
            if scene is None or not m.isfile():continue
            tail=Path(*parts[parts.index(scene)+1:])
            if '..' in tail.parts or tail.is_absolute():raise RuntimeError('unsafe archive member')
            if not (tail.parts and (tail.parts[0] in ('images','sparse') or str(tail)=='cameras.npz')):continue
            total+=m.size
            if total>6*1024**3 or m.size>512*1024**2:raise RuntimeError('selected extraction cap exceeded')
            target=DATA/'inputs'/scene/tail
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with tar.extractfile(m) as src,target.open('xb') as out:shutil.copyfileobj(src,out,1<<20)
            if target.stat().st_size!=m.size:raise RuntimeError('extraction mismatch')
            extracted.append(dict(member=m.name,path=str(target),bytes=m.size,sha256=sha(target)))
    result=dict(host=socket.gethostname(),files=records,extracted=extracted,extracted_bytes=total,
                scenes=SCENES,reference_downloaded=False,protocol_sha256=sha(ROOT/'EXPERIMENT_PROTOCOL.md'))
    manifest=DATA/'INPUT_MANIFEST.json'
    with manifest.open('x') as f:json.dump(result,f,indent=2)
    print('INPUTS_READY',len(extracted),'members',total,'bytes',flush=True)
if __name__=='__main__':main()

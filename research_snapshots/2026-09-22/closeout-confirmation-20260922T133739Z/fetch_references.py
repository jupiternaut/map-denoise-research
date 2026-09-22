"""Retrieve evaluator-only members AFTER all method outputs have been sealed."""
import concurrent.futures,json,zipfile
from pathlib import Path
import data_probe
from scene_adapter import sha,save
ROOT=Path(__file__).resolve().parent
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1')
data_probe.LIMIT=512*1024**2
def scene(sid,metadata):
    out=DATA/'evaluation_only'/f'scan{sid}';out.mkdir(parents=True,exist_ok=True);records=[]
    for kind in ('reference','masks'):
        m=metadata[kind];remote=data_probe.MetadataZip(m['url'])
        with zipfile.ZipFile(remote) as z:
            for item in m['scenes'][str(sid)]['members']:
                info=z.getinfo(item['name'])
                if (info.file_size,info.CRC)!=(item['bytes'],item['crc32']):raise RuntimeError('remote metadata changed')
                path=out/Path(item['name']).name
                if not path.exists():
                    part=path.with_suffix(path.suffix+'.part')
                    if part.exists():raise RuntimeError('partial reference requires explicit recovery')
                    with z.open(info) as src,part.open('xb') as dest:
                        count=0
                        while b:=src.read(1<<20):
                            count+=len(b)
                            if count>info.file_size:raise RuntimeError('size cap')
                            dest.write(b)
                    if count!=info.file_size:raise RuntimeError('short reference')
                    part.rename(path)
                if path.stat().st_size!=info.file_size:raise RuntimeError('reference size mismatch')
                records.append(dict(scene=sid,kind=kind,path=str(path),url=m['url'],member=item['name'],bytes=info.file_size,sha256=sha(path)))
        print('REFERENCE_BYTES_READY',sid,kind,remote.transferred,flush=True)
    return records
def main():
    seals={}
    for sid in (40,55,65,69):
        p=ROOT/('adaptation' if sid==40 else 'confirmation')/f'scan{sid}'/'SEALED.json'
        if not p.exists():raise RuntimeError('method output not sealed '+str(sid))
        seals[str(sid)]=sha(p)
    metadata=json.loads((ROOT/'DATA_READINESS.json').read_text())['results']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=sum(list(pool.map(lambda sid:scene(sid,metadata),(40,55,65,69))),[])
    save(DATA/'REFERENCE_MANIFEST.json',dict(records=records,method_seals_before_retrieval=seals))
    print('REFERENCES_READY',flush=True)
if __name__=='__main__':main()

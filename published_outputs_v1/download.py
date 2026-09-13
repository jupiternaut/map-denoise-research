"""Download published artifacts; remotely extract selected ZIP members, verify CRC."""
import concurrent.futures, hashlib, json, struct, urllib.request, zlib, time
from pathlib import Path

ROOT = Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v1')
ZIP = 'https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip'
HF = 'https://huggingface.co/Fictionary/GeoSVR/resolve/main/'

def request(url, start=None, end=None):
    headers = {'User-Agent': 'map-denoise-research/1.0'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    if start is not None and r.status != 206:
        r.close()
        raise RuntimeError('Server did not honor range; refusing whole archive')
    return r

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def zip_entries():
    with urllib.request.urlopen(urllib.request.Request(ZIP,method='HEAD'),timeout=60) as r:
        length=int(r.headers['Content-Length'])
    with request(ZIP,length-131072,length-1) as r:data=r.read()
    entries=[];pos=0
    while True:
        pos=data.find(b'PK\x01\x02',pos)
        if pos<0:break
        fields=struct.unpack_from('<4s6H3I5H2I',data,pos)
        _,_,_,flags,method,_,_,crc,compressed,raw,n,e,c,_,_,_,offset=fields
        name=data[pos+46:pos+46+n].decode()
        if name in ('room/cameras.json','room/cfg_args','room/point_cloud/iteration_30000/point_cloud.ply'):
            extra=data[pos+46+n:pos+46+n+e];ep=0
            while ep+4<=len(extra):
                kind,size=struct.unpack_from('<HH',extra,ep);payload=extra[ep+4:ep+4+size]
                if kind==1:
                    zp=0
                    if raw==0xffffffff:raw=struct.unpack_from('<Q',payload,zp)[0];zp+=8
                    if compressed==0xffffffff:compressed=struct.unpack_from('<Q',payload,zp)[0];zp+=8
                    if offset==0xffffffff:offset=struct.unpack_from('<Q',payload,zp)[0]
                ep+=4+size
            entries.append(dict(name=name,method=method,crc=crc,compressed=compressed,raw=raw,offset=offset))
        pos+=46+n+e+c
    assert len(entries)==3,entries
    return entries

def extract(entry):
    path=ROOT/entry['name'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.stat().st_size==entry['raw']
        return dict(path=str(path),source=ZIP,member=entry,sha256=digest(path))
    with request(ZIP,entry['offset'],entry['offset']+29) as r:head=r.read()
    assert head[:4]==b'PK\x03\x04'
    n,e=struct.unpack_from('<HH',head,26);start=entry['offset']+30+n+e
    temp=path.with_suffix(path.suffix+'.part');crc=0;count=0
    decompressor=zlib.decompressobj(-15) if entry['method']==8 else None
    assert entry['method'] in (0,8)
    with request(ZIP,start,start+entry['compressed']-1) as r,temp.open('wb') as f:
        while b:=r.read(1<<20):
            out=decompressor.decompress(b) if decompressor else b
            f.write(out);crc=zlib.crc32(out,crc);count+=len(out)
        if decompressor:
            out=decompressor.flush();f.write(out);crc=zlib.crc32(out,crc);count+=len(out)
            assert decompressor.eof
    assert count==entry['raw'] and crc==entry['crc'],(count,crc,entry)
    temp.rename(path)
    print('verified ZIP member',path,count,flush=True)
    return dict(path=str(path),source=ZIP,member=entry,sha256=digest(path),crc_verified=True)

def mesh(name,remote,expected):
    path=ROOT/name;url=HF+remote;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with request(url) as r,path.with_suffix('.part').open('wb') as f:
            while b:=r.read(1<<20):f.write(b)
        assert path.with_suffix('.part').stat().st_size==expected
        path.with_suffix('.part').rename(path)
    assert path.stat().st_size==expected
    print('downloaded mesh',name,expected,flush=True)
    return dict(path=str(path),source=url,bytes=expected,sha256=digest(path))

def main():
    ROOT.mkdir(parents=True,exist_ok=True);t=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(extract,e) for e in zip_entries()]
        jobs += [pool.submit(mesh,'scan24_mesh.ply','meshes_complete/DTU/scan24_mesh.ply',39831230),
                 pool.submit(mesh,'Courthouse_mesh.ply','meshes_complete/TnT/Courthouse_mesh.ply',652102212)]
        results=[j.result() for j in jobs]
    (ROOT/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(dict(files=results,seconds=time.monotonic()-t),indent=2))
    print('ALL COMPLETE',time.monotonic()-t,flush=True)

if __name__=='__main__':main()

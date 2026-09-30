"""Only after prediction seal: retrieve two independent official DTU members."""
import importlib.util, shutil, time, zipfile
from mvs_transfer import ROOT, load, dump, verify, verify_lock, sha

def main():
    verify_lock(); seal=load(ROOT/'PREDICTIONS_SEALED.json'); verify(seal['files'])
    path='/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z/data_probe.py'
    spec=importlib.util.spec_from_file_location('_locked_remote_zip',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.LIMIT=512*1024**2
    url='https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip'
    files=[]; dest=ROOT/'references'; dest.mkdir(exist_ok=True)
    remote=module.MetadataZip(url)
    with zipfile.ZipFile(remote) as z:
        for sid,size,crc in [(118,83342093,3008898086),(122,73742567,2738697435)]:
            name=f'Points/stl/stl{sid:03d}_total.ply'; info=z.getinfo(name)
            assert info.file_size==size and info.CRC==crc
            target=dest/f'stl{sid:03d}_total.ply'
            if not target.exists():
                part=target.with_suffix('.part')
                with z.open(info) as src,part.open('xb') as out:shutil.copyfileobj(src,out,1<<20)
                assert part.stat().st_size==size; part.rename(target)
            files.append(dict(scene=sid,path=str(target),url=url,member=name,bytes=size,crc32=crc,sha256=sha(target)))
            print('REFERENCE_ACQUIRED',sid,size,flush=True)
    dump(dest/'MANIFEST.json',dict(files=files,prediction_seal_sha256=sha(ROOT/'PREDICTIONS_SEALED.json'),
         downloaded_after_prediction_seal=True,http_bytes=remote.transferred,
         created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
if __name__=='__main__':main()

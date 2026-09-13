"""Seekable HTTP range ZIP reader; refuses accidental full-archive downloads."""
import io,urllib.request,zipfile,json,hashlib
from pathlib import Path
class Remote(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.cache=None
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
            self.size=int(r.headers['Content-Length'])
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        if self.cache and self.cache[0]<=self.pos and self.pos+n<=self.cache[1]:
            a,b,d=self.cache;out=d[self.pos-a:self.pos-a+n];self.pos+=n;return out
        end=min(self.size,self.pos+max(n,65536));assert end-self.pos<20_000_000
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{end-1}'})
        with urllib.request.urlopen(req,timeout=60) as r:
            assert r.status==206,'Refusing full archive'
            data=r.read(end-self.pos)
        assert len(data)==end-self.pos
        self.cache=(self.pos,end,data);out=data[:n];self.pos+=n;return out

def extract(url,member,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(Remote(url)) as z:
        info=z.getinfo(member);assert info.file_size<512_000_000
        if not dest.exists():
            with z.open(member) as src,dest.with_suffix(dest.suffix+'.part').open('wb') as out:
                while b:=src.read(1<<20):out.write(b)
            assert dest.with_suffix(dest.suffix+'.part').stat().st_size==info.file_size
            dest.with_suffix(dest.suffix+'.part').rename(dest)
        h=hashlib.sha256()
        with dest.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return dict(url=url,member=member,bytes=info.file_size,sha256=h.hexdigest(),path=str(dest))

if __name__=='__main__':
    for name in ('Points','SampleSet'):
        url=f'https://roboimagedata2.compute.dtu.dk/data/MVS/{name}.zip'
        with zipfile.ZipFile(Remote(url)) as z:
            print(name,[(i.filename,i.file_size,i.compress_size) for i in z.infolist() if ('024' in i.filename or '24' in i.filename) and ('stl' in i.filename or 'ObsMask' in i.filename)],flush=True)

"""Bounded public metadata probe. Never extracts an archive member or loads GT.

Usage: python -B data_probe.py
Only this workspace's DATA_READINESS.json is written. Maximum ZIP metadata
payload: 8 MiB per archive; HTTP 200 to a range request is rejected before read.
"""
import concurrent.futures
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENES = (40, 55, 65, 69)
LIMIT = 8 * 1024 * 1024
URLS = {
    'images': 'https://drive.usercontent.google.com/download?id=1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ&export=download&confirm=t',
    'reference': 'https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip',
    'masks': 'https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip',
}

class MetadataZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.transferred, self.cache = url, 0, 0, None
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=25) as r:
            if r.status != 206:
                raise RuntimeError(f'Range unsupported (HTTP {r.status}); no full body downloaded')
            match = re.fullmatch(r'bytes 0-0/(\d+)', r.headers.get('Content-Range', ''))
            if not match:
                raise RuntimeError('Missing exact Content-Range')
            self.size = int(match.group(1))
            if len(r.read(2)) != 1:
                raise RuntimeError('Incorrect probe payload')
            self.transferred = 1

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('negative offset')
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b''
        if self.cache and self.cache[0] <= self.pos and self.pos+n <= self.cache[1]:
            start, _, payload = self.cache
            answer = payload[self.pos-start:self.pos-start+n]
            self.pos += n
            return answer
        end = min(self.size, self.pos + max(n, 65536))
        size = end-self.pos
        if self.transferred + size > LIMIT:
            raise RuntimeError('ZIP central-directory metadata budget exceeded')
        request = urllib.request.Request(self.url, headers={
            'Range': f'bytes={self.pos}-{end-1}', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=25) as r:
            expected = f'bytes {self.pos}-{end-1}/{self.size}'
            if r.status != 206 or r.headers.get('Content-Range') != expected:
                raise RuntimeError('Refusing unsupported or incorrect HTTP range')
            payload = r.read(size+1)
        if len(payload) != size: raise RuntimeError('Incorrect range payload length')
        self.transferred += size
        self.cache = self.pos, end, payload
        self.pos += n
        return payload[:n]

def wanted(name, kind, scene):
    if kind == 'images':
        return f'/scan{scene}/' in '/'+name and (
            '/images/' in name or name.endswith('cameras.npz') or '/sparse/' in name)
    if kind == 'reference': return name.endswith(f'stl{scene:03d}_total.ply')
    return name.endswith(f'ObsMask{scene}_10.mat') or name.endswith(f'Plane{scene}.mat')

def probe_zip(kind, url):
    remote = None
    result = {'url': url, 'status': 'UNAVAILABLE', 'content_extracted': False}
    try:
        remote = MetadataZip(url)
        with zipfile.ZipFile(remote) as z:
            infos = z.infolist()
            result.update(status='METADATA_READY', archive_bytes=remote.size, scenes={})
            for scene in SCENES:
                chosen = [i for i in infos if not i.is_dir() and wanted(i.filename, kind, scene)]
                result['scenes'][str(scene)] = {
                    'members': [{'name': i.filename, 'bytes': i.file_size,
                                 'compressed_bytes': i.compress_size, 'crc32': i.CRC} for i in chosen],
                    'bytes': sum(i.file_size for i in chosen),
                    'compressed_bytes': sum(i.compress_size for i in chosen),
                }
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    result['metadata_bytes_read'] = 0 if remote is None else remote.transferred
    return kind, result

def probe_meshes():
    url = 'https://huggingface.co/api/models/Fictionary/GeoSVR/tree/main/meshes_complete/DTU?recursive=false&expand=false'
    result = {'url': url, 'status': 'UNAVAILABLE', 'content_extracted': False}
    try:
        with urllib.request.urlopen(url, timeout=25) as r: payload = r.read(LIMIT+1)
        if len(payload) > LIMIT: raise RuntimeError('Metadata budget exceeded')
        items = json.loads(payload)
        result.update(status='METADATA_READY', metadata_bytes_read=len(payload), scenes={})
        for scene in SCENES:
            selected = [i for i in items if i['path'].endswith(f'/scan{scene}_mesh.ply')]
            result['scenes'][str(scene)] = selected
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    return 'meshes', result

def probe_images():
    # This historical GeoSVR input is tar.gz, not a ZIP with a seekable directory.
    url = URLS['images']
    result = {'url': url, 'status': 'UNAVAILABLE', 'content_extracted': False,
              'member_sizes_known': False, 'format': 'tar.gz'}
    try:
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-15', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(req, timeout=25) as r:
            match = re.fullmatch(r'bytes 0-15/(\d+)', r.headers.get('Content-Range', ''))
            if r.status != 206 or not match:
                raise RuntimeError('Range metadata unsupported; refusing full download')
            first = r.read(17)
            if len(first) != 16 or first[:2] != b'\x1f\x8b':
                raise RuntimeError('Archive is not gzip as expected')
            result.update(status='ARCHIVE_METADATA_ONLY', archive_bytes=int(match.group(1)),
                          metadata_bytes_read=16, content_disposition=r.headers.get('Content-Disposition'))
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    return 'images', result

def main():
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(probe_zip, k, v) for k, v in URLS.items() if k != 'images']
        jobs += [pool.submit(probe_meshes), pool.submit(probe_images)]
        for future in concurrent.futures.as_completed(jobs):
            k, v = future.result()
            results[k] = v
            print(k, v['status'], v.get('error', ''), flush=True)
    output = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'public metadata only; no scene contents or reference accessed',
              'proposed_adaptation': 40, 'proposed_confirmation': [55,65,69],
              'results': results}
    (ROOT/'DATA_READINESS.json').write_text(json.dumps(output, indent=2)+'\n')

if __name__ == '__main__': main()

"""Check staged files for known credential patterns, oversized blobs, and copied evidence hashes."""
from pathlib import Path
import subprocess,re,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0');paths=[p for p in paths if p]
    staged={entry.split('\t',1)[1]:entry.split('\t',1)[0].split()[1] for entry in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).decode().split('\0') if entry}
    patterns=[rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{30,}',
        rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'\bAKIA[0-9A-Z]{16}\b',
        rb'sk-[A-Za-z0-9_-]{30,}',rb'https?://[^\s/@:]+:[^\s/@]+@',
        rb'(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*[\x22\x27][^\x22\x27\n]{8,}[\x22\x27]']
    hits=[];large=[];total=0
    for path in paths:
        data=(ROOT/path).read_bytes();total+=len(data)
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==staged[path],f'staged/working bytes differ: {path}'
        if len(data)>50*1024**2:large.append(path)
        if path=='tools/check_git_snapshot.py':continue
        if any(re.search(p,data) for p in patterns):hits.append(path)
    manifest=json.loads((ROOT/'evidence/EXPORT_MANIFEST.json').read_text())
    for item in manifest['files']:
        p=ROOT/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
    print(json.dumps(dict(tracked_files=len(paths),total_bytes=total,secret_pattern_files=hits,oversized_files=large,
        copied_files_verified=len(manifest['files']),scan_scope='Known-pattern scan, not exhaustive secret detection; no credential values printed'),indent=2))
    if hits or large:raise SystemExit(1)
if __name__=='__main__':main()

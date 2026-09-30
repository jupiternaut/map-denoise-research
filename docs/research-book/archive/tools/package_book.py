#!/usr/bin/env python3
"""Seal this local book and create a portable ZIP, without publishing."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'PACKAGE_MANIFEST.json'
ARCHIVE=ROOT.parent/'geometry-meta-research-gitbook-20260929.zip'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=MANIFEST and not any(x in p.relative_to(ROOT).parts for x in ('.git','__pycache__','site')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:
        data=json.loads(MANIFEST.read_text())
        expected={r['path'] for r in data['files']}
        assert expected=={p.relative_to(ROOT).as_posix() for p in files()},'File set changed since seal'
        for r in data['files']:assert digest(ROOT/r['path'])==r['sha256'],r['path']
        with zipfile.ZipFile(ARCHIVE) as z:
            assert z.testzip() is None
            for r in data['files']:
                assert hashlib.sha256(z.read(ROOT.name+'/'+r['path'])).hexdigest()==r['sha256'],r['path']
        print(json.dumps({'sealed_files':len(data['files']),'archive_sha256':digest(ARCHIVE),'status':'passed'},ensure_ascii=False));return
    if ARCHIVE.exists():raise SystemExit(f'Archive already exists; use a new explicit filename instead of overwriting: {ARCHIVE}')
    data={'host':'liekkas','root':str(ROOT),'publication':'local_only','conversation_cutoff_utc':'2026-09-29T11:56:40.445Z',
          'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files()]}
    MANIFEST.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files()+[MANIFEST]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    print(json.dumps({'archive':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'sha256':digest(ARCHIVE),'sealed_files':len(data['files'])},ensure_ascii=False))

if __name__=='__main__':main()

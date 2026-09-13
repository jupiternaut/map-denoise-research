"""Copy final delivery code/docs next to immutable raw outputs; no old snapshot edits."""
from pathlib import Path
import sys,json,hashlib,shutil,socket


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(path):
    run=Path(path);source=Path(__file__).resolve().parent
    final=run/'final_delivery';final.mkdir()
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):
            shutil.copyfile(p,final/p.name)
    files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()}
    old=json.loads((run/'V15_HASHES_BEFORE.json').read_text())
    assert all(sha(Path(p))==h for p,h in old.items())
    with (run/'FINAL_MANIFEST.json').open('x') as f:
        json.dump(dict(host=socket.gethostname(),files=files,v15_files_unchanged=len(old),
                       note='Final code/docs copied separately; original source snapshot and scoring repair preserved.'),f,indent=2)
    print('FINAL',run,'files',len(files),'bytes',sum((run/p).stat().st_size for p in files),flush=True)


if __name__=='__main__':main(sys.argv[1])

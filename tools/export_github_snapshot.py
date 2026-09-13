"""Export compact experiment evidence and frozen legacy sources without modifying originals."""
from pathlib import Path
import json,hashlib,shutil,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
LEGACY=Path('/home/grf/Documents/Codex/2026-09-11')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    dest=ROOT/'evidence';dest.mkdir(exist_ok=False);manifest=[]
    def copy(p,q):
        if p.is_symlink():raise ValueError(f'no symlink export: {p}')
        q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        digest=sha(p);assert sha(q)==digest
        manifest.append(dict(source=str(p),path=str(q.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest))
    runroots=[RUNS]+[RUNS.parent/name for name in ('published-outputs-v1','published-outputs-v2','published-outputs-v3') if (RUNS.parent/name).is_dir()]
    for runroot in runroots:
        candidates=[]
        if runroot==RUNS:
            candidates=[(folder,p) for folder in sorted(runroot.iterdir()) if folder.is_dir() and folder.name not in ('checkpoints','patches','synthetics','preview','logs') for p in folder.iterdir() if p.is_file()]
        else:candidates=[(runroot,p) for p in runroot.iterdir() if p.is_file()]
        for folder,p in candidates:
            name=p.name.upper()
            selected=(p.suffix=='.csv' or p.suffix=='.md' or (p.suffix=='.json' and any(s in name for s in ('SUMMARY','AUDIT','TEST_RESULTS','TIMING','LOCK','MANIFEST','REPAIR','REPLACEMENT'))))
            if selected and p.stat().st_size<=5*1024**2:copy(p,dest/'runs'/folder.name/p.name)
        # Retain ready-made explanatory figures, not the raw model/point arrays.
        for p in runroot.glob('*/*/*.png') if runroot==RUNS else runroot.glob('*/*.png'):
            if p.parent.name not in ('figures','readout','plots'):continue
            if p.stat().st_size<=5*1024**2:copy(p,dest/'runs'/p.relative_to(runroot))
    # Source-only snapshots of the explicit pre-pilot dependency chain.
    for name in ('map-denoise-correlation-v1','map-denoise-six-track-v1','map-denoise-t2-boundary-v1'):
        folder=LEGACY/name
        for p in sorted(folder.rglob('*')):
            rel=p.relative_to(folder)
            if any(x in ('.git','__pycache__','data','pilot','real_results','synthetic_results','outputs','runs') for x in rel.parts):continue
            if p.is_file() and p.suffix in ('.py','.md') and p.stat().st_size<2*1024**2:
                copy(p,ROOT/'legacy_sources'/name/rel)
    versions={}
    for name in ('numpy','scipy','matplotlib','open3d','pymeshlab','torch'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    info=dict(project=str(ROOT),host=platform.node(),python=platform.python_version(),package_versions=versions,
        export_policy='Current project source plus compact run evidence and source-only explicit legacy dependencies. No raw point clouds, training assets, Python environments, or full per-point output archives.',
        files=manifest,total_exported_bytes=sum(x['bytes'] for x in manifest))
    with (dest/'EXPORT_MANIFEST.json').open('x') as f:json.dump(info,f,indent=2)
    print(json.dumps(dict(files=len(manifest),bytes=info['total_exported_bytes']),indent=2))

if __name__=='__main__':main()

"""Verify frozen inputs and inventory A15 outputs without changing experiments."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def main():
    assert socket.gethostname()=='liekkas'
    assert ROOT==Path('/srv/slam-research/grf/map-denoise/runs/lovelace-proof-reuse-20261014T074405Z').resolve()
    checks={}
    for name in ('FREEZE_tolerance.json','FREEZE_switches.json','FREEZE_supplement.json'):
        f=load(ROOT/name);checks[name]=all(sha(Path(p))==h for p,h in f['files'].items())
    assert all(checks.values())
    for stage in ('tolerance','switches'):
        ready=load(ROOT/'outputs'/stage/'READY.json')
        assert all(sha(ROOT/r['result'])==r['sha256'] and r['returncode']==0 for r in ready['receipts'])
        assert load(ROOT/'outputs'/stage/'VERIFICATION.json')['all_valid']
    assert load(ROOT/'outputs/switches/STATE_EQUIVALENCE.json')['all_equal']
    boundary=load(ROOT/'outputs/boundary-tolerance.json')
    assert boundary['all_targets_completed'] and len(boundary['records'])==72
    assert all(len(r['tiles'])==4 and all(t['independent_validation']['verified'] for t in r['tiles']) for r in boundary['records'])
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '.aris' not in p.parts and '__pycache__' not in p.parts
           and p.name not in ('ARTIFACTS.json','MANIFEST.md')]
    inventory={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    result={'utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'freeze_checks':checks,
            'files':inventory,'file_count':len(files),'private_review_traces_excluded':True}
    with (ROOT/'ARTIFACTS.json').open('x') as f:json.dump(result,f,indent=2)
    with (ROOT/'MANIFEST.md').open('x') as f:
        f.write('# Research Output Manifest\n\n| Timestamp | Skill | File | Stage | Description |\n|---|---|---|---|---|\n')
        for p in files:
            name=str(p.relative_to(ROOT));stamp=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()
            skill='experiment-audit' if 'AUDIT' in p.name else 'experiment-plan/ablation-planner'
            f.write(f'| {stamp} | {skill} | {name} | implementation | SHA256 in ARTIFACTS.json |\n')
        f.write('| '+result['utc']+' | experiment-plan | ARTIFACTS.json | implementation | complete inventory |\n')
    print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in inventory.values()),'freeze_checks':checks}))
if __name__=='__main__':main()

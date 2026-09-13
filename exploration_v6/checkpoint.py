"""Protect previous checkpoints while V6 experiments live in new directories."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUNS = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
V5_RUN_NAMES = (
    'exploration-v5-development-vdbkdouk',
    'exploration-v5-confirmation-p0oymcit',
    'real-transfer-v5-zrtui_f5',
    'diagnostic-v5-91id9auq',
)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def manifest():
    # Reuse the prior explicit protection set, then include complete V5 artifacts.
    spec = importlib.util.spec_from_file_location('_v6_readonly_protect_v5', PROJECT/'exploration_v5/run_v5.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.protected_manifest()
    for root in (PROJECT/'exploration_v5', *(RUNS/n for n in V5_RUN_NAMES)):
        for p in root.rglob('*'):
            if p.is_file(): result[str(p)] = digest(p)
    # Protect original project code/data documentation as well, excluding V6.
    for p in PROJECT.rglob('*'):
        if p.is_file() and not p.is_relative_to(HERE): result[str(p)] = digest(p)
    return result

def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify', type=Path)
    args=parser.parse_args()
    if socket.gethostname() != 'liekkas': raise RuntimeError('exact target must be liekkas')
    current=manifest()
    if args.verify:
        run=args.verify.resolve()
        before=json.loads((run/'PROTECTED_BEFORE.json').read_text())
        missing=sorted(set(before)-set(current))
        changed=sorted(p for p in before if p in current and before[p]!=current[p])
        added=sorted(set(current)-set(before))
        report=dict(host=socket.gethostname(),protected_files=len(before),missing=missing,
                    changed=changed,added=added,unchanged=not(missing or changed or added))
        save(run/'PROTECTED_AFTER.json',current)
        save(run/'VERIFICATION.json',report)
        print(json.dumps(report,indent=2))
        if not report['unchanged']: raise RuntimeError('protected checkpoint changed')
    else:
        run=Path(tempfile.mkdtemp(prefix='session-v6-',dir=RUNS))
        save(run/'PROTECTED_BEFORE.json',current)
        save(run/'SESSION.json',dict(host=socket.gethostname(),project=str(PROJECT),
            protected_files=len(current),scope='direction and conservative oracle-split experiments'))
        print(str(run))
        print('Protected files:',len(current))

if __name__=='__main__': main()

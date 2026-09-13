"""V7 checkpoint: protect the explicit historical set, not newly created work."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
RUNS = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
V6_SESSION = RUNS / 'session-v6-pt14hi1q'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('wrong target host')
    if args.verify:
        dest = args.verify.resolve()
        before = json.loads((dest/'PROTECTED_BEFORE.json').read_text())
        after = {p: digest(p) for p in before if Path(p).is_file()}
        report = dict(host=socket.gethostname(), protected_files=len(before),
                      missing=sorted(set(before)-set(after)),
                      changed=[p for p in before if p in after and before[p] != after[p]])
        report['unchanged'] = not (report['missing'] or report['changed'])
        save(dest/'PROTECTED_AFTER.json', after)
        save(dest/'VERIFICATION.json', report)
        print(json.dumps(report, indent=2))
        if not report['unchanged']:
            raise RuntimeError('historical file changed')
        return
    historical = json.loads((V6_SESSION/'PROTECTED_BEFORE.json').read_text())
    changed = [p for p, h in historical.items() if not Path(p).is_file() or digest(p) != h]
    if changed:
        raise RuntimeError('earlier checkpoint already mismatched: '+repr(changed[:10]))
    roots = [PROJECT/'exploration_v6', PROJECT/'planning_v7', V6_SESSION]
    roots += [RUNS/name for name in ('association-v6-bgcq8zff', 'direction-v6-edpxxdsb',
                                    'direction-support-diag-v6-smuzp3nl')]
    paths = set(historical)
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths.update(str(p) for p in root.rglob('*') if p.is_file())
    dest = Path(tempfile.mkdtemp(prefix='session-v7-', dir=RUNS))
    save(dest/'PROTECTED_BEFORE.json', {p: digest(p) for p in sorted(paths)})
    save(dest/'SESSION.json', dict(host=socket.gethostname(), project=str(PROJECT),
         historical_files=len(paths), earlier_v6_files_verified=len(historical),
         scope='GT-free reassociation and real coordinate diagnostics; V7 outputs are new'))
    print(str(dest), flush=True)
    print('Protected files:', len(paths), flush=True)

if __name__ == '__main__':
    main()

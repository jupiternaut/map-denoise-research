"""Project historical request metadata into a candidate-free extraction interface."""
import hashlib
import itertools
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert socket.gethostname() == 'liekkas'
    protocol = json.loads((ROOT/'PROTOCOL.json').read_text())
    plan_path = Path(protocol['source_plan'])
    source_path = Path(protocol['request_source'])
    plan = json.loads(plan_path.read_text())
    source = json.loads(source_path.read_text())
    scenes = {}
    for sid, s in plan['scenes'].items():
        names = sorted(set(s['Q'] + s['Q_extra']))
        def separation(pair):
            a,b=(s['cameras'][n]['center'] for n in pair)
            return sum((x-y)**2 for x,y in zip(a,b))
        pair = sorted(itertools.combinations(names,2),key=lambda p:(-separation(p),p))[0]
        cameras = {}
        for name in [s['reference'],*pair]:
            c = s['cameras'][name]
            cameras[name] = {k:c[k] for k in ['K_half','R','center','image_path','width','height']}
            assert Path(c['image_path']).is_file()
            cameras[name]['image_sha256'] = sha(c['image_path'])
        scenes[sid] = dict(reference=s['reference'],sources=list(pair),
                           depth_range_mm=s['depth_ranges']['wide'],cameras=cameras,
                           source_baseline_mm=separation(pair)**0.5)
    rows=[{k:r[k] for k in ['scene','roi','query','pixel_xy']} for r in source['rows']]
    assert len(rows)==21 and len({(r['roi'],r['query']) for r in rows})==21
    out = dict(schema_version=1,created_at=datetime.now(timezone.utc).isoformat(),
               scenes=scenes,rows=rows,description='Only image metadata and fixed queries; no old candidate coordinates, scores, decisions or labels.')
    dump(ROOT/'REQUESTS.json',out)
    dump(ROOT/'PREPARATION_SEAL.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files={str(p):sha(p) for p in [plan_path,source_path]},
        files={str(p):sha(p) for p in [ROOT/'REQUESTS.json',ROOT/'PROTOCOL.json',ROOT/'prepare.py',ROOT/'AGENTS.md']},
        historical_request_selection=True,gt_accessed=False))
    print(json.dumps({'requests':len(rows),'scenes':{s:v['sources'] for s,v in scenes.items()}}))


if __name__=='__main__':main()

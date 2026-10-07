"""Persist the subordinate mechanical hash check; no integrity judgment."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket

ROOT=Path('/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z')
assert socket.gethostname()=='liekkas'
queue=collections.deque(ROOT/n for n in ('RUN_LOCK.json','INPUTS.json','PREDICTIONS_SEALED.json','evaluation/SEALED.json'))
seen=set(); bindings=[]; unresolved=[]; actuals={}
def digest(path):
    if path not in actuals:
        actuals[path]=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return actuals[path]
while queue:
    p=queue.popleft()
    if p in seen or p.suffix!='.json' or not p.is_file():continue
    seen.add(p);d=json.loads(p.read_text())
    def walk(x,trail=()):
        if isinstance(x,dict):
            for k,v in x.items():
                if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v):
                    q=None
                    if k.startswith('/'):q=Path(k)
                    elif trail and trail[-1]=='sha256':q=(p.parent.parent if p.name=='LOCK.json' else p.parent)/k
                    elif k=='lock_sha256':q=p.parent/'LOCK.json'
                    elif k=='request_sha256':q=p.parent.parent/'REQUESTS.json'
                    elif k=='protocol_sha256':q=p.parent.parent/'PROTOCOL.json'
                    elif k=='image_sha256' and 'image_path' in x:q=Path(x['image_path'])
                    if q is None:unresolved.append((str(p),trail+(k,),v));continue
                    h=digest(q)
                    bindings.append(dict(manifest=str(p),field=list(trail+(k,)),path=str(q),expected=v,actual=h,status='match' if h==v else 'missing' if h is None else 'mismatch'))
                    queue.append(q)
                else:walk(v,trail+(k,))
        elif isinstance(x,list):
            for i,v in enumerate(x):walk(v,trail+(i,))
    walk(d)
by_path=collections.defaultdict(set)
for b in bindings:by_path[b['path']].add(b['expected'])
conflicts={p:sorted(hs) for p,hs in by_path.items() if len(hs)>1}
chronology=[]
for p in [ROOT/n for n in ('INPUTS.json','RUN_LOCK.json','DECISIONS.json','PREDICTIONS_SEALED.json','evaluation/RESULTS.json','evaluation/SEALED.json')]:
    chronology.append(dict(path=str(p),created_at=json.loads(p.read_text())['created_at'],mtime=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()))
assert not unresolved and not conflicts and all(b['status']=='match' for b in bindings)
out=dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),
    implementation='Subordinate read-only mechanical verifier algorithm, independently rerun by auditor',
    bindings_count=len(bindings),unique_bound_paths=len(actuals),visited_json=len(seen),bindings=bindings,
    unresolved=unresolved,conflicts=conflicts,chronology=chronology,input_sha256={str(p):h for p,h in actuals.items()})
with (ROOT/'audit/HASH_BINDINGS.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:out[k] for k in ('bindings_count','unique_bound_paths','visited_json','unresolved','conflicts')}))

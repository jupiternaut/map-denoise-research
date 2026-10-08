"""Exercise exact locked guard functions extracted via AST; no writes/network."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path('/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z')
if len(sys.argv)==1:
    out={}
    for mode in ('observation','prediction'):
        p=subprocess.run([sys.executable,'-B',__file__,mode],capture_output=True,text=True,check=True)
        out[mode]=json.loads(p.stdout)
    print(json.dumps(out,indent=2))
else:
    mode=sys.argv[1];tree=ast.parse((ROOT/'run_experiment.py').read_text());names={'os':os,'sys':sys,'Path':Path,'ROOT':ROOT}
    if mode=='observation':
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observation_guard')
    else:
        infer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='infer')
        node=next(n for n in infer.body if isinstance(n,ast.FunctionDef) and n.name=='no_truth')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'locked_guard','exec'),names)
    allowed=ROOT/'REQUESTS.json'
    if mode=='observation': names['observation_guard']([allowed])
    else: sys.addaudithook(names['no_truth'])
    assert allowed.read_bytes()
    targets=[ROOT/'evaluation/POINT_METRICS.csv',ROOT.parent/'colmap-transfer-20260930T180000Z/references/stl118_total.ply']
    if mode=='observation':targets.append(ROOT/'INPUTS.json')
    else: assert (ROOT/'INPUTS.json').read_bytes()
    blocked=[]
    for p in targets:
        try:p.read_bytes()
        except PermissionError:blocked.append(str(p))
        else:raise AssertionError('guard allowed '+str(p))
    # Synthetic audit events exercise r+ without opening, modifying, or creating a file.
    update_mode_allowed=True
    try:sys.audit('open',str(ROOT/'INPUTS.json'),'r+',os.O_RDWR)
    except PermissionError:update_mode_allowed=False
    print(json.dumps(dict(exact_locked_function=node.name,ordinary_forbidden_reads_blocked=blocked,request_read_permitted=True,candidate_read_permitted=(mode=='prediction'),synthetic_local_candidate_rplus_event_allowed=update_mode_allowed,scope='ordinary Python opens; not an OS sandbox')))

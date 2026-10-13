"""One-shot forward material intervention; separate from public inference."""
from fractions import Fraction as Q
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1')
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
import baseline_geometry as geometry
ETA_VALUES=(Q(0),Q(1,10))
PATTERN=((-1,0,1),(0,1,-1),(1,-1,0))
exact=lambda value: Q(value['numerator'],value['denominator']) if isinstance(value,dict) else Q(value)
def encode(value):
    value=exact(value)
    return {'numerator':value.numerator,'denominator':value.denominator}
def load(path):
    return json.loads(Path(path).read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT/'data'):
        raise ValueError('outside registered data')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
def view_darkness(colors,eta):
    colors=tuple(map(exact,colors));eta=exact(eta)
    if len(colors)!=3 or not 0<=eta<=1 or any(not 0<=c<=1 for c in colors):
        raise ValueError('physical three-material colors and eta required')
    dark=tuple(1-c for c in colors)
    return tuple(d+eta*Q(sign)*min(d,1-d) for row in PATTERN for d,sign in zip(dark,row))
def render_views(fact,eta):
    dark=view_darkness(fact['colors'],eta)
    offsets=tuple(map(exact,fact['actual_offsets']))
    depth=exact(fact['truth']);baseline=exact(fact['baseline'])
    noise=tuple(map(exact,fact['image_noise']))
    bias=tuple(map(exact,fact['actual_view_bias']))
    raw=[]
    for view in range(3):
        image=geometry.point_rgb(fact['rear_branch'],depth,offsets,
                                 tuple(1-d for d in dark[3*view:3*view+3]),baseline)
        raw.extend(image[144*view:144*(view+1)])
    observed=[encode(min(Q(1),max(Q(0),value+error+bias[i//144])))
              for i,(value,error) in enumerate(zip(raw,noise))]
    return observed,dark
def build(source_public,source_tasks,source_facts):
    observations,tasks,facts=[],[],[]
    tm={row['task_id']:row for row in source_tasks}
    fm={row['observation_id']:row for row in source_facts}
    for variation,eta in enumerate(ETA_VALUES):
        for public in source_public:
            if public['unit_id']!='baseline_80':
                continue
            base=public['observation_id'];fact=fm[base]
            image,dark=render_views(fact,eta)
            if eta==0 and image!=public['observation']:
                raise RuntimeError('zero-variation forward identity failed')
            oid=base+f'e{variation}'
            observations.append({**public,'observation_id':oid,'base_observation_id':base,
                                 'unit_id':'appearance_eta_'+('0' if variation==0 else '1_10'),
                                 'variation_index':variation,'eta':encode(eta),'observation':image})
            facts.append({**fact,'observation_id':oid,'base_observation_id':base,
                          'unit_id':observations[-1]['unit_id'],'variation_index':variation,
                          'eta':encode(eta),'view_colors':[encode(1-d) for d in dark],
                          'actual_material_spread':encode(max(max(dark[k::3])-min(dark[k::3]) for k in range(3))),
                          'image_unchanged':image==public['observation']})
            for state in range(9):
                old=tm[base+f's{state}']
                tasks.append({**old,'task_id':oid+f's{state}','observation_id':oid,
                              'base_task_id':old['task_id']})
    return observations,tasks,facts
def prepare(deadline):
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED or (ROOT/'data').exists():
        raise RuntimeError('wrong target or existing one-shot data')
    gate=load(ROOT/'.aris/REVIEW_GATE.json')
    if gate.get('status')!='PASS':
        raise RuntimeError('source review must pass before generation')
    if any(sha(path)!=digest for path,digest in gate['files'].items()):
        raise RuntimeError('reviewed source changed')
    if time.time()>=deadline:
        raise TimeoutError('shared budget before preparation')
    observations,tasks,facts=build(load(PARENT/'data/observations.json'),
        load(PARENT/'data/tasks.json'),load(PARENT/'data/private/truth.json'))
    if (len(observations),len(tasks),len(facts))!=(36,324,36):
        raise RuntimeError('wrong fixed generation population')
    dump(ROOT/'data/observations.json',observations)
    dump(ROOT/'data/tasks.json',tasks)
    dump(ROOT/'data/private/truth.json',facts)
    dump(ROOT/'data/GENERATION.json',{'stage':'A4-V','source_seed':61005,'eta_values':['0','1/10'],
        'pattern':[list(row) for row in PATTERN],'observations':36,'tasks':324,
        'forward':'per-view exact integral + frozen noise + frozen bias, then one clip',
        'no_new_random_draws':True,'public_bound':'1/10','bound_source':'analytic intervention design, not learned',
        'deadline_epoch':deadline})
    if time.time()>=deadline:
        raise TimeoutError('shared budget after preparation')
    print(json.dumps({'prepared':str(ROOT/'data'),'observations':36,'tasks':324}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--deadline',type=float,required=True)
    prepare(parser.parse_args().deadline)


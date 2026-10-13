"""One-shot independent noisy acquisitions; NOT an inference module."""
import argparse, hashlib, json, random, socket, sys, time
from fractions import Fraction as Q
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/independent-anchors-1')
P = ROOT.parent
OLD = P/'appearance-variation-1/data/private/truth.json'
PUBLIC = P/'point-feasibility-1/outputs/point-v1/observations.json'
def exact(x): return Q(x['numerator'],x['denominator']) if isinstance(x,dict) else Q(x)
def enc(x):
    x=exact(x);return {'numerator':x.numerator,'denominator':x.denominator}
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p)
    if not p.resolve().is_relative_to(ROOT/'data'): raise ValueError('wrong output target')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def clip(x): return min(Q(1),max(Q(0),x))
def tape(rng):
    draw=lambda bound: Q(rng.randint(-bound,bound),100000)
    return {'camera_k':enc(draw(1000)), 'camera_noise':[enc(draw(1000)) for _ in range(3)],
            'appearance_t':[enc(Q(1,100)-draw(1000)) for _ in range(3)],
            'coupon_transfer':[enc(draw(500)) for _ in range(9)],
            'coupon_noise':[enc(draw(500)) for _ in range(9)],
            'reference_noise':[enc(draw(500)) for _ in range(3)]}
def acquire(fact,t):
    es=list(map(exact,fact['actual_offsets']));k=exact(t['camera_k'])
    nu=list(map(exact,t['camera_noise']));ts=list(map(exact,t['appearance_t']))
    colors=list(map(exact,fact['view_colors']))
    qs=list(map(exact,t['coupon_transfer']));ns=list(map(exact,t['coupon_noise']))
    rn=list(map(exact,t['reference_noise']))
    return {'observation_id':fact['observation_id'],
        'camera':{'u':[enc((-c-es[v]-k-nu[v])/1000) for v,c in enumerate((-80,0,80))],
                  'Xf':enc(0),'Zf':enc(1000),'survey_bound':enc(Q(1,100)),
                  'localization_bound':enc(Q(1,100))},
        'appearance':{'coupons':[enc(clip(colors[i]+Q(1,100)-ts[i//3]+qs[i]+ns[i])) for i in range(9)],
                      'references':[enc(clip(Q(1,2)+Q(1,100)-ts[v]+rn[v])) for v in range(3)],
                      'bias_bound':enc(Q(1,100)),'transfer_bound':enc(Q(1,200)),
                      'noise_bound':enc(Q(1,200))}}
def main(deadline):
    if socket.gethostname()!='liekkas' or ROOT!=EXPECTED: raise RuntimeError('wrong host/root')
    if (ROOT/'data').exists(): raise RuntimeError('one-shot acquisition already exists')
    gate=load(ROOT/'.aris/REVIEW_GATE.json')
    if gate['status']!='PASS' or any(sha(p)!=h for p,h in gate['files'].items()):
        raise RuntimeError('review gate/source drift')
    if time.time()>=deadline: raise TimeoutError('shared acquisition deadline')
    publics=load(PUBLIC);facts={f['observation_id']:f for f in load(OLD)}
    rng=random.Random(load(ROOT/'REGISTRATION.json')['seed']);tapes={};readings=[];private=[]
    for p in publics:
        oid=p['observation_id'];base=p['base_observation_id']
        if base not in tapes:tapes[base]=tape(rng)
        t=tapes[base];readings.append(acquire(facts[oid],t))
        private.append({'observation_id':oid,'base_observation_id':base,**t})
    if len(readings)!=36 or len(tapes)!=18: raise RuntimeError('frozen population')
    save(ROOT/'data/anchors.json',readings)
    save(ROOT/'data/private/measurements.json',private)
    save(ROOT/'data/GENERATION.json',{'stage':'A8','seed':61006,'observations':36,'base_tapes':18,
        'camera_scalars_per_observation':3,'appearance_scalars_per_observation':12,
        'old_private_sha256':sha(OLD),'old_public_sha256':sha(PUBLIC),
        'forward':'independent shared survey, independent appearance acquisition bias, bounded errors, clip once',
        'paired_across_eta':True,'deadline_epoch':deadline})
    if time.time()>=deadline:raise TimeoutError('shared deadline after generation')
    print(json.dumps({'stage':'A8-acquire','observations':36,'base_tapes':18}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',type=float,required=True)
    main(ap.parse_args().deadline)

"""Cold-start adapters to frozen A9/A10/A11. No saved fits or hidden worlds."""
from fractions import Fraction as Q
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
P = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z').resolve()
LATEST = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z').resolve()
sys.path.insert(0,str(LATEST))
import accumulation_solver as a11
a10 = a11.kernel
a9 = a10.baseline
NAMES = ('coarse', 'depth', 'accumulated')

def rational(x):
    if isinstance(x,list):
        return [rational(v) for v in x]
    if isinstance(x,dict):
        return {k:rational(v) for k,v in x.items()}
    if isinstance(x,(int,float)):
        return Q(str(x))
    return x

def legacy_inputs(obs):
    photo={'observation_id':'measurement','observation':rational(obs['photo']),
           'radius':Q(1,5),'baseline':Q(80),'clipped':True}
    anchors={'observation_id':'measurement','camera':rational(obs['camera']),
             'appearance':rational(obs['appearance'])}
    # A5 public bias range t=beta-b. Background-derived narrowing is recovered
    # from the same photos, with the exact frozen conservative geometry.
    geom=a9.geometry
    all_bounds=[geom.geometry_bounds(b,Q(1,5),Q(540),Q(660),Q(80)) for b in a9.BRANCHES]
    mask=[all(g[i][k]==0 for g in all_bounds for k in (1,3,5)) for i in range(144)]
    # Calculate photo-bias constraints directly from known white background.
    limits=[[Q(0),Q(1,50)] for _ in range(3)]
    for p,white in enumerate(mask):
        if not white:
            continue
        v=p//48
        for y in photo['observation'][3*p:3*p+3]:
            if y>0:
                limits[v][1]=min(limits[v][1],1+Q(1,100)+Q(1,100)-y)
            if y<1:
                limits[v][0]=max(limits[v][0],1+Q(1,100)-Q(1,100)-y)
    oldarm={'calibration_intervals':limits,'appearance_model':'bounded','drift_bound':Q(1,10)}
    return photo,oldarm,anchors

def prepare_method(name,observation,budget_seconds):
    if name not in NAMES:
        raise ValueError(name)
    started=time.monotonic()
    end=started+float(budget_seconds)
    a11.clear_geometry_cache()
    photo,old,anchors=legacy_inputs(observation)
    stages={'coarse':1,'depth':2,'accumulated':3}[name]
    history=[]
    branches=[]
    for bi,branch in enumerate(a9.BRANCHES):
        # Per-branch block allocations: no uncharged inherited support.
        previous=None
        for stage in range(stages):
            fraction=(bi*stages+stage+1)/(3*stages)
            stop=min(end,started+float(budget_seconds)*fraction)
            if stage==0:
                record=a9.solve_interval_branch(photo,old,anchors,'both',branch,deadline=stop)
                record['branch']=branch
            elif stage==1:
                record=a10.solve_branch(photo,old,anchors,previous,'depth',deadline=stop)
            else:
                record=a11.solve_branch(photo,old,anchors,previous,'D_then_DC',deadline=stop)
            history.append({'branch':branch,'stage':stage,'record':record})
            previous=record
        branches.extend(previous['outer_intervals'])
    support=a9.merge_intervals(branches)
    return {'method':name,'hull':None if not support else [float(support[0][0]),float(support[-1][1])],
            'outer_intervals':[[float(a),float(b)] for a,b in support],
            'seconds':time.monotonic()-started,'trace':history,
            'stage_deadline_overruns':[{'branch':x['branch'],'stage':x['stage'],
                'seconds':x['record']['seconds']-x['record'].get('search_budget_seconds',0.)}
                for x in history if x['record']['seconds']-x['record'].get('search_budget_seconds',0.)>0.25],
            'calibration_intervals':[[str(a),str(b)] for a,b in old['calibration_intervals']]}

def decode(state,incumbent):
    h=state.get('hull')
    return float(incumbent) if not h else min(h[1],max(h[0],float(incumbent)))

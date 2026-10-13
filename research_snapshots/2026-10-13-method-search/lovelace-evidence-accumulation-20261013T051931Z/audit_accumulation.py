"""Independent public proof admission, then exact simulation-only evaluation."""
import csv
from fractions import Fraction as Q
import time
from a11_io_contract import *
import verify_accumulation as check
ARMS=('A10_D','D_then_D','D_then_DC')
def pairs(xs): return tuple(tuple(map(check.exact,p)) for p in xs)
def contains(xs,x): return any(lo<=x<=hi for lo,hi in pairs(xs))
def nested(xs,ys): return all(any(a<=lo<=hi<=b for a,b in pairs(ys)) for lo,hi in pairs(xs))
def index(records,fields):
    out={}
    for r in records:
        key=tuple(r[f] for f in fields);require(key not in out,'duplicate identity');out[key]=r
    return out
def groups(fact,photo,initial):
    out=['all','shifted' if initial else 'correct']
    if initial:
        out.append('negative' if fact['case']=='equal_full_negative' else 'informative')
        if fact['case']!='equal_full_negative' and check.exact(fact['truth'])==Q(9007,14) and photo['variation_index']==1:
            out.append('primary')
    return out
def summarize(rows):
    cells={}
    for r in rows:
        for unit in ('ALL',r['unit_id']):
            for group in r['groups']:cells.setdefault((r['arm'],unit,group),[]).append(r)
    out=[]
    for (arm,unit,group),cell in sorted(cells.items()):
        a=[Q(r['initial_ae_mm']) for r in cell];b=[Q(r['final_ae_mm']) for r in cell];n=len(a)
        mse0=sum(x*x for x in a)/n;mse=sum(x*x for x in b)/n
        out.append(dict(arm=arm,policy='HULL',unit=unit,group=group,n=n,
            initial_MAE_mm=str(sum(a)/n),final_MAE_mm=str(sum(b)/n),
            initial_MSE_mm2=str(mse0),final_MSE_mm2=str(mse),
            MSE_reduction_percent=None if not mse0 else float(100*(1-mse/mse0)),
            improved=sum(y<x for x,y in zip(a,b)),equal=sum(y==x for x,y in zip(a,b)),
            worsened=sum(y>x for x,y in zip(a,b))))
    return out
def comparisons(rows):
    ix=index(rows,('task_id','arm'));out=[]
    for left,right in (('A10_D','D_then_D'),('A10_D','D_then_DC'),('D_then_D','D_then_DC')):
        for group in ('all','shifted','correct','informative','negative','primary'):
            cell=[r for r in rows if r['arm']==left and group in r['groups']]
            aa=[Q(r['final_ae_mm']) for r in cell];bb=[Q(ix[r['task_id'],right]['final_ae_mm']) for r in cell]
            out.append(dict(baseline=left,candidate=right,group=group,n=len(cell),
                better=sum(b<a for a,b in zip(aa,bb)),same=sum(b==a for a,b in zip(aa,bb)),
                worse=sum(b>a for a,b in zip(aa,bb)),
                MSE_gain_mm2=str(sum(a*a-b*b for a,b in zip(aa,bb))/len(cell)),
                better_ids=[r['task_id'] for r,a,b in zip(cell,aa,bb) if b<a],
                worse_ids=[r['task_id'] for r,a,b in zip(cell,aa,bb) if b>a]))
    return out
def main():
    target();check.check_target();validate_old()
    frozen=load(OUT/'SOURCE_LOCK.json')['files'];locked=load(OUT/'INPUT_PRELOCK.json')['files']
    require(sources()==frozen and inputs()==locked,'preaudit frozen bytes')
    require(load(OUT/'RUN_STATUS.json')['status']=='COMPLETE','public method incomplete')
    receipt=load(OUT/'PREDICTION_RECEIPT.json')
    require(receipt['private_parsed'] is False and receipt['sets_sha256']==sha(OUT/'sets.json') and
        receipt['predictions_sha256']==sha(OUT/'predictions.json'),'public prediction receipt')
    photos,anchors,oldarms,source=public_inputs();photo_by={p['observation_id']:p for p in photos}
    a9={s['observation_id']:s for s in load(OLD9/'sets.json') if s['arm']=='both'}
    tasks=load(OLD9/'tasks.json');task_ix=index(tasks,('task_id',))
    sets=index(load(OUT/'sets.json'),('observation_id','arm'))
    actions=index(load(OUT/'predictions.json'),('task_id','arm'))
    require(len(tasks)==324 and set(sets)=={(oid,arm) for oid in source for arm in check.ARMS} and
        set(actions)=={(t['task_id'],arm) for t in tasks for arm in check.ARMS},'exact public matrix')
    stats=[]
    for photo in photos:
        oid=photo['observation_id'];joint=oldarms[oid]['box']['joint'];anchor=anchors[oid]
        check.clear_geometry_cache()
        require([b['branch'] for b in a9[oid]['branches']]==list(check.BRANCHES),'A9 branch order')
        require([b['branch'] for b in source[oid]['branches']]==list(check.BRANCHES),'A10 branch order')
        old9=[check._a9.verify_branch(photo,joint,anchor,'both',b['branch'],b) for b in a9[oid]['branches']]
        check.aggregate_branches(old9,a9[oid])
        admitted=[check.verify_source_branch(photo,joint,anchor,d['branch'],b,d)
            for b,d in zip(a9[oid]['branches'],source[oid]['branches'])]
        check.aggregate_branches(admitted,source[oid])
        for arm in check.ARMS:
            record=sets[oid,arm]
            require([b['branch'] for b in record['branches']]==list(check.BRANCHES),'A11 branch order')
            checked=[check.verify_branch(photo,joint,anchor,arm,b['branch'],d,b)
                for d,b in zip(source[oid]['branches'],record['branches'])]
            check.aggregate_branches(checked,record)
            require(nested(record['outer_intervals'],source[oid]['outer_intervals']),'cumulative nesting')
            xs=pairs(record['outer_intervals']);branches=record['branches']
            stats.append(dict(observation_id=oid,arm=arm,components=len(xs),
                union_length_mm=str(sum((hi-lo for lo,hi in xs),Q(0))),
                hull_width_mm=None if not xs else str(xs[-1][1]-xs[0][0]),
                classification_calls=sum(b['classification_calls'] for b in branches),
                seconds=sum(b['seconds'] for b in branches),
                terminations=[b['termination'] for b in branches],
                retained_status_counts=[c['retained_status_counts'] for c in checked],
                classification_status_counts=[c['classification_status_counts'] for c in checked],
                source_D_nested=True))
        print(json.dumps(dict(stage='public-audit',observation=oid,sets=len(stats))),flush=True)
    for (tid,arm),r in actions.items():
        task=task_ix[tid,];require(r['observation_id']==task['observation_id'],'action identity')
        check.verify_selection(task['incumbent'],sets[task['observation_id'],arm]['outer_intervals'],'HULL',r)
    save(OUT/'verification/PUBLIC_GATE.json',dict(status='PASS',sets=72,trees=216,actions=648,
        historical_trees_reverified=216,completed_epoch=time.time(),private_parsed=False,
        prediction_receipt_sha256=sha(OUT/'PREDICTION_RECEIPT.json')))
    # Private truth is parsed only after every new action and public proof has passed.
    facts={f['observation_id']:f for f in load(P/'appearance-variation-1/data/private/truth.json')}
    measures={m['observation_id']:m for m in load(A8/'data/private/measurements.json')}
    require(set(facts)==set(measures)==set(source),'private identities')
    coverage=[]
    for photo in photos:
        oid=photo['observation_id'];fact=facts[oid];measurement=measures[oid]
        truth,branch=check.exact(fact['truth']),fact['rear_branch'];joint=oldarms[oid]['box']['joint']
        witness=check._a8.actual_witness(fact,measurement,'both')
        check._a8.verify_forward(photo,joint,anchors[oid],truth,branch,'both',witness,'A11 actual sealed world')
        for arm in ARMS:
            record=source[oid] if arm=='A10_D' else sets[oid,arm]
            generating=next(b for b in record['branches'] if b['branch']==branch)
            require(contains(generating['outer_intervals'],truth) and contains(record['outer_intervals'],truth),'true world excluded')
            coverage.append(dict(observation_id=oid,arm=arm,truth_mm=str(truth),generating_branch=branch,
                branch_covered=True,union_covered=True,forward_verified=True))
    rows=[]
    for task in tasks:
        tid,oid=task['task_id'],task['observation_id'];photo=photo_by[oid];fact=facts[oid]
        truth,old=check.exact(fact['truth']),check.exact(task['incumbent']);initial=abs(old-truth)
        hull=source[oid]['hull'];require(hull is not None,'covered D hull')
        baseline=min(max(old,check.exact(hull[0])),check.exact(hull[1]));base_error=abs(baseline-truth)
        for arm in ARMS:
            selected=baseline if arm=='A10_D' else check.exact(actions[tid,arm]['selected'])
            final=abs(selected-truth)
            require(final<=initial and final<=base_error,'conditional HULL nonregression violated')
            rows.append(dict(task_id=tid,observation_id=oid,arm=arm,policy='HULL',unit_id=photo['unit_id'],
                variation_index=photo['variation_index'],case=fact['case'],truth_mm=str(truth),
                incumbent_mm=str(old),selected_mm=str(selected),initial_ae_mm=str(initial),final_ae_mm=str(final),
                ae_gain_mm=str(initial-final),displacement_mm=str(abs(selected-old)),moved=selected!=old,
                groups=groups(fact,photo,initial)))
    summaries=summarize(rows)
    for arm in ARMS:
        cell={r['group']:r['n'] for r in summaries if r['arm']==arm and r['unit']=='ALL'}
        require(cell==dict(all=324,shifted=288,correct=36,primary=40,negative=48,informative=240),'group denominators')
    old_unfixed=[r['task_id'] for r in rows if r['arm']=='A10_D' and 'informative' in r['groups'] and Q(r['ae_gain_mm'])==0]
    require(len(old_unfixed)==87,'frozen old informative residual denominator')
    residuals={arm:{'old_unfixed':87,'newly_improved':sum(r['arm']==arm and r['task_id'] in old_unfixed and Q(r['ae_gain_mm'])>0 for r in rows)} for arm in check.ARMS}
    save(OUT/'evaluation/ROWS.json',rows);save(OUT/'evaluation/SUMMARY.json',summaries)
    save(OUT/'evaluation/COMPARISONS.json',comparisons(rows));save(OUT/'evaluation/COVERAGE.json',coverage)
    save(OUT/'evaluation/OUTER_STATS.json',stats);save(OUT/'evaluation/OLD_RESIDUALS.json',residuals)
    with (OUT/'evaluation/ROWS.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    require(sources()==frozen and inputs()==locked,'postevaluation locks')
    save(OUT/'verification/VERIFY.json',dict(status='PASS',sets=72,trees=216,actions=648,
        historical_trees_reverified=216,true_world_checks=108,rows=972,epoch=time.time(),
        public_gate_sha256=sha(OUT/'verification/PUBLIC_GATE.json'),
        assurance='Independent reconstruction, same-family; finite synthetic model only'))
    print(json.dumps(dict(stage='audit',status='PASS',actions=648,rows=972)),flush=True)
if __name__=='__main__':main()

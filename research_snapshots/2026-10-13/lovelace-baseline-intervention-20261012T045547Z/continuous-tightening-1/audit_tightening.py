"""Independent inherited/new proof gate precedes private evaluation."""
import csv
from fractions import Fraction as Q
import time
from io_contract import *
import verify_tightening as check
# Only sealed grouping/arithmetic utilities; this does not import inference.
sys.path.insert(0,str(A9))
from audit_continuous import group_names,summarize,contains,covered_by,index
def main():
    target(); check.check_target(); validate_old()
    frozen=load(OUT/'SOURCE_LOCK.json')['files']; locked=load(OUT/'INPUT_PRELOCK.json')['files']
    require(sources()==frozen and inputs()==locked,'frozen bytes')
    require(load(OUT/'RUN_STATUS.json')['status']=='COMPLETE','public run incomplete')
    receipt=load(OUT/'PREDICTION_RECEIPT.json')
    require(receipt['private_parsed'] is False and receipt['probes_started'] is False and
        receipt['sets_sha256']==sha(OUT/'sets.json') and receipt['predictions_sha256']==sha(OUT/'predictions.json'),'prediction receipt')
    observations,anchors,oldarms,baseline=public_inputs(); photos={p['observation_id']:p for p in observations}
    tasks=index(load(OLD/'tasks.json'),('task_id',))
    sets=index(load(OUT/'sets.json'),('observation_id','arm'))
    predictions=index(load(OUT/'predictions.json'),('task_id','arm','policy'))
    require(set(sets)=={(o,a) for o in photos for a in check.ARMS},'complete support matrix')
    require(set(predictions)=={(t[0],a,'HULL') for t in tasks for a in check.ARMS},'complete action matrix')
    stats=[]
    for oi,(oid,photo) in enumerate(photos.items()):
        check.clear_geometry_cache()
        inherited=[check._a9.verify_branch(photo,oldarms[oid]['box']['joint'],anchors[oid],'both',b['branch'],b)
                   for b in baseline[oid]['branches']]
        check._a9.aggregate_branches(inherited,baseline[oid])
        for arm in check.ARMS:
            record=sets[oid,arm]
            require([b['branch'] for b in record['branches']]==list(check.BRANCHES),'ordered three branches')
            checked=[check.verify_branch(photo,oldarms[oid]['box']['joint'],anchors[oid],arm,b['branch'],old,b)
                     for old,b in zip(baseline[oid]['branches'],record['branches'])]
            aggregation=check.aggregate_branches(checked,record)
            require(covered_by(record['outer_intervals'],baseline[oid]['outer_intervals']),'nested support')
            hull=record['hull']
            stats.append({'observation_id':oid,'arm':arm,'branches':checked,
                 'hull_width_mm':None if hull is None else str(check.exact(hull[1])-check.exact(hull[0])),
                 'baseline_hull_width_mm':None if baseline[oid]['hull'] is None else str(check.exact(baseline[oid]['hull'][1])-check.exact(baseline[oid]['hull'][0]))})
        print(json.dumps({'stage':'public verification','observations':oi+1}),flush=True)
    for (tid,arm,policy),record in predictions.items():
        task=tasks[tid,]
        require(record['observation_id']==task['observation_id'] and record['incumbent']==task['incumbent'],'action identity')
        check.verify_selection(task['incumbent'],sets[task['observation_id'],arm]['outer_intervals'],policy,record)
    probes=load(OUT/'probes.json')
    expected=[{'observation_id':r['observation_id'],**pos} for r in load(OUT/'sets.json') if r['arm']=='camera_tiles'
              for pos in probe_positions(r,check.exact)]
    require(len(probes)==len(expected)<=72,'probe matrix')
    for probe,pos in zip(probes,expected):
        require({k:v for k,v in probe.items() if k!='result'}==pos,'public midpoint probe selection')
        result=probe['result']; oid=probe['observation_id']
        require(result['status'] in ('FORWARD_WITNESS','UNKNOWN') and 0<=result['search_budget_seconds']<=.5,'probe contract')
        if result['status']=='FORWARD_WITNESS':
            check._a8.verify_forward(photos[oid],oldarms[oid]['box']['joint'],anchors[oid],check.exact(probe['depth']),
                  probe['branch'],'both',result['witness'],'A10 public endpoint probe')
        else: require(result['witness'] is None,'UNKNOWN cannot assert full world')
    save(OUT/'verification/PUBLIC_GATE.json',{'status':'PASS','old_supports':36,'new_supports':72,'actions':648,
         'probes':len(probes),'forward_probe_successes':sum(p['result']['status']=='FORWARD_WITNESS' for p in probes),
         'private_parsed':False,'completed_epoch':time.time()})
    # PRIVATE FIRST PARSE after all public proof and coordinate checks.
    facts={f['observation_id']:f for f in load(P/'appearance-variation-1/data/private/truth.json')}
    measurements={m['observation_id']:m for m in load(A8/'data/private/measurements.json')}
    coverage=[]
    for oid,photo in photos.items():
        fact,measure=facts[oid],measurements[oid]; truth=check.exact(fact['truth']); branch=fact['rear_branch']
        witness=check._a8.actual_witness(fact,measure,'both')
        check._a8.verify_forward(photo,oldarms[oid]['box']['joint'],anchors[oid],truth,branch,'both',witness,'A10 actual world')
        for arm in check.ARMS:
            record=sets[oid,arm]; generating=next(b for b in record['branches'] if b['branch']==branch)
            require(contains(generating['outer_intervals'],truth) and contains(record['outer_intervals'],truth),'actual world excluded')
            coverage.append({'observation_id':oid,'arm':arm,'truth_mm':str(truth),'generating_branch':branch,'covered':True})
    rows=[]
    # Evaluate frozen A9 baseline with the same independent HULL arithmetic.
    base_predictions=[]
    for (tid,),task in tasks.items():
        hull=baseline[task['observation_id']]['hull']; old=check.exact(task['incumbent'])
        new=old if hull is None else min(max(old,check.exact(hull[0])),check.exact(hull[1]))
        base_predictions.append({'task_id':tid,'observation_id':task['observation_id'],'arm':'a9_both','policy':'HULL','selected':str(new)})
    for record in base_predictions+list(predictions.values()):
        tid,arm=record['task_id'],record['arm']; task=tasks[tid,]; oid=task['observation_id']
        photo,fact=photos[oid],facts[oid]
        truth,old,new=map(check.exact,(fact['truth'],task['incumbent'],record['selected']))
        initial,final=abs(old-truth),abs(new-truth)
        require(final<=initial and (initial!=0 or new==old),'HULL nonregression/correct input')
        rows.append({'task_id':tid,'observation_id':oid,'arm':arm,'policy':'HULL',
             'unit_id':photo['unit_id'],'variation_index':photo['variation_index'],'case':fact['case'],
             'truth_mm':str(truth),'incumbent_mm':str(old),'selected_mm':str(new),'initial_ae_mm':str(initial),
             'final_ae_mm':str(final),'ae_gain_mm':str(initial-final),'displacement_mm':str(abs(new-old)),
             'moved':new!=old,'groups':group_names(fact,photo,initial)})
    base={r['task_id']:r for r in rows if r['arm']=='a9_both'}
    for r in rows:
        if r['arm']!='a9_both': require(Q(r['final_ae_mm'])<=Q(base[r['task_id']]['final_ae_mm']),'nested HULL must not regress A9')
    comparisons=[]
    for arm in check.ARMS:
        for group in ('all','shifted','informative','negative','correct','primary','old_informative_unchanged'):
            selected=[r for r in rows if r['arm']==arm and (group in r['groups'] if group!='old_informative_unchanged' else
                ('informative' in r['groups'] and Q(base[r['task_id']]['ae_gain_mm'])==0))]
            before=[Q(base[r['task_id']]['final_ae_mm']) for r in selected]; after=[Q(r['final_ae_mm']) for r in selected]
            comparisons.append({'arm':arm,'group':group,'n':len(selected),'improved_over_a9':sum(b<a for a,b in zip(before,after)),
                 'equal_to_a9':sum(b==a for a,b in zip(before,after)),'worsened_over_a9':sum(b>a for a,b in zip(before,after)),
                 'MAE_gain_over_a9_mm':str(sum((a-b for a,b in zip(before,after)),Q(0))/len(selected))})
    probe_eval=[{'observation_id':p['observation_id'],'side':p['side'],'status':p['result']['status'],
        'depth_mm':p['depth'],'truth_mm':facts[p['observation_id']]['truth'],
        'distance_from_true_mm':str(abs(Q(p['depth'])-check.exact(facts[p['observation_id']]['truth'])))} for p in probes]
    save(OUT/'evaluation/ROWS.json',rows);save(OUT/'evaluation/SUMMARY.json',summarize(rows))
    save(OUT/'evaluation/COMPARISONS.json',comparisons);save(OUT/'evaluation/COVERAGE.json',coverage)
    # Verification results contain Fraction tuples; use the checker's encoded fields instead.
    for stat in stats:
        for b in stat['branches']: b.pop('intervals',None)
    save(OUT/'evaluation/OUTER_STATS.json',stats);save(OUT/'evaluation/PROBES.json',probe_eval)
    with (OUT/'evaluation/ROWS.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    require(sources()==frozen and inputs()==locked,'postaudit bytes')
    save(OUT/'verification/VERIFY.json',{'status':'PASS','new_supports':72,'old_supports':36,'actions':648,
         'actual_world_coverage':72,'probes_verified':len(probes),'epoch':time.time()})
    print(json.dumps({'stage':'A10 audit','status':'PASS'}),flush=True)
if __name__=='__main__':main()

"""Post-run exact arithmetic and residual classification, not used for decisions."""
from collections import Counter,defaultdict
from datetime import datetime
from fractions import Fraction as Q
import json
from io_contract import ROOT,OUT,load,require,sha,inputs,sources,validate_old
def exact(x):
    return Q(x['numerator'],x['denominator']) if isinstance(x,dict) else Q(x)
def counts(before,after):
    return {'improved':sum(b<a for a,b in zip(before,after)),'unchanged':sum(a==b for a,b in zip(before,after)),
            'worsened':sum(b>a for a,b in zip(before,after))}
def metrics(rows):
    before=[exact(r['initial_ae_mm']) for r in rows]; after=[exact(r['final_ae_mm']) for r in rows]
    n=len(rows); m=sum(after,Q(0))/n; mse=sum((v*v for v in after),Q(0))/n
    imse=sum((v*v for v in before),Q(0))/n
    return {'n':n,'MAE_mm_exact':str(m),'MAE_mm':float(m),'MSE_mm2_exact':str(mse),'MSE_mm2':float(mse),
        'MSE_reduction_percent':float(100*(1-mse/imse)) if imse else None,**counts(before,after)}
def main():
    require(load(OUT/'STAGE_STATUS.json')['status']=='COMPLETE','only analyze completed verified batch')
    validate_old(); require(sources()==load(OUT/'SOURCE_LOCK.json')['files'] and inputs()==load(OUT/'INPUT_PRELOCK.json')['files'],'freeze')
    rows=load(OUT/'evaluation/ROWS.json'); sets=load(OUT/'sets.json'); probes=load(OUT/'probes.json')
    lookup={(r['task_id'],r['arm']):r for r in rows}
    arms=['a9_both','depth','camera_tiles']; groups=['all','correct','shifted','informative','negative','primary']
    summary={a:{g:metrics([r for r in rows if r['arm']==a and g in r['groups']]) for g in groups} for a in arms}
    ddc=[]
    for group in groups:
        rr=[r for r in rows if r['arm']=='depth' and group in r['groups']]
        ddc.append({'group':group,'n':len(rr),**counts([Q(r['final_ae_mm']) for r in rr],
            [Q(lookup[r['task_id'],'camera_tiles']['final_ae_mm']) for r in rr])})
    family=[]
    for a in arms[1:]:
        for comparison in ('a9_both','depth'):
            byobs=defaultdict(list);bypair=defaultdict(list)
            for r in (r for r in rows if r['arm']==a and 'shifted' in r['groups']):
                gain=Q(lookup[r['task_id'],comparison]['final_ae_mm'])-Q(r['final_ae_mm'])
                byobs[r['observation_id']].append(gain)
                bypair[r['observation_id'].rsplit('e',1)[0]].append(gain)
            require(len(byobs)==36 and len(bypair)==18,'paired denominator')
            for unit,values in (('observations',byobs),('paired_realizations',bypair)):
                sums=[sum(v,Q(0)) for v in values.values()]
                family.append({'arm':a,'compared_with':comparison,'unit':unit,'n':len(sums),
                    'positive':sum(s>0 for s in sums),'zero':sum(s==0 for s in sums),'negative':sum(s<0 for s in sums)})
    search={}; residual=[]
    for arm in arms[1:]:
        ar=[s for s in sets if s['arm']==arm]; branches=[b for s in ar for b in s['branches']]
        classified=[c['classification'] for b in branches for n in b['trace'] for c in n['checks'] if c['classification']]
        search[arm]={'calls':sum(b['classification_calls'] for b in branches),'seconds':sum(b['seconds'] for b in branches),
            'termination':dict(Counter(b['termination'] for b in branches)),
            'classifications':dict(Counter(c['status'] for c in classified)),
            'overrun_branches':sum(b['seconds']>6 for b in branches),'max_branch_seconds':max(b['seconds'] for b in branches)}
        si={s['observation_id']:s for s in ar}
        rr=[r for r in rows if r['arm']==arm and 'informative' in r['groups'] and Q(lookup[r['task_id'],'a9_both']['ae_gain_mm'])==0]
        remain=[r for r in rr if Q(r['ae_gain_mm'])==0]
        categories=Counter()
        for r in remain:
            point=Q(r['incumbent_mm'])
            hits=[n for b in si[r['observation_id']]['branches'] for n in b['trace'] if n['status'].startswith('retained_') and exact(n['lo'])<=point<=exact(n['hi'])]
            categories['+'.join(sorted(set(n['status'] for n in hits))) if hits else 'hull_gap']+=1
        residual.append({'arm':arm,'old_unchanged_informative':len(rr),'newly_improved':len(rr)-len(remain),
             'still_unchanged':len(remain),'retained_reasons_at_incumbent':dict(categories)})
    pe=load(OUT/'evaluation/PROBES.json'); successes=[p for p in pe if p['status']=='FORWARD_WITNESS']
    successpairs=[]
    for oid in sorted({p['observation_id'] for p in successes}):
        pp=[p for p in successes if p['observation_id']==oid]
        if len(pp)==2:
            gap=abs(Q(pp[0]['depth_mm'])-Q(pp[1]['depth_mm']))
            successpairs.append({'observation_id':oid,'separation_mm':str(gap),'minimax_absolute_lower_bound_mm':str(gap/2),
                                'minimax_squared_lower_bound_mm2':str(gap*gap/4)})
    cases={r['observation_id']:r['case'] for r in rows}
    probe_groups={}
    for label in ('informative','negative'):
        subset=[p for p in pe if (cases[p['observation_id']]=='equal_full_negative')==(label=='negative')]
        good=[p for p in subset if p['status']=='FORWARD_WITNESS']
        both=[p for p in successpairs if (cases[p['observation_id']]=='equal_full_negative')==(label=='negative')]
        probe_groups[label]={'attempts':len(subset),'success':len(good),'observations_with_success':len({p['observation_id'] for p in good}),
             'observations_with_two_successes':len(both),'largest_certified_separation_mm':str(max([Q(0)]+[Q(p['separation_mm']) for p in both]))}
    public_probe_stats={'attempts':len(probes),'success':len(successes),'unknown':len(probes)-len(successes),'groups':probe_groups,
        'success_observations':len({p['observation_id'] for p in successes}),'both_endpoints_success':successpairs,
        'max_success_true_distance_mm':str(max([Q(0)]+[Q(p['distance_from_true_mm']) for p in successes])),
        'generating_world_vs_probe_pairs':[{'observation_id':p['observation_id'],'side':p['side'],
             'separation_mm':p['distance_from_true_mm'],
             'conditional_minimax_absolute_lower_bound_mm':str(Q(p['distance_from_true_mm'])/2)} for p in successes],
        'seconds':sum(p['result']['seconds'] for p in probes),'over_half_second':sum(p['result']['seconds']>.5 for p in probes)}
    analysis={'summary':summary,'DC_vs_D':ddc,'cluster_gain_counts':family,'search':search,'old118':residual,'probes':public_probe_stats,
              'comparisons':load(OUT/'evaluation/COMPARISONS.json'),'formal':load(OUT/'STAGE_STATUS.json')}
    body=json.dumps(analysis,indent=2,ensure_ascii=False)+'\n'; stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    version=ROOT/('ANALYSIS_'+stamp+'.json');version.write_text(body);(ROOT/'ANALYSIS.json').write_text(body)
    print(json.dumps({'summary':summary,'DC_vs_D':ddc,'search':search,'old118':residual,'probe_summary':{k:v for k,v in public_probe_stats.items() if k!='both_endpoints_success'}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

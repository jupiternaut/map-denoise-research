"""Post-verification descriptive analysis. Not in the scientific source graph."""
from collections import Counter
from fractions import Fraction as Q
from datetime import datetime,timezone,timedelta
import hashlib,json,socket
from pathlib import Path
R=Path(__file__).resolve().parent.parent;O=R/'outputs/anchor-v1'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def group(x,name):
    wrong=Q(x['initial_ae_mm'])!=0
    return (name=='all' or name=='shifted' and wrong or name=='correct' and not wrong or
            name=='primary' and wrong and x['variation_index']==1 and x['truth_mm']=='9007/14' and x['case']!='equal_full_negative' or
            name=='negative' and wrong and x['case']=='equal_full_negative')
def main():
    if socket.gethostname()!='liekkas':raise RuntimeError('wrong host')
    if load(O/'STAGE_STATUS.json')['status']!='COMPLETE' or load(O/'verification/VERIFY.json')['status']!='PASS':
        raise RuntimeError('formal stage gate before descriptive analysis')
    rows=load(O/'results.json');records=load(O/'diagnoses.json');names=('none','camera','appearance','both')
    cells={};paired={}
    for name in names:
        for g,n in [('all',324),('shifted',288),('correct',36),('primary',40),('negative',48)]:
            rs=[x for x in rows if x['arm']==name and group(x,g)]
            if len(rs)!=n:raise RuntimeError('denominator '+name+g)
            cells[name+':'+g]={'n':n,'status':dict(Counter(x['status'] for x in rs)),
                'old_a6_depth_exclusions':sum(x['a6_status']=='CERTIFIED_INFEASIBLE' for x in rs),
                'additional_depth_exclusions_vs_a6':sum(x['new_depth_exclusion'] for x in rs),
                'a7_forward_tasks':sum(x['a7_status']=='FORWARD_WITNESS' for x in rs),
                'a7_forward_now_certified':sum(x['a7_status']=='FORWARD_WITNESS' and x['status']=='CERTIFIED_INFEASIBLE' for x in rs),
                'a7_forward_still_forward':sum(x['a7_status']=='FORWARD_WITNESS' and x['status']=='FORWARD_WITNESS' for x in rs),
                'saved_branch_rejections':sum(x['saved_witness_rejected_branches'] for x in rs),
                'refitted_forward_branches':sum(x['refitted_forward_branches'] for x in rs)}
    rm={(x['task_id'],x['arm']):x for x in rows}
    for name in names[1:]:
        for g in ('all','shifted','correct','primary','negative'):
            rs=[x for x in rows if x['arm']==name and group(x,g)]
            pair=[(rm[x['task_id'],'none'],x) for x in rs]
            paired[name+':'+g]={'n':len(pair),
                'new_exclusions_vs_none':sum(a['status']!='CERTIFIED_INFEASIBLE' and b['status']=='CERTIFIED_INFEASIBLE' for a,b in pair),
                'lost_certificates_vs_none':sum(a['status']=='CERTIFIED_INFEASIBLE' and b['status']!='CERTIFIED_INFEASIBLE' for a,b in pair),
                'none_forward_now_certified':sum(a['status']=='FORWARD_WITNESS' and b['status']=='CERTIFIED_INFEASIBLE' for a,b in pair),
                'transitions':dict(Counter(a['status']+' -> '+b['status'] for a,b in pair))}
    coverage=load(O/'coverage.json')
    verify=load(O/'verification/VERIFY.json');budget=load(O/'BUDGET.json')
    result={'stage':'A8','formal_status':'COMPLETE','verification_status':'PASS','cells':cells,'paired':paired,
        'branch_status':{name:dict(Counter(b['status'] for r in records if r['arm']==name for b in r['branches'])) for name in names},
        'actual_coverage':{name:sum(c['arm']==name and c['actual_photo_and_enabled_anchors_verified'] for c in coverage) for name in names},
        'search_terminations':dict(Counter(b['termination'] for r in records for b in r['branches'])),
        'whole_seconds':load(O/'STAGE_STATUS.json')['seconds'],'budget':budget,'source_count':verify['source_files_verified'],
        'input_hashes':{str(p):sha(p) for p in (O/'results.json',O/'diagnoses.json',O/'coverage.json',O/'verification/VERIFY.json')}}
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    serialized=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    for path in (R/('ANALYSIS_'+stamp+'.json'),R/'ANALYSIS.json'):
        with path.open('x') as f:f.write(serialized)
    print(json.dumps({'status':'SAVED_ANALYSIS','file':str(R/'ANALYSIS.json'),
        'primary':{name:cells[name+':primary'] for name in names},
        'coverage':result['actual_coverage'],'whole_seconds':result['whole_seconds']},ensure_ascii=False))
if __name__=='__main__':main()

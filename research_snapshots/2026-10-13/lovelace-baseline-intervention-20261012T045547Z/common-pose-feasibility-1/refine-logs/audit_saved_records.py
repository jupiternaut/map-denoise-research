"""Audit saved A7 numerical records without converting an INCOMPLETE run to COMPLETE.
Reuses the independently authored verifier primitives; same original deadline.
No solver is imported or executed; no original scientific receipt is overwritten.
"""
import sys,time,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import verify_pose as v
OUT=ROOT/'outputs/pose-v1'
DEST=ROOT/'diagnostic-audit/SAVED_RECORD_AUDIT.json'
def main():
    assert not DEST.exists()
    v.check_target(OUT)
    with v.deadline_guard(OUT) as deadline:
        start=time.time()
        sources,inputs=v.verify_locks(OUT)
        v.same(sources,v.load(ROOT/'.aris/REVIEW_GATE.json')['files'],'restored full reviewed source graph')
        reg=v.load(ROOT/'REGISTRATION.json')
        v.verify_registration(reg)
        v.same(v.load(OUT/'REGISTRATION.json'),reg,'original frozen registration')
        run_audit,execute_audit=v.audit_runner_source(),v.audit_executor_source()
        observations,tasks,original_public,_=v._old.verify_public_population(OUT)
        v.same(v.load(OUT/'observations.json'),v.load(v.PRIOR/'observations.json'),'unchanged36observations')
        v.same(v.load(OUT/'tasks.json'),v.load(v.PRIOR/'tasks.json'),'unchanged324tasks')
        records=v.load(OUT/'diagnoses.json')
        v.verify_record_population(records,observations,tasks)
        old_records=v.load(v.PRIOR/'diagnoses.json')
        v._point.verify_record_population(old_records,observations,tasks)
        old_joint=v.indexed([r for r in old_records if r['method']=='joint'],lambda r:r['task_id'],'old324joint')
        arms=v.load(v.A5_OUTPUT/'outers.json')
        v.same(list(arms),list(observations),'36oldarms')
        counts=Counter();search_seconds=0.0
        for i,record in enumerate(records):
            oid,tid=record['observation_id'],record['task_id'];source=old_joint[tid]
            v.same(source['observation_id'],oid,'prior observation')
            v.same(source['incumbent'],tasks[tid]['incumbent'],'prior depth')
            v.same([r['branch'] for r in source['global']],list(v.BRANCHES),'prior3branches')
            derived=v._point.aggregate_status([r['point_classification'] for r in source['global']])
            v.same(source['global_status'],derived,'prior necessary union')
            v.same(record['a6_status'],derived,'unchangedA6status')
            for branch,old_branch in zip(record['branches'],source['global']):
                verified=v.verify_branch(branch,old_branch,observations[oid],arms[oid]['box']['joint'],tasks[tid])
                search_seconds+=verified.pop('search_seconds');counts.update(verified);counts['branches']+=1
            counts['tasks']+=1
            if i%27==26:print(json.dumps({'stage':'saved_record_diagnostic_audit','tasks_complete':i+1,'remaining_seconds':round(deadline-time.time(),1)}),flush=True)
        v.require(counts['tasks']==324 and counts['branches']==972,'complete saved public gate')
        pages=OUT/'checkpoint'
        v.same(sorted(p.name for p in pages.iterdir()), sorted(oid+'.json' for oid in observations),'exact36pages')
        for oid in observations:
            v.same(v.load(pages/(oid+'.json')),{'observation_id':oid,'records':[r for r in records if r['observation_id']==oid]},'saved page '+oid)
        # First diagnostic private parse occurs only after full saved public verification.
        facts,original_checks=v._old.verify_private_population(observations,tasks,original_public)
        coverage=[v.actual_coverage(p,arms[oid]['box']['joint'],facts[oid],records) for oid,p in observations.items()]
        rows=[v.evaluation_row(r,tasks[r['task_id']],observations[r['observation_id']],facts[r['observation_id']]) for r in records]
        v.same(v.load(OUT/'results.json'),rows,'324independent evaluation rows')
        v._geometry.verify_csv(OUT/'results.csv',rows)
        summary=v.summary_rows(rows);v.verify_summary_denominators(summary)
        v.same(v.load(OUT/'summary.json'),summary,'independent exact summary')
        status=v.load(OUT/'RUN_STATUS.json');stage=v.load(OUT/'STAGE_STATUS.json')
        v.require(status['status']=='INCOMPLETE' and stage['status']=='INCOMPLETE','original failed receipts preserved')
        v.require(status['observations_complete']==36 and status['diagnoses_complete']==324,'all records saved before failure')
        v.same(v.scientific_files(),sources,'restored reviewed source bytes unchanged during audit')
        v.same(v.input_files(),inputs,'input bytes unchanged during audit')
        v.verify_budget(v.load(OUT/'BUDGET.json'))
        result={'status':'SAVED_RECORD_NUMERICAL_PASS','formal_stage_status':'INCOMPLETE','formal_acceptance':False,
                'scope':'exact saved-record diagnostics only; original source-set failure is not retroactively accepted',
                'host':'liekkas','output':str(OUT),'tasks':324,'branches':972,'counts':dict(sorted(counts.items())),
                'actual_forward_coverage':coverage,'original_forward_checks':original_checks,
                'source_files_now_matching_lock':len(sources),'input_files_matching_lock':len(inputs),'old_inventory_artifacts_verified':94,
                'source_parse_order_audit':run_audit,'shared_budget_source_audit':execute_audit,
                'search_seconds':search_seconds,'audit_seconds':time.time()-start,
                'original_deadline_epoch':deadline,'elapsed_since_original_start_seconds':time.time()-v.load(OUT/'BUDGET.json')['start_epoch'],
                'workflow_failure':'unregistered posthoc root analyze_pose.py temporarily changed glob source set; relocated without modifying any frozen source'}
        DEST.parent.mkdir(exist_ok=True)
        with DEST.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({k:r for k,r in result.items() if k not in ('actual_forward_coverage','original_forward_checks')}),flush=True)
if __name__=='__main__':main()

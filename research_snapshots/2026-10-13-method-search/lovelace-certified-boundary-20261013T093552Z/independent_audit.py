"""A13 independent saved-artifact audit. No inference or generation reruns.

Uses the standalone verifier's reconstructed *complete* constraint arrays for
an independent full-row rational margin sum; replays the stored physical world
directly here, without loading the generator or either estimator.
"""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import resource
import socket
import time

ROOT = Path(__file__).resolve().parent
LOGICAL = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z')
OLD = LOGICAL.parent / 'lovelace-representation-search-20261013T080031Z'
FOLDER = ROOT / 'outputs/formal-v1'
FILES = {}
ERRORS = []


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    FILES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def check(condition, message):
    if not condition:
        ERRORS.append(message)


def exact(value):
    return Q(value['numerator'], value['denominator']) if isinstance(value, dict) else Q(value)


def flatten_float(value):
    if isinstance(value, dict):
        if set(value) == {'numerator', 'denominator'}:
            return float(exact(value))
        return {k: flatten_float(v) for k, v in value.items()}
    return [flatten_float(v) for v in value] if isinstance(value, list) else value


def aggregate(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    initial = sum(r['initial_se'] for r in rows) / n
    final = sum(r['final_se'] for r in rows) / n
    return dict(n=n, worlds=len({r['observation_id'] for r in rows}), mse=final,
                mae=sum(r['final_ae'] for r in rows)/n, identity_mse=initial,
                mse_improvement_percent=100*(1-final/initial) if initial else None,
                improved=sum(r['initial_ae']-r['final_ae'] > 1e-9 for r in rows),
                same=sum(abs(r['initial_ae']-r['final_ae']) <= 1e-9 for r in rows),
                harmed=sum(r['final_ae']-r['initial_ae'] > 1e-9 for r in rows),
                max_error=max(r['final_ae'] for r in rows), failures=sum(r['failure'] for r in rows))


def equal_tree(a, b, context):
    if isinstance(a, dict) and isinstance(b, dict):
        check(set(a) == set(b), context + ': keys differ')
        for k in set(a) & set(b):
            equal_tree(a[k], b[k], context + '.' + k)
    elif isinstance(a, list) and isinstance(b, list):
        check(len(a) == len(b), context + ': lengths differ')
        for i, (x, y) in enumerate(zip(a, b)):
            equal_tree(x, y, context + f'[{i}]')
    elif isinstance(a, float) and isinstance(b, (int, float)):
        check(math.isclose(a, b, rel_tol=2e-13, abs_tol=1e-13), context + f': {a} != {b}')
    else:
        check(a == b, context + f': {a} != {b}')


def replay_physical(fact, observation):
    """Direct point rendering/acquisition from saved latents, exact arithmetic."""
    z = exact(fact['truth']); branch = fact['rear_branch']
    colors = list(map(exact, fact['view_colors']))
    offsets = list(map(exact, fact['actual_offsets']))
    noise = list(map(exact, fact['image_noise']))
    bias = list(map(exact, fact['actual_view_bias']))
    t = {k: list(map(exact, v)) if isinstance(v, list) else exact(v)
         for k, v in fact['anchor_tape'].items()}
    check(Q(540) <= z <= 660 and offsets[1] == 0 and all(abs(x) <= Q(1, 5) for x in offsets), 'latent camera/depth bounds')
    check(all(0 <= x <= 1 for x in colors), 'latent colors')
    check(all(max(colors[k::3])-min(colors[k::3]) <= Q(1, 10) for k in range(3)), 'latent drift')
    check(all(abs(x) <= Q(1, 100) for x in noise+bias), 'latent photographic errors')
    clip = lambda x: min(Q(1), max(Q(0), x))
    def area(rect, pixel):
        return max(Q(0), min(rect[1], pixel[1])-max(rect[0], pixel[0])) * max(Q(0), min(rect[3], pixel[3])-max(rect[2], pixel[2])) * 750
    def project(rect, depth, camera):
        return ((rect[0]-camera)/depth, (rect[1]-camera)/depth, Q(rect[2])/depth, Q(rect[3])/depth)
    def intersect(a, b):
        return max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    rears = {'none': None, 'partial': (-20,36,-17,23), 'full': (-100,100,-80,80)}
    pixels = [(Q(-1,5)+Q(j,20), Q(-1,5)+Q(j+1,20), Q(-2,25)+Q(2*i,75), Q(-2,25)+Q(2*(i+1),75)) for i in range(6) for j in range(8)]
    rendered = []
    for v, center in enumerate((-80,0,80)):
        camera = center+offsets[v]
        front = project((-20,20,-15,15), z, camera)
        left = project((-20,0,-15,15), z, camera)
        right = project((0,20,-15,15), z, camera)
        rear = project(rears[branch], Q(720), camera) if rears[branch] else None
        for p, pixel in enumerate(pixels):
            areas = [area(left,pixel), area(right,pixel), Q(0) if rear is None else area(rear,pixel)-area(intersect(rear,front),pixel)]
            ideal = 1-sum(a*(1-c) for a,c in zip(areas,colors[3*v:3*v+3]))
            for channel in range(3):
                rendered.append(clip(ideal+bias[v]+noise[v*144+p*3+channel]))
    check(rendered == list(map(exact, fact['photo_exact'])), fact['observation_id']+': exact stored photo')
    check(list(map(float, rendered)) == observation['photo'], fact['observation_id']+': public photo transport')
    u = [(-center-offsets[v]-t['camera_k']-t['camera_noise'][v])/1000 for v,center in enumerate((-80,0,80))]
    coupons = [clip(colors[i]+Q(1,100)-t['appearance_t'][i//3]+t['coupon_transfer'][i]+t['coupon_noise'][i]) for i in range(9)]
    refs = [clip(Q(1,2)+Q(1,100)-t['appearance_t'][v]+t['reference_noise'][v]) for v in range(3)]
    for values, key, obj in ((u,'u','camera'), (coupons,'coupons','appearance'), (refs,'references','appearance')):
        check(values == list(map(exact, fact['anchor_exact'][obj][key])), fact['observation_id']+': exact anchor '+key)
        check(list(map(float,values)) == observation[obj][key], fact['observation_id']+': public anchor '+key)
    check(flatten_float(fact['anchor_exact']['camera']) == observation['camera'], 'camera packet')
    check(flatten_float(fact['anchor_exact']['appearance']) == observation['appearance'], 'appearance packet')
    return [1-c for c in colors]+bias+[Q(1,100)-x for x in t['appearance_t']]


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    started = time.perf_counter(); cpu = time.process_time()
    check(socket.gethostname() == 'liekkas' and ROOT == LOGICAL.resolve(), 'exact target identity')
    import verify_certificates as verifier
    freeze = read(ROOT/'SOURCE_FREEZE.json')
    for path, digest in freeze['files'].items():
        check(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'source freeze '+path)
    generation = read(ROOT/'data/confirm/GENERATION.json')
    public = read(ROOT/'data/confirm/public.json')['observations']
    tasks = read(ROOT/'data/confirm/tasks.json')
    facts = read(ROOT/'data/confirm/private/truth.json')
    observations = {p['observation_id']: p['observation'] for p in public}
    truth = {f['observation_id']: f for f in facts}
    check(len(observations) == len(public) == len(truth) == len(facts) == 18, '18 unique worlds')
    check(len(tasks) == len({t['task_id'] for t in tasks}) == 126, '126 unique tasks')
    check(set(observations) == set(truth) == {t['observation_id'] for t in tasks}, 'population identities')
    check(generation['source_freeze_sha256'] == FILES[str(ROOT/'SOURCE_FREEZE.json')], 'generation freeze binding')
    check(freeze['utc'] < generation['utc'], 'freeze precedes generation')
    for fact in facts:
        oid = fact['observation_id']; point = replay_physical(fact, observations[oid])
        rowset, bounds = verifier.reconstruct_constraints(observations[oid], fact['rear_branch'], exact(fact['truth']), exact(fact['truth']))
        check(all(a <= x <= b for x,(a,b) in zip(point,bounds)), oid+': generating point nuisance box')
        check(all(sum(point[j]*a for j,a in row) <= b for row,b in rowset), oid+': all generating point necessary constraints')
        selected = [t for t in tasks if t['observation_id'] == oid]
        check(len(selected)==7 and sum(exact(t['incumbent_exact'])==exact(fact['truth']) for t in selected)==1, oid+': one correct plus six shifts')
        check(all(float(exact(t['incumbent_exact']))==t['incumbent'] for t in selected), oid+': task binary64 transport')
    manifest = read(FOLDER/'MANIFEST.json'); ready = read(FOLDER/'PREDICTIONS_READY.json')
    saved_summary = read(FOLDER/'SUMMARY.json'); saved_costs = read(FOLDER/'COSTS.json')
    saved_analysis = read(FOLDER/'ANALYSIS.json'); saved_verification = read(FOLDER/'INDEPENDENT_VERIFICATION.json')
    saved_paired = read(FOLDER/'PAIRED_RESIDUALS.json')
    check(ready['status']=='COMPLETE' and ready['calls']==len(manifest)==len(ready['receipts'])==108, '108 complete calls')
    check(ready['source_freeze_sha256']==FILES[str(ROOT/'SOURCE_FREEZE.json')], 'ready freeze binding')
    rows=[]; costs=[]; states={}; coverage=defaultdict(list); proof_counts=Counter(); support=Counter(); cert_cache={}; verification=[]
    for i,(job, receipt) in enumerate(zip(manifest, ready['receipts'])):
        check(receipt == read(FOLDER/f'jobs/{i:03}.receipt.json'), f'receipt {i} equality')
        check(job == read(FOLDER/f'jobs/{i:03}.job.json'), f'job {i} equality')
        out = read(ROOT/receipt['result'])
        check(FILES[str(ROOT/receipt['result'])] == receipt['sha256'], f'result {i} hash')
        check(receipt['returncode'] == 0, f'worker {i} return code')
        check(read(FOLDER/f'state/{i:03}-running.json') == {'index':i,'status':'running'}, f'running {i}')
        check(read(FOLDER/f'state/{i:03}-terminal.json') == {'index':i,'status':'completed'}, f'terminal {i}')
        for k in ('arm','budget_seconds','observation_id'):
            check(job[k] == out[k] == receipt[k], f'identity {i} {k}')
        arm,budget,oid = out['arm'],out['budget_seconds'],out['observation_id']
        expected_arm = ('numeric','certified','legacy')[(i%3+(list(observations).index(oid)+int(budget==24))%3)%3]
        check(arm == expected_arm, f'rotated order {i}')
        check(job['observation'] == observations[oid] and set(job['observation'])=={'photo','camera','appearance'}, f'public input {i}')
        expected_tasks=[{'task_id':t['task_id'],'incumbent':t['incumbent']} for t in tasks if t['observation_id']==oid]
        check(job['tasks']==expected_tasks, f'tasks {i}')
        check(len(out['predictions'])==7, f'predictions {i}')
        check([(p['task_id'],p['incumbent']) for p in out['predictions']]==[(t['task_id'],t['incumbent']) for t in expected_tasks], f'prediction task provenance {i}')
        state=out['state']; states[(arm,budget,oid)]=out
        hull=state.get('hull') if arm=='legacy' else [state['lower'],state['upper']]
        fact=truth[oid]; z=float(exact(fact['truth'])); label=f'{arm}@{budget:g}'
        coverage[label].append(hull is not None and Q(hull[0])<=exact(fact['truth'])<=Q(hull[1]))
        costs.append({k:receipt[k] for k in ('arm','budget_seconds','observation_id','process_seconds','process_cpu_seconds','returncode')} | out['cost'])
        for p in out['predictions']:
            selected=p['incumbent'] if p['failure'] or hull is None else min(hull[1],max(hull[0],p['incumbent']))
            check(selected==p['selected'], f'decoder {i} {p["task_id"]}')
            a=abs(p['incumbent']-z); b=abs(p['selected']-z)
            rows.append(dict(arm=arm,budget_seconds=budget,observation_id=oid,task_id=p['task_id'],truth=z,incumbent=p['incumbent'],selected=p['selected'],initial_ae=a,final_ae=b,initial_se=a*a,final_se=b*b,correct_input=a<=1e-9,failure=p['failure'] is not None,branch=fact['rear_branch'],stratum=fact['stratum'],negative_control=fact['negative_control'],seed_block=fact['seed_block']))
        if arm=='certified':
            result=verifier.verify_state(observations[oid],state)
            check(result['valid'] and result['partitioned'], f'formal state verifier {i}')
            check(state['certificate_attempts']==state['exclusions'], f'no hidden rejected proposals {i}')
            verification.append({'job':i,**result})
            for e in state['exclusions']:
                cert=e['certificate']; proof_counts[str(int(budget))]+=1; support[len(cert['multipliers'])]+=1
                key=json.dumps(cert,sort_keys=True)
                if key not in cert_cache:
                    cs,bs=verifier.reconstruct_constraints(observations[oid],e['branch'],e['lower'],e['upper'])
                    coef=[Q(0)]*15; rhs=Q(0)
                    for index,w in cert['multipliers']:
                        w=Q(w); row,b=cs[index]; rhs+=w*b
                        for j,a in row: coef[j]+=w*a
                    margin=sum(a*(lo if a>=0 else hi) for a,(lo,hi) in zip(coef,bs))-rhs
                    check(margin==Q(cert['margin']) and margin>0, f'full-row exact certificate {i}')
                    cert_cache[key]=str(margin)
    check(len(rows)==len({(r['arm'],r['budget_seconds'],r['task_id']) for r in rows})==ready['prediction_count']==756,'756 unique predictions')
    check(len(states)==108 and len(list((FOLDER/'jobs').glob('*.result.json')))==108, 'complete output cross-product')
    csv_path=FOLDER/'METRICS.csv'; FILES[str(csv_path)]=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    saved_csv=list(csv.DictReader(csv_path.open()))
    check(len(saved_csv)==len(rows),'CSV row count')
    for i,(expected,saved) in enumerate(zip(rows,saved_csv)):
        check({k:str(v) for k,v in expected.items()}==saved,f'CSV exact row {i}')
    summary={}
    for arm in ('numeric','certified','legacy'):
        for budget in (6.,24.):
            label=f'{arm}@{budget:g}'; rr=[r for r in rows if r['arm']==arm and r['budget_seconds']==budget]; cc=[c for c in costs if c['arm']==arm and c['budget_seconds']==budget]
            summary[label]={'all':aggregate(rr),'shifted':aggregate([r for r in rr if not r['correct_input']]),'correct':aggregate([r for r in rr if r['correct_input']]),'negative':aggregate([r for r in rr if r['negative_control']]),'by_branch':{b:aggregate([r for r in rr if r['branch']==b]) for b in ('none','partial','full')},'by_stratum':{s:aggregate([r for r in rr if r['stratum']==s]) for s in ('contrast','lowcontrast')},'covered_worlds':sum(coverage[label]),'worlds':18,'cost':{k:sum(x.get(k,0) for x in cc) for k in ('process_seconds','process_cpu_seconds','prepare_seconds','prepare_cpu_seconds','overrun_gt_0_25')}}
    equal_tree(summary,saved_summary,'SUMMARY'); equal_tree(costs,saved_costs,'COSTS')
    ratios={}; proof_components={}
    for budget in (6.,24.):
        ns,cs,ls=[summary[f'{a}@{budget:g}'] for a in ('numeric','certified','legacy')]
        attempts=[e for (a,b,o),r in states.items() if a=='certified' and b==budget for e in r['state']['certificate_attempts']]
        ratios[f'{budget:g}']={'legacy_over_certified_process_ratio':ls['cost']['process_seconds']/cs['cost']['process_seconds'],'certified_over_numeric_process_ratio':cs['cost']['process_seconds']/ns['cost']['process_seconds'],'certified_vs_legacy_shifted_mse_difference':cs['shifted']['mse']-ls['shifted']['mse'],'certified_vs_legacy_shifted_mse_percent':100*(cs['shifted']['mse']/ls['shifted']['mse']-1)}
        for k,v in ratios[f'{budget:g}'].items(): equal_tree(v,saved_analysis[f'{budget:g}'][k],f'ANALYSIS {budget} {k}')
        proof_components[f'{budget:g}']={k:sum(e.get(k,0) for e in attempts) for k in ('geometry_seconds','certificate_search_seconds','exact_check_seconds','wall_seconds','cpu_seconds')}
        equal_tree(proof_components[f'{budget:g}'],saved_analysis[f'{budget:g}']['proof_component_totals'],f'proof component totals {budget}')
        check(len(attempts)==saved_analysis[f'{budget:g}']['certificate_attempts']==saved_analysis[f'{budget:g}']['verified_exclusions'], f'proof totals {budget}')
        sameh=samep=0
        for oid in observations:
            c,n=[states[(a,budget,oid)] for a in ('certified','numeric')]
            check(c['state']['cells']==n['state']['cells'], f'full retained cells equal {budget} {oid}')
            check([tuple(e[k] for k in ('branch','lower','upper')) for e in c['state']['exclusions']]==[tuple(e[k] for k in ('branch','lower','upper')) for e in n['state']['exclusions']],f'proposal sequences equal {budget} {oid}')
            sameh += [c['state']['lower'],c['state']['upper']]==[n['state']['lower'],n['state']['upper']]
            samep += sum(a['selected']==b['selected'] for a,b in zip(c['predictions'],n['predictions']))
        check(sameh==saved_analysis[f'{budget:g}']['exact_same_hulls_vs_numeric']==18 and samep==saved_analysis[f'{budget:g}']['exact_same_predictions_vs_numeric']==126, f'numeric paired equality {budget}')
    diag=read(ROOT/'outputs/diagnostic-v1/RESULTS.json')['records']; dev=read(OLD/'data/dev/public.json')['observations']; devobs={x['observation_id']:x['observation'] for x in dev}
    diagnostic=[]
    for r in diag:
        check(r['attempted']==len(r['proofs'])==len(r['numeric_state']['exclusions']), 'dev attempt completeness')
        check([tuple(e[k] for k in ('branch','lower','upper')) for e in r['proofs']]==[tuple(e[k] for k in ('branch','lower','upper')) for e in r['numeric_state']['exclusions']], 'dev exact same proposals')
        check(r['verified']==sum(p['verified'] for p in r['proofs']), 'dev verified count')
        state=dict(r['numeric_state']); state['exclusions']=[p for p in r['proofs'] if p['verified']]
        state['cells']=state['cells']+[p for p in r['proofs'] if not p['verified']]
        state['lower'],state['upper']=r['verified_replay_hull']
        v=verifier.verify_state(devobs[r['observation_id']],state); check(v['valid'],'dev partition/certificates')
        diagnostic.append({'observation_id':r['observation_id'],**v})
    check(len(diag)==6 and sum(r['verified'] for r in diag)==89,'six dev worlds / 89 proofs')
    check(saved_verification['all_valid'] and len(saved_verification['formal_states'])==36 and len(saved_verification['diagnostic_states'])==6,'saved independent verification completeness')
    for i,x in enumerate(verification):
        y=saved_verification['formal_states'][i]
        check(x['checked_exclusions']==y['checked_exclusions'] and x['hull']==y['hull'] and y['valid'] and y['decoder_valid'],'saved formal verification values')
    ordered=[ROOT/'SOURCE_FREEZE.json',ROOT/'data/confirm/GENERATION.json',FOLDER/'MANIFEST.json',FOLDER/'PREDICTIONS_READY.json',FOLDER/'INDEPENDENT_VERIFICATION.json',FOLDER/'METRICS.csv',FOLDER/'SUMMARY.json',FOLDER/'ANALYSIS.json']
    mtimes={str(p.relative_to(ROOT)):datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat() for p in ordered}
    check(all(a.stat().st_mtime<=b.stat().st_mtime for a,b in zip(ordered,ordered[1:])),'file chronology')
    check(all((ROOT/r['result']).stat().st_mtime <= (FOLDER/'PREDICTIONS_READY.json').stat().st_mtime for r in ready['receipts']),'all predictions before ready')
    residual_sums=defaultdict(float); residual_counts=defaultdict(int); labels={}
    for row in rows:
        if row['budget_seconds']==24. and row['arm'] in ('certified','legacy'):
            k=(row['observation_id'],row['arm']); residual_sums[k]+=row['final_se']; residual_counts[k]+=1
            labels[row['observation_id']]={'branch':row['branch'],'stratum':row['stratum'],'negative_control':str(row['negative_control'])}
    paired_rows=[{'observation_id':oid,'certified_minus_legacy_mse':(residual_sums[(oid,'certified')]-residual_sums[(oid,'legacy')])/residual_counts[(oid,'certified')],**group} for oid,group in labels.items()]
    negative=sum(residual_sums[(o,'certified')] for o,g in labels.items() if g['negative_control']=='True')
    total=sum(residual_sums[(o,'certified')] for o in labels)
    paired={'scope':'post-hoc descriptive analysis of frozen predictions; no parameter selection','worlds':paired_rows,'certified_better':sum(x['certified_minus_legacy_mse'] < -1e-12 for x in paired_rows),'certified_worse':sum(x['certified_minus_legacy_mse'] > 1e-12 for x in paired_rows),'same':sum(abs(x['certified_minus_legacy_mse']) <= 1e-12 for x in paired_rows),'negative_control_fraction_of_certified_squared_error':negative/total}
    equal_tree(paired,saved_paired,'PAIRED_RESIDUALS')
    output={'schema':'a13-independent-integrity-audit-v1','generated_at':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'logical_root':str(LOGICAL),'canonical_root':str(ROOT),'evaluation_type':'simulation_only','errors':ERRORS,'all_deterministic_checks_pass':not ERRORS,'worlds':18,'seed_blocks':dict(Counter(f['seed'] for f in facts)),'negative_worlds':sum(f['negative_control'] for f in facts),'predictions':len(rows),'calls':len(states),'freeze_files_checked':len(freeze['files']),'summary_recomputed':summary,'ratios_recomputed':ratios,'proof_component_totals':proof_components,'formal_exclusion_occurrences':sum(proof_counts.values()),'formal_exclusions_per_budget':dict(proof_counts),'distinct_formal_certificates':len(cert_cache),'formal_support_size_counts':dict(support),'formal_verification':verification,'diagnostic_verification':diagnostic,'diagnostic_proofs':89,'formal_batch_seconds':ready['seconds'],'total_saved_process_seconds':sum(c['process_seconds'] for c in costs),'total_saved_process_cpu_seconds':sum(c['process_cpu_seconds'] for c in costs),'artifact_mtimes_utc':mtimes,'audited_input_hashes':FILES,'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,'limitations':['No solver inference or new worlds generated.','Metric and certificate recomputation is deterministic; semantic source assessment remains same-family/provisional.','Complete-row certificate sum reuses standalone verifier geometry reconstruction; independent physical rendering/acquisition replay is authored in this script.','Chronology is supported by saved code, hashes and filesystem timestamps, not adversarial tamper-proof historical logs.']}
    output['paired_residuals_recomputed']=paired
    with (ROOT/'INDEPENDENT_AUDIT_RESULTS_FINAL.json').open('x') as f: json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({k:output[k] for k in ('all_deterministic_checks_pass','errors','calls','predictions','formal_exclusion_occurrences','distinct_formal_certificates','formal_support_size_counts','diagnostic_proofs','total_saved_process_seconds','total_saved_process_cpu_seconds','wall_seconds','cpu_seconds')}))
    for label,s in summary.items(): print(json.dumps({'arm':label,'shifted':s['shifted'],'correct':s['correct'],'negative':s['negative'],'cost':s['cost']}))
    return int(bool(ERRORS))


if __name__=='__main__':
    raise SystemExit(main())

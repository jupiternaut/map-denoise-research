"""Fixed-A evaluation reuses sealed exact distances, never recomputes candidates."""
from common import *
import csv,time
from threadpoolctl import threadpool_limits
from evaluate_replay import measure,summarize,matched_random,NUMERIC


def main():
    check_host();start=time.monotonic()
    for s in SCENES:verify_seal(ROOT/'inference'/f'scan{s}')
    verify_seal(PREV/'evaluation')
    out=ROOT/'evaluation';out.mkdir(exist_ok=False)
    sources={str(p):sha(p) for p in (ROOT/'evaluate_reserved.py',PREV/'evaluate_replay.py',
              PREV/'evaluation/SEALED.json',ROOT/'PROTOCOL.md')}
    save_json(out/'LOCK.json',dict(source_sha256=sources,
               data_role='EXPOSED_REPLAY',same_frozen_A_identity_and_native_support=True))
    with (PREV/'evaluation/METRICS.csv').open() as f:
        old={(int(r['scene']),r['roi'],r['condition'],r['arm']):r for r in csv.DictReader(f)}
    rows=[];checks=[];ply_checks=[]
    for sid in SCENES:
        for path in cases_for_scene(sid):
            rid,condition=path.name.split('__');dest=out/path.name;dest.mkdir()
            error_path=PREV/'evaluation'/path.name/'point_errors.npz'
            with np.load(error_path,allow_pickle=False) as z:
                d0,d1,support,move2=(z[k] for k in ('d0','d1','support','movement_squared'))
            inf=ROOT/'inference'/f'scan{sid}'/path.name
            with np.load(inf/'decisions.npz',allow_pickle=False) as z:decisions={k:z[k] for k in z.files}
            for mode in ('count','bin'):
                for seed in range(10):
                    decisions[f'random__{mode}__seed{seed}']=matched_random(
                        decisions['reserved_aug__balanced'],support,np.sqrt(move2),mode,SEED+seed)
            decisions['oracle_fixed_A']=d1<d0
            save_npz(dest/'diagnostic_decisions.npz',**{k:v for k,v in decisions.items()
                     if k.startswith('random__') or k=='oracle_fixed_A'})
            save_json(dest/'ERROR_SOURCE.json',dict(path=str(error_path),sha256=sha(error_path),
                       fixed_candidates=str(path),scope='unchanged row-preserving A or identity'))
            for arm,mask in decisions.items():
                if mask.shape!=d0.shape or mask.dtype!=bool:raise AssertionError('invalid decision')
                m=measure(d0,d1,support,move2,mask)
                rows.append(dict(scene=sid,roi=rid,condition=condition,arm=arm,n_rows=len(d0),
                                 n_source=int(support.sum()),**m))
                names={'identity':'identity','A_all':'A_all','frozen_gain':'frozen_gain',
                       'previous_normalized':'normalized_gain__balanced',
                       'cached64__balanced':'normalized_gain__balanced'}
                if arm in names:
                    previous=old[(sid,rid,condition,names[arm])]
                    err=max(abs(m[k]-float(previous[k])) for k in NUMERIC)
                    if err>1e-9:raise AssertionError('previous baseline mismatch')
                    checks.append(dict(case=path.name,arm=arm,max_metric_difference=err))
            p,a=read_points(path/'identity.ply'),read_points(path/'A_all.ply')
            if p.shape!=a.shape or len(p)!=len(d0):raise AssertionError('geometry row mismatch')
            if not np.allclose(np.sum((a-p)**2,axis=1),move2,atol=0,rtol=0):raise AssertionError('old A changed')
            for arm in ('fit_aug__balanced','reserved_aug__balanced','both_aug__balanced','previous_reserved_veto'):
                q=read_points(inf/(arm+'.ply'));expected=np.where(decisions[arm][:,None],a,p)
                err=float(np.max(np.abs(q-expected)))
                if err!=0:raise AssertionError('export changed coordinates')
                ply_checks.append(dict(case=path.name,arm=arm,max_coordinate_difference_mm=err))
            print('SCORED',path.name,len(decisions),'arms',flush=True)
    with (out/'METRICS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=summarize(rows);summary.update(wall_seconds=time.monotonic()-start,
        aggregation='ROI then scene equal weight',not_new_confirmation=True)
    save_json(out/'SUMMARY.json',summary)
    save_json(out/'REPRODUCTION.json',dict(status='PASS',metric_checks=checks,ply_checks=ply_checks))
    seal(out,sources);print('EVALUATION SEALED',len(rows),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

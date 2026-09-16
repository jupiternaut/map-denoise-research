"""E0-F: bounded incumbent-support construction; historical artifacts read-only."""
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PREV = Path('/home/grf/Documents/Codex/2026-09-16/e0-e-selective-revision-20260915T160016Z')
sys.path.insert(0, str(PREV))
from experiment_support import (OLD, prepared_population, calibrate_with_roughness,
    linear_projection, evaluate, write_json, write_ply, sha, snapshot, arrays_hash)
from selective_operator import choose_outputs, curve_features, veto_mask
from anchored_operator import construct, anchored_prominence


class ZeroNoise:
    def standard_normal(self, size):
        return np.zeros(size)


def history_hashes():
    result = snapshot()
    result.update({str(p): sha(p) for p in PREV.rglob('*') if p.is_file()})
    return result


def recovered_ledger(baseline, output, truth, subset, mm, strata):
    recover = (baseline == 0) & (output != 0)
    improvement = np.abs(baseline-truth)-np.abs(output-truth)
    masks = {'ALL': np.ones(len(truth), bool)}
    for name in np.unique(subset):
        masks[str(name)] = subset == name
        for k in range(3):
            masks[f'{name}:contrast{k}'] = (subset == name) & (strata == k)
    masks['big:single'] = (subset == 'big') & ~mm
    rows = {}
    for name, selection in masks.items():
        take = selection & recover
        rows[name] = dict(n=int(take.sum()), beneficial=int((take & (improvement>1e-9)).sum()),
            harmful=int((take & (improvement < -1e-9)).sum()),
            gain_mm=float(np.maximum(improvement[take],0).sum()),
            harm_mm=float(np.maximum(-improvement[take],0).sum()),
            net_mae_contribution=float(-improvement[take].sum()/len(truth)))
    return rows


def noise_diagnostic(pop, strata, feature, observations, threshold):
    # Evaluation-only reconstruction of the same latent signal. No method receives it.
    c0 = OLD.make_curves(ZeroNoise(), pop['tstar'], pop['subset'], pop['mm'], pop)
    feature0 = curve_features(c0, OLD.TGRID)
    # Freeze roughness and original TEST/proposal; contrast and width recomputed.
    v0, _ = veto_mask(feature0, observations['sigma_hf'])
    p0, _ = anchored_prominence(c0, OLD.TGRID, feature0.width_mm, 1.)
    threshold0 = np.maximum(.25*feature0.contrast, 3*observations['sigma_hf'])
    a0 = (p0>0) & (p0>=threshold0)
    pn, _ = anchored_prominence(pop['c'], OLD.TGRID, feature.width_mm, 1.)
    an = (pn>0) & (pn>=threshold)
    rows=[]
    masks = {str(k):pop['subset']==k for k in np.unique(pop['subset'])}
    masks['big:single']=(pop['subset']=='big') & ~pop['mm']
    for name, sel in masks.items():
        for k in range(3):
            take = sel & (strata==k) & observations['test']
            rows.append(dict(group=name, stratum=k, n=int(take.sum()),
                noisy_global=int((take & observations['veto']).sum()),
                noiseless_global=int((take & v0).sum()),
                noisy_anchor=int((take & an).sum()), noiseless_anchor=int((take & a0).sum()),
                global_noise_added=int((take & observations['veto'] & ~v0).sum()),
                global_noise_removed=int((take & ~observations['veto'] & v0).sum())))
    return rows, dict(oracle_noiseless_global=v0, oracle_noiseless_anchor=a0)


def run_case(dest, phase, seed, fbig, kind, pop, strata, cuts, props, feature):
    key=f'{phase}_seed{seed}_fbig{fbig:.2f}_{kind}'
    cal, sigma, chash=calibrate_with_roughness(seed, fbig, kind, cuts, props, .2)
    oldout, obs=choose_outputs(OLD, linear_projection, pop['c'], strata, cal, sigma, feature)
    out, masks, extra=construct(pop['c'], OLD.TGRID, feature, obs, oldout)
    if phase=='development':
        name=f'development_seed{seed}_fbig{fbig:.2f}_{kind}_a0.2'
        with np.load(PREV/'results/cases'/name/'outputs.npz') as data:
            for arm in oldout:
                np.testing.assert_array_equal(oldout[arm], data['out_'+arm])
    xyz=np.column_stack([pop['x'],pop['y'],pop['z0']])
    metrics={arm:evaluate(theta,xyz,pop['tstar'],pop['subset'],pop['mm']) for arm,theta in out.items()}
    # A reference denominator, not an additional candidate chosen from truth.
    denom=evaluate(oldout['no_bh_projection'],xyz,pop['tstar'],pop['subset'],pop['mm'])
    ledger={arm:recovered_ledger(out['test_veto_projection'],out[arm],pop['tstar'],
                               pop['subset'],pop['mm'],strata)
            for arm in ('anchor1_projection','anchor2_projection')}
    rows,oracle=noise_diagnostic(pop,strata,feature,obs,obs['threshold']) if phase=='development' else ([],{})
    folder=dest/'cases'/key
    folder.mkdir(parents=True)
    np.savez_compressed(folder/'outputs.npz', xyz0=xyz,evaluation_truth_delta=pop['tstar'],
        evaluation_subset=pop['subset'].astype(str),evaluation_mm=pop['mm'],strata=strata,
        test_mask=obs['test'],global_veto=obs['veto'],threshold=obs['threshold'],
        **{'veto_'+k:v for k,v in masks.items()},**extra,**oracle,
        **{'out_'+k:v for k,v in out.items()},out_no_bh_projection=oldout['no_bh_projection'])
    for name,theta in out.items():
        cloud=xyz.copy();cloud[:,2]+=theta
        write_ply(folder/(name+'.ply'),cloud)
    ref=xyz.copy();ref[:,2]+=pop['tstar']
    write_ply(folder/'evaluation_reference.ply',ref)
    row=dict(key=key,phase=phase,seed=seed,fbig=fbig,calibration=kind,n=len(xyz),
        public_hash=arrays_hash(pop['c'],strata),calibration_hash=chash,
        metrics=metrics,recovered_ledger=ledger,noise_oracle_diagnostic=rows,
        back_harm_denominator=denom['groups']['twosheet_back']['harm_sum'])
    write_json(folder/'metrics.json',row)
    print(key,{k:round(v['groups']['ALL']['delta_mae'],6) for k,v in metrics.items()},flush=True)
    return row


def stress(dest):
    _,_,_,_,cuts,props=prepared_population(OLD,101,.1)
    cal,sigma,_=calibrate_with_roughness(101,.1,'exch',cuts,props,.2)
    rows=[]
    for shift in (0.,.75,1.5,2.5):
        for ratio in (.15,.30,.60):
            rng=np.random.default_rng([991,int(shift*100),int(ratio*100)])
            n=256
            amplitude=rng.uniform(.5,.8,n)
            main=np.exp(-.5*(OLD.TGRID-6.)**2)
            secondary=ratio*np.exp(-.5*(OLD.TGRID-shift)**2)
            c=amplitude[:,None]*(1-np.maximum(main,secondary))+.03*OLD.smooth_noise(rng,n)
            strata=np.digitize(np.ptp(c,axis=1),cuts)
            feat=curve_features(c,OLD.TGRID)
            oldout,obs=choose_outputs(OLD,linear_projection,c,strata,cal,sigma,feat)
            out,masks,extra=construct(c,OLD.TGRID,feat,obs,oldout)
            # No truth is passed into either call; these are deliberately identical observations.
            repeat,_,_=construct(c.copy(),OLD.TGRID.copy(),feat,obs,oldout)
            for name in out:np.testing.assert_array_equal(out[name],repeat[name])
            metrics={}
            for world,truth in [('authentic_alternative',shift),('ghost',6.)]:
                metrics[world]={name:dict(mae=float(np.abs(theta-truth).mean()),
                     delta_mae=float(np.abs(theta-truth).mean()-abs(truth))) for name,theta in out.items()}
            key=f'offset{shift:.2f}_ratio{ratio:.2f}'
            folder=dest/'stress'/key;folder.mkdir(parents=True)
            np.savez_compressed(folder/'outputs.npz',curves=c,grid=OLD.TGRID,
                truth_authentic=np.full(n,shift),truth_ghost=np.full(n,6.),
                **{'out_'+k:v for k,v in out.items()})
            row=dict(key=key,shift=shift,ratio=ratio,n=n,metrics=metrics,
                paired_outputs_equal=True,global_veto=float(obs['veto'].mean()),
                anchor1_veto=float(masks['anchor1'].mean()),anchor2_veto=float(masks['anchor2'].mean()))
            write_json(folder/'metrics.json',row);rows.append(row)
    return rows


def summarize(runs):
    groups={}
    for row in runs:groups.setdefault((row['phase'],row['fbig'],row['calibration']),[]).append(row)
    result=[]
    for (phase,fbig,kind),rr in groups.items():
        arms={}
        for name in rr[0]['metrics']:
            arms[name]={k:float(np.mean([r['metrics'][name]['groups']['ALL'][k] for r in rr]))
                        for k in ('mae','delta_mae','move_rate','harm_rate')}
            arms[name]['seed_delta']=[r['metrics'][name]['groups']['ALL']['delta_mae'] for r in rr]
            for k in ('nn_accuracy','nn_completeness','nn_symmetric','precision_1mm','recall_1mm'):
                arms[name][k]=float(np.mean([r['metrics'][name]['geometry'][k] for r in rr]))
            d=sum(r['back_harm_denominator'] for r in rr)
            g=sum(r['metrics']['test_projection']['groups']['big']['gain_sum'] for r in rr)
            arms[name]['back_harm_ratio']=sum(r['metrics'][name]['groups']['twosheet_back']['harm_sum'] for r in rr)/d if d else None
            arms[name]['big_gain_retained']=sum(r['metrics'][name]['groups']['big']['gain_sum'] for r in rr)/g if g else None
        primary=arms['anchor1_projection']
        checks=dict(noninferior=primary['delta_mae']<=0,
                    back_damage_tenth=primary['back_harm_ratio']<=.1,
                    retain_big_80=primary['big_gain_retained']>=.8)
        result.append(dict(phase=phase,fbig=fbig,calibration=kind,seeds=[r['seed'] for r in rr],
                           arms=arms,primary_checks=checks))
    gate=all(all(r['primary_checks'].values()) for r in result if r['phase']=='confirmation' and r['calibration']=='exch')
    return dict(groups=result,advance_to_real_candidate=gate)


def main():
    if platform.node()!='liekkas':raise RuntimeError('wrong host')
    dest=ROOT/'results'
    if dest.exists():raise SystemExit('results exists; refusing overwrite')
    before=history_hashes()
    own={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md')}
    write_json(ROOT/'SOURCE_LOCK.json',dict(history=before,implementation=own))
    dest.mkdir()
    test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (dest/'TEST_LOG.txt').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError('tests failed: see TEST_LOG.txt')
    start=time.perf_counter();runs=[]
    for phase,seeds in [('development',(101,102)),('confirmation',(701,702,703,704))]:
        for seed in seeds:
            for fbig in (.01,.10):
                pop,_,_,strata,cuts,props=prepared_population(OLD,seed,fbig)
                feat=curve_features(pop['c'],OLD.TGRID)
                for kind in ('exch','sfm'):
                    runs.append(run_case(dest,phase,seed,fbig,kind,pop,strata,cuts,props,feat))
                del pop,feat
    paired=stress(dest)
    summary=summarize(runs)
    write_json(dest/'RESULTS.json',dict(runs=runs,stress=paired,summary=summary))
    write_json(dest/'SUMMARY.json',summary)
    rows=[]
    for r in runs:
        for arm,block in r['metrics'].items():
            for group,values in block['groups'].items():
                rows.append(dict(key=r['key'],phase=r['phase'],seed=r['seed'],fbig=r['fbig'],calibration=r['calibration'],arm=arm,group=group,**values))
    with (dest/'METRICS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    after=history_hashes()
    changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
    own_changed=[p for p,h in own.items() if sha(ROOT/p)!=h]
    write_json(dest/'VERIFICATION.json',dict(historical_files=len(before),historical_changed=changed,
        implementation_changed=own_changed,tests_returncode=test.returncode,development_exact_replays=8,
        main_configurations=len(runs),same_evidence_pairs=len(paired),
        wall_seconds=time.perf_counter()-start,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=sys.version,numpy=np.__version__))
    assert not changed and not own_changed
    print('DONE',len(runs),'configurations; advance=',summary['advance_to_real_candidate'],flush=True)


if __name__=='__main__':main()


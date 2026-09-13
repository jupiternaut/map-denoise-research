"""Frozen-development challenge, exposed recheck, then same-family new-seed confirmation."""
from pathlib import Path
import sys, json, tempfile, time, traceback, socket, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'exploration_v14'))
import experiment as v14
from spatial_filter import filter_frozen as legacy
from constant_solver import filter_frozen as constant

OLD = v14.RUNS/'effect-v14-taip2cld'
FIXED = ['identity', 'bias_only', 'old_original', 'apss_4.0', 'rimls_8.0', 'spatial_36', 'constant_36']
SEARCH = [f'{kind}_{n}' for kind in ('constant', 'spatial') for n in (36,108,324)] + ['constant_lbfgs64']
SEARCH += [f'rimls_{s:.1f}' for s in (.5,1,2,4,8,16,32,64,128)]
DEV_METHODS = list(dict.fromkeys(FIXED + SEARCH))
FIELDS = ['phase','case','seed','gap','sigma','bias','retain','n','method','status','seconds','reused']


def apply(state, sigma, method):
    if 'design' not in state:
        return state['world'].copy(), dict(status='UNSUPPORTED'), np.zeros(len(state['world']),bool)
    if method == 'constant_lbfgs64':
        return constant(state, sigma)
    if method.startswith('constant_'):
        out, info, art = legacy(state, sigma, 'global_free', int(method.split('_')[1]))
        return out, info, art['support_mask']
    return v14.apply(state, sigma, method)


def worker(job, methods, dest, prior):
    inp = v14.load(job['input'])
    state = None
    records = []
    for method in methods:
        r = dict(**job, method=method, input_sha256=v14.sha(job['input']), reused=False)
        start = time.perf_counter()
        if method in prior:
            old = prior[method]
            assert old['input_sha256'] == r['input_sha256']
            if old['status'] != 'FAILED':
                assert v14.sha(old['output']) == old['output_sha256']
            r.update({k:old[k] for k in ('output','output_sha256','support','status','info','seconds','error') if k in old})
            r.update(reused=True, reused_from=str(OLD), wall_this_call_seconds=time.perf_counter()-start)
        else:
            try:
                if state is None:
                    state = v14.v7.freeze(inp['xyz_world'], inp['scan_id'], job['sigma'])
                out, info, mask = apply(state, job['sigma'], method)
                assert out.shape == inp['xyz_world'].shape and np.isfinite(out).all()
                np.testing.assert_array_equal(out[~mask], inp['xyz_world'][~mask])
                path = Path(dest)/'outputs'/f'{job["phase"]}__{job["case"]}__{method}.npz'
                with path.open('xb') as f:
                    np.savez_compressed(f, xyz_world=out, support_mask=mask)
                r.update(output=str(path), output_sha256=v14.sha(path), support=float(mask.mean()),
                         status=info.get('status','OK'), info=info)
            except Exception:
                r.update(status='FAILED', error=traceback.format_exc())
            r['seconds'] = time.perf_counter()-start
        records.append(r)
    v14.save(Path(dest)/'records'/f'{job["phase"]}__{job["case"]}.json', records)
    return records


def structural(out, e):
    z = out[:,2]*1000
    labels = e['gt_layer']
    if int(e['n_true_layers']) == 1:
        return dict(single_normal_span_90_10_mm=float(np.quantile(z,.9)-np.quantile(z,.1)))
    gap = float(e['true_gap_mm'])
    observed = float(z[labels==1].mean()-z[labels==0].mean())
    centers = [float(np.mean(e['gt_clean_xyz_world'][labels==k,2])*1000) for k in (0,1)]
    mid = np.mean(centers)
    crossing = .5*(np.mean(z[labels==0] >= mid) + np.mean(z[labels==1] <= mid))
    return dict(gap_absolute_relative_error=abs(observed/gap-1),
                severe_gap_contraction=float(observed/gap < .5),
                balanced_source_midplane_crossing=float(crossing))


def score(records):
    rows = []
    for r in records:
        row = {k:r[k] for k in FIELDS}
        if r['status'] != 'FAILED':
            q = v14.load(r['output'])['xyz_world']
            e = v14.load(r['evaluation'])
            e.update(json.loads(e.pop('json').tobytes().decode()))
            row.update(v14.synthetic_geometry(q, r.get('info',{}), e))
            row.update(v14.layer_scores(q,e))
            row.update(structural(q,e))
            row['support'] = r['support']
            info = r.get('info',{})
            for key in ('score','initial_two_plane_nll','final_two_plane_nll','k'):
                if key in info:
                    row['fit_'+key] = info[key]
        rows.append(row)
    return rows


def aggregate(rows):
    result = v14.aggregate(rows)
    keys = ['source_group_gap_error_mm','gap_absolute_relative_error','severe_gap_contraction',
            'balanced_source_midplane_crossing','single_normal_span_90_10_mm']
    for method, a in result.items():
        for key in keys:
            values = [r[key] for r in rows if r['method']==method and r.get(key) is not None]
            if values:
                a[key] = float(np.mean(values))
                a[key+'_n'] = len(values)
        a['reused_records'] = sum(r['method']==method and r['reused'] for r in rows)
    return result


def old_jobs(phase):
    source_phase = 'development' if phase == 'development' else 'confirmation'
    jobs = json.loads((OLD/f'{source_phase}_JOBS.json').read_text())
    records = json.loads((OLD/f'{source_phase}_SEALED_BEFORE_GT.json').read_text())
    lookup = {}
    for r in records:
        lookup.setdefault(r['case'],{})[r['method']] = r
    return [dict(j,phase=phase) for j in jobs], lookup


def phase(dest, name, methods, workers=4):
    if name == 'confirmation':
        jobs = v14.jobs_for(dest, name, (9151101,9151111,9151121))
        lookup = {}
    else:
        jobs, lookup = old_jobs(name)
    v14.save(dest/f'{name}_JOBS.json', jobs)
    v14.save(dest/f'{name}_METHODS.json', methods)
    records = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker,j,methods,str(dest),lookup.get(j['case'],{})):j for j in jobs}
        for i,future in enumerate(as_completed(futures),1):
            batch = future.result()
            records.extend(batch)
            print(name, i, '/', len(jobs), 'sealed', futures[future]['case'],
                  'failed', sum(r['status']=='FAILED' for r in batch), flush=True)
    records.sort(key=lambda r:(r['case'],methods.index(r['method'])))
    v14.save(dest/f'{name}_SEALED_BEFORE_GT.json', records)
    rows = score(records)
    v14.save(dest/f'{name}_ROWS.json', rows)
    v14.csvsave(dest/f'{name}_RESULTS.csv', rows)
    agg = aggregate(rows)
    v14.save(dest/f'{name}_AGGREGATES.json', agg)
    v14.save(dest/f'{name}_TIMING.json', dict(wall_seconds=time.perf_counter()-started, workers=workers,
               timing_note='parallel phase including scoring; cached records retain original timings; not a speed comparison'))
    return rows, agg


def choose(agg):
    selected, structural_choice = {}, {}
    key = lambda m:(round(agg[m]['surface_accuracy_mean_mm'],12),agg[m]['matched_point_rms_mm'],m)
    for family in ('constant_','spatial_','rimls_'):
        eligible = [m for m in agg if m.startswith(family) and agg[m]['failed']==0]
        if not eligible:
            raise RuntimeError(f'No valid method: {family}')
        selected[family] = min(eligible,key=key)
        good = [m for m in eligible if agg[m]['gap_absolute_relative_error'] <= .10
                and agg[m]['severe_gap_contraction'] == 0]
        structural_choice[family] = min(good,key=key) if good else None
    return dict(primary=selected, structural=structural_choice,
                rule='global development MAE; structural: mean abs relative gap error <= .10 and no <50% gap cases')


def main():
    assert socket.gethostname() == 'liekkas'
    dest = Path(tempfile.mkdtemp(prefix='challenge-v15-',dir=v14.RUNS))
    print('RUN', dest, flush=True)
    for sub in ('source','outputs','records'):
        (dest/sub).mkdir()
    sources = {str(p):v14.sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md')}
    v14.save(dest/'SOURCES_BEFORE.json',sources)
    for name in sources:
        target = dest/'source'/Path(name).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target)
    _, development = phase(dest,'development',DEV_METHODS)
    lock = choose(development)
    v14.save(dest/'SELECTION_LOCK.json',lock)
    print('LOCKED', json.dumps(lock),flush=True)
    methods = list(dict.fromkeys(FIXED+list(lock['primary'].values())+[m for m in lock['structural'].values() if m]))
    # Freeze before either the exposed old confirmation or new-seed confirmation is scored.
    phase(dest,'exposed_recheck',methods)
    _, confirm = phase(dest,'confirmation',methods)
    assert all(v14.sha(p)==h for p,h in sources.items())
    v14.save(dest/'SUMMARY.json',dict(status='COMPLETE',source_count=len(sources),methods=methods,
        old_confirmation_scope='exposed recheck, not new evidence',
        confirmation_scope='three new seeds, same generator family, supplied sigma',
        new_real_geometry=False))
    print('COMPLETE',dest,flush=True)
    print(json.dumps(confirm,indent=2),flush=True)


if __name__ == '__main__':
    main()

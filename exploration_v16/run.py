"""V16 input-sealed paired mechanism experiment and exposed V15 trace replay."""
from pathlib import Path
import sys, json, tempfile, time, traceback, socket, shutil, resource, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'exploration_v14')]
import experiment as v14
from models import METHODS, fit
from generator import make, SEEDS
from metrics import score_new, score_replay, dependence_diagnostics

OLD = v14.RUNS/'challenge-v15-tchfksi_'


def npzsave(path, data):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **data)


def model_artifact(m):
    return {k:np.asarray(m[k]) for k in ('k','means','slope','groups','gate','posterior','prediction')}


def new_jobs(dest):
    jobs = []
    for seed in SEEDS:
        for gap in (0., 2., 8.):
            for sigma in (1., 2.):
                for lam in ((0.,) if gap == 0 else (0., .5, 1.)):
                    case = f's{seed}_g{gap:g}_n{sigma:g}_l{lam:g}'
                    inp, truth = make(seed, gap, sigma, lam)
                    ip, ep = dest/'inputs'/f'{case}.npz', dest/'evaluation'/f'{case}.npz'
                    npzsave(ip, inp)
                    npzsave(ep, truth)
                    jobs.append(dict(case=case, seed=seed, gap=gap, sigma=sigma, dependence=lam, n=768,
                                     input=str(ip), evaluation=str(ep), input_sha256=v14.sha(ip), evaluation_sha256=v14.sha(ep)))
    return jobs


def worker(job, dest):
    inp = v14.load(job['input'])
    records = []
    for method in METHODS:
        started = time.perf_counter()
        r = dict(**job, method=method)
        try:
            # This constant is the explicit known-slope intervention; evaluation files are never loaded here.
            kwargs = dict(oracle_slope=np.zeros(2)) if method.endswith('oracle_slope') else {}
            m = fit(inp['design'], inp['height_mm'], float(inp['sigma_mm']), method, **kwargs)
            out = inp['xyz_world'].copy()
            out[:,2] = m['prediction']/1000.
            art = model_artifact(m)
            art['xyz_world'] = out
            path = Path(dest)/'outputs'/f'{job["case"]}__{method}.npz'
            npzsave(path, art)
            info = {k:v for k,v in m.items() if not isinstance(v, np.ndarray)}
            r.update(output=str(path), output_sha256=v14.sha(path), status='OK', info=info)
        except Exception:
            r.update(status='FAILED', error=traceback.format_exc())
        r.update(seconds=time.perf_counter()-started, worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(r)
    v14.save(Path(dest)/'records'/f'{job["case"]}.json', records)
    return records


def replay_worker(job, dest):
    inp = v14.load(job['input'])
    state = v14.v7.freeze(inp['xyz_world'], inp['scan_id'], job['sigma'])
    records = []
    for method, oldmethod in (('constant_free','constant_lbfgs64'),('spatial_free','spatial_36')):
        old = job['prior'][oldmethod]
        r = {k:v for k,v in job.items() if k != 'prior'}
        r.update(method=method, phase='v15_exposed_replay', previous_output=old['output'], previous_output_sha256=old['output_sha256'])
        started = time.perf_counter()
        try:
            m = fit(state['design'], state['corrected'], job['sigma'], method)
            take = np.flatnonzero(state['support'])
            rows = state['order'][take]
            out = state['world'].copy()
            displacement = m['prediction'][take]-state['local'][take,2]
            if method == 'spatial_free':
                out[rows] += displacement[:,None]*state['normal']/1000.
            else:
                out[rows] += (displacement/1000.)[:,None]*state['normal']
            previous = v14.load(old['output'])['xyz_world']
            assert v14.sha(old['output']) == old['output_sha256']
            delta = float(np.max(abs(out-previous))*1000.)
            # Same functions, operations and ordered inputs, expected bitwise equality.
            np.testing.assert_array_equal(out, previous)
            art = model_artifact(m)
            art.update(xyz_world=out, **{k:state[k] for k in ('order','basis','center','normal','design','local','corrected','support','common_scale')})
            path = Path(dest)/'replay'/f'{job["case"]}__{method}.npz'
            npzsave(path, art)
            r.update(output=str(path), output_sha256=v14.sha(path), status='OK', replay_max_difference_mm=delta,
                     info={k:v for k,v in m.items() if not isinstance(v, np.ndarray)})
        except Exception:
            r.update(status='FAILED', error=traceback.format_exc())
        r['seconds'] = time.perf_counter()-started
        r['worker_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        records.append(r)
    v14.save(Path(dest)/'records'/f'replay__{job["case"]}.json', records)
    return records


def run_pool(jobs, worker_fn, dest, name, workers):
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_fn, job, str(dest)):job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            batch = future.result()
            records.extend(batch)
            if i % 10 == 0 or i == len(jobs):
                print(name, i, '/', len(jobs), 'failed', sum(r['status']=='FAILED' for r in records), flush=True)
    records.sort(key=lambda r:(r['case'], r['method']))
    v14.save(dest/f'{name}_SEALED_BEFORE_GT.json', records)
    v14.save(dest/f'{name}_TIMING.json', dict(wall_seconds=time.perf_counter()-started, workers=workers,
               parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               maximum_worker_peak_rss_kib=max(r['worker_peak_rss_kib'] for r in records),
               memory_note='parent and maximum single-worker lifetime high-water RSS; not aggregate simultaneous peak',
               scope='phase fit/save wall clock, excludes scoring/audit/code authoring; no GPU'))
    return records


def score_all(dest, records, replay=False):
    rows = []
    for r in records:
        row = {k:r[k] for k in ('case','seed','gap','sigma','method','status','seconds')}
        if 'dependence' in r:
            row['dependence'] = r['dependence']
        if r['status'] == 'OK':
            inp, truth, art = map(v14.load, (r['input'],r['evaluation'],r['output']))
            if replay:
                truth.update(json.loads(truth.pop('json').tobytes().decode()))
                metrics, extra = score_replay(inp, truth, art, art, art['xyz_world'])
                row.update(metrics, replay_max_difference_mm=r['replay_max_difference_mm'])
                if extra:
                    npzsave(dest/'diagnostics'/f'{r["case"]}__{r["method"]}.npz', extra)
            else:
                row.update(score_new(inp, truth, art))
                row.update(dependence_diagnostics(inp, truth))
        rows.append(row)
    name = 'replay' if replay else 'mechanism'
    v14.save(dest/f'{name}_ROWS.json', rows)
    v14.csvsave(dest/f'{name}_RESULTS.csv', rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assert socket.gethostname() == 'liekkas'
    dest = Path(tempfile.mkdtemp(prefix='mechanism-v16-', dir=v14.RUNS))
    print('RUN', dest, flush=True)
    for d in ('source','inputs','evaluation','outputs','records','replay','diagnostics'):
        (dest/d).mkdir()
    sources = {str(p):v14.sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md')}
    v14.save(dest/'SOURCES_BEFORE.json', sources)
    for name in sources:
        target = dest/'source'/Path(name).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    old_hashes = {str(p):v14.sha(p) for p in OLD.rglob('*') if p.is_file()}
    v14.save(dest/'V15_HASHES_BEFORE.json', old_hashes)
    jobs = new_jobs(dest)
    v14.save(dest/'JOBS_LOCK.json', jobs)
    v14.save(dest/'METHODS_LOCK.json', dict(methods=METHODS, protocol_sha256=v14.sha(HERE/'PROTOCOL.md'),
                 no_selection_or_tuning=True, oracle_fields=['shared_slope'], normal_and_noise_known=True))
    records = run_pool(jobs, worker, dest, 'mechanism', args.workers)
    score_all(dest, records)
    old_jobs = json.loads((OLD/'confirmation_JOBS.json').read_text())
    old_records = json.loads((OLD/'confirmation_SEALED_BEFORE_GT.json').read_text())
    lookup = {}
    for r in old_records:
        lookup.setdefault(r['case'], {})[r['method']] = r
    replay_jobs = [dict(j, prior=lookup[j['case']]) for j in old_jobs]
    old_replays = run_pool(replay_jobs, replay_worker, dest, 'replay', args.workers)
    score_all(dest, old_replays, replay=True)
    assert all(v14.sha(p) == h for p,h in sources.items())
    assert all(v14.sha(p) == h for p,h in old_hashes.items())
    failures = sum(r['status']=='FAILED' for r in records+old_replays)
    v14.save(dest/'SUMMARY.json', dict(status='COMPLETE' if not failures else 'COMPLETE_WITH_FAILURES',
                 mechanism_inputs=len(jobs), mechanism_outputs=len(records), v15_replay_outputs=len(old_replays),
                 failed=failures, old_source_hashes_unchanged=len(sources), v15_hashes_unchanged=len(old_hashes),
                 scope='10 new seeds, one statistical family; V15 replays exposed; no real-geometry result'))
    print('COMPLETE', dest, 'failed', failures, flush=True)


if __name__ == '__main__':
    main()

"""Frozen V20 diagnostic/confirmation runner; archives fits before GT scoring."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import retained_gate as op

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'exploration_v19'))
import common
OLD = Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/parallel-v19-9ob96cyr')
RUNS = OLD.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    def default(v):
        if isinstance(v, (np.ndarray, np.generic)):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    with Path(path).open('x') as handle:
        json.dump(data, handle, default=default, indent=2, allow_nan=False)


def cases(seed):
    for gap in (0., 2., 8.):
        for sigma in (1., 2.):
            for ratio in ((.5,) if gap == 0 else (.2, .5, .8)):
                for dep in ((0.,) if gap == 0 else (0., 1.)):
                    yield (seed, gap, sigma, ratio, dep)


def prepare(dest, phase):
    params = list(cases(9190101)) if phase == 'exposed' else list(cases(9201101)) + list(cases(9201111))
    if phase == 'exposed':
        params += [(s, 8., 1., .5, 1.) for s in (9191101, 9191111, 9191121, 9191131)]
    jobs = []
    for seed, gap, sigma, ratio, dep in params:
        case = f's{seed}_g{gap:g}_n{sigma:g}_p{ratio:g}_l{dep:g}'
        inp, truth = common.make(seed, gap, sigma, ratio, dep)
        ip, ep = dest/'inputs'/f'{case}.npz', dest/'evaluation'/f'{case}.npz'
        with ip.open('xb') as handle:
            np.savez_compressed(handle, **inp)
        with ep.open('xb') as handle:
            np.savez_compressed(handle, **truth)
        jobs.append(dict(case=case, seed=seed, gap=gap, sigma=sigma, supplied_sigma=sigma,
                         ratio=ratio, dependence=dep, kind='statistical', phase=phase,
                         input=str(ip), evaluation=str(ep), input_sha256=sha(ip), evaluation_sha256=sha(ep)))
    save(dest/f'{phase}_JOBS.json', jobs)
    return jobs


def work(job, dest):
    dest = Path(dest)
    start = time.perf_counter()
    try:
        assert sha(job['input']) == job['input_sha256']
        with np.load(job['input'], allow_pickle=False) as inp:
            x, y = inp['design'], inp['height_mm']
        sigma = job['sigma']
        _, full = op.v18.construct(x, y, sigma)
        baseline = op.v18.arrays(full['retained'])
        cache = dest/'cache'/f'{job["case"]}.json'
        save(cache, dict(x=x, y=y, sigma=sigma, baseline=baseline, state=None))
        job = dict(job, cache=str(cache), cache_sha256=sha(cache))
        save(dest/'diagnostics'/f'{job["case"]}__full_pool.json', full)
        outputs = dict(v18_map=baseline)
        outputs.update({'old_'+name: model for name, model in op.old.run(x, y, sigma, baseline).items()})
        repaired, diagnostics = op.run(x, y, sigma, baseline, full)
        outputs.update(repaired)
        save(dest/'diagnostics'/f'{job["case"]}__repaired_gate.json', diagnostics)
        candidates = {'pool_'+key: value for key, value in op.family_pool(full, x, y, sigma).items()}
        candidates['legacy_spatial'] = op.v18.arrays(full['old'])
        candidates['retained'] = baseline
        outputs.update({'candidate_'+key: op.v18.project(model, x, 1.) for key, model in candidates.items()})
        outputs['identity'] = dict(identity=True, k=0)
        replay = {}
        if job['phase'] == 'exposed':
            previous = json.loads((OLD/'cache'/f'{job["case"]}.json').read_text())
            for key, value in (('x', x), ('y', y)):
                np.testing.assert_array_equal(previous[key], value)
            np.testing.assert_allclose(previous['baseline']['prediction'], baseline['prediction'], atol=1e-8, rtol=0)
            replay['v18_max_difference_mm'] = float(np.max(abs(np.array(previous['baseline']['prediction'])-baseline['prediction'])))
            for old_name in ('insample_gate_map', 'crossfit_gate_map'):
                prev = json.loads((OLD/'outputs'/f'{job["case"]}__a_{old_name}.json').read_text())
                np.testing.assert_allclose(prev['prediction'], outputs['old_'+old_name]['prediction'], atol=1e-8, rtol=0)
        records = []
        for name, model in outputs.items():
            path = dest/'outputs'/f'{job["case"]}__{name}.json'
            save(path, model)
            records.append(dict(method=name, output=str(path), sha256=sha(path)))
        result = dict(job=job, status='OK', records=records, replay=replay, shared_seconds=time.perf_counter()-start)
    except Exception:
        result = dict(job=job, status='FAILED', error=traceback.format_exc(), shared_seconds=time.perf_counter()-start)
    save(dest/'records'/f'{job["case"]}.json', result)
    return result


def score_phase(dest, phase, results):
    rows, choices = [], []
    for result in results:
        job = result['job']
        if result['status'] != 'OK':
            rows.append(dict(case=job['case'], phase=phase, status='FAILED', error=result['error']))
            continue
        assert sha(job['evaluation']) == job['evaluation_sha256']
        local = []
        for rec in result['records']:
            assert sha(rec['output']) == rec['sha256']
            model = json.loads(Path(rec['output']).read_text())
            row = {k:job[k] for k in ('case','phase','seed','gap','sigma','ratio','dependence')}
            row.update(method=rec['method'], status='OK', **common.score(job, model))
            rows.append(row)
            local.append(row)
        candidate = min((r for r in local if r['method'].startswith('candidate_')),
                        key=lambda r:(r['balanced_source_mae_mm'], r['method']))
        default = next(r for r in local if r['method'] == 'v18_map')
        choices.append(dict(case=job['case'], seed=job['seed'], gap=job['gap'],
                            oracle_method=candidate['method'], default=default, oracle=candidate,
                            headroom_source_mae_mm=default['balanced_source_mae_mm']-candidate['balanced_source_mae_mm']))
    save(dest/f'{phase}_ROWS.json', rows)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (dest/f'{phase}_RESULTS.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    save(dest/f'{phase}_ORACLE_DIAGNOSTIC.json', choices)
    return rows, choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assert socket.gethostname() == 'liekkas'
    dest = Path(tempfile.mkdtemp(prefix='candidate-repair-v20-', dir=RUNS))
    for name in ('inputs', 'evaluation', 'cache', 'outputs', 'records', 'diagnostics', 'source'):
        (dest/name).mkdir()
    tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
    protected = [ROOT/p for p in tracked if p] + [p for p in (ROOT/'exploration_v19').rglob('*') if p.is_file()]
    before = {str(p):sha(p) for p in protected}
    save(dest/'OLD_FILES_BEFORE.json', before)
    sources = list(HERE.glob('*.py')) + [HERE/'PROTOCOL.md']
    sources += [p for i in (11,14,15,16,17,18,19) for p in (ROOT/f'exploration_v{i}').rglob('*.py')]
    lock = {str(p):sha(p) for p in sources}
    save(dest/'SOURCE_LOCK.json', lock)
    for p in sources:
        target = dest/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    save(dest/'IDENTITY.json', dict(host=socket.gethostname(), project=str(ROOT), python=sys.executable,
                                  workers=args.workers, run=str(dest), gpu=False))
    print('RUN', dest, flush=True)
    summary = {}
    for phase in ('exposed', 'confirmation'):
        assert all(sha(p)==h for p,h in lock.items())
        jobs = prepare(dest, phase)
        start = time.perf_counter()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(work, job, str(dest)) for job in jobs]):
                results.append(future.result())
                print(phase, len(results), '/', len(jobs), 'failed', sum(r['status']!='OK' for r in results), flush=True)
        results.sort(key=lambda r:r['job']['case'])
        save(dest/f'{phase}_SEALED_BEFORE_GT.json', results)
        rows, choices = score_phase(dest, phase, results)
        summary[phase] = dict(cases=len(jobs), failed=sum(r['status']!='OK' for r in results),
                              outputs=len(rows), wall_seconds=time.perf_counter()-start,
                              mean_oracle_headroom_mm=float(np.mean([c['headroom_source_mae_mm'] for c in choices])))
        print('SUMMARY', phase, summary[phase], flush=True)
    assert all(sha(p)==h for p,h in lock.items())
    assert all(sha(p)==h for p,h in before.items())
    summary['old_files_unchanged'] = len(before)
    save(dest/'SUMMARY.json', summary)
    print('DONE', dest, flush=True)


if __name__ == '__main__':
    main()

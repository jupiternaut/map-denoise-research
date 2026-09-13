"""Post-hoc scope correction: score archived raw R without fitting anything."""
import sys
import json
from pathlib import Path
import numpy as np
import run as runner


def main():
    dest = Path(sys.argv[1])
    out = dest / 'posthoc_raw_R'
    out.mkdir()
    records = []
    # All added predictions are sealed before accessing evaluation arrays.
    for phase in ('exposed', 'confirmation'):
        for bundle in json.loads((dest/f'{phase}_SEALED_BEFORE_GT.json').read_text()):
            job = bundle['job']
            full_path = dest/'diagnostics'/f'{job["case"]}__full_pool.json'
            full = json.loads(full_path.read_text())
            cache = json.loads(Path(job['cache']).read_text())
            model = runner.op.v18.project(runner.op.v18.arrays(full['pool']['R']), np.asarray(cache['x']), 1.)
            path = out/f'{job["case"]}.json'
            runner.save(path, model)
            records.append(dict(job=job, output=str(path), sha256=runner.sha(path),
                                source=str(full_path), source_sha256=runner.sha(full_path)))
    runner.save(out/'SEALED_ADDED_PREDICTIONS.json', records)
    all_choices, summaries = [], {}
    for phase in ('exposed', 'confirmation'):
        rows = json.loads((dest/f'{phase}_ROWS.json').read_text())
        choices = []
        for rec in records:
            job = rec['job']
            if job['phase'] != phase:
                continue
            assert runner.sha(rec['output']) == rec['sha256']
            assert runner.sha(rec['source']) == rec['source_sha256']
            assert runner.sha(job['evaluation']) == job['evaluation_sha256']
            model = json.loads(Path(rec['output']).read_text())
            raw = dict(case=job['case'], method='candidate_raw_R', **runner.common.score(job, model))
            pool = [r for r in rows if r['case']==job['case'] and r['method'].startswith('candidate_')]
            before = min(pool, key=lambda r:(r['balanced_source_mae_mm'], r['method']))
            winner = min(pool+[raw], key=lambda r:(r['balanced_source_mae_mm'], r['method']))
            default = next(r for r in rows if r['case']==job['case'] and r['method']=='v18_map')
            choices.append(dict(case=job['case'], gap=job['gap'], seed=job['seed'], phase=phase,
                                raw_R=raw, default=default, oracle=winner,
                                extra_gain_mm=before['balanced_source_mae_mm']-winner['balanced_source_mae_mm'],
                                headroom_mm=default['balanced_source_mae_mm']-winner['balanced_source_mae_mm']))
        summaries[phase] = {}
        for name, cc in [('all',choices),('dual',[c for c in choices if c['gap']>0])]:
            summaries[phase][name] = dict(n=len(cc), G_mm=float(np.mean([c['headroom_mm'] for c in cc])),
                raw_R_improved_cases=sum(c['extra_gain_mm']>1e-10 for c in cc),
                **{k:float(np.mean([c['oracle'][k] for c in cc])) for k in
                   ('balanced_source_mae_mm','surface_mae_mm','matched_rms_mm','coverage_1mm')})
        all_choices.extend(choices)
    runner.save(out/'COMPLETE_ARCHIVED_POOL_ORACLE.json', dict(
        scope='Raw S/C/L/R, legacy spatial, retained baseline; not all optimizer starts or all possible models.',
        posthoc=True, refitted=False, added_outputs=len(records), summaries=summaries, choices=all_choices))
    print(json.dumps(summaries, indent=2))


if __name__=='__main__':
    main()

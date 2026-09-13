"""Resume the sealed first V16 run after a dict.update reporting-only error.

The original run.py is already archived in source/. Do not repeat or select model fits.
"""
from pathlib import Path
import json, sys, difflib
import run as runner


def main(path):
    dest=Path(path)
    sources=json.loads((dest/'SOURCES_BEFORE.json').read_text())
    changed=[]
    for p,h in sources.items():
        current=runner.v14.sha(p)
        if current!=h:
            old=dest/'source'/Path(p).relative_to(runner.ROOT)
            before=old.read_text();after=Path(p).read_text()
            expected=before.replace('row.update(score_new(inp, truth, art), dependence_diagnostics(inp, truth))',
                                    'row.update(score_new(inp, truth, art))\n                row.update(dependence_diagnostics(inp, truth))')
            assert p==str(runner.HERE/'run.py') and expected==after
            changed.append(dict(path=p,before=h,after=current,scope='dict.update reporting-only fix, after 560 model outputs sealed, before any scoring completed',
                                diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True)))))
    runner.v14.save(dest/'SCORING_REPAIR.json',dict(changes=changed,model_refits=0,method_changes=0))
    records=json.loads((dest/'mechanism_SEALED_BEFORE_GT.json').read_text())
    assert len(records)==560 and all(r['status']=='OK' for r in records)
    runner.score_all(dest,records)
    old_jobs=json.loads((runner.OLD/'confirmation_JOBS.json').read_text())
    old_records=json.loads((runner.OLD/'confirmation_SEALED_BEFORE_GT.json').read_text())
    lookup={}
    for r in old_records:lookup.setdefault(r['case'],{})[r['method']]=r
    jobs=[dict(j,prior=lookup[j['case']]) for j in old_jobs]
    replays=runner.run_pool(jobs,runner.replay_worker,dest,'replay',4)
    runner.score_all(dest,replays,replay=True)
    expected=dict(sources)
    for r in changed:expected[r['path']]=r['after']
    assert all(runner.v14.sha(p)==h for p,h in expected.items())
    old_hashes=json.loads((dest/'V15_HASHES_BEFORE.json').read_text())
    assert all(runner.v14.sha(p)==h for p,h in old_hashes.items())
    failures=sum(r['status']=='FAILED' for r in records+replays)
    runner.v14.save(dest/'SUMMARY.json',dict(status='COMPLETE' if not failures else 'COMPLETE_WITH_FAILURES',
        mechanism_inputs=140,mechanism_outputs=len(records),v15_replay_outputs=len(replays),failed=failures,
        source_files=len(sources),source_changes=changed,v15_hashes_unchanged=len(old_hashes),
        scope='10 new seeds, conditional statistical family; exposed V15 replays; no independent real geometry'))
    print('COMPLETE',dest,'failed',failures,flush=True)


if __name__=='__main__':main(sys.argv[1])

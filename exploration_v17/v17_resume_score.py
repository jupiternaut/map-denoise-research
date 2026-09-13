"""Resume sealed bridge scoring after dict.update API repair; never refits."""
import sys,json
from pathlib import Path
import v17_run as run

dest=Path(sys.argv[1])
original=dest/'source/exploration_v17/v17_run.py'
current=Path(run.__file__)
before=original.read_text();after=current.read_text()
assert before.replace('row.update(geometry,layer)','row.update(geometry);row.update(layer)')==after
records=json.loads((dest/'bridge_SEALED_BEFORE_GT.json').read_text())
for r in records:assert run.v14.sha(r['output'])==r['output_sha256']
run.v14.save(dest/'SCORING_REPAIR.json',dict(error='TypeError: update expected at most 1 argument, got 2',
    original_source=str(original),original_sha256=run.v14.sha(original),corrected_source=str(current),corrected_sha256=run.v14.sha(current),
    change='row.update(geometry,layer) -> row.update(geometry);row.update(layer)',
    predictions_unchanged=len(records),estimator_changed=False,refits=0))
rows=run.score(dest,'bridge',records)
print('SCORED',len(rows),'sealed outputs; no refitting')

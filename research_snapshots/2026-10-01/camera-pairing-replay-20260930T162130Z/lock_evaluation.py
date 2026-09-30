"""Freeze the evaluation definition before the first new GT evaluation."""
from pathlib import Path
import hashlib,json,time
ROOT=Path(__file__).resolve().parent
names=['AGENTS.md','refine-logs/EXPERIMENT_PLAN.md','INITIALIZER_PROTOCOL.md',
       'evaluate_replay.py','PREDICTIONS_SEALED.json','lock_evaluation.py']
record=dict(files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
            locked_at_unix=time.time(),before_new_GT_evaluation=not (ROOT/'evaluation').exists())
assert record['before_new_GT_evaluation']
with (ROOT/'EVALUATION_PROTOCOL_LOCK.json').open('x') as f:json.dump(record,f,indent=2)

"""One-time pre-run code correction: account only for own CUDA PID, not all Python."""
from run_colmap import ROOT,load,sha,dump
old=ROOT/'RUN_LOCK.json'
data=load(old)
assert not (ROOT/'job_records').exists()
before=data['files'][str(ROOT/'run_colmap.py')]
old.rename(ROOT/'RUN_LOCK_PRE_GPU_GATE.json')
data['files'][str(ROOT/'run_colmap.py')]=sha(ROOT/'run_colmap.py')
data['files'][str(ROOT/'evaluate_mvs.py')]=sha(ROOT/'evaluate_mvs.py')
data['pre_run_correction']={'old_script_hash':before,'new_script_hash':sha(ROOT/'run_colmap.py'),
    'reason':'GPU availability recognizes own PID, never exempts unrelated Python jobs','before_any_mvs':True}
dump(old,data)

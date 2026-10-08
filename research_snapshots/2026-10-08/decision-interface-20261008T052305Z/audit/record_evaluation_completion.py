"""Record sealed evaluation completion after a stdout-only serialization error.

Does not evaluate, alter source, recompute predictions, or overwrite any output.
The original failed receipt remains intact and is linked from the new receipt.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
def load(p):return json.loads(p.read_text())
def main(stage):
    failed=ROOT/'phase_events'/f'evaluate-{stage}_failed.json'
    failure=load(failed)
    assert failure['error']=="TypeError('Object of type bool is not JSON serializable')",failure
    d=ROOT/stage/'evaluation';seal=load(d/'SEAL.json')
    for p,h in seal['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    for name in ('ROWS.json','SUMMARY.json','GATE.json','RESULTS.csv'):assert str(d/name) in seal['files']
    assert load(d/'ROWS.json') and load(d/'SUMMARY.json')['overall']
    if stage=='confirmation':assert str(d/'PAIRED_BOOTSTRAP.json') in seal['files']
    out=ROOT/'phase_events'/f'evaluate-{stage}_artifacts_completed.json'
    with out.open('x') as f:json.dump(dict(stage=stage,status='artifacts_complete_stdout_error',original_failure=str(failed),sealed_files=len(seal['files']),finished=datetime.now(timezone.utc).isoformat(),scope='Existence and hash receipt; independent metrics checked by verify_results.py; frozen source unchanged'),f,indent=2)
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['replay','confirmation']);main(p.parse_args().stage)

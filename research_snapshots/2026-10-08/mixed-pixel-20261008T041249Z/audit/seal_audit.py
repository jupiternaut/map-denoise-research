"""Finalize audit artifacts after checking all declared audited input hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket

ROOT=Path('/srv/slam-research/grf/map-denoise/runs/mixed-pixel-20261008T041249Z')
assert socket.gethostname()=='liekkas'
assert Path(__file__).resolve().parent==ROOT/'audit'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
audit=json.loads((ROOT/'audit/EXPERIMENT_AUDIT.json').read_text())
for relative,expected in audit['audited_input_hashes'].items():
    assert sha(ROOT/relative)==expected.removeprefix('sha256:'),relative
assert (ROOT/audit['full_response_path']).is_file()
assert audit['deterministic_verification']['main_checks']+audit['deterministic_verification']['diagnostic_checks']==11307
files={str(p):sha(p) for p in sorted((ROOT/'audit').iterdir()) if p.is_file() and p.name!='SEAL.json'}
with (ROOT/'audit/SEAL.json').open('x') as stream:
    json.dump(dict(created=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),
        verdict=audit['verdict'],audited_declared_hashes_match=True,files=files),stream,indent=2)
print(json.dumps(dict(audit_verdict=audit['verdict'],audited_hashes=len(audit['audited_input_hashes']),sealed_audit_files=len(files))))

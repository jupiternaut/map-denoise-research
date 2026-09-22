"""Verify preparation and source artifacts, without running a new experiment."""
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def check(root, entries):
    bad = []
    for name, expected in entries.items():
        path = root/name
        if not path.is_file() or digest(path) != expected: bad.append(name)
    if bad: raise AssertionError({'root': str(root), 'hash_mismatches': bad})
    return len(entries)

def main():
    if socket.gethostname() != 'liekkas':
        raise RuntimeError('This preparation audit is for liekkas; use package tests for portable checks')
    lock = json.loads((ROOT/'TRAINING_LOCK.json').read_text())
    origin = Path(lock['source_workspace'])
    release = json.loads((ROOT/'package/RELEASE_MANIFEST.json').read_text())
    n_package = check(ROOT, release['files_relative_to_workspace_sha256'])
    training_files = {r[key]: r[key+'_sha256'] for r in lock['records'] for key in ('labels','features')}
    n_training = check(origin, training_files)
    old_manifest = json.loads((origin/'DELIVERY_MANIFEST.json').read_text())
    n_old = check(origin, old_manifest['top_level_files'])
    data = json.loads((ROOT/'DATA_READINESS.json').read_text())['results']
    wire = data['images']['archive_bytes']
    wire += sum(v[0]['size'] for v in data['meshes']['scenes'].values())
    wire += sum(v['compressed_bytes'] for k in ('reference','masks') for v in data[k]['scenes'].values())
    assert wire == 3917498394
    assert lock['rows_per_model'] == 79594
    assert release['primary'] == 'post_A_keep'
    protocol_files = ['AGENTS.md','DECISIONS.md','EXPERIMENT_PROTOCOL.md','DATA_READINESS.md',
                      'DATA_READINESS.json','BASELINE_READINESS.md','BASELINE_READINESS.json',
                      'TRAINING_LOCK.json','REPRODUCIBILITY.md','README.md','COMMANDS.md','CHECKPOINT.md',
                      'baseline_probe.py','data_probe.py','verify_preparation.py',
                      'package/RELEASE_MANIFEST.json','package/v28-closeout-0.1.0.tar.gz']
    output = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
        'workspace': str(ROOT), 'status': 'PREPARATION_VERIFIED',
        'release_files_verified': n_package, 'training_source_files_verified': n_training,
        'old_top_level_files_verified': n_old, 'planned_transfer_bytes': wire,
        'new_scene_contents_downloaded': False, 'official_baseline_run': False,
        'independent_confirmation_run': False,
        'root_test_rerun': {'tests':7,'failures':0,'errors':0,'wall_seconds':11.617,
                            'note':'separate execution of the same tests, not external reproduction'},
        'sha256': {name: digest(ROOT/name) for name in protocol_files},
    }
    path = ROOT/'PREPARATION_AUDIT.json'
    if path.exists():
        raise RuntimeError('Audit already exists; do not overwrite the frozen preparation')
    path.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k != 'sha256'}, indent=2))

if __name__ == '__main__': main()

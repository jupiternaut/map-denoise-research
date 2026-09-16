"""Record publication checks without editing any original experiment directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
from check_snapshot import check

ROOT = Path(__file__).resolve().parents[1]
PY = '/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'
ORACLE = ROOT / 'research_snapshots/2026-09-16/pnp-oracle-lab-20260916T112018Z'
JOBS = [
    ('oracle_units', ORACLE, [sys.executable, '-B', '-m', 'unittest', 'discover', '-v']),
    ('oracle_audit', ORACLE, [sys.executable, '-B', '-m', 'unittest', 'audit_checks', '-v']),
] + [(name, ROOT, [PY, '-B', '-m', 'unittest', 'discover', '-s', name, '-p', 'test_*.py', '-v'])
     for name in ('loss_alignment_v23', 'field_budget_v24', 'real_support_bridge_v1')]


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    records = []
    for name, cwd, command in JOBS:
        print('Running ' + name, flush=True)
        proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
        log = ROOT / f'publication/{name}.log'
        log.write_text(proc.stdout + proc.stderr)
        records.append({'name': name, 'cwd': str(cwd.relative_to(ROOT)), 'command': command,
                        'exit_code': proc.returncode, 'log': str(log.relative_to(ROOT))})
        print(proc.stderr[-500:], flush=True)
    receipt = {'date': '2026-09-16', 'snapshot_checks': check(), 'jobs': records,
               'scope': 'Publication integrity and selected unit/audit tests, not a full rerun of all experiments. Geometry tests use the existing local environment; old absolute dependencies remain.',
               'not_run': ['V25 full experiment', 'E0/E0-D/E0-E/E0-F full experiments',
                           'Hermes/open-world full experiments', 'cold-machine portability']}
    receipt['passed'] = receipt['snapshot_checks']['passed'] and all(r['exit_code'] == 0 for r in records)
    (ROOT / 'publication/VALIDATION_20260916.json').write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(not receipt['passed'])


if __name__ == '__main__':
    main()

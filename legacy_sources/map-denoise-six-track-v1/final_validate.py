"""Read-only final checks of saved work; write one new validation record."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = '/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python'
CHECKPOINT = Path('/srv/slam-research/grf/map-denoise/checkpoints/20260911-1404-pre-paper')


def main():
    destination = ROOT / 'FINAL_VALIDATION.json'
    if destination.exists():
        raise FileExistsError(destination)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    checks = [
        ('t1_t2_tests', ROOT/'tracks/t1_t2', [PYTHON, '-B', '-m', 'unittest', '-v', 'test_experiment.py']),
        ('t3_t6_tests', ROOT/'tracks/t3_t6', [PYTHON, '-B', 'test_track.py']),
        ('t4_tests', ROOT, [PYTHON, '-B', '-m', 'unittest', 'discover', '-s', 'tracks/t4', '-p', 'test_*.py']),
        ('t5_saved_outputs', ROOT, [PYTHON, '-B', 'tracks/t5/verify_saved.py']),
        ('old_sources', ROOT, [PYTHON, '-B', 'audit_sources.py', 'check', 'BASELINE_SOURCES.json']),
        ('old_checkpoint', CHECKPOINT, ['sha256sum', '-c', 'SHA256SUMS']),
    ]
    results = []
    for name, cwd, command in checks:
        proc = subprocess.run(command, cwd=cwd, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        results.append({'check': name, 'cwd': str(cwd), 'command': command,
                        'returncode': proc.returncode, 'output': proc.stdout})
        print(name, 'PASS' if proc.returncode == 0 else 'FAIL', flush=True)
    known = {
        'tracks/t3_t6/local_operator.py': 'aa7b204b3de556e8ee7fb036166280aa32bcbcc997e55ac4ba0c9e071ecdc877',
        'tracks/t3_t6/geometry.py': '190f839e5afa447da4b9e4a37713fa48586b17af33bb7eefcde0f243c2fd1945',
        'tracks/t1_t2/experiment.py': '40df63459fe620788acb714f44bd794757f8197542eb03411e685e33ad384f57',
    }
    frozen = {name: {'sha256': hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                     'expected_sha256': digest}
              for name, digest in known.items()}
    for row in frozen.values():
        row['match'] = row['sha256'] == row['expected_sha256']
    passed = all(item['returncode'] == 0 for item in results) and all(x['match'] for x in frozen.values())
    record = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'all_passed': passed, 'checks': results, 'frozen_candidate_sources': frozen}
    with destination.open('x') as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print(json.dumps({'all_passed': passed, 'record': str(destination)}))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

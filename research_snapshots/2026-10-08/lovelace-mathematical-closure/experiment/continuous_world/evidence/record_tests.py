"""Capture developer test results as a reproducible receipt."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / 'continuous_world/outputs/confirmation-v1'


def main():
    command = [sys.executable, '-B', '-m', 'unittest',
               'continuous_world.test_interval_model', 'continuous_world.test_interval_solver', '-v']
    result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, encoding='utf-8')
    log = result.stdout + result.stderr
    (OUTPUT / 'tests.log').write_text(log, encoding='utf-8')
    files = ['continuous_world/test_interval_model.py', 'continuous_world/test_interval_solver.py']
    receipt = {'schema': 1, 'exit_code': result.returncode, 'passed': result.returncode == 0,
               'command': command, 'python': sys.version, 'expected_tests': 33,
               'test_hashes': {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in files},
               'log_sha256': hashlib.sha256((OUTPUT / 'tests.log').read_bytes()).hexdigest(),
               'scope': 'hand-computable and adversarial development checks, separate from 72 confirmation configurations'}
    (OUTPUT / 'TESTS.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(log)
    print(json.dumps(receipt, indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

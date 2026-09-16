"""Final verification and a fresh-output deterministic replay; no primary overwrites."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/grf/.hermes/attachments/outputs/20260914T142434Z/CHECKPOINT.md')
EXPECTED = {
    ROOT / 'research_plan.md': 'c06d34af5ba8637f19565d46cd6d91b36376b608697870f3a5b46516a97842f0',
    ROOT / 'COMPOSITION_PROTOCOL.md': 'c37484eb9d2416ddd43d1a93a8eee19f4c9d5eca4409d81db0f4e3cc55a6b359',
    OLD: '86cea9000349650e4a51c7ffa25ea3e09e9a7354bd00d5bfb932d8343ca1eb1b',
}


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def run(command, name, logs):
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    (logs / (name + '.log')).write_text(result.stdout + '\n' + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{name} failed: {result.stderr}')
    tests = re.search(r'Ran (\d+) tests?', result.stderr)
    return {'command': command, 'exit_code': result.returncode,
            'seconds': time.perf_counter() - start,
            'tests': int(tests.group(1)) if tests else None}


def compare_csv(first, second, ignored=()):
    with first.open(newline='') as a, second.open(newline='') as b:
        left, right = csv.DictReader(a), csv.DictReader(b)
        if left.fieldnames != right.fieldnames:
            raise AssertionError(f'Schema differs: {first.name}')
        count = 0
        while True:
            x, y = next(left, None), next(right, None)
            if x is None or y is None:
                if x != y:
                    raise AssertionError(f'Row counts differ: {first.name}')
                break
            for key in x:
                if key not in ignored and x[key] != y[key]:
                    raise AssertionError(f'{first.name} row{count} field{key} differs')
            count += 1
    return {'rows_equal': count, 'ignored_fields': list(ignored)}


def main():
    if platform.node() != 'liekkas':
        raise RuntimeError('Wrong target host; do not substitute an environment')
    checksums = {str(path): digest(path) for path in EXPECTED}
    if checksums != {str(path): value for path, value in EXPECTED.items()}:
        raise AssertionError('Frozen protocol or old checkpoint changed')
    logs = ROOT / 'verification_logs'
    logs.mkdir(exist_ok=True)
    receipt = {'host': platform.node(), 'workspace': str(ROOT),
               'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
               'frozen_checksums': checksums, 'commands': []}
    receipt['commands'].append(run([sys.executable, '-m', 'unittest', 'discover', '-v'], 'unit_tests', logs))
    receipt['commands'].append(run([sys.executable, 'audit_checks.py'], 'independent_audit', logs))
    parent = ROOT / 'reproduction'
    parent.mkdir(exist_ok=True)
    replay = Path(tempfile.mkdtemp(prefix='fresh-', dir=parent))
    receipt['replay_directory'] = str(replay)
    for name, args in [
        ('A', ['learning.py', '--output-dir', str(replay)]),
        ('B', ['planning.py', '--output', str(replay)]),
        ('C', ['composition.py', '--input', str(replay / 'learning_rows.csv'), '--output', str(replay)]),
        ('D', ['tie_audit.py', '--source', str(replay / 'composition_rows.csv'), '--output', str(replay)]),
    ]:
        receipt['commands'].append(run([sys.executable, *args], 'replay_' + name, logs))
    receipt['reproducibility'] = {}
    for filename, skip in [('learning_rows.csv', ('fit_seconds',)),
                           ('planning_rows.csv', ('runtime_seconds',)),
                           ('planning_traces.csv', ()), ('composition_rows.csv', ()),
                           ('tie_audit_pairs.csv', ())]:
        receipt['reproducibility'][filename] = compare_csv(ROOT / 'results' / filename, replay / filename, skip)
    receipt['passed'] = True
    (ROOT / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    manifest = {}
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name != 'SHA256SUMS.json':
            manifest[str(path.relative_to(ROOT))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    (ROOT / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()


"""Read-only checksum and relocated scientific-evidence check of this snapshot."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'experiment'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    manifest = read(ROOT / 'PUBLICATION_MANIFEST.json')
    failures = []
    for name, record in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            failures.append({'kind': 'publication_file', 'path': name})
    evidence = EXPERIMENT / 'continuous_world/outputs/confirmation-v1'
    algorithm = read(evidence / 'SOURCE_LOCK.json')
    for name, expected in algorithm['hashes'].items():
        if sha(EXPERIMENT / name) != expected or sha(evidence / 'source' / name) != expected:
            failures.append({'kind': 'algorithm_source_lock', 'path': name})
    checker = read(evidence / 'CHECKER_LOCK.json')
    if sha(EXPERIMENT / checker['checker']) != checker['sha256']:
        failures.append({'kind': 'checker_lock'})
    for stage in (1, 2):
        archive = ROOT / f'Lovelace-mathematical-closure-stage{stage}-20261008.zip'
        receipt = read(archive.with_suffix('.receipt.json'))
        if sha(archive) != receipt['sha256'] or archive.stat().st_size != receipt['bytes']:
            failures.append({'kind': 'archive_receipt', 'stage': stage})
        with zipfile.ZipFile(archive) as packed:
            if packed.testzip() is not None:
                failures.append({'kind': 'archive_crc', 'stage': stage})
            for name in packed.namelist():
                # Stage 1's original README remains in that archive; current
                # experiment README only adds navigation to stage 2.
                if stage == 1 and name == 'README.md':
                    continue
                if packed.read(name) != (EXPERIMENT / name).read_bytes():
                    failures.append({'kind': 'archive_workspace_bytes', 'stage': stage, 'path': name})
    scientific = {}
    if not failures:
        tests = subprocess.run([sys.executable, '-B', '-m', 'unittest',
                                'continuous_world.test_interval_model', 'continuous_world.test_interval_solver'],
                               cwd=EXPERIMENT, capture_output=True, text=True, encoding='utf-8')
        scientific['developer_tests'] = {'exit_code': tests.returncode, 'log': tests.stdout + tests.stderr}
        code = ('from pathlib import Path; import json; '
                'from continuous_world.verify_continuous import verify_archive; '
                'r=verify_archive(Path("continuous_world/outputs/confirmation-v1").resolve()); '
                'print(json.dumps({"passed":r["passed"],"failures":r["failures"],"counts":r["counts"]})); '
                'raise SystemExit(0 if r["passed"] else 1)')
        checked = subprocess.run([sys.executable, '-B', '-c', code], cwd=EXPERIMENT,
                                 capture_output=True, text=True, encoding='utf-8')
        scientific['trace_checker'] = {'exit_code': checked.returncode,
                                        'result': json.loads(checked.stdout) if checked.returncode == 0 else checked.stdout,
                                        'stderr': checked.stderr}
        if tests.returncode or checked.returncode:
            failures.append({'kind': 'relocated_scientific_validation'})
    report = {'passed': not failures, 'publication_files': len(manifest['files']),
              'frozen_algorithm_files': len(algorithm['hashes']), 'failures': failures,
              'scientific': scientific, 'scope': 'checksummed published snapshot at its current path; no evidence files overwritten'}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

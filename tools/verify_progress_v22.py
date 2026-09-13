"""Publication checks only; no model fitting or rewriting frozen run evidence."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = ROOT / 'evidence/progress_v22'
    manifest = json.loads((dest / 'EXPORT_MANIFEST.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256']
    lock = json.loads((dest / 'runs/reconstruction-v22-qayc8gft/SOURCE_LOCK.json').read_text())
    documentation_changes = []
    for path, expected in lock.items():
        current = Path(path)
        actual = hashlib.sha256(current.read_bytes()).hexdigest()
        if actual != expected:
            assert current in (ROOT / 'README.md', ROOT / 'REPOSITORY_SNAPSHOT.md'), path
            saved = dest / 'pre_publication_docs' / current.name
            assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected
            documentation_changes.append(dict(path=path, before=expected, after=actual))
    suites = ('exploration_v18', 'exploration_v19', 'exploration_v19/track_a',
              'exploration_v19/track_b', 'exploration_v20', 'real_closure_v21', 'reconstruction_v22')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    results = []
    for suite in suites:
        p = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                            '-s', suite, '-p', 'test_*.py', '-v'],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        results.append(dict(suite=suite, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
        print(suite, p.returncode, p.stderr[-120:], flush=True)
    report = dict(copied_evidence_verified=len(manifest['files']),
                  original_v22_locked_files_checked=len(lock),
                  intentional_publication_document_changes=documentation_changes,
                  tests=results,
                  scope='Publication regression suites, not a rerun of full experiments; original APSS historical replay failure remains recorded.')
    with (dest / 'PUBLICATION_CHECKS.json').open('x') as f:
        json.dump(report, f, indent=2)
    if any(r['returncode'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Read-only checks of the exported subset; needs only the Python standard library."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check():
    manifest = json.loads((ROOT / 'publication/SNAPSHOT_20260916.json').read_text())
    errors = []
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append('missing or changed: ' + row['path'])
    roots = ['publication', 'research_snapshots', 'meta_research', 'field_budget_v24',
             'loss_alignment_v23', 'real_support_bridge_v1', 'evidence/progress_20260916']
    py_count = json_count = 0
    secret = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                        r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
                        r'\bsk-[A-Za-z0-9_-]{32,}')
    for root in roots:
        for path in (ROOT / root).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            try:
                if path.suffix == '.py':
                    ast.parse(path.read_text(), filename=str(path))
                    py_count += 1
                if path.suffix == '.json':
                    json.loads(path.read_text())
                    json_count += 1
                if path.suffix in {'.py', '.json', '.md', '.txt', '.log', '.sh', '.yaml', '.toml'}:
                    if secret.search(path.read_text()):
                        errors.append('possible credential: ' + str(path.relative_to(ROOT)))
            except (SyntaxError, ValueError, UnicodeError) as exc:
                errors.append(str(path.relative_to(ROOT)) + ': ' + str(exc))
    result = {'passed': not errors, 'snapshot_files': len(manifest['files']),
              'snapshot_bytes': sum(r['bytes'] for r in manifest['files']),
              'python_parsed': py_count, 'json_parsed': json_count, 'errors': errors,
              'scope': 'hashes, syntax, JSON and high-confidence secret patterns; not a security proof or full scientific rerun'}
    return result


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(not result['passed'])

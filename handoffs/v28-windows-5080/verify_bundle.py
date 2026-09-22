"""Read-only stdlib verifier; does not load joblib or execute the reference."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    seen = set()
    for row in manifest['files']:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or row['path'] in seen:
            failures.append({'path': row['path'], 'reason': 'unsafe_or_duplicate_path'})
            continue
        seen.add(row['path'])
        p = ROOT.joinpath(*rel.parts)
        if p.is_symlink() or not p.is_file() or ROOT not in p.resolve().parents:
            failures.append({'path': row['path'], 'reason': 'missing_or_unsafe_file'})
            continue
        b = p.read_bytes()
        if len(b) != row['bytes'] or hashlib.sha256(b).hexdigest() != row['sha256']:
            failures.append({'path': row['path'], 'reason': 'content_mismatch'})
    print(json.dumps({'checked': len(seen), 'ok': not failures, 'failures': failures}, indent=2))
    raise SystemExit(1 if failures else 0)

if __name__ == '__main__':
    main()

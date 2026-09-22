"""Publish the frozen closeout evidence, not third-party datasets or dense outputs."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z')
DEST = ROOT / 'research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z'
ALLOW = {'.md', '.py', '.json', '.csv', '.txt', '.toml', '.joblib', '.png', '.svg'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert socket.gethostname() == 'liekkas'
    rows, excluded = [], []
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file():
            continue
        rel = src.relative_to(SOURCE)
        if src.is_symlink() or '__pycache__' in rel.parts or src.suffix not in ALLOW:
            excluded.append({'source': str(src), 'bytes': src.stat().st_size,
                             'reason': 'dense output/cache/duplicate archive/non-allowlisted artifact; retained at source'})
            continue
        if src.stat().st_size > 50_000_000:
            raise RuntimeError(f'Unexpected large publication file: {rel}')
        if src.suffix != '.joblib':
            content = src.read_text(errors='replace')
            patterns = [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                        r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
                        r'https?://[^\s/@:]+:[^\s/@]+@']
            if any(re.search(p, content) for p in patterns):
                raise RuntimeError(f'Credential-pattern review required: {rel}')
        if rel.name == 'AGENTS.md':
            rel = rel.with_name('SOURCE_AGENTS.md')
        dst = DEST / rel
        if dst.exists() and sha(dst) != sha(src):
            raise RuntimeError(f'Refusing overwrite: {dst}')
        rows.append({'source': str(src), 'path': str(dst.relative_to(ROOT)),
                     'sha256': sha(src), 'bytes': src.stat().st_size})
    for row in rows:
        dst = ROOT / row['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(row['source'], dst)
        assert sha(dst) == row['sha256']
    manifest = {'scope': 'Frozen closeout source, models, protocols and tabular evidence; not full data backup',
                'source_immutable': True, 'historical_instructions': 'AGENTS.md renamed SOURCE_AGENTS.md',
                'files': rows, 'excluded': excluded}
    target = ROOT / 'publication/CLOSEOUT_20260922_MANIFEST.json'
    data = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    if target.exists() and target.read_text() != data:
        raise RuntimeError('Existing manifest differs')
    target.write_text(data)
    print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'excluded': len(excluded)}))

if __name__ == '__main__':
    main()

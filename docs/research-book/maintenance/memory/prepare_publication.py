#!/usr/bin/env python3
"""Create a fixed visible-message prefix and a complete, portable book copy.

The original sealed book is never modified. Session bytes are read only by the
existing visible-message exporter and are never copied into the public package.
"""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOK = Path('/home/grf/Documents/Codex/2026-09-29/geometry-meta-research-gitbook')
SERVICE = BOOK.parent / 'research-book-lan-service'
REPO = Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
DEST = REPO / 'docs/research-book'
FORK = Path('/home/grf/.codex/sessions/2026/09/30/rollout-2026-09-30T04-56-18-01a0eef4-3b61-7cc2-ada6-74e67175c7f5.jsonl')
SECRET_PATTERNS = {
    'credential_token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b'),
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential_url': re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def freeze_snapshot(name, cutoff):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('simple snapshot name required')
    target = ROOT / 'snapshots' / name / 'SNAPSHOT.json'
    if target.exists():
        raise ValueError('existing snapshot is immutable; reuse through exporter, not preparation')
    base = json.loads((BOOK / 'conversations/MANIFEST.json').read_text())
    entries = []
    sources = [(Path(x['path']), base['thread_id'], 'parent') for x in base['inputs']]
    sources.append((FORK, '01a0eef4-3b61-7cc2-ada6-74e67175c7f5', 'fork'))
    for path, thread, kind in sources:
        count = 0
        with path.open('rb') as handle:
            for line in handle:
                if not line.endswith(b'\n'):
                    break
                record = json.loads(line)
                if record.get('timestamp', '') > cutoff:
                    break
                count += len(line)
        entries.append({'path': str(path), 'limit_bytes': count,
                        'expected_thread_id': thread, 'kind': kind})
    save(target, {'snapshot': name, 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'base_cutoff_utc': base['cutoff_utc'], 'requested_visible_cutoff_utc': cutoff,
                  'parent_thread_id': base['thread_id'], 'inputs': entries})
    print(json.dumps({'snapshot_config': str(target), 'cutoff': cutoff}))

def package():
    from bs4 import BeautifulSoup
    entries = []
    transformed = []
    expected = set()

    def emit(rel, data, source=None, change='byte_exact'):
        for kind, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                raise ValueError(f'publication pattern requires review: {rel}: {kind}')
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        expected.add(rel)
        item = {'path': rel, 'bytes': len(data), 'sha256': digest(data), 'operation': change}
        if source:
            item['source'] = str(source)
            item['source_sha256'] = digest(source.read_bytes())
        entries.append(item)
        if change != 'byte_exact':
            transformed.append({'path': rel, 'operation': change})

    original = json.loads((BOOK / 'PACKAGE_MANIFEST.json').read_text())
    original_paths = [x['path'] for x in original['files']] + ['PACKAGE_MANIFEST.json']
    for item in original['files']:
        if digest((BOOK / item['path']).read_bytes()) != item['sha256']:
            raise ValueError('sealed original changed: ' + item['path'])
    for rel in original_paths:
        emit('archive/' + rel, (BOOK / rel).read_bytes(), BOOK / rel)

    overlay = ROOT / 'published'
    increment = json.loads((overlay / 'MANIFEST.json').read_text())
    for item in increment['files']:
        source = overlay / item['path']
        if digest(source.read_bytes()) != item['sha256']:
            raise ValueError('increment changed: ' + item['path'])
        emit('increment/' + item['path'], source.read_bytes(), source)
    emit('increment/MANIFEST.json', (overlay / 'MANIFEST.json').read_bytes(), overlay / 'MANIFEST.json')

    # Explicitly allow only already-filtered memory products, never raw sessions.
    selected = [ROOT / n for n in ('memory.py', 'test_memory.py', 'prepare_publication.py', 'verify_publication.py',
                                    'README.md', 'REPORT.md', 'LATEST_STATE.md', 'VALIDATION.json')]
    for group in ('registry', 'snapshots'):
        selected += [p for p in (ROOT / group).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for source in sorted(selected):
        if source.suffix not in {'.py', '.md', '.json', '.jsonl', '.png', '.jpg', '.jpeg', '.webp', '.gif'}:
            raise ValueError('unexpected maintenance artifact: ' + str(source))
        emit('maintenance/memory/' + source.relative_to(ROOT).as_posix(), source.read_bytes(), source)
    for name in ('serve.py', 'render.py', 'README.md', 'RENDER_RECEIPT.json'):
        source = SERVICE / name
        emit('maintenance/service/' + name, source.read_bytes(), source)

    source = SERVICE / 'reader.html'
    soup = BeautifulSoup(source.read_text(), 'html.parser')
    for tag in soup.find_all(True):
        for attr in ('href', 'src'):
            value = tag.get(attr)
            if isinstance(value, str) and value.startswith('/') and not value.startswith('//'):
                tag[attr] = value[1:] if value.startswith('/increment/') else 'archive/' + value[1:]
    banner = soup.select_one('.banner')
    notice = soup.new_tag('p')
    notice.string = '完整公开包装：封存原书与全部登记增量均包含；历史“未公开”描述是原记录时点。当前出版范围和未收录项见 README.md。'
    banner.insert(0, notice)
    data = str(soup).encode()
    emit('READ.html', data, source, 'root_asset_urls_to_relative_and_publication_banner')
    emit('index.html', data, source, 'same_portable_reader_as_READ.html')
    summary = '# 研究记忆书目录\n\n* [当前公开版本](README.md)\n* [最新状态](increment/docs/LATEST_STATE.md)\n* [全部增量索引](increment/INDEX.md)\n\n## 封存原书\n\n'
    original_summary = (BOOK / 'SUMMARY.md').read_text()
    summary += re.sub(r'\]\(([^)#]+\.md)\)', lambda m: '](archive/' + m[1] + ')', original_summary)
    summary += '\n## 后续增量\n\n'
    for item in json.loads((overlay / 'NAVIGATION.json').read_text()):
        summary += '* [' + item['title'] + '](increment/' + item['path'] + ')\n'
    emit('SUMMARY.md', summary.encode(), change='generated_merged_navigation')
    emit('.gitbook.yaml', b'root: ./\nstructure:\n  readme: README.md\n  summary: SUMMARY.md\n', change='generated_gitbook_layout')

    readme = (ROOT / 'PUBLICATION_README.md').read_bytes()
    emit('README.md', readme, ROOT / 'PUBLICATION_README.md', 'maintained_publication_frontmatter')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'target_repository': 'https://github.com/jupiternaut/map-denoise-research',
                'publication_status_at_package_build': 'local_package_verified; git push verification is recorded separately by the publishing task',
                'scope': 'complete sealed book + complete published increment + filtered provenance and maintenance sources; not all raw research datasets',
                'base_files': len(original_paths), 'increment_files': len(increment['files']) + 1,
                'base_visible_messages': 1807, 'increment_visible_messages': increment['new_visible_messages'],
                'total_experiment_records': increment['total_records'], 'files': entries,
                'transformations': transformed,
                'excluded': [
                    {'scope': 'raw Codex/Hermes session logs, hidden reasoning, tool payloads, system/developer records, subagent internal messages', 'reason': 'not user-visible research book content'},
                    {'scope': '__pycache__, .aris, runtime databases, credentials, environments, unpublished proposals', 'reason': 'not public book content'},
                    {'scope': 'large raw pointcloud/photo datasets and model binaries not already in the book', 'reason': 'book never claimed to contain the full experiment data archive; see source/download manifests'},
                    {'scope': 'original September 29 ZIP', 'reason': 'all 694 unpacked original files are included byte-for-byte; redundant container not required'},
                    {'scope': 'eight unavailable original screenshot paths', 'reason': 'historical missing-source status retained; existing embedded images remain archived'}],
                'secret_pattern_scan': 'no matches; not a proof of absence of all sensitive information',
                'historical_absolute_links': 'preserved verbatim in archived evidence; portable reader/navigation link available archived sources, not all external local artifacts',
                'manifest_self_hash': 'excluded to avoid circularity; external expected manifest includes this file'}
    save(DEST / 'PUBLICATION_MANIFEST.json', manifest)
    expected.add('PUBLICATION_MANIFEST.json')
    actual = {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('unexpected stale/missing package files: ' + repr(sorted(actual ^ expected)))
    # All local resource URLs and Markdown navigation targets must resolve.
    failures = []
    for tag in soup.find_all(True):
        for attr in ('href', 'src'):
            value = tag.get(attr)
            if not value or value.startswith(('#', 'http://', 'https://', 'mailto:')):
                continue
            if not (DEST / value.split('#')[0]).is_file():
                failures.append(value)
    for rel in re.findall(r'\]\(([^)#]+\.md)\)', summary):
        if not (DEST / rel).is_file():
            failures.append(rel)
    if failures:
        raise ValueError('portable link failures: ' + repr(sorted(set(failures))))
    expected_items = [{'path': 'docs/research-book/' + rel,
                       'sha256': digest((DEST / rel).read_bytes())} for rel in sorted(expected)]
    save(ROOT / 'PUBLICATION_EXPECTED_MANIFEST.json', {'files': expected_items})
    save(ROOT / 'PUBLICATION_VALIDATION.json', {'files': len(expected),
          'bytes': sum((DEST / p).stat().st_size for p in expected), 'missing_local_urls': [],
          'original_files_byte_exact': len(original_paths), 'increment_files_byte_exact': len(increment['files']) + 1,
          'expected_index_manifest': str(ROOT / 'PUBLICATION_EXPECTED_MANIFEST.json')})
    print((ROOT / 'PUBLICATION_VALIDATION.json').read_text())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('name')
    freeze.add_argument('--through-utc', required=True)
    sub.add_parser('package')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze_snapshot(args.name, args.through_utc)
    else:
        package()

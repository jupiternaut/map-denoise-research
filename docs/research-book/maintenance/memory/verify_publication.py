#!/usr/bin/env python3
"""Verify a downloaded public book without the original machine or datasets."""
import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if key in {'href', 'src'} and value)


def verify(root):
    root = root.resolve(strict=True)
    manifest = json.loads((root / 'PUBLICATION_MANIFEST.json').read_text())
    expected = {x['path']: x for x in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('duplicate manifest path')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and '.git' not in p.parts}
    issues = ['missing:' + p for p in sorted(set(expected) - actual)]
    issues += ['unexpected:' + p for p in sorted(actual - set(expected) - {'PUBLICATION_MANIFEST.json'})]
    for rel, item in expected.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            issues.append('invalid-path:' + rel)
            continue
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            issues.append('byte-mismatch:' + rel)
    parser = Links()
    parser.feed((root / 'READ.html').read_text())
    for value in parser.targets:
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (root / unquote(parsed.path)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            issues.append('reader-resource:' + value)
    result = {'status': 'PASS' if not issues else 'FAIL', 'checked_files': len(expected),
              'base_files': manifest['base_files'], 'increment_files': manifest['increment_files'],
              'issues': sorted(set(issues)),
              'scope': 'packaged file set, bytes and local reader resources; not scientific recomputation, credentials or GitHub publication status'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return not issues


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.root) else 1)

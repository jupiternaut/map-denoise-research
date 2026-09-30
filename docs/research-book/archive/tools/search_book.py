#!/usr/bin/env python3
"""Offline plain-text search. No network or model required."""
import argparse
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('query');parser.add_argument('--limit',type=int,default=40)
parser.add_argument('--scope',choices=['all','experiments','conversations','chapters','topics'],default='all')
args=parser.parse_args();count=0
base=root if args.scope=='all' else root/args.scope
for p in sorted(base.rglob('*.md')):
    if 'sources/files' in str(p):continue
    for n,line in enumerate(p.read_text(errors='replace').splitlines(),1):
        if args.query.casefold() in line.casefold():
            print(f'{p.relative_to(root)}:{n}: {line[:300]}')
            count+=1
            if count>=args.limit:raise SystemExit(0)
print(f'匹配 {count} 行。')

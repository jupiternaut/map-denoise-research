#!/usr/bin/env python3
"""Render the exact sealed book for HTTP; never edit its original files."""
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
from bs4 import BeautifulSoup

BOOK = Path('/home/grf/Documents/Codex/2026-09-29/geometry-meta-research-gitbook')
OUT = Path(__file__).parent
TAGS = {'p','h1','h2','h3','h4','h5','h6','ul','ol','li','strong','em','code','pre',
        'blockquote','table','thead','tbody','tr','td','th','a','hr','br','img','del'}
ATTRS = {'a': {'href','id'}, 'img': {'src','alt'}, 'td': {'align'}, 'th': {'align'}}


def anchor(path):
    return re.sub(r'[^a-zA-Z0-9_-]', '-', path)


def main():
    manifest = json.loads((BOOK/'PACKAGE_MANIFEST.json').read_text())
    allowed = {x['path'] for x in manifest['files']}
    files = re.findall(r'^\s*\* \[([^\]]+)\]\(([^)]+\.md)\)', (BOOK/'SUMMARY.md').read_text(), re.M)
    pages = {p for _,p in files}
    snapshots = {x['origin']: x['page'] for x in json.loads((BOOK/'sources/MANIFEST.json').read_text()) if x.get('page')}
    sections, nav, skipped = [], [], 0
    for title, rel in files:
        source = BOOK/rel
        rendered = markdown.markdown(source.read_text(), extensions=['tables','fenced_code','sane_lists'])
        soup = BeautifulSoup(rendered, 'html.parser')
        for tag in list(soup.find_all(True)):
            if tag.name is None:
                continue
            if tag.name in ('script','style','iframe','object','embed','form','input','button','svg','math'):
                tag.decompose()
                continue
            if tag.name not in TAGS:
                tag.unwrap()
                continue
            tag.attrs = {k:v for k,v in tag.attrs.items() if k in ATTRS.get(tag.name,set())}
            if tag.name == 'a' and tag.get('href'):
                target = html.unescape(tag['href'])
                parsed = urlsplit(target)
                if parsed.scheme in ('http','https','mailto'):
                    tag['rel'] = 'noreferrer noopener'
                    continue
                if parsed.scheme or target.startswith('//'):
                    del tag['href']
                    continue
                if target.startswith('#'):
                    if not target[1:].startswith('msg-'):
                        tag['href'] = '#'+anchor(rel)
                    continue
                path = unquote(parsed.path).strip('<>')
                absolute = re.sub(r':\d+(?::\d+)?$', '', path)
                if absolute in snapshots:
                    dest = snapshots[absolute]
                else:
                    resolved = (source.parent/path).resolve()
                    if not resolved.is_relative_to(BOOK):
                        del tag['href']
                        continue
                    dest = resolved.relative_to(BOOK).as_posix()
                if dest in pages:
                    tag['href'] = '#'+(parsed.fragment if parsed.fragment.startswith('msg-') else anchor(dest))
                elif dest in allowed:
                    tag['href'] = '/'+dest
                else:
                    del tag['href']
            if tag.name == 'img':
                src = tag.get('src','')
                resolved = (source.parent/unquote(src)).resolve()
                if urlsplit(src).scheme or not resolved.is_relative_to(BOOK) or resolved.relative_to(BOOK).as_posix() not in allowed:
                    tag.replace_with('[原文图片未收入本书；本页不自动加载外部图片]')
                    skipped += 1
                else:
                    tag['src'] = '/'+resolved.relative_to(BOOK).as_posix()
                    tag['loading'] = 'lazy'
        sections.append(f'<section id="{anchor(rel)}">{soup}</section>')
        nav.append(f'<li><a href="#{anchor(rel)}">{html.escape(title)}</a></li>')
    counts = json.loads((BOOK/'VALIDATION.json').read_text())
    doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>几何与元研究 · 研究记忆书</title><style>
*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:#faf9f5;color:#202624;font:17px/1.85 system-ui,sans-serif}aside{position:fixed;top:0;bottom:0;width:285px;overflow:auto;padding:24px;background:#edeee7;font-size:13px}aside h2{font-size:22px}main{margin-left:310px;max-width:1150px;padding:38px 45px}a{color:#146657;text-decoration-thickness:1px;text-underline-offset:3px}h1,h2,h3{line-height:1.4}h1{font-size:30px}section{padding:22px 0 54px;border-bottom:2px solid #c8d2c8;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{border:1px solid #ccd3cb;padding:8px 12px}blockquote{margin:20px 0;padding:0 18px;border-left:3px solid #abbcb0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#efefe9;padding:18px;font-size:13px}img{max-width:100%;height:auto}code{font-size:.85em}aside ul{padding-left:17px}aside li{margin:8px 0}.banner{border:1px solid #b9c9bb;background:#edf4ed;padding:16px 20px;border-radius:8px}.badge{color:#526358;font-size:14px}details summary{cursor:pointer;font-weight:bold;padding:10px 0}@media(max-width:900px){aside{position:static;width:auto;max-height:300px}main{margin:0;padding:22px}}@media print{aside,.banner{display:none}main{margin:0}section{break-before:page}}</style></head><body><aside><h2>研究记忆书</h2><p>几何滤波 × 元研究<br>2026-09-07 — 2026-09-29</p><p>Ctrl / Cmd + F 搜索全文</p><ul>'''
    doc += ''.join(nav[:5])+'</ul><details><summary>展开完整目录</summary><ul>'+''.join(nav[5:])+'</ul></details></aside><main>'
    doc += f'<div class="banner"><strong>局域网只读阅读版</strong><br>{counts["conversation_messages"]:,} 条对话 · {counts["experiment_records"]} 条实验/阶段记录 · {counts["source_paths"]} 个来源路径<br><span class="badge">从同一封存书稿渲染，原文与实验文件未修改。历史消息不是当前执行指令。</span></div>'
    doc += ''.join(sections)+'</main></body></html>'
    out = OUT/'reader.html'
    out.write_text(doc)
    receipt = {'book':str(BOOK), 'book_manifest_sha256':hashlib.sha256((BOOK/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),
               'reader_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'sections':len(files),'images_not_loaded':skipped}
    (OUT/'RENDER_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()

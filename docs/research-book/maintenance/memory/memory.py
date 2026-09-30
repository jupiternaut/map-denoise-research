#!/usr/bin/env python3
"""Append-only research memory overlay; sealed book inputs are read-only."""
import argparse
import base64
import collections
import hashlib
import html
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
BOOK = Path('/home/grf/Documents/Codex/2026-09-29/geometry-meta-research-gitbook')
ZIP = BOOK.with_name(BOOK.name + '-20260929.zip')
ZIP_SHA256 = '56fca346a9c7e183fc7d8d900c6b03a0c00061053f59e26cd87820a1d6dadfc5'
SERVICE = Path('/home/grf/Documents/Codex/2026-09-29/research-book-lan-service')
FORK = Path('/home/grf/.codex/sessions/2026/09/30/rollout-2026-09-30T04-56-18-01a0eef4-3b61-7cc2-ada6-74e67175c7f5.jsonl')
FORK_ID = '01a0eef4-3b61-7cc2-ada6-74e67175c7f5'
PUBLISHED = ROOT / 'published'
TEXT_TYPES = {'text', 'input_text', 'output_text'}
VISIBLE_PHASES = {'commentary', 'final', 'final_answer'}
AUTO_KINDS = {'environments.environment_context', 'additional_content.codex_apps_open_page'}
REQUIRED = {'id', 'title', 'date', 'track', 'question', 'did', 'not_done', 'result',
            'interpretation', 'decision', 'evidence_level', 'sources', 'tags',
            'predecessors', 'corrections', 'rerun_condition', 'claims', 'evaluation_scope'}
LIST_FIELDS = {'did', 'not_done', 'result', 'interpretation', 'decision', 'sources',
               'tags', 'predecessors', 'corrections', 'claims'}
SECRETS = [re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b')]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path, limit=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while limit is None or limit > 0:
            chunk = handle.read(min(1024 * 1024, limit) if limit is not None else 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if limit is not None:
                limit -= len(chunk)
    if limit is not None and limit:
        raise ValueError('source shorter than frozen prefix')
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def redact(text):
    count = 0
    for pattern in SECRETS:
        text, n = pattern.subn('[REDACTED_CREDENTIAL]', text)
        count += n
    return text, count


def visible_message(record):
    """Whitelist visible message roles/phases, excluding automatic context."""
    payload = record.get('payload', {})
    if record.get('type') != 'response_item' or payload.get('type') != 'message':
        return None, 'not_visible_message'
    role = payload.get('role')
    if role not in {'user', 'assistant'}:
        return None, 'non_visible_role'
    if role == 'assistant' and payload.get('phase') not in VISIBLE_PHASES:
        return None, 'non_visible_assistant_phase'
    text = '\n\n'.join(x.get('text', '') for x in payload.get('content', []) if x.get('type') in TEXT_TYPES)
    metadata = payload.get('internal_chat_message_metadata_passthrough', {})
    if not isinstance(metadata, dict):
        metadata = {}
    kinds = set(metadata.get('content_item_kinds', []))
    stripped = text.lstrip()
    automatic = (role == 'user' and (bool(kinds & AUTO_KINDS)
                 or stripped.startswith('<environment_context>')
                 or stripped.startswith('<external_codex_apps_open_page>')
                 or (stripped.startswith('# AGENTS.md instructions') and '<environment_context>' in text)))
    if automatic:
        return None, 'automatic_context'
    if not payload.get('id'):
        raise ValueError('visible message without stable ID')
    return {'id': payload['id'], 'timestamp': record.get('timestamp', ''), 'role': role,
            'phase': payload.get('phase'), 'text': text, 'parts': payload.get('content', [])}, None


def base_inventory():
    records = []
    for source in sorted((BOOK / 'records').glob('*.json')):
        value = json.loads(source.read_text())
        if isinstance(value, list):
            records.extend(value)
    return records


def base_check():
    manifest = json.loads((BOOK / 'PACKAGE_MANIFEST.json').read_text())
    checked = 0
    for item in manifest['files']:
        source = BOOK / item['path']
        if not source.is_file() or sha_file(source) != item['sha256']:
            raise ValueError('sealed book mismatch: ' + item['path'])
        checked += 1
    zip_hash = sha_file(ZIP)
    if zip_hash != ZIP_SHA256:
        raise ValueError('sealed ZIP changed')
    return {'files_checked': checked, 'manifest_sha256': sha_file(BOOK / 'PACKAGE_MANIFEST.json'),
            'zip_sha256': zip_hash}


def export_snapshot(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('snapshot name must be a simple filename')
    base = json.loads((BOOK / 'conversations/MANIFEST.json').read_text())
    target = ROOT / 'snapshots' / name
    target.mkdir(parents=True, exist_ok=True)
    frozen_path = target / 'SNAPSHOT.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
    else:
        entries = []
        for item in base['inputs']:
            entries.append({'path': item['path'], 'limit_bytes': Path(item['path']).stat().st_size,
                            'expected_thread_id': base['thread_id'], 'kind': 'parent'})
        entries.append({'path': str(FORK), 'limit_bytes': FORK.stat().st_size,
                        'expected_thread_id': FORK_ID, 'kind': 'fork'})
        frozen = {'snapshot': name, 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'base_cutoff_utc': base['cutoff_utc'], 'parent_thread_id': base['thread_id'], 'inputs': entries}
        save_json(frozen_path, frozen)
    base_ids = {json.loads(line)['id'] for line in (BOOK / 'conversations/messages.jsonl').read_text().splitlines()}
    seen, rows, sources = {}, [], []
    previous_receipt = json.loads((target / 'MANIFEST.json').read_text()) if (target / 'MANIFEST.json').exists() else None
    redactions = 0
    for entry in frozen['inputs']:
        source = Path(entry['path'])
        consumed = 0
        excluded = collections.Counter()
        digest = hashlib.sha256()
        visible_count = 0
        metadata_checked = False
        records_scanned = 0
        incomplete_tail = False
        with source.open('rb') as handle:
            for line_no, raw in enumerate(handle, 1):
                if consumed >= entry['limit_bytes']:
                    break
                raw = raw[:entry['limit_bytes'] - consumed]
                consumed += len(raw)
                digest.update(raw)
                if not raw.endswith(b'\n'):
                    incomplete_tail = True
                    break
                record = json.loads(raw)
                records_scanned += 1
                if record.get('type') == 'session_meta':
                    meta = record.get('payload', {})
                    if meta.get('id') != entry['expected_thread_id'] or meta.get('cwd') != '/home/grf':
                        raise ValueError('session identity mismatch')
                    if entry['kind'] == 'fork' and meta.get('forked_from_id') != base['thread_id']:
                        raise ValueError('fork ancestry mismatch')
                    metadata_checked = True
                if record.get('timestamp', '') <= base['cutoff_utc']:
                    continue
                message, reason = visible_message(record)
                if message is None:
                    excluded[reason] += 1
                    continue
                if message['id'] in base_ids:
                    excluded['already_archived'] += 1
                    continue
                text_hash = sha(message['text'].encode())
                if message['id'] in seen:
                    if seen[message['id']] != text_hash:
                        raise ValueError('duplicate visible ID with different text')
                    excluded['duplicate_id'] += 1
                    continue
                seen[message['id']] = text_hash
                clean, count = redact(message['text'])
                redactions += count
                media = []
                for part in message['parts']:
                    if part.get('type') not in {'image', 'input_image'}:
                        continue
                    url = part.get('image_url', part.get('url', ''))
                    if isinstance(url, dict):
                        url = url.get('url', '')
                    if isinstance(url, str) and url.startswith('data:image/'):
                        header, encoded = url.split(',', 1)
                        blob = base64.b64decode(encoded, validate=True)
                        extension = header.split(';', 1)[0].split('/', 1)[1].replace('jpeg', 'jpg')
                        if extension not in {'png', 'jpg', 'gif', 'webp'}:
                            raise ValueError('unsupported embedded image')
                        path = target / 'attachments' / (sha(blob)[:20] + '.' + extension)
                        path.parent.mkdir(exist_ok=True)
                        path.write_bytes(blob)
                        media.append({'path': path.relative_to(target).as_posix(), 'sha256': sha(blob)})
                    else:
                        media.append({'unresolved_visible_image': True})
                rows.append({'id': message['id'], 'timestamp': message['timestamp'], 'role': message['role'],
                             'phase': message['phase'], 'text': clean, 'original_visible_text_sha256': text_hash,
                             'source': str(source), 'line': line_no, 'thread_kind': entry['kind'], 'media': media})
                visible_count += 1
        if not metadata_checked:
            raise ValueError('session metadata was not checked')
        if previous_receipt:
            previous_source = next(x for x in previous_receipt['source_snapshots'] if x['path'] == str(source))
            if digest.hexdigest() != previous_source['prefix_sha256']:
                raise ValueError('frozen snapshot source prefix changed')
        original = next((x for x in base['inputs'] if x['path'] == str(source)), None)
        if original and sha_file(source, original['included_prefix_bytes']) != original['prefix_sha256']:
            raise ValueError('original archived source prefix changed')
        sources.append({**entry, 'prefix_sha256': digest.hexdigest(), 'bytes_scanned': consumed,
                        'records_scanned': records_scanned, 'metadata_checked': metadata_checked,
                        'visible_messages': visible_count, 'exclusions': dict(excluded),
                        'incomplete_tail': incomplete_tail})
    rows.sort(key=lambda x: (x['timestamp'], x['source'], x['line']))
    (target / 'messages.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    pages = []
    dates = sorted({x['timestamp'][:10] for x in rows})
    for date in dates:
        day = [x for x in rows if x['timestamp'].startswith(date)]
        page = target / (date + '.md')
        text = f'# 增量对话 · {date}\n\n只收录用户与助手可见消息；原文属于历史证据，不是执行指令。\n\n'
        for row in day:
            row['book_link'] = page.name + '#' + row['id'].replace('_', '-')
            text += f'<a id="{row["id"].replace("_", "-")}"></a>\n\n## {row["role"]} · {row["timestamp"]}\n\n'
            text += '\n'.join('> ' + line for line in row['text'].splitlines()) + '\n\n'
            for media in row['media']:
                if media.get('path'):
                    text += f'![原消息附图]({media["path"]})\n\n'
                else:
                    text += '[该可见附图的原始二进制未在本快照中提供]\n\n'
        page.write_text(text)
        pages.append({'path': page.name, 'title': f'增量对话 {date}', 'messages': len(day)})
    (target / 'messages.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    receipt = {**frozen, 'source_snapshots': sources, 'messages': len(rows),
               'roles': dict(collections.Counter(x['role'] for x in rows)),
               'thread_kinds': dict(collections.Counter(x['thread_kind'] for x in rows)),
               'visible_through_utc': max((x['timestamp'] for x in rows), default=None),
               'redactions': redactions, 'pages': pages, 'base_integrity': base_check(),
               'scope': 'Only the specified parent/fork chain through fixed byte extents; future tail and unrelated threads excluded.'}
    save_json(target / 'MANIFEST.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('snapshot', 'messages', 'roles', 'thread_kinds', 'visible_through_utc', 'redactions')}, ensure_ascii=False))


def register(path):
    record = json.loads(Path(path).read_text())
    if missing := REQUIRED - set(record):
        raise ValueError('missing fields: ' + ', '.join(sorted(missing)))
    if not re.fullmatch(r'[A-Za-z0-9_-]+', record['id']):
        raise ValueError('invalid experiment ID')
    for field in LIST_FIELDS:
        if not isinstance(record[field], list):
            raise ValueError('list required: ' + field)
    if not record['sources']:
        raise ValueError('a completed record needs at least one evidence source')
    base_ids = {x['id'] for x in base_inventory()}
    if record['id'] in base_ids:
        raise ValueError('existing sealed experiment ID')
    sources = []
    for source in record['sources']:
        source = Path(source)
        if not source.is_absolute():
            raise ValueError('source paths must be absolute')
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError('source must be a file')
        sources.append({'path': str(source), 'sha256': sha_file(source), 'bytes': source.stat().st_size})
    record['source_checks'] = sources
    record['registered_utc'] = datetime.now(timezone.utc).isoformat()
    target = ROOT / 'registry' / (record['id'] + '.json')
    if target.exists():
        existing = json.loads(target.read_text())
        comparable = dict(record)
        comparable['registered_utc'] = existing['registered_utc']
        if existing != comparable:
            raise ValueError('record already registered; use a new ID and predecessor for correction')
        print('already registered: ' + record['id'])
        return
    save_json(target, record)
    print('registered: ' + record['id'])


def all_records():
    return base_inventory() + [json.loads(x.read_text()) for x in sorted((ROOT / 'registry').glob('*.json'))]


def portable_report_images(text, source, checks):
    """Copy only explicitly registered and hash-bound local report images."""
    known = {str(Path(x['path']).resolve()): x for x in checks}
    def replace(match):
        image = (source.parent / match[2]).resolve()
        item = known.get(str(image))
        if item is None or image.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
            return match[0]
        if sha_file(image) != item['sha256']:
            raise ValueError('registered image changed: ' + str(image))
        rel = 'media/' + item['sha256'] + image.suffix.lower()
        dest = PUBLISHED / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image.read_bytes())
        return '![' + match[1] + '](../' + rel + ')'
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace, text)


def search(query, limit=10):
    tokens = {x.lower() for x in re.findall(r'[A-Za-z0-9_]+|[\u4e00-\u9fff]{2}', query)}
    hits = []
    for record in all_records():
        corpus = ' '.join(str(record.get(k, '')) for k in ('title', 'question', 'tags', 'did', 'not_done', 'predecessors', 'rerun_condition')).lower()
        matched = sorted(x for x in tokens if x in corpus)
        if matched:
            hits.append({'id': record['id'], 'title': record['title'], 'matched_terms': matched,
                         'predecessors': record['predecessors'], 'did': record['did'],
                         'not_done': record['not_done'], 'rerun_condition': record['rerun_condition']})
    hits.sort(key=lambda x: (-len(x['matched_terms']), x['id']))
    print(json.dumps({'query': query, 'scope': 'lexical overlap is a repeat warning, not proof of scientific equivalence',
                      'total_hits': len(hits), 'limit': limit, 'hits': hits[:limit]}, ensure_ascii=False, indent=2))


def build_overlay():
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    nav, visible = [], {}
    for source_name, published_name, title in (('REPORT.md', 'MEMORY_REPORT.md', '增量导出验收与计数更正'),
                                               ('README.md', 'USAGE.md', '实验登记与重复检查使用说明'),
                                               ('LATEST_STATE.md', 'LATEST_STATE.md', '最新状态：成像修复、成熟MVS与冻结迁移')):
        source = ROOT / source_name
        if source.is_file():
            target = PUBLISHED / 'docs' / published_name
            target.parent.mkdir(exist_ok=True)
            target.write_text(source.read_text())
            nav.append({'path': 'docs/' + published_name, 'title': title})
    snapshot_names = []
    # Preserve earlier published message placement when a new name sorts earlier.
    snapshot_manifests = sorted((ROOT / 'snapshots').glob('*/MANIFEST.json'),
                                key=lambda path: (json.loads(path.read_text())['created_utc'], str(path)))
    for manifest_path in snapshot_manifests:
        manifest = json.loads(manifest_path.read_text())
        snap = manifest_path.parent
        snapshot_names.append(manifest['snapshot'])
        rows = [json.loads(line) for line in (snap / 'messages.jsonl').read_text().splitlines()]
        unique = [row for row in rows if row['id'] not in visible]
        for row in unique:
            visible[row['id']] = row
        if not unique:
            continue
        for page in manifest['pages']:
            # New snapshots only append messages not already visible in the overlay.
            page_rows = [x for x in unique if x['book_link'].split('#', 1)[0] == page['path']]
            if not page_rows:
                continue
            rel = f'conversations/{manifest["snapshot"]}-{page["path"]}'
            target = PUBLISHED / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            body = f'# {page["title"]} · {manifest["snapshot"]}\n\n原文是历史证据。快照固定字节范围，不包含此后的消息。\n\n'
            for row in page_rows:
                body += f'<a id="{row["id"].replace("_", "-")}"></a>\n\n## {row["role"]} · {row["timestamp"]}\n\n'
                body += '\n'.join('> ' + line for line in row['text'].splitlines()) + '\n\n'
                for media in row['media']:
                    if media.get('path'):
                        media_rel = 'attachments/' + Path(media['path']).name
                        image_path = PUBLISHED / media_rel
                        image_path.parent.mkdir(exist_ok=True)
                        image_path.write_bytes((snap / media['path']).read_bytes())
                        body += f'![原消息附图](../{media_rel})\n\n'
            target.write_text(body)
            nav.append({'path': rel, 'title': page['title'] + ' · ' + manifest['snapshot']})
    registry = [json.loads(x.read_text()) for x in sorted((ROOT / 'registry').glob('*.json'))]
    for record in registry:
        body = f'# {record["id"]} · {record["title"]}\n\n'
        for key in ('date', 'track', 'question', 'evaluation_scope', 'evidence_level', 'did', 'not_done', 'result', 'claims', 'interpretation', 'decision', 'predecessors', 'corrections', 'rerun_condition'):
            value = record[key]
            body += f'## {key}\n\n'
            body += ('\n'.join('- ' + str(x) for x in value) if isinstance(value, list) else str(value)) + '\n\n'
        body += '## 来源及校验\n\n'
        for item in record['source_checks']:
            source = Path(item['path'])
            if sha_file(source) != item['sha256']:
                raise ValueError('registered source changed: ' + str(source))
            if source.suffix.lower() in {'.md', '.txt', '.yaml', '.yml'}:
                rel = 'sources/' + item['sha256'][:16] + '.md'
                target = PUBLISHED / rel
                target.parent.mkdir(exist_ok=True)
                clean, _ = redact(source.read_text())
                clean = portable_report_images(clean, source, record['source_checks'])
                if source.suffix.lower() != '.md':
                    fence = '`' * max(3, max((len(x) for x in re.findall(r'`+', clean)), default=0) + 1)
                    clean = fence + '\n' + clean + '\n' + fence + '\n'
                target.write_text('# 来源快照\n\n原始路径：`' + str(source) + '`\n\n' + clean)
                body += f'- [{source.name}](../{rel}) · SHA-256 `{item["sha256"]}`\n'
                nav.append({'path': rel, 'title': record['id'] + ' 来源 ' + source.name})
            elif source.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
                rel = 'media/' + item['sha256'] + source.suffix.lower()
                target = PUBLISHED / rel
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(source.read_bytes())
                body += f'- [{source.name}](../{rel}) · SHA-256 `{item["sha256"]}`（登记图件原字节）\n'
            else:
                body += f'- `{source}` · SHA-256 `{item["sha256"]}`（校验记录，不在 LAN 提供原始二进制）\n'
        rel = 'records/' + record['id'] + '.md'
        target = PUBLISHED / rel
        target.parent.mkdir(exist_ok=True)
        target.write_text(body)
        nav.append({'path': rel, 'title': record['id'] + ' · ' + record['title']})
    records = all_records()
    save_json(PUBLISHED / 'EXPERIMENT_INDEX.json', records)
    save_json(PUBLISHED / 'VISIBLE_MESSAGES.json', list(visible.values()))
    body = '# 增量研究记忆\n\n封存原书截至 2026-09-29T11:56:40.445Z；以下是固定字节快照的增量。\n\n'
    body += f'新增可见消息 {len(visible)} 条；新增实验记录 {len(registry)} 条；与原书合并索引 {len(records)} 条。\n\n'
    body += '计数更正：旧 audit 的父线程 18 条含一条自动环境消息；合法可见对话实际为 17 条（用户3、助手14）。原书截止前1,807条保持不变。\n\n'
    body += '## 快照与边界\n\n'
    for name in snapshot_names:
        m = json.loads((ROOT / 'snapshots' / name / 'MANIFEST.json').read_text())
        body += f'- `{name}`：固定范围内 {m["messages"]} 条，最后可见时间 `{m["visible_through_utc"]}`。\n'
    body += '\n## 新增章节\n\n' + '\n'.join(f'- [{x["title"]}]({x["path"]})' for x in nav) + '\n\n'
    body += '## 重复研究检查\n\n执行 `python memory/memory.py search "纹理 归属 候选 校准"`；搜索会返回既往 did/not_done、前驱与重访条件。关键词重叠只用于提醒，不代替数学等价判断。\n\n'
    body += '## 所有实验索引\n\n| ID | 日期 | 问题 | 作用域 |\n|---|---|---|---|\n'
    for record in records:
        title = str(record['title']).replace('|', '／')
        scope = str(record.get('evaluation_scope', record.get('evidence_level', 'unknown'))).replace('|', '／')
        body += f'| {record["id"]} | {record["date"]} | {title} | {scope} |\n'
    (PUBLISHED / 'INDEX.md').write_text(body)
    nav.insert(0, {'path': 'INDEX.md', 'title': '9月30日增量：对话、实验与重复检查'})
    save_json(PUBLISHED / 'NAVIGATION.json', nav)
    files = [{'path': x.relative_to(PUBLISHED).as_posix(), 'sha256': sha_file(x), 'bytes': x.stat().st_size}
             for x in sorted(PUBLISHED.rglob('*')) if x.is_file() and x.name != 'MANIFEST.json']
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'base_book': str(BOOK),
                'base_integrity': base_check(), 'new_visible_messages': len(visible),
                'new_records': len(registry), 'total_records': len(records), 'snapshots': snapshot_names, 'files': files}
    save_json(PUBLISHED / 'MANIFEST.json', manifest)
    return manifest


def publish():
    manifest = build_overlay()
    subprocess.run([sys.executable, str(SERVICE / 'render.py')], check=True, capture_output=True, text=True)
    import markdown
    from bs4 import BeautifulSoup
    reader = SERVICE / 'reader.html'
    soup = BeautifulSoup(reader.read_text(), 'html.parser')
    nav = json.loads((PUBLISHED / 'NAVIGATION.json').read_text())
    links = BeautifulSoup('<h3>新增记忆</h3><ul></ul>', 'html.parser')
    for item in nav:
        rel = item['path']
        identifier = 'increment-' + re.sub(r'[^A-Za-z0-9_-]', '-', rel)
        li = soup.new_tag('li')
        a = soup.new_tag('a', href='#' + identifier)
        a.string = item['title']
        li.append(a)
        links.ul.append(li)
        rendered = markdown.markdown((PUBLISHED / rel).read_text(), extensions=['tables', 'fenced_code', 'sane_lists'])
        section = BeautifulSoup('<section></section>', 'html.parser').section
        section['id'] = identifier
        content = BeautifulSoup(rendered, 'html.parser')
        for tag in list(content.find_all(True)):
            if not tag.name:
                continue
            if tag.name in {'script', 'style', 'iframe', 'object', 'embed', 'form', 'input', 'button', 'svg', 'math'}:
                tag.decompose()
                continue
            if tag.name not in {'p','h1','h2','h3','h4','h5','h6','ul','ol','li','strong','em','code','pre','blockquote','table','thead','tbody','tr','td','th','a','hr','br','img','del'}:
                tag.unwrap()
                continue
            tag.attrs = {k:v for k,v in tag.attrs.items() if k in ({'href', 'id'} if tag.name == 'a' else {'src','alt'} if tag.name == 'img' else set())}
            if tag.name == 'a' and tag.get('href'):
                href = tag['href']
                if href.startswith('#'):
                    pass
                elif href.startswith(('http://', 'https://', 'mailto:')):
                    tag['rel'] = 'noreferrer noopener'
                else:
                    dest = (PUBLISHED / rel).parent.joinpath(href).resolve()
                    if dest.is_relative_to(PUBLISHED) and dest.is_file():
                        target = dest.relative_to(PUBLISHED).as_posix()
                        tag['href'] = '#increment-' + re.sub(r'[^A-Za-z0-9_-]', '-', target)
                    else:
                        del tag['href']
            if tag.name == 'img':
                target = (PUBLISHED / rel).parent.joinpath(tag.get('src','')).resolve()
                if target.is_relative_to(PUBLISHED) and target.is_file():
                    tag['src'] = '/increment/' + target.relative_to(PUBLISHED).as_posix()
                else:
                    tag.replace_with('[未收入本增量的附图]')
        section.extend(content.contents)
        soup.main.append(section)
    soup.aside.insert(2, links)
    banner = soup.select_one('.banner')
    banner.append(BeautifulSoup(f'<br><strong>已追加 {manifest["new_visible_messages"]} 条可见消息、{manifest["new_records"]} 条新记录。</strong> 封存原书保持不变；新增部分的精确时间/字节边界见“新增记忆”。', 'html.parser'))
    reader.write_text(str(soup))
    receipt = json.loads((SERVICE / 'RENDER_RECEIPT.json').read_text())
    receipt.update({'reader_sha256': sha_file(reader), 'overlay': str(PUBLISHED),
                    'overlay_manifest_sha256': sha_file(PUBLISHED / 'MANIFEST.json'),
                    'new_visible_messages': manifest['new_visible_messages'], 'new_records': manifest['new_records'],
                    'sections': receipt['sections'] + len(nav)})
    save_json(SERVICE / 'RENDER_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def verify():
    issues = []
    manifest = json.loads((PUBLISHED / 'MANIFEST.json').read_text())
    integrity = base_check()
    source_prefixes = 0
    for snapshot in manifest['snapshots']:
        receipt = json.loads((ROOT / 'snapshots' / snapshot / 'MANIFEST.json').read_text())
        for item in receipt['source_snapshots']:
            if sha_file(item['path'], item['limit_bytes']) != item['prefix_sha256']:
                issues.append('frozen source prefix mismatch: ' + item['path'])
            source_prefixes += 1
    for item in manifest['files']:
        if sha_file(PUBLISHED / item['path']) != item['sha256']:
            issues.append('published file hash mismatch: ' + item['path'])
    messages = json.loads((PUBLISHED / 'VISIBLE_MESSAGES.json').read_text())
    from bs4 import BeautifulSoup
    reader = SERVICE / 'reader.html'
    soup = BeautifulSoup(reader.read_text(), 'html.parser')
    anchor_counts = collections.Counter(tag.get('id') for tag in soup.find_all(attrs={'id': True}))
    for message in messages:
        if anchor_counts[message['id'].replace('_', '-')] != 1:
            issues.append('message anchor missing/duplicated: ' + message['id'])
    base_messages = [json.loads(line) for line in (BOOK / 'conversations/messages.jsonl').read_text().splitlines()]
    for message in base_messages:
        if anchor_counts[message['id'].replace('_', '-')] != 1:
            issues.append('original message anchor missing/duplicated: ' + message['id'])
    with urlopen('http://192.168.6.201:8765/', timeout=15) as response:
        status = response.status
        if sha(response.read()) != sha_file(reader):
            issues.append('served reader differs')
    with urlopen('http://192.168.6.201:8765/increment/EXPERIMENT_INDEX.json', timeout=15) as response:
        index_status = response.status
        if sha(response.read()) != sha_file(PUBLISHED / 'EXPERIMENT_INDEX.json'):
            issues.append('served index differs')
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed' if not issues else 'failed',
              'issues': issues, 'base_integrity': integrity, 'http_status': status, 'index_http_status': index_status,
              'new_visible_messages': len(messages), 'message_anchors_checked': len(messages),
              'base_message_anchors_checked': len(base_messages),
              'frozen_source_prefixes_checked': source_prefixes,
              'new_records': manifest['new_records'], 'total_records': manifest['total_records'],
              'reader_sha256': sha_file(reader)}
    save_json(ROOT / 'VALIDATION.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--snapshot', required=True)
    add = sub.add_parser('register')
    add.add_argument('record', type=Path)
    find = sub.add_parser('search')
    find.add_argument('query')
    find.add_argument('--limit', type=int, default=10)
    sub.add_parser('publish')
    sub.add_parser('verify')
    args = parser.parse_args()
    if args.command == 'export':
        export_snapshot(args.snapshot)
    elif args.command == 'register':
        register(args.record)
    elif args.command == 'search':
        search(args.query, args.limit)
    elif args.command == 'publish':
        publish()
    else:
        raise SystemExit(verify())


if __name__ == '__main__':
    main()

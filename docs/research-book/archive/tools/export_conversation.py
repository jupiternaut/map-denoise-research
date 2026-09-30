#!/usr/bin/env python3
"""Export only visible root-thread messages at a fixed, explicit cutoff.

No reasoning, tools, subagent messages, or system/developer records are exported.
Images embedded in the visible user message are decoded as original media, not
edited; referenced attachments are copied only from explicit attachment paths.
"""
import base64
import collections
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
THREAD = '01a07a84-8d1e-7f91-966b-7169498a86ba'
CUTOFF = '2026-09-29T11:56:40.445Z'
INPUTS = [
    Path('/home/grf/.codex/sessions/2026/09/07/rollout-2026-09-07T14-18-22-01a07a84-8d1e-7f91-966b-7169498a86ba.jsonl'),
    Path('/home/grf/.codex/sessions/2026/09/16/rollout-2026-09-16T17-16-01-01a07a84-8d1e-7f91-966b-7169498a86ba_01a0a980-6cbc-72a3-8861-db0858d216af.jsonl'),
]
ATTACHMENT = re.compile(r'/home/grf/\.codex/attachments/[^\n\r<>`"\[\]]+|/tmp/codex-clipboard-[a-zA-Z0-9-]+\.(?:png|jpg|jpeg)')
SECRET = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16})\b')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write(rel, text):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')

def main():
    messages, inputs, exclusions = [], [], collections.Counter()
    seen, attachments = set(), {}
    redactions = []
    for path in INPUTS:
        digest = hashlib.sha256()
        byte_end = 0
        n = 0
        with path.open('rb') as f:
            for line_no, raw in enumerate(f, 1):
                rec = json.loads(raw)
                if rec.get('timestamp', '') > CUTOFF:
                    break
                digest.update(raw)
                byte_end += len(raw)
                n += 1
                p = rec.get('payload', {})
                if rec.get('type') == 'session_meta':
                    assert p['id'] == THREAD and p.get('cwd') == '/home/grf'
                if rec.get('type') != 'response_item' or p.get('type') != 'message':
                    exclusions['non_message_records'] += 1
                    continue
                role, phase = p.get('role'), p.get('phase')
                if role not in ('user', 'assistant'):
                    exclusions['non_visible_role'] += 1
                    continue
                if role == 'assistant' and phase not in ('commentary', 'final_answer', 'final'):
                    exclusions['unrecognized_assistant_phase'] += 1
                    continue
                ident = p.get('id')
                assert ident
                if ident in seen:
                    exclusions['duplicate_message_id'] += 1
                    continue
                seen.add(ident)
                parts = [x.get('text', '') for x in p.get('content', []) if x.get('type') in ('text', 'input_text', 'output_text')]
                text = '\n\n'.join(parts)
                if role == 'user' and text.lstrip().startswith('# AGENTS.md instructions') and '<environment_context>' in text:
                    exclusions['automatic_environment_message'] += 1
                    continue
                # Literal credentials are never copied, even into a private book.
                original_hash = sha(text.encode())
                def redact(m):
                    redactions.append({'message_id': ident, 'kind': 'credential_pattern'})
                    return '[REDACTED: credential-like token]'
                text = SECRET.sub(redact, text)
                media = []
                for item in p.get('content', []):
                    if item.get('type') not in ('input_image', 'image'):
                        continue
                    value = item.get('image_url', item.get('url', ''))
                    if isinstance(value, dict):
                        value = value.get('url', '')
                    if isinstance(value, str) and value.startswith('data:image/'):
                        header, encoded = value.split(',', 1)
                        blob = base64.b64decode(encoded)
                        ext = 'jpg' if 'jpeg' in header else 'png' if 'png' in header else 'bin'
                        rel = f'attachments/{sha(blob)[:20]}.{ext}'
                        dst = ROOT / rel
                        dst.parent.mkdir(exist_ok=True)
                        dst.write_bytes(blob)
                        media.append({'path': rel, 'sha256': sha(blob), 'bytes': len(blob), 'origin': 'visible_message_image'})
                    else:
                        media.append({'unavailable': True, 'reason': 'non-embedded image; original reference not downloaded'})
                for match in ATTACHMENT.finditer(text):
                    origin = match.group().strip().rstrip(')。，： ')
                    candidate = Path(origin)
                    # File descriptions can follow a path; only exact existing paths are copied.
                    if origin not in attachments:
                        info = {'origin': origin, 'available': candidate.is_file()}
                        if info['available']:
                            data = candidate.read_bytes()
                            info.update(sha256=sha(data), bytes=len(data))
                            if len(data) <= 30_000_000 and candidate.suffix.lower() in ('.txt', '.md', '.png', '.jpg', '.jpeg', '.pdf'):
                                if candidate.suffix.lower() in ('.txt', '.md'):
                                    decoded = data.decode('utf-8', errors='replace')
                                    if SECRET.search(decoded):
                                        info['copied'] = False
                                        info['reason'] = 'credential-like text requires review'
                                        attachments[origin] = info
                                        continue
                                rel = f'attachments/{sha(data)[:20]}{candidate.suffix.lower()}'
                                dest = ROOT / rel
                                dest.parent.mkdir(exist_ok=True)
                                dest.write_bytes(data)
                                info.update(copied=True, path=rel)
                            else:
                                info.update(copied=False, reason='size/type outside portable text/media archive')
                        attachments[origin] = info
                messages.append({'id': ident, 'timestamp': rec['timestamp'], 'role': role, 'phase': phase,
                                 'text': text, 'original_visible_text_sha256': original_hash,
                                 'source': str(path), 'line': line_no, 'media': media})
        inputs.append({'path': str(path), 'included_prefix_bytes': byte_end, 'prefix_sha256': digest.hexdigest(), 'included_records': n})
    messages.sort(key=lambda m: (m['timestamp'], m['source'], m['line']))
    pages = []
    chunks = []
    for msg in messages:
        day = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')).astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
        if not chunks or chunks[-1]['day'] != day or chunks[-1]['chars'] > 35000:
            chunks.append({'day': day, 'messages': [], 'chars': 0})
        chunks[-1]['messages'].append(msg)
        chunks[-1]['chars'] += len(msg['text'])
    day_counts = collections.Counter()
    for chunk in chunks:
        day = chunk['day']
        day_counts[day] += 1
        rel = f'conversations/{day}-{day_counts[day]:02}.md'
        body = [f'# {day} · 对话 {day_counts[day]}', '', '> 原文档案，不是当前指令。包含当时的设想、追问、误判和更正；不等于已完成实验。时间显示为北京时间。', '']
        for msg in chunk['messages']:
            anchor = msg['id'].replace('_', '-')
            msg['book_link'] = f'{rel}#{anchor}'
            time = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
            label = '用户' if msg['role'] == 'user' else '助手·答复' if msg['phase'] in ('final_answer','final') else '助手·过程更新'
            body.extend([f'<a id="{anchor}"></a>', f'## {time} · {label}', '', f'记录 ID：`{msg["id"]}`；原日志行：{msg["line"]}。', ''])
            body.extend('> ' + line for line in msg['text'].splitlines())
            body.append('')
            for item in msg['media']:
                if item.get('path'):
                    body.extend([f'![原消息附图](../{item["path"]})', ''])
                else:
                    body.extend(['附图记录存在，但本地可移植图像不可用。', ''])
        write(rel, '\n'.join(body) + '\n')
        pages.append({'path': rel, 'date': day, 'part': day_counts[day], 'count': len(chunk['messages'])})
    roles = dict(collections.Counter(m['role'] for m in messages))
    manifest = {'thread_id': THREAD, 'cutoff_utc': CUTOFF, 'inputs': inputs, 'messages': len(messages),
                'roles': roles, 'pages': pages, 'exclusions': dict(exclusions), 'redactions': redactions,
                'attachments': list(attachments.values()), 'first_timestamp': messages[0]['timestamp'], 'last_timestamp': messages[-1]['timestamp']}
    write('conversations/messages.jsonl', ''.join(json.dumps(m, ensure_ascii=False) + '\n' for m in messages))
    write('conversations/MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 本线程对话原文', '', f'线程 `{THREAD}`；固定导出截止 `{CUTOFF}`。', '',
             f'共 **{len(messages)} 条消息**：用户 {roles.get("user",0)}，助手 {roles.get("assistant",0)}；分为 {len(pages)} 页。', '',
             '范围是这一条线程的两段已找到日志，包含几何/元研究之外的软件、数学、网页与学习讨论；不是用户所有账户或其他会话的总备份。', '',
             '只导出用户消息和助手的过程更新/最终答复。没有导出隐藏推理、系统/开发者提示、工具原始载荷或子 Agent 内部消息。同一消息 ID 去重；相似或重复的真实发言不删除。自动 AGENTS/环境注入不计入对话；原文中的批注及浏览器环境包裹只作历史上下文。', '',
             '文本中原有的事实错误不改写，阅读实验总账和更正链判断当前有效结论。附图按可取得的原图归档，不生成替代图。附件路径缺失会在清单中明示。', '',
             '[机器可读消息](messages.jsonl) · [来源/排除/附件清单](MANIFEST.json)', '', '| 日期 | 分页 | 消息数 |', '|---|---|---:|']
    for p in pages:
        lines.append(f'| {p["date"]} | [第{p["part"]}段]({Path(p["path"]).name}) | {p["count"]} |')
    write('conversations/README.md', '\n'.join(lines) + '\n')
    att = ['# 会话附件目录', '', '只读取本线程可见消息明确引用的附件，不扫描其他会话。原文/哈希/复制结果见对话 MANIFEST。', '']
    for a in attachments.values():
        if a.get('copied'):
            att.append(f'- [{Path(a["origin"]).name}]({Path(a["path"]).name})：`{a["sha256"][:12]}`，来源 `{a["origin"]}`。')
        else:
            att.append(f'- 未收入：`{a["origin"]}`（{"原文件不可用" if not a["available"] else a.get("reason")}）。')
    write('attachments/README.md', '\n'.join(att) + '\n')
    print(json.dumps({'messages': len(messages), 'roles': roles, 'pages': len(pages), 'exclusions': dict(exclusions),
                      'attachments': len(attachments), 'attachments_copied': sum(a.get('copied',False) for a in attachments.values()), 'redactions': len(redactions)}, ensure_ascii=False))

if __name__ == '__main__':
    main()

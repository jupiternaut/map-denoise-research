#!/usr/bin/env python3
"""Build portable evidence snapshots, ledger, topic index and offline reader.

Only documentation is generated. No experiment scripts or archived commands run.
"""
import argparse
import collections
import hashlib
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path('/home/grf/Documents/Codex/2026-09-28/unified-revision-theory-20260928T114318Z')
EXTRA = [HISTORY / n for n in ('UNIFIED_THEORY.md', 'EVIDENCE_MATRIX.md', 'EVIDENCE_MATRIX.json', 'CLAIMS.json', 'HISTORY_COVERAGE.md', 'REVISION_LOG.md')]
FIELDS = {'id','title','date','track','question','did','not_done','result','interpretation','decision','evidence_level','sources','tags','predecessors','corrections','rerun_condition'}
TOPICS = {
    '01-information': ('信息、可辨识与观测归属', ['不可辨识','可辨识','关联','归属','信息','双世界']),
    '02-candidates': ('候选、表示、Oracle 与全局最优', ['候选','Oracle','oracle','表示','全局最优','P=NP','精确']),
    '03-spatial': ('空间混合、斜率与薄层保持', ['空间','斜率','层距','薄层','双层','混合']),
    '04-render': ('渲染、遮挡、纹理来源', ['渲染','遮挡','纹理','donor','ZNCC','可见性']),
    '05-learning': ('机器学习、收益预测、KEEP/MOVE', ['学习','预测器','双头','回归','KEEP','MOVE','门控']),
    '06-meta': ('元研究、修复能力与自适应实验', ['元研究','修复器','自适应','开环','目录','CEGIS','调度']),
    '07-cpr': ('CPR、校准、BH 与投影', ['CPR','conformal','BH','校准','投影','覆盖率']),
    '08-engineering': ('软件范式、TLA+、接口与 AI 协作', ['TLA','RUNTIME','runtime','AGENT','SDK','MCP','CLI','TUI','HOOK','状态机']),
    '09-thesis': ('结题、论文与同行评审', ['论文','博士','硕士','结题','导师','审稿','贡献','复现']),
    '10-mathematics': ('数学桥梁与学习讨论', ['范畴','测度','柯里','希尔伯特','高斯','kernel','ISA','模糊数学','认知']),
    '11-web': ('网页、个人博客与计算史', ['网页','Brutalism','masswerk','两朵云','Hilbert','千禧年','Bojie','01.me']),
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def put(rel, text):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')

def read_records():
    rows = []
    for p in sorted((ROOT / 'records').glob('*.json')):
        if p.name == 'INDEX.json':
            continue
        value = json.loads(p.read_text())
        assert isinstance(value, list), p
        for row in value:
            assert FIELDS <= set(row), (p, row.get('id'), FIELDS-set(row))
            for key in ('did','not_done','result','interpretation','decision','sources','tags','predecessors','corrections'):
                assert isinstance(row[key], list), (p,row['id'],key)
            assert row['evidence_level'] in ('report_read','artifact_checked','historical_index','dialogue_only'), row['id']
        rows.extend(value)
    assert len({r['id'] for r in rows}) == len(rows), 'duplicate IDs'
    return rows

def portable_links(text, mapping, current_page):
    """Link archived absolute citations to portable source pages; raw stays raw."""
    import os
    def replace(match):
        label,target=match.group(1),match.group(2).strip('<>')
        origin=re.sub(r':\d+(?::\d+)?$','',target.split('#')[0])
        if origin not in mapping:return match.group(0)
        rel=os.path.relpath(ROOT/mapping[origin],current_page.parent)
        return f'[{label}]({rel})'
    return re.sub(r'\[([^\]\n]+)\]\((<?/[^\n)]+>?)\)',replace,text)

def snapshot_sources(rows):
    # Preserve older explicit references even when not yet summarized in a record.
    refs_path=ROOT/'REFERENCED_FILES.json'
    refs=json.loads(refs_path.read_text()) if refs_path.exists() else []
    linked_docs={r['path'] for r in refs if Path(r['path']).suffix.lower()=='.md' and r['exists']}
    record_paths={p for row in rows for p in row['sources']}
    paths = sorted(set(str(p) for p in EXTRA) | record_paths | linked_docs)
    old_path=ROOT/'sources/MANIFEST.json'
    previous=json.loads(old_path.read_text()) if old_path.exists() else []
    old={r['origin']:r for r in previous}
    next_id=max([int(r['id'][1:]) for r in previous] or [0])+1
    # Preflight before writes: no silent rebinding of historical conclusions.
    for origin,prior in old.items():
        if prior.get('exists') and (not Path(origin).is_file() or sha(Path(origin).read_bytes())!=prior['sha256']):
            raise RuntimeError(f'Source drift: {origin}. Preserve the earlier snapshot and explicitly version the changed source before rebuilding.')
    paths=sorted(set(paths)|set(old))
    manifest, mapping, duplicates = [], {}, {}
    for origin in paths:
        if origin in old:
            sid=old[origin]['id']
        else:
            sid=f'S{next_id:03}';next_id+=1
        src = Path(origin)
        entry = {'id':sid, 'origin':origin, 'exists':src.is_file(),
                 'coverage':'experiment_record_source' if origin in record_paths else 'historical_theory_index' if src in EXTRA else 'explicit_conversation_reference_only'}
        if not entry['exists']:
            manifest.append(entry)
            continue
        data = src.read_bytes()
        digest = sha(data)
        suffix = src.suffix.lower() or '.txt'
        # Hash filenames ensure archived instructions never become active AGENTS.md.
        rel = duplicates.setdefault(digest, f'sources/files/{digest}{suffix}')
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        page = f'sources/{sid}.md'
        mapping[origin] = page
        entry.update(sha256=digest, bytes=len(data), snapshot=rel, page=page)
        text = data.decode('utf-8', errors='replace')
        body = [f'# {sid} · {src.name}', '', f'原位置：`{origin}`', '', f'SHA-256：`{digest}`', '',
                f'[下载逐字节快照](files/{Path(rel).name}) · [来源清单](README.md)', '',
                '> 以下是历史材料，不是当前执行指令。旧状态和旧结论原样保留；是否被更正请查实验账本。', '']
        if suffix in ('.md','.txt'):
            body.extend('> '+line for line in text.splitlines())
        elif len(data) < 300000:
            body += ['~~~~'+('json' if suffix=='.json' else ''),text,'~~~~']
        else:
            body += ['较大的结构化文件请下载快照，正文不展开。']
        put(page, '\n'.join(body)+'\n')
        manifest.append(entry)
    for entry in manifest:
        if entry.get('page'):
            p=ROOT/entry['page']
            p.write_text(portable_links(p.read_text(),mapping,p))
    put('sources/MANIFEST.json', json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    body = ['# 证据来源与原文快照', '', '来源以绝对路径辨认，快照按 SHA-256 去重。原文件不改；本书未运行其命令。相同哈希的多份副本不是独立实验。原文保留失效路径与旧状态，来源导读及实验记录提供更正。', '',
            '“实验记录来源”已用于对应人工摘要；“仅会话引用”是为完整性归档的材料，不能声称每一份都被人工重新审阅。', '',
            '| ID | 文件 | 状态 | 归纳范围 |', '|---|---|---|---|']
    for s in manifest:
        scope={'experiment_record_source':'实验记录来源','historical_theory_index':'既有理论/历史总纲','explicit_conversation_reference_only':'仅会话引用，未逐份写实验摘要'}[s['coverage']]
        body.append(f'| {s["id"]} | [{Path(s["origin"]).name}]({s["id"]}.md) | 已归档 `{s["sha256"][:12]}` | {scope} |' if s['exists'] else f'| {s["id"]} | `{s["origin"]}` | 原文件缺失 | {scope} |')
    put('sources/README.md', '\n'.join(body)+'\n')
    return manifest,mapping

def build_experiments(rows,mapping):
    index = ['# 实验总账', '', '每条记录都是一个明确问题或阶段；不是独立样本计数。历史读报不冒充本轮复算。“未做”包括明确未执行、未证明及未找到记录，按每项措辞区分。', '',
             '| ID | 日期 | 分支 | 问题 / 实验 | 证据级别 |','|---|---|---|---|---|']
    labels = [('question','问题'),('did','实际做了什么'),('not_done','没做什么 / 没有证明什么'),('result','报告记录的结果'),('interpretation','解释及边界'),('decision','研究决定'),('corrections','更正 / 被撤回的解释'),('predecessors','前项'),('rerun_condition','什么情况下值得重做')]
    corrections = ['# 更正与撤回链', '', '更正不是删除失败，而是限制旧结论的有效范围。以下从实验记录自动聚合；完整上下文和原版本仍可阅读。','']
    for row in rows:
        rel = f'experiments/{row["id"]}.md'
        body = [f'# {row["id"]} · {row["title"]}', '', f'日期：{row["date"]} · 分支：{row["track"]} · 证据：`{row["evidence_level"]}`', '', '[返回实验总账](../EXPERIMENTS.md)', '']
        for key,label in labels:
            value = row[key]
            body += [f'## {label}', '']
            if isinstance(value,list):
                body += ['- '+str(x) for x in value] if value else ['本记录未列出；不推断为不存在。']
            else:
                body.append(str(value))
            body.append('')
        body += ['## 来源', '']
        for p in row['sources']:
            body.append(f'- [{Path(p).name}](../{mapping[p]}) · 原位置 `{p}`' if p in mapping else f'- 缺失来源：`{p}`')
        body += ['', '主题：'+ ' / '.join(row['tags'])]
        put(rel,'\n'.join(body)+'\n')
        index.append(f'| [{row["id"]}]({rel}) | {row["date"]} | {row["track"]} | {row["title"]} | {row["evidence_level"]} |')
        if row['corrections']:
            corrections += [f'## [{row["id"]} · {row["title"]}]({rel})', '']+['- '+x for x in row['corrections']]+['']
    put('EXPERIMENTS.md','\n'.join(index)+'\n')
    put('CORRECTIONS.md','\n'.join(corrections)+'\n')
    put('records/INDEX.json', json.dumps({'generated':True,'count':len(rows),'ids':[r['id'] for r in rows],
        'records_sha256':sha(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()),
        'pages_sha256':{r['id']:sha((ROOT/f'experiments/{r["id"]}.md').read_bytes()) for r in rows}},ensure_ascii=False,indent=2)+'\n')

def build_topics(rows,messages):
    index = ['# 按问题检索，不按版本重新开始', '', '自动关键词索引用于发现线索，不宣称命中就表示实验相同。相关实验全列；对话只列用户问题与助手最终答复，过程更新仍在原文页。', '']
    for slug,(title,words) in TOPICS.items():
        match = lambda s:any(w.casefold() in s.casefold() for w in words)
        selected = [r for r in rows if match(json.dumps(r,ensure_ascii=False))]
        hits = [m for m in messages if (m['role']=='user' or m['phase'] in ('final_answer','final')) and match(m['text'])]
        body = [f'# {title}', '', f'关键词：{"、".join(words)}。自动命中 {len(selected)} 条实验记录、{len(hits)} 条对话。', '', '## 实验与更正入口', '']
        body += [f'- [{r["id"]} · {r["title"]}](../experiments/{r["id"]}.md)' for r in selected] or ['本轮未找到对应结构化记录，查原对话。']
        body += ['', '## 当时怎样讨论', '']
        for m in hits:
            compact = re.sub(r'<[^>]+>',' ',m['text'])
            if '## My request:' in compact:
                compact = compact.split('## My request:')[-1]
            compact = re.sub(r'\s+',' ',compact).strip().replace('[','［').replace(']','］')
            body.append(f'- {m["timestamp"][:10]} · {"用户" if m["role"]=="user" else "助手"}：[ {compact[:100]} ](../{m["book_link"]})')
        put(f'topics/{slug}.md','\n'.join(body)+'\n')
        index.append(f'- [{title}]({slug}.md)：{len(selected)} 条记录 / {len(hits)} 条对话')
    put('topics/README.md','\n'.join(index)+'\n')

def build_navigation(rows,conversation,manifest):
    lines=['# 目录','','* [首页](README.md)','* [当前状态](CURRENT_STATE.md)','* [旧待办对账](STATUS_RECONCILIATION.md)','* [怎样避免重复](REPEAT_CHECK.md)','* [交给下一位 Agent](NEXT_AGENT.md)', '', '## 研究叙事', '']
    for p in sorted((ROOT/'chapters').glob('*.md')):
        lines.append(f'* [{p.read_text().splitlines()[0].lstrip("# ")}]({p.relative_to(ROOT)})')
    lines += ['', '## 证据与检索', '', '* [问题索引](topics/README.md)']
    for slug,(title,_) in TOPICS.items(): lines.append(f'  * [{title}](topics/{slug}.md)')
    lines += ['* [实验总账](EXPERIMENTS.md)']
    for r in rows: lines.append(f'  * [{r["id"]} {r["title"]}](experiments/{r["id"]}.md)')
    lines += ['* [更正链](CORRECTIONS.md)','* [历史原始来源](sources/README.md)']
    for s in manifest:
        if s['exists']:lines.append(f'  * [{s["id"]} {Path(s["origin"]).name}]({s["page"]})')
    lines += ['', '## 对话档案', '', '* [对话范围与原文](conversations/README.md)']
    for p in conversation['pages']:lines.append(f'  * [{p["date"]} · {p["part"]}]({p["path"]})')
    lines += ['* [附件](attachments/README.md)','','## 维护与边界','','* [覆盖范围](COVERAGE.md)','* [对话引用文件补漏](REFERENCED_FILES.md)','* [导入 GitBook 与维护](PUBLISHING.md)','* [本次建书记录](BUILD_LOG.md)','* [核验结果](VALIDATION.md)']
    put('SUMMARY.md','\n'.join(lines)+'\n')

def coverage(rows,conversation,manifest):
    ev=collections.Counter(r['evidence_level'] for r in rows)
    missing=[x for x in manifest if not x['exists']]
    attachment_missing=[x for x in conversation['attachments'] if not x.get('copied')]
    body=['# 覆盖范围与已知缺口','',f'本书截止对话时间：`{conversation["cutoff_utc"]}`。书稿构建日期：2026-09-29。','',
          '## 已收进书里的内容','',
          f'- 本线程两段日志：{conversation["messages"]} 条可见消息，{len(conversation["pages"])} 个分页；用户 {conversation["roles"]["user"]}，助手 {conversation["roles"]["assistant"]}。',
          f'- {len(rows)} 条结构化实验/阶段记录，{len(manifest)} 个来源路径，{len(set(x.get("sha256") for x in manifest if x["exists"]))} 份唯一内容快照。',
          f'- 证据级别：`{dict(ev)}`。每项写明做了什么、没做/未证明什么、更正与重做条件。',
          '- 既有统一理论、93条历史证据矩阵、命题状态与修订记录作为原文快照保留。它们与本书索引不是新增独立实验。',
          '- 本线程明确链接且可得的Markdown也做了原文快照；来源表区分“实验摘要依据”和“仅对话引用”，后者不是已人工逐份审计。','',
          '## 明确没有声称覆盖的内容','',
          '- 不是所有 ChatGPT/Cursor/Hermes/其他账户会话的完整导出；本书导出了当前 Codex 线程，Hermes/Fable 材料依其报告与本线程粘贴记录。',
          '- 日志开始之前的外部线程没有静默导入。QCE与道路重定位以明确引用的交接/报告补记。',
          '- 没有逐个重读所有源码、重算所有CSV或复制全部数据集/PLY/模型权重。来源快照保存研究记录，大型工件仍在来源列明的位置。',
          '- 论文草稿、软件架构与网页学习等讨论保留在原对话/主题索引；不将它们伪装成几何实验。',
          '- 所有实验记录是一轮人工证据整理，不保证已发现每个未命名子实验。缺项可用记录追加，不覆盖历史。另见 [对话引用文件盘点](REFERENCED_FILES.md)，明确区分已归档与尚未逐项归档的支线。',
          '- 同线程日志完整到导出截止，不等于运行平台必然保留了此前从未写入日志的消息。','',
          '## 附件与原始路径','',f'原文明确引用附件 {len(conversation["attachments"])} 项；已复制 {len(conversation["attachments"])-len(attachment_missing)}，未收入 {len(attachment_missing)}。消息内嵌图另按原始媒体保存。',
          f'原报告来源缺失 {len(missing)} 项。详细状态见 [来源清单](sources/MANIFEST.json) 和 [会话清单](conversations/MANIFEST.json)。','',
          '## 隐私与当前发布状态','',
          '本书仅在 liekkas 的新工作区生成，没有 GitHub/GitBook 发布。对话含个人交流、内部路径和研究意见；公开前需要单独选择公开版范围。已排除隐藏推理、系统/开发者记录、工具原始载荷和子 Agent 内部通信。自动凭据模式检查不能替代公开前人工隐私审查。']
    put('COVERAGE.md','\n'.join(body)+'\n')

def reader():
    """Simple offline document reader; no remote JS, scripts or images execute."""
    import markdown
    try:
        import bleach
    except ImportError:
        bleach=None
    summary=(ROOT/'SUMMARY.md').read_text()
    entries=re.findall(r'^\s*\* \[([^\]]+)\]\(([^)]+\.md)\)',summary,re.M)
    page_set={rel for _,rel in entries}
    # Avoid expanding every raw source twice; source pages and transcripts remain standalone.
    sections=[]; nav=[]
    for title,rel in entries:
        p=ROOT/rel
        if not p.is_file():continue
        anchor=re.sub(r'[^a-zA-Z0-9_-]','-',rel)
        nav.append(f'<li><a href="#{anchor}">{html.escape(title)}</a></li>')
        text=p.read_text()
        # Rendering Markdown is safe only after removing scripts/active HTML.
        rendered=markdown.markdown(text,extensions=['tables','fenced_code','sane_lists'])
        if bleach:
            tags={'p','h1','h2','h3','h4','h5','h6','ul','ol','li','strong','em','code','pre','blockquote','table','thead','tbody','tr','td','th','a','hr','br','img'}
            rendered=bleach.clean(rendered,tags=tags,attributes={'a':['href','id'],'img':['src','alt'],'th':['align'],'td':['align']},protocols=['http','https','mailto'],strip=True)
        else:
            rendered='<pre>'+html.escape(text)+'</pre>'
        # No external images: opening the local book does not contact tracking servers.
        rendered=re.sub(r'<img\b[^>]*src=["\']https?://[^>]*>', '<p>[远程图片未自动加载；见原文链接]</p>', rendered)
        def local_link(m):
            target=html.unescape(m.group(1))
            if target.startswith(('https://','http://','mailto:','#')):return m.group(0)
            path,sep,frag=target.partition('#')
            resolved=(p.parent/unquote(path)).resolve()
            try:new=resolved.relative_to(ROOT).as_posix()
            except ValueError:return 'href="'+html.escape(target,quote=True)+'"'
            if new in page_set:
                return 'href="#'+(frag if frag.startswith('msg-') else re.sub(r'[^a-zA-Z0-9_-]','-',new))+'"'
            return 'href="'+html.escape(new,quote=True)+'"'
        rendered=re.sub(r'href="([^"]+)"',local_link,rendered)
        # Original message IDs provide deep links into the combined reader.
        for mid in re.findall(r'<a id="([^"]+)"></a>',text):
            if f'id="{mid}"' not in rendered:rendered=f'<a id="{mid}"></a>'+rendered
        # Resolve copied image paths relative to the combined reader.
        def local_image(m):
            src=html.unescape(m.group(1))
            resolved=(p.parent/unquote(src)).resolve()
            try:relimg=resolved.relative_to(ROOT).as_posix()
            except ValueError:return 'src=""'
            if not resolved.is_file():return 'src=""'
            return 'src="'+html.escape(relimg,quote=True)+'"'
        rendered=re.sub(r'src="([^"]+)"',local_image,rendered)
        sections.append(f'<section id="{anchor}">{rendered}</section>')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>研究记忆书 · 本地阅读</title><style>
body{margin:0;background:#faf9f5;color:#202624;font:17px/1.8 system-ui,sans-serif}aside{position:fixed;top:0;bottom:0;width:260px;overflow:auto;padding:24px;background:#eeeee7;font-size:13px}main{margin-left:310px;max-width:980px;padding:40px}a{color:#146657}h1,h2,h3{line-height:1.35}section{padding:22px 0 54px;border-bottom:2px solid #b4c4ba}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{border:1px solid #ccd3cb;padding:8px 12px}blockquote{margin:20px 0;padding:0 18px;border-left:3px solid #abbcb0;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#efefe9;padding:18px;font-size:13px}img{max-width:100%;height:auto}code{font-size:.85em;overflow-wrap:anywhere}aside ul{padding-left:18px}aside li{margin:7px 0}@media(max-width:900px){aside{position:static;width:auto;max-height:240px}main{margin:0;padding:22px}}@media print{aside{display:none}main{margin:0}section{break-before:page}}</style><aside><h2>研究记忆书</h2><p>本地私有 · Ctrl/Cmd+F 搜索全文<br>无外部脚本或远程图片自动加载</p><ul>'''+''.join(nav)+'</ul></aside><main><p>这是 GitBook 兼容书稿的离线阅读页，不是已发布的 GitBook 网站。原始 Markdown、来源快照及检索工具随书保存。</p>'+''.join(sections)+'</main></html>'
    put('READ.html',doc)

def validate():
    rows=read_records()
    record_index=json.loads((ROOT/'records/INDEX.json').read_text())
    assert record_index['ids']==[r['id'] for r in rows], 'Records changed; rebuild the book first'
    assert record_index['records_sha256']==sha(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()),'Record contents changed; rebuild first'
    manifest=json.loads((ROOT/'sources/MANIFEST.json').read_text())
    assert {p for r in rows for p in r['sources']} <= {s['origin'] for s in manifest}
    checks=[]
    for s in manifest:
        if s['exists']:
            assert sha((ROOT/s['snapshot']).read_bytes())==s['sha256'],s['id']
            assert sha(Path(s['origin']).read_bytes())==s['sha256'],s['origin']
    checks.append('所有来源快照哈希与原文件一致')
    summary=(ROOT/'SUMMARY.md').read_text()
    nav=re.findall(r'\]\(([^)]+\.md)\)',summary)
    for link in nav:assert (ROOT/link).is_file(),link
    for r in rows:
        rel=f'experiments/{r["id"]}.md'
        assert rel in nav,r['id']
        assert sha((ROOT/rel).read_bytes())==record_index['pages_sha256'][r['id']],r['id']
    checks.append('记录内容、生成页、INDEX、导航与来源清单一致；修改记录后须重建')
    checks.append(f'SUMMARY 的 {len(nav)} 个 Markdown 入口存在')
    required=['README.md','CURRENT_STATE.md','STATUS_RECONCILIATION.md','REPEAT_CHECK.md','NEXT_AGENT.md','PUBLISHING.md','COVERAGE.md']+[str(p.relative_to(ROOT)) for p in (ROOT/'chapters').glob('*.md')]
    broken=[]
    for rel in required+[str(p.relative_to(ROOT)) for p in (ROOT/'experiments').glob('*.md')]+[str(p.relative_to(ROOT)) for p in (ROOT/'topics').glob('*.md')]:
        p=ROOT/rel
        for target in re.findall(r'\]\(([^\n)]+)\)',p.read_text()):
            target=target.strip('<>').split('#')[0]
            if target and not target.startswith(('http:','https:','mailto:','/')):
                if not (p.parent/unquote(target)).exists():broken.append((rel,target))
    assert not broken,broken[:20]
    checks.append('首页、当前状态、实验和主题内部文件链接无缺失')
    messages=[json.loads(l) for l in (ROOT/'conversations/messages.jsonl').read_text().splitlines()]
    ids=[m['id'] for m in messages]
    assert len(ids)==len(set(ids))
    assert all(m['role'] in ('user','assistant') for m in messages)
    assert all(m['role']=='user' or m['phase'] in ('commentary','final_answer','final') for m in messages)
    assert all((ROOT/m['book_link'].split('#')[0]).is_file() for m in messages)
    cm=json.loads((ROOT/'conversations/MANIFEST.json').read_text())
    assert len(messages)==cm['messages']
    # Verify the exact frozen prefixes and independently match visible text only.
    exported={m['id']:m for m in messages}
    validated_ids=set()
    for src in cm['inputs']:
        h=hashlib.sha256();left=src['included_prefix_bytes']
        with Path(src['path']).open('rb') as f:
            for data in f:
                if not left:break
                assert len(data)<=left
                h.update(data);left-=len(data)
                rec=json.loads(data);p=rec.get('payload',{})
                if rec.get('type')=='response_item' and p.get('type')=='message' and p.get('id') in exported:
                    m=exported[p['id']]
                    text='\n\n'.join(x.get('text','') for x in p.get('content',[]) if x.get('type') in ('text','input_text','output_text'))
                    assert sha(text.encode())==m['original_visible_text_sha256'],m['id']
                    assert m['role']==p.get('role') and m['phase']==p.get('phase')
                    if not any(red['message_id']==m['id'] for red in cm['redactions']):assert text==m['text'],m['id']
                    validated_ids.add(m['id'])
        assert left==0
        assert h.hexdigest()==src['prefix_sha256']
    assert validated_ids==set(exported)
    checks.append(f'{len(messages)} 条可见消息唯一、角色/阶段合规、原日志冻结前缀哈希匹配')
    for a in cm['attachments']:
        if a.get('copied'):assert sha((ROOT/a['path']).read_bytes())==a['sha256']
    for m in messages:
        page=(ROOT/m['book_link'].split('#')[0]).read_text()
        assert m['id'] in page and m['book_link'].split('#')[1] in page
        quoted='\n'.join('> '+line for line in m['text'].splitlines())
        assert quoted in page,m['id']
        for media in m['media']:
            if media.get('path'):assert sha((ROOT/media['path']).read_bytes())==media['sha256']
    checks.append('每条会话文本对原日志逐项匹配；分页正文和内嵌附图一致')
    checks.append('已复制的消息附件逐字节哈希匹配')
    out={'status':'passed','experiment_records':len(rows),'source_paths':len(manifest),'conversation_messages':len(messages),'navigation_pages':len(nav),'checks':checks,
         'limits':['未重新执行科学实验','没有声称每条历史结论均正确','本轮不检验所有历史原文链接或外部URL可达性','自动凭据检查不替代公开前隐私审查']}
    put('VALIDATION.json',json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    put('VALIDATION.md','# 书稿核验\n\n状态：通过。这里检查档案和索引的一致性，不验证科学假说。\n\n'+'\n'.join('- '+s for s in checks)+'\n\n## 未声称\n\n'+'\n'.join('- '+s for s in out['limits'])+'\n')
    print(json.dumps(out,ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    if args.validate:
        validate();return
    # INDEX is generated metadata, not a schema record file.
    rows=read_records()
    cm=json.loads((ROOT/'conversations/MANIFEST.json').read_text())
    messages=[json.loads(l) for l in (ROOT/'conversations/messages.jsonl').read_text().splitlines()]
    manifest,mapping=snapshot_sources(rows)
    for p in (ROOT/'chapters').glob('*.md'):
        p.write_text(portable_links(p.read_text(),mapping,p))
    subprocess.run([sys.executable,str(ROOT/'tools/inventory_references.py')],check=True)
    build_experiments(rows,mapping)
    build_topics(rows,messages)
    coverage(rows,cm,manifest)
    build_navigation(rows,cm,manifest)
    if not (ROOT/'VALIDATION.md').exists():put('VALIDATION.md','# 待核验\n')
    validate()
    reader()
    print('Built READ.html and GitBook Markdown; no publication performed.')

if __name__=='__main__':main()

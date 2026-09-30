#!/usr/bin/env python3
"""List explicitly linked local files in exported visible messages.

This inventories path existence, not contents or unrelated directories.
"""
import collections
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
messages=[json.loads(x) for x in (ROOT/'conversations/messages.jsonl').read_text().splitlines()]
sources=json.loads((ROOT/'sources/MANIFEST.json').read_text())
indexed={x['origin']:x for x in sources}
found={}
for m in messages:
    targets=re.findall(r'\]\(<?((?:/home/grf/|/srv/slam-research/)[^\n)]+)>?\)',m['text'])
    targets+=re.findall(r'`((?:/home/grf/|/srv/slam-research/)[^\n`]+\.(?:md|json|csv|py|pdf))`',m['text'])
    for p in targets:
        p=p.rstrip('>')
        p=re.sub(r':\d+(?::\d+)?$','',p)
        if not Path(p).suffix:continue
        row=found.setdefault(p,{'path':p,'exists':Path(p).is_file(),'source_index':indexed.get(p,{}).get('id'),'messages':[]})
        row['messages'].append(m['book_link'])
rows=list(found.values())
(ROOT/'REFERENCED_FILES.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
docs=[r for r in rows if Path(r['path']).suffix.lower()=='.md']
body=['# 对话引用文件盘点','',
      '这是补漏索引：只提取已导出消息中明确给出的本地 Markdown 链接/代码路径，不扫描其他项目。文件存在不代表本次已读；未入账不代表没做过。它帮助后续补齐未命名实验、论文、软件和交接支线。','',
      f'识别 {len(rows)} 个不同文件路径，其中 Markdown {len(docs)} 个。这个计数不是实验数。完整结构见 [JSON](REFERENCED_FILES.json)。','',
      '| 文件 | 本地存在 | 本书原文快照 | 当时消息 |','|---|---|---|---|']
for r in docs:
    source=f'[{r["source_index"]}](sources/{r["source_index"]}.md)' if r['source_index'] else '未逐项归档/待补索引'
    body.append(f'| `{r["path"]}` | {"是" if r["exists"] else "否"} | {source} | [最早引用]({r["messages"][0]}) |')
(ROOT/'REFERENCED_FILES.md').write_text('\n'.join(body)+'\n')
print(json.dumps({'explicit_file_paths':len(rows),'markdown_paths':len(docs),'markdown_not_archived':sum(not r['source_index'] for r in docs),'missing_paths':sum(not r['exists'] for r in rows)},ensure_ascii=False))

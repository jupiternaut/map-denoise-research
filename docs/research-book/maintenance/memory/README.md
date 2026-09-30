# 增量研究记忆

## 2026-10-01 追加与公开包装

新增LG036–LG039：构造照片留出及成像冲突、相机接口修复交叉回放、官方COLMAP固定像素对照、两个预留新物体的冻结迁移；原报告与更正链分别登记，没有只保留最新好数字。迁移直接对原CSV汇总并纳入最终独立审查，保留搜索域不对称、回退残差和误改。

用户本轮已明确授权完整GitBook上传现有GitHub仓库。公开包装脚本为 `prepare_publication.py`，输出到既有仓库 `docs/research-book/`，原书/ZIP不变。它复制完整原书和全部发布增量，合并可移植导航、网页与已过滤可见对话，并生成外部 `PUBLICATION_EXPECTED_MANIFEST.json` 供暂存索引核验。脚本不commit/push。

本轮新对话快照使用显式截止 `2026-09-30T17:53:00Z`，在当前迁移执行之前冻结，不追逐本轮工具/推理或未来消息。实际收录增量167条（相对上次新增65条），最后可见时间17:50:13.602Z；加原书1807条共1974条。现为120+8=128条实验记录。原快照和旧消息锚点保留。

新增图件按登记来源哈希复制，报告内相对图片转换到同一增量的 `media/`；没有扫描并复制未登记的任意图片或运行数据库。

该分支已更新原局域网服务 [研究记忆书](http://192.168.6.201:8765/)。它以只读 overlay 追加新内容，原书 693 个文件和原 ZIP 保持不变。

## 当前交付

- 固定字节快照：`snapshots/initial-20260930/`。
- 初始增量可见对话：28 条，父线程 17 条、当前分叉 11 条；最后可见时间 `2026-09-29T21:34:11.828Z`，即北京时间 9月30日 05:34:11.828。该快照原件保持不变。
- 本次新固定快照：`snapshots/skill-comparison-20260930/`，共 85 条（用户14、助手71；父17、当前分叉68），最后可见时间 `2026-09-30T07:45:42.264Z`，即北京时间15:45:42.264。与初始快照去重后本次新增57条，发布层共85条；不追入此后消息。
- 实验登记：完整 `did/not_done`、前驱、更正、结论、评价作用域与源文件哈希契约。
- 已登记 LG032、LG033、LG034、LG035 共4条增量记录，与原书120条合并索引124条。LG033/034新增23项来源哈希绑定；LG035另绑定34项第二批完成来源，时间作用域和原作业失败/恢复测量分离。
- 第二批收尾新固定快照：`snapshots/batch2-closeout-20260930/`，共102条（用户14、助手88；父17、当前分叉85），最后可见时间`2026-09-30T08:56:56.578Z`，即北京时间16:56:56.578。相对上一已发布85条新增17条；按快照创建顺序去重追加，原消息落点保持。未来消息不追入同名快照。
- [最新状态](LATEST_STATE.md)：LG032已实际完成纹理见证；LG034更新启动前“Shinka未运行”的历史断点；LG035完成第二批及gen1测量技术准入，开发略胜未兑现回放优势，联合目标仍未过，不升级默认。
- 实际提案配置及有效gen3、最后完成gen5的提示词/响应是程序工件，经凭据模式核查后单独保留来源；不是本会话system/developer或隐藏推理导出。
- 合并重复研究索引：旧书 120 条记录加所有实际登记的新记录。关键词匹配是重访提醒，不声称数学等价。
- 验证：9 项单元测试；原书文件/ZIP、原消息锚点、新消息锚点、固定来源前缀及同一 HTTP 返回字节。

## 精确来源与计数更正

父线程源是原书已登记的两段日志。分叉 ID 为 `01a0eef4-3b61-7cc2-ada6-74e67175c7f5`，元数据的 `forked_from_id` 与原根一致。每个导出只读取 `SNAPSHOT.json` 固定的字节上限；后续日志尾部不加入同名快照。

上一份完整性审计写“父线程新增 18 条”。本次增加 `content_item_kinds` 过滤后，实际可见对话为 17 条（用户 3、助手 14）；多出的 `msg_01a0ee61-cd27-7d53-aa9c-4f1fd32ad61b` 是 `environments.environment_context` 自动上下文。它不是用户发言。旧书截止前 1,807 条原消息不受影响。

导出不收入 system/developer/tool/analysis、子 Agent 日志、自动环境/页面消息，也不把原始完整日志或 session 的基础指令写入发布文件。正文内用户引用的文档仍是可见证据，不因像指令而删除。

## 命令

在项目根目录执行：

```bash
/usr/bin/python3 -B memory/memory.py export --snapshot initial-20260930
/usr/bin/python3 -B memory/memory.py register completed-experiment.json
/usr/bin/python3 -B memory/memory.py publish
/usr/bin/python3 -B memory/memory.py verify
/usr/bin/python3 -B memory/memory.py search '纹理 候选 校准' --limit 10
```

新实验记录 JSON 必须包括以下字段；`claims` 是有证据支持的具体主张，不是目标数字。

```json
{
  "id": "NEW_UNIQUE_ID",
  "title": "实际已完成的实验名称",
  "date": "2026-09-30",
  "track": "geometry",
  "question": "本实验回答的具体问题",
  "did": ["实际执行的步骤"],
  "not_done": ["尚未执行的范围"],
  "result": ["实测结果及单位"],
  "interpretation": ["与事实分开的解释"],
  "decision": ["本实验改变的决策"],
  "evidence_level": "artifact_checked",
  "sources": ["/absolute/path/to/completed/REPORT.md"],
  "tags": ["geometry", "calibration"],
  "predecessors": ["LG031"],
  "corrections": [],
  "rerun_condition": "什么真实差异才值得重访",
  "claims": ["已测得的有限范围主张"],
  "evaluation_scope": "已暴露场景回放，不是新增独立确认"
}
```

同一 ID 的完全一致登记可重跑；不允许同 ID 改写事实或来源哈希。更正使用新 ID，并引用原前驱。

下一次更新对话应明确给出**新快照名**；不要在运行中反复导出新快照追上本轮自身消息。`publish` 只汇总已冻结快照和已完成记录，不自动抓取日志。

## LAN 行为

主入口仍是 `http://192.168.6.201:8765/`，新导航位于左侧“新增记忆”。机器索引在 `/increment/EXPERIMENT_INDEX.json`。发布路径实行 allowlist；源日志、运行时脚本、未登记路径和目录遍历不提供访问。

现服务仍是原用户级 transient `research-book-lan.service`：跨聊天继续运行，但没有新增开机自启配置。本分支未公开发布或推送 GitHub。

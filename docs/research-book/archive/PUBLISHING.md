# 浏览、检索、GitBook 导入与维护

## 现在就读

- 打开本目录 `READ.html`，是可离线阅读的整本书；Ctrl/Cmd+F 搜索正文。没有外部脚本，不会自动载入远程图片。
- 用 Markdown 阅读器打开 `SUMMARY.md`，按章节进入。
- 精确检索：

```bash
cd /home/grf/Documents/Codex/2026-09-29/geometry-meta-research-gitbook
python tools/search_book.py '纹理' --scope experiments
python tools/search_book.py '共同移动' --scope conversations --limit 30
python tools/build_book.py --validate
```

检索不需要调用 AI，也不需要重新装载全部会话。每条记录附原来源与快照哈希，可追溯而不必重跑。

## 导入 GitBook

本目录有 `README.md`、`SUMMARY.md` 与 `.gitbook.yaml`，是 GitBook 的 Git Sync 兼容布局。GitBook 以 SUMMARY.md 建立目录，可以连接 GitHub/GitLab 同步 Markdown；连接目标仓库仍需用户在 GitBook 中授权。[GitBook 目录说明](https://gitbook.com/docs/help-center/integrations/integrations-troubleshooting/git-sync/my-table-of-contents-is-not-correctly-structured)、[官方 Git Sync 导入指南](https://gitbook.com/docs/guides/editing-and-publishing-documentation/import-or-migrate-your-content-to-gitbook-with-git-sync)

**当前没有创建在线 Space、没有连接 GitBook 账户、没有上传这本书。** 这是已构建的兼容书稿，不是宣称远程网站已上线。

推荐先导入私有空间。全文包含个人研究交流、历史错误、原始本地路径和账户相关语境；公开项目仓库与私人研究笔记应分开。公开前可单独生成去隐私版，仅保留实验、来源及必要讨论。

## 更新方式

1. 新实验完成后，在 `records/` 添加数组文件或追加原分支数组，遵守根目录 `AGENTS.md` 的 schema。
2. 写真实来源、已做/未做、后续更正；不覆盖旧源文件。沿用已有ID，新增工作使用新ID。
3. 执行 `python tools/build_book.py` 重建索引、来源快照、目录和离线阅读页，再执行 `--validate`。
4. 新增对话需要显式调整 `tools/export_conversation.py` 的日志集合和截止时间，然后导出、构建、核验。第一次导出的截止及前缀哈希保留在版本历史中，不伪装成恒定的全部会话。
5. 公开发布、推送或上传附件需要单独确认目标和范围。

来源哈希会锁定：如果某份旧报告后来被改写，重建会报告 `Source drift`，不把新文本悄悄绑定到旧结论。应先保留该来源的新旧版本并登记更正，再更新源路径；不要关闭校验或覆盖历史快照。

## 包含与不包含

本书携带对话文本、可得的会话附件、报告/协议/证据矩阵快照和账本。大型数据集、PLY输出与模型权重不在书里复制。来源中的运行路径用于找回它们，这不是完整可复现实验数据包。

## 工具依赖

导出、账本和哈希检查只用 Python 标准库。`READ.html` 生成使用 Python `markdown`；若安装 `bleach` 则清理历史 HTML 标签，否则以转义纯文本显示。Markdown 本身是权威书稿，HTML 是派生阅读副本。

本次使用 `map-denoise-research` 技能将实验能力、机制解释与部署主张分开；使用 `git-evolution-workflow` 技能保留来源、纠错链、未完成项和非破坏性版本边界。

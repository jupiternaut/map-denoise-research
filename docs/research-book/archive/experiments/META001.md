# META001 · Hermes mismatch：分歧采样没有稳定胜过覆盖式失配发现

日期：unknown/not documented · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

同库、同拟合器、同初始观测时，按候选分歧查询是否更快发现模型失配？

## 实际做了什么

- 历史实验：64位置、H0/H1、180新任务，按三种模型充分性分层；random每任务3次重复。
- 本次读取原报告，未重跑。

## 没做什么 / 没有证明什么

- 没有开放式发明新表示；没有在C1为空之后增加新修复族。
- 原始运行确切日期 unknown/not documented；晚于此项的审计为2026-09-14。

## 报告记录的结果

- H1足够的type1在预算4时disagreement L1约8.3，covering约151.7；但发现率0.983低于covering 1.000。
- 连H1也错的type2预算4：covering发现率0.583、L1 68.2，disagreement为0.300、117.3。

## 解释及边界

- type1收益是足够类内主动采样的条件收益，不能外推为开放科研或稳定更强的失配发现。

## 研究决定

- 原报告停止推广disagreement作为研究诊断策略；若继续，研究候选耗尽后的修复动作。

## 更正 / 被撤回的解释

- 隔离与随机性归因随后由META002修正；原数值保留。

## 前项

本记录未列出；不推断为不存在。

## 什么情况下值得重做

只有新增失配机制、修复动作或明确改变的诊断终点才形成新问题；同生成器换选择器应先解释META002。

## 来源

- [MISMATCH_REPORT.md](../sources/S009.md) · 原位置 `/home/grf/.hermes/attachments/outputs/MISMATCH_REPORT.md`

主题：Hermes / mismatch / same-capability

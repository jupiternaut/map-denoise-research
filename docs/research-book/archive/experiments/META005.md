# META005 · Hermes fitter-first：固定历史后可逆隔离恢复持续污染任务

日期：2026-09-14 · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

不增加新观测，仅允许隔离一个错误约束，能修复多少旧失败？

## 实际做了什么

- 历史实验A冻结Q/QR历史，比较硬约束拟合器与H0/H1×isolate-cap-1。
- 隔离免费可逆、至少保留2个可信观测；枚举和排序规则均为设定。
- 本次读取报告。

## 没做什么 / 没有证明什么

- 没有两脏点能力、没有软权重；原记录不应被删除。

## 报告记录的结果

- persistent的Q历史预算8/16：旧MAE 8.334/3.397，新均0。
- transient的Q历史同样变为0；model类不误隔离且新旧一致。
- QR预算8的persistent新MAE仍0.866，预算16归零。

## 解释及边界

- 主要收益属于修复器能力，非无区分实验、非调度收益。

## 研究决定

- 采用可逆隔离能力，在共享拟合器后评价策略增量。

## 更正 / 被撤回的解释

- 该轮append-only实现及候选保留细节仍由META007后续清理。

## 前项

- META004

## 什么情况下值得重做

只有isolate-cap-1确实成为限制或新证据机制不同，才有新增软权重/隔离能力实验的理由。

## 来源

- [REPORT.md](../sources/S004.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T084840Z/REPORT.md`

主题：repair-capability / isolate / frozen-history

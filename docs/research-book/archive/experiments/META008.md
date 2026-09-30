# META008 · Hermes DP修复：按终点分别求解、完整历史条件化和空支持回退

日期：2026-09-14 · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

早期DP的负结果来自策略不足还是比较与回退实现？

## 实际做了什么

- 历史复现旧代码问题；H4/H6分别求解。
- belief使用orig及复测结果；空支持标记support_failed并cover回退。
- 用新seed 20000+确认，未重用已看holdout；本次读取报告。

## 没做什么 / 没有证明什么

- 没有第二生成机制；新确认中大量空支持不证明目录Bayes迁移。

## 报告记录的结果

- 目录18世界H4/H6最优MAE均0；cover分别0.917/0.292。
- 新同族18世界确认H4最优0.583、H6为0；多数轨迹进入covering回退。
- 旧holdout空支持后停止改为cover：model 0.292→0，dirt 1.3125→0.625。

## 解释及边界

- 旧预算4劣势不是该终点的精确最优失败；旧holdout崩溃包含stop-on-empty缺陷。

## 研究决定

- 策略问题尚未关闭；下一锁定应让目录支撑初始记录，而非继续在不完整支持上夸大最优性。

## 更正 / 被撤回的解释

- 替代META007中budget4精确DP较差和holdout退化独归于先验迁移的解释。

## 前项

- META007

## 什么情况下值得重做

任何有限最优比较必须同终点、同历史条件、非空支持或明示回退；重复只为验证这些契约。

## 来源

- [REPORT.md](../sources/S006.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T124618Z/REPORT.md`

主题：correction / horizon / empty-support / history

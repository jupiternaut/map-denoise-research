# META003 · Hermes conflict：复测、扩展与诊断在预算和污染机制上有不同表现

日期：2026-09-14 · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

同信息和修复工具下，按冲突选择retest或expand能否优于固定流程？

## 实际做了什么

- 历史实验比较expand_first、retest_first、diagnose，180新任务、三原因、预算8/16。
- 历史补充改动作成本及生成位置，不混入锁定主排名。
- 本次读取报告并结合后续更正解释。

## 没做什么 / 没有证明什么

- 没有软权重、没有第二类脏点修复；未发现新的模型族。

## 报告记录的结果

- 预算16主终点diagnose与retest逐原因一致，不能称额外主终点收益。
- 预算8等混合次级MAE：diagnose 3.607、expand 4.098、retest 4.130。
- persistent上expand仍有残余，但比消耗多次无效复测的策略误差低。

## 解释及边界

- 预算8混合收益可保留；未知原因策略不需要逐原因胜过事后挑出的每个专门基线才有价值。

## 研究决定

- 停止扩大冲突点排序器，检查拟合器是否能撤销错误约束。

## 更正 / 被撤回的解释

- 原报告的无可区分动作、H1折衷拟合解释被META004取代；数值不撤销。

## 前项

- META002

## 什么情况下值得重做

必须改变可用修复或明确的新污染机制；若只是换冲突排序，先给出其可改变主终点的残余案例。

## 来源

- [REPORT.md](../sources/S002.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T051316Z/REPORT.md`
- [CORRECTION.md](../sources/S003.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T084840Z/CORRECTION.md`

主题：conflict / retest / budget

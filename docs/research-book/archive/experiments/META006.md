# META006 · Hermes共享新拟合器：lookahead改了轨迹却未改善预算16终点

日期：2026-09-14 · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

修复器共享之后，主动选择下一实验还增加多少价值？

## 实际做了什么

- 历史实验B比较cover、短复测后cover、1步lookahead。
- 全部共享isolate拟合器和观察接口；本次读取报告。

## 没做什么 / 没有证明什么

- lookahead先验与重测混合概率未被证明正确；不是一般科研策略。

## 报告记录的结果

- 预算16各原因三策略MAE均0，全60任务打平。
- 预算8 model：cover 0.397、lookahead 0.150；dirt对cover没有增量。
- 查询序列全部改变且评分更慢，改变轨迹不等于改变最终误差。

## 解释及边界

- 修复能力和低预算调度是不同效应，主终点已被简单流程解决。

## 研究决定

- 保留有限预算次级收益，停止在此主终点下扩大战略宣传。

## 更正 / 被撤回的解释

- 后续META007修正lookahead概率及候选保留实现，不能将其作为精确最优基线。

## 前项

- META005

## 什么情况下值得重做

须明确未解决的预算和终点；新策略不能独占更强拟合器。

## 来源

- [REPORT.md](../sources/S004.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T084840Z/REPORT.md`

主题：scheduling / negative-result / same-fitter

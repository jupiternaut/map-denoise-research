# META010 · Hermes最终终点：H4严格自适应增量，H6开环已为零

日期：2026-09-14 · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

完整目录中后续标签改下一行动的终点收益是否严格为正？

## 实际做了什么

- 历史H4/H6分别求解；最终报告复述锁定数值并重解具名H4例子。
- 本次读取132630Z和142434Z报告，未重新运行DP。

## 没做什么 / 没有证明什么

- 最终审计未重跑H6；只解H4/H6，未证明H6是最小零误差预算。

## 报告记录的结果

- 2340世界H4：openloop 0.0102564、adaptive 0；72残余均model。
- H6全部世界openloop=adaptive=0；同族确认H4增量0.0264、H6增量0。
- 具名初始记录后先query3，新标签决定query0或query6，最后行动实际改变。

## 解释及边界

- H4是真实而窄的预算依赖增量；H6主要收益由目录设计和足够预算解释。
- cover在model类已0不取消未知原因混合策略增量；不可事后挑专门基线。

## 研究决定

- 冻结该生成器调度竞赛；保留H4事实，不称通用研究策略或迁移。

## 更正 / 被撤回的解释

- 纠正fixed等于non-adaptive的混称：retest_then_cover已有反馈；openloop可用初始观测。

## 前项

- META009

## 什么情况下值得重做

H6同损失已0，新的调度器无法改善该终点；需要另定能力/信息/任务问题，不能只重赛。

## 来源

- [REPORT.md](../sources/S007.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T132630Z/REPORT.md`
- [REPORT.md](../sources/S008.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T142434Z/REPORT.md`

主题：H4 / H6 / strict-adaptivity / stop

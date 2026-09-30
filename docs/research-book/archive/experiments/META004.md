# META004 · Hermes纠错：同点复测等价不等于所有新位置查询等价

日期：2026-09-14 · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

persistent与model真的没有任何可用观测能够区分吗？

## 实际做了什么

- 历史更正从已有EVAL_TRAJ和原代码复核60对任务，不是新eval。
- 本次读取CORRECTION.md。

## 没做什么 / 没有证明什么

- 没有给新策略硬编码x=48的授权；没有证明一般可识别性。

## 报告记录的结果

- 全部60对在x=32初读和复测相同，但后来查询x=48的标签不同。
- 拟合器获得分裂观测后仍把脏标签保留为不可撤销硬约束。
- expand的较低误差来自多查询改善最近邻回退，代码没有H1 compromise分支。

## 解释及边界

- 失败是已有信息未被修复器使用；只对同点复测成立的观测等价不能扩大成所有允许实验的等价。

## 研究决定

- 下一步允许可逆isolate，之后才比较调度。

## 更正 / 被撤回的解释

- 明确替代META003原报告的no available action separates与H1 compromise解释。

## 前项

- META003

## 什么情况下值得重做

只有允许观测集合或拟合约束改变时再论证辨识边界；必须逐项说明新证据是否被拟合器使用。

## 来源

- [CORRECTION.md](../sources/S003.md) · 原位置 `/home/grf/.hermes/attachments/outputs/20260914T084840Z/CORRECTION.md`

主题：correction / identifiability / interface

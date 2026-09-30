# META013 · Open-world旧记忆接口缺口：有状态表示仍从m0=0预测

日期：unknown/not documented · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

增加隐藏记忆状态是否足以预测同可见初态下的不同未来？

## 实际做了什么

- 历史construction单列同(x,v)、同未来输入而不同隐藏记忆的两分支。
- 本次读取记忆诊断与失败解释。

## 没做什么 / 没有证明什么

- 未以真实m0直接输入方法；此处没有完成预驱动估计。

## 报告记录的结果

- C2的memory3普通轨迹NMSE约0.0006–0.0017，两个记忆分支约0.59/0.57。
- 候选总从m0=0展开；C4带其他初始记忆时选错obs_drift。

## 解释及边界

- 表示新增状态与接口可估计初态是两个条件；同可见瞬时状态不等于全部历史等价。

## 研究决定

- 将公开预驱动历史与初态估计列为合法下一修复，后续META014已部分执行。

## 更正 / 被撤回的解释

- 当前实现不能继续按旧m0=0限制描述，见META014。

## 前项

- META012

## 什么情况下值得重做

重查此问题前检查当前历史接口；仅重演旧初态缺口不构成新发现。

## 来源

- [REPORT.md](../sources/S099.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/outputs/construction/REPORT.md`

主题：hidden-state / history / interface-gap

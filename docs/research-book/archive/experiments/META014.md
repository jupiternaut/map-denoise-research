# META014 · Open-world capability_v1：16结构库存与历史初态修复已存在

日期：unknown/not documented · 分支：meta-research · 证据：`artifact_checked`

[返回实验总账](../EXPERIMENTS.md)

## 问题

当前代码是否仍限于旧四模型与零记忆初态，历史接口实际改善了吗？

## 实际做了什么

- 本次核对host/artifact.py的公开原语、legal_structures和DECAYS，以及保存SUMMARY.json。
- 只检查指定源码与保存汇总，未执行模拟、重新计算NMSE或检查全部数组。

## 没做什么 / 没有证明什么

- 完整运行日期 unknown/not documented；没有证明无界表达或开放科学新颖性。
- 本次未复验沙箱、隔离或全盲评。

## 报告记录的结果

- 两个可选动力学项×是否含记忆×两观测映射形成16结构，DECAYS为0.25/0.35/0.5/0.7。
- 保存汇总T2/library NMSE：无历史0.3967492388→有历史0.0017404743。
- T4/library：0.3270157374→0.0022132049；真实m0不是直接参数。

## 解释及边界

- 旧四类与当前可组合库存是不同版本；收益支持这些任务族上的历史利用，不等于一般可辨识或调度增量。

## 研究决定

- 以当前能力作为新实验基线，不重复把已修复的历史接口宣称为新机制。

## 更正 / 被撤回的解释

- 更新META012/META013中只对construction成立的四命名构造与m0=0状态。

## 前项

- META013

## 什么情况下值得重做

若改历史可见范围、初态估计或原语库存，应分别隔离对照；已有T2/T4大收益不能再次记为新发现。

## 来源

- [artifact.py](../sources/S097.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/host/artifact.py`
- [SUMMARY.json](../sources/S098.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/outputs/capability_v1/SUMMARY.json`

主题：capability / 16-structures / history-repair

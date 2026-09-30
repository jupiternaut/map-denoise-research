# META015 · Open-world残余：漂移未解决，library/propose共享覆盖采样

日期：unknown/not documented · 分支：meta-research · 证据：`artifact_checked`

[返回实验总账](../EXPERIMENTS.md)

## 问题

历史修复和更大结构库存是否等于漂移或自适应研究问题已解决？

## 实际做了什么

- 本次读取SUMMARY.json中T3及policies/capability.py全部策略包装。
- 核对library/propose都先covering(session)，再fit(all)/fit(propose)。

## 没做什么 / 没有证明什么

- 未独立重算动力学轨迹；未见12/24系统锁定盲评完成证据。

## 报告记录的结果

- T3_drift library与propose均为有历史NMSE0.2546616806、无历史0.2485210275。
- 两策略共用实验计划，差异是候选搜索/选择；新增历史未改善此漂移分支。

## 解释及边界

- 不能把候选提议与库搜索差异说成自适应采样；局部能力修复不表示整个open-world挑战结束。

## 研究决定

- 保留漂移残余，下一研究须说明改观测模型、历史接口还是调度。

## 更正 / 被撤回的解释

- 收窄将capability收益统称为反馈/开放研究成功的解释。

## 前项

- META014

## 什么情况下值得重做

若只换proposer，要共享采样和原语；若测调度，必须真改变取证并给强开环基线同能力。

## 来源

- [SUMMARY.json](../sources/S098.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/outputs/capability_v1/SUMMARY.json`
- [capability.py](../sources/S100.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/policies/capability.py`
- [README.md](../sources/S096.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/README.md`

主题：drift / same-schedule / remaining-gap

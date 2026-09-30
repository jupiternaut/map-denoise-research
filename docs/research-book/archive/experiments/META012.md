# META012 · Open-world construction：四个预定义构造器跑通，未达盲评目标

日期：unknown/not documented · 分支：meta-research · 证据：`report_read`

[返回实验总账](../EXPERIMENTS.md)

## 问题

离开完整目录后，增加模型结构与顺序干预分别能带来什么？

## 实际做了什么

- 历史建设运行六系统、五策略、预算8；四命名构造linear2/nl2/memory3/obs_drift。
- A/C共享covering；C/D共享构造器；B/D执行承诺预测再干预。
- 本次读取construction报告与README。

## 没做什么 / 没有证明什么

- 12系统开发锁定、24系统盲评在所读材料中未完成；确切运行日期 unknown/not documented。
- 没有OS沙箱、没有官方LLM-ACES/PySR复现；不是闭卷LLM测试。

## 报告记录的结果

- C在C2较A改善NMSE约0.490；C4反而较差。
- D−C在C0–C4约0，C5更差约0.054；20%挑战未达成。
- 漂移/另一记忆初态上结构选错，ridge更强。

## 解释及边界

- 四预定义构造与开放科学发明不同；系统能力收益不应算到调度。

## 研究决定

- 保留构造装置和条件性状态收益；不声称20%胜利、迁移或强基线复现完成。

## 更正 / 被撤回的解释

- 旧README同时保留任务包未启动与六系统已运行文字，以运行报告限定阶段；当前能力由META014更新。

## 前项

- META010
- META011

## 什么情况下值得重做

先明确新模型原语和共享基线，再锁定新系统；旧六系统只能作开发/回归，不能改称盲评。

## 来源

- [REPORT.md](../sources/S099.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/outputs/construction/REPORT.md`
- [README.md](../sources/S096.md) · 原位置 `/home/grf/Documents/meta-research-open-world-challenge/README.md`

主题：open-world / construction / four-models

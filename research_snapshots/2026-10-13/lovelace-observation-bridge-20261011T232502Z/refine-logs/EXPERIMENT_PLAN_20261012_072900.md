# Experiment Plan

**Problem**: 保留有效几何、修复不足；解释共同材料修复能力在亮度偏差批中的损失。
**Method Thesis**: 固定几何与动作器，配对分解观测处理/误差界/传播近似，而不是立即扩大算法自由度。
**Date**: 2026-10-12（Asia/Shanghai）

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1 配对处理归因 | 避免把裁剪或宽界变化全算成亮度偏差代价 | 全54配对、全初态、同图像/同界差异；正负全报 | A |
| C2 传播余量 | 确定是否值得在旧传播器继续投入 | 4/8/16同图同界外包/动作/误差与成本 | B |

## Paper Storyline

主文：共享解释仍有效，后续修复能力缺口具体定位；附录：模板/种子与计时；不宣称真实泛化或新学习器优势。

## Experiment Blocks

A：协议六配对单元，其中四新增、两历史复用。固定中间β=.01；MAE/MSE与修复/覆盖优先。零差也是机制测量结论，不能将它判成运行失败。
B：CBdelta传播4/8/16，8复用，4/16新增。逐例外包和动作一致/不同，与终点配对分开报告。稳定只限定这些设置。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 实现/参数/历史谱系/隔离 | 两作者单测与非确认冒烟 | 实际测试通过 | 单CPU短测 | raw越界与裁剪顺序混淆 |
| M1 | 非作者源码审查 | 1轮，必要1次修复后续审 | 无已知阻塞 | 代理审查 | 新线程额度可能不足 |
| M2 | 固定正式批次 | 6新配置×54，2只读复用 | 7776动作完整 | 预计约0.45–1.20 CPU小时含核验 | 超5400秒标未完成 |
| M3 | 独立核验 | 全树/全动作/全评价 | 核验PASS | 纳入总窗口 | 共享历史几何内核边界 |
| M4 | 归因结算 | 完整配对与报告 | 记录实测不追阈值 | 只读算术 | 错误将KEEP回退当模型有效 |

## Compute and Data Budget

单CPU、0GPU小时、API0、安装0。一个有限运行器、无≥10独立作业或调度器依赖；不是多job排队。初始估计不是测时承诺。

## Final Checklist

- [x] Main paper tables are covered
- [x] Novelty isolation not expanded beyond known shared mechanism
- [x] Simplicity tested via propagation/config controls
- [x] Frontier primitive explicitly not claimed
- [x] C/D/E/F future work kept out of must-run

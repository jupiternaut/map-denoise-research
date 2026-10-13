# Research Contract：共享层级消融

## Selected Idea

前轮同信息共享材料改善115/432，但逐通道臂有大量预算未决。恢复完整511节点并放松共享维度，以决定可继承的数学约束，不再堆收益预测器。

## Core Claims

1. 完整盒预算下共享约束仍产生认证修复增量。
2. 删除对照可以分辨跨像素、跨视图及灰度相等的贡献。

## Method Summary

同54观测回放、旧几何界和hybrid动作。必要半空间按global/view/pixel/RGB分组；每组三变量精确传播；任一组矛盾排除，否则保留。不给求解器真值或校准样本。

## Experiment Design

6臂×54观测×9初态=2916决策，每臂511盒/分支、8轮、噪声1/100、半径1/5；CPU单线程5400秒上限。MAE/MSE、改善/同分/恶化、覆盖、KEEP、负控、未决叶与冷缓存成本。

## Baselines

KEEP与channel511；旧255记录为冻结已核验来源，不是再次确认。无训练、LLM或额外传感器。

## Current Results

未运行。上轮115改善/317同分/0恶化只作设计背景，不作本轮成绩。

## Key Decisions

先完整预算/共享分解，不在本批引入LP/曝光/真实迁移。六臂统计全部报告，禁止按真值挑应用位置；旧文件不变。

## Status

- [x] Idea selected
- [ ] Implementation / review / run / verification

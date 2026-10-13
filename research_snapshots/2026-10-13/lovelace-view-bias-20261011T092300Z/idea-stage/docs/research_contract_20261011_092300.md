# Research Contract：有界视图亮度偏差

## Selected Idea

上一轮完整深度树中共享灰度修复115/432、MAE降低22.35%；RGB放松仍修复94。此批检验共同材料的带界外观适用性，不用旧结果预设新收益。

## Core Claims

C1：同公开ε+β下，共享在至少一个锁定β上保留覆盖、正确KEEP及零损伤，且优于channel修复与MAE。
C2：严格ε诊断的合同失配能按覆盖/损伤准确呈现，不作为合法修复成绩。

## Method Summary

复用旧grouped solver，视图偏差(−β,0,+β)经裁剪；合法残差界ε+β。求解只读图像和公开参数，偏差实际值不进入求解。保持旧几何与动作器，无训练。

## Experiment Design

3β×54观测×9初态×4臂=5832决策。单CPU，5400秒cap，511深度盒/分支、8轮、容差15/32；全部动作落盘后真实评价，再独立核验。

## Baselines

KEEP和channel_wide；shared_strict故意失配仅诊断。

## Current Results

尚未运行。报告不把计划数字当结果。

## Key Decisions

只做有界加性偏差；不做其他后续清单。输出不可覆盖，历史只读；不发布不部署。

## Status

- [x] Idea selected / protocol frozen
- [ ] Implementation / review / run / verification


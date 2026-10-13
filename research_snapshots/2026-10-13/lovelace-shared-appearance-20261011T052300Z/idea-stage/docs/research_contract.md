# Research Contract：共享外观约束

## Selected Idea
原照片保留不变，把view-invariant材料灰度从逐像素独立解释恢复为三个共享未知量。冻结旧hybrid修复器，检验同信息算法收益。

## Core Claims
1. 必要约束的精确矛盾排除不丢失合同内真深度。
2. 共享约束在六个受控模板的新种子深度上优于逐通道松弛；这是待检验假说。

## Method Summary
三种遮挡解释分别构造连续几何盒的面积上下界。将每个通道变成非负线性半空间，八轮边界传播。留存未决盒，分支外包合并后原样调用旧hybrid。

## Experiment Design
四臂、54配置/486初态；单CPU1800秒，无训练/GPU/API；详情PROTOCOL.md。MAE/MSE、改善/恶化比例、覆盖、正确点KEEP、负控与成本。

## Current Results
未运行。

## Key Decisions
不使用额外材料样本来选择shared；calibrated仅信息对照。不宣称任意纹理/拓扑普适性，不改真实部署。

## Status
- [x] Idea selected
- [ ] Implementation / tests / confirmation / verification

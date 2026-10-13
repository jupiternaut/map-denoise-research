# Experiment Plan

**Problem**：保留有效几何并修复不足。
**Method Thesis**：在覆盖真实深度的观测外包上选择安全动作，并检验其修复范围，而非只降低照片残差。
**Date**：2026-10-11

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：投影最大化标量最坏收益 | 检验近边界小偏移是否因中点提案漏修 | 解析证明、边界单测、MOVE覆盖不低于中点 | B1,B3 |
| C2：双层遮挡可保留连续覆盖 | 接近历史表面归属问题但保持可验证模型 | 全部真值在外包内；双层正确点无伤且区间内坏点可修 | B2,B3 |

## Paper Storyline
- 必要：不回归保证和非平凡修复同时出现；同误差信息基线。
- 辅助：小模型CPU成本与无信息负控。
- 删去：GPU部署、ML学习器、未知纹理、真实图册大跑、完整角色；这些没有本轮数据支持。

## Experiment Blocks
### B1：边界动作与算子单测
- 检验C1；固定外包，以可手算端点收益验证最大最坏收益及中点漏修。
- 验收：单测通过；不把已知公式误称新理论。
- Priority: MUST-RUN。
### B2：两层成像与外包
- 检验C2；partial灰/同色和完全覆盖同色负控，按面积先遮挡再积分。
- 验收：手算像素、点区间等价、参数样本覆盖诊断以及解析连续界。
- Priority: MUST-RUN。
### B3：冻结确认与强对照
- 五模板×三相机预算×三新深度×三噪声种子。9初态、4臂；参数详见PROTOCOL.md。
- 终点：MAE/MSE、改善/恶化/不变、正确点保留；偏移大小和先验内外分开。
- 固定baseline .5mm深度/五档相机误差，有限profiled残差；禁止称全局最优。
- 验收：135覆盖、证书零伤、投影MOVE不低于中点；真实收益分账。
- Priority: MUST-RUN。
### B4：独立记录复核
- 不调用root决策函数重算外包端点收益、动作、真实误差、报告汇总。
- 验收：与归档JSON/CSV一致，不把共享成像内核称第二套独立物理实现。
- Priority: MUST-RUN。

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 旧单测和新单测 | 1 | 测试通过 | <2min CPU | 接口匹配 |
| M1 | 新baseline和代码复核 | 1 | fresh-agent无blocking | 1–3min | float基线tie |
| M2 | 封存→确认 | 1 | 不改协议 | 5–15min CPU | 区间过松 |
| M3 | 逐条复核→报告 | 1 | 零未解释差异 | <3min | 样本重复计数 |

## Compute and Data Budget
- GPU-hours：0；local CPU，最多2个同时活动的计算进程。
- 公开已知几何与新合成测量，无新增真实数据。
- 最慢部分预计pose-profiled网格精确预测预计算。

## Risks and Mitigations
- 双层外包过松：保留KEEP和未决盒，报告修复覆盖，不事后放松保证。
- 同色负控凭先验获改善：区间内外单列。
- 原始数据仍未知材质：本轮结论只对应明确合同。

## Final Checklist
- [x] 主终点、强对照与无信息负控覆盖
- [x] 简化/近边界提案对照覆盖
- [x] 没有强加LLM/训练/GPU组件
- [x] 确认参数在结果之前固定

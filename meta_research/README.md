# 元研究材料索引

两个不同装置，不能混成一个成功证明：

1. `hermes/`：有限函数族与污染机制下，比较实验选择、修复和规划；包括原始记录与修正。
2. `open_world/`：可干预动力系统的受限模型构造与初态估计。历史 proposal 不是全部完成的盲评。

## Hermes 时间线

| 路径（hermes/outputs 下） | 内容 |
| --- | --- |
| [MISMATCH_REPORT.md](hermes/outputs/MISMATCH_REPORT.md) | 候选分歧查询与失配发现。 |
| [20260914T051316Z](hermes/outputs/20260914T051316Z/REPORT.md) | 冲突诊断与复测/扩展；后续纠正了失败归因。 |
| [20260914T084840Z](hermes/outputs/20260914T084840Z/REPORT.md) | 可逆隔离修复器；更正见同目录 CORRECTION.md。 |
| [20260914T111013Z](hermes/outputs/20260914T111013Z/REPORT.md) | 初次目录 DP；不是最终结论。 |
| [20260914T124618Z](hermes/outputs/20260914T124618Z/REPORT.md) | 分预算求解、完整历史、空集回退的更正。 |
| [20260914T132630Z](hermes/outputs/20260914T132630Z/REPORT.md) | 2,340 世界、预先计划与自适应。 |
| [20260914T142434Z](hermes/outputs/20260914T142434Z/REPORT.md) | 初始设计条件化审计；本生成器最终解释。 |

更早的 `20260914T022716Z` 和根目录实验也按原记录收录。TLA+ 只证明报告列出的有限契约检查项，不证明实验科学结论或实际权限隔离。

## 开放模型挑战

- [建设阶段](open_world/outputs/construction/REPORT.md)
- [能力修复阶段摘要](open_world/outputs/capability_v1/SUMMARY.json)
- [工件模型](open_world/host/artifact.py)、[评分接口](open_world/host/score.py)

后一个阶段提供预驱动历史/初态估计、组合原语，不应仍概括成建设阶段的四固定模型；也不能称为无限开放发现。
原运行产物的声明与数值保留，本次发布不重新进行开发/盲评。

历史 `AGENTS.md` 重命名为 `SOURCE_AGENTS.md`，仅供研究指令审计，不作用于当前公开仓库。
模拟器与答案代码公开用于复现；阅读过它们的 Agent 不再满足原来的未暴露参与者条件。

# 几何证据与选择器贡献：2026-10-07实验快照

## 最新：固定证据下的选择器消融

- [贡献报告](selector-attribution-20261007T180539Z/REPORT.md)
- [冻结协议](selector-attribution-20261007T180539Z/PROTOCOL.json)
- [纯选择器](selector-attribution-20261007T180539Z/policies.py) · [28项单测](selector-attribution-20261007T180539Z/test_policies.py)
- [全部17臂结果](selector-attribution-20261007T180539Z/evaluation/RESULTS.json) · [逐项贡献](selector-attribution-20261007T180539Z/evaluation/ATTRIBUTION.csv)
- [独立审计](selector-attribution-20261007T180539Z/EXPERIMENT_AUDIT.md)
- [发布范围与跨机复算](../../publication/SELECTOR_ATTRIBUTION_20261008.md)

新star证据配均值选择改善30.19%，最坏收益选择29.63%；两者仍有两处好点损伤。旧场景回溯消融，不是新场景确认或部署升级。

## 前轮：三视图轨迹与方向判别

- [报告](track-discrimination-20261007T160716Z/REPORT.md)
- [条件理论](track-discrimination-20261007T160716Z/theory/THEORY.md) · [数学核](track-discrimination-20261007T160716Z/theory/kernel.py)
- [观测构造](track-discrimination-20261007T160716Z/observation/README.md) · [离散化诊断](track-discrimination-20261007T160716Z/observation/DIAGNOSTIC.md)
- [合成机制汇总](track-discrimination-20261007T160716Z/theory/results/summary.csv)
- [真实回放结果](track-discrimination-20261007T160716Z/evaluation/RESULTS.json) · [审计更正](track-discrimination-20261007T160716Z/AUDIT_RESOLUTION.md)

这两轮原始AGENTS.md仅改名SOURCE_AGENTS.md以避免激活历史任务指令；其余收录字节不变。原机绝对路径保留用于追溯，不是跨机运行入口；使用发布核验器。

## 更早：邻域射线区间支持

- [完整实验报告](plane-support-20261007T084206Z/REPORT.md)
- [数学模型与候选排除条件](plane-support-20261007T084206Z/theory/THEORY.md)
- [可运行数学内核](plane-support-20261007T084206Z/theory/kernel.py)
- [选择器](plane-support-20261007T084206Z/selector.py)
- [合成反例](plane-support-20261007T084206Z/counterexamples/README.md)
- [原始逐点结果](plane-support-20261007T084206Z/evaluation/POINT_METRICS.csv)
- [独立审查结论](plane-support-20261007T084206Z/EXPERIMENT_AUDIT.md)
- [后续工作交接](plane-support-20261007T084206Z/CHECKPOINT.md)
- [发布范围、路径映射和复算命令](../../publication/PLANE_SUPPORT_20261007.md)

源文件保持原字节，主脚本依赖原机路径；使用发布核验器可在其他机器只读重现表格与候选决策。主验收未通过，部署默认不变。`dependencies/`只包含明确列出的轻量前项，不是所有历史实验的完整备份。

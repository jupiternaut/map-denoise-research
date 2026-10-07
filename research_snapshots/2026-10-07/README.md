# 邻域射线区间支持：2026-10-07快照

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

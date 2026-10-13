# Experiment Tracker：A+B完成

2026-10-12，Asia/Shanghai。确切host liekkas，目标为保留有效几何、修复不足。

| 阶段 | 最终状态 | 实际证据 |
|---|---|---|
| M0 实现、测试、非确认冒烟 | PASS | 根代理14+23 tests通过，六新配置smoke exit=0；M0_ACCEPTANCE_20261012_074657.md |
| M1 非作者源码审查 | PASS / LAUNCH_ALLOWED | 四科学文件及PROTOCOL审查无blocker；全文EXPERIMENT_CODE_REVIEW.md；同族内部provisional |
| M2 冻结正式批次 | COMPLETE / exit=0 | 324新共享观测+108历史复用；7776动作；RUN_STATUS总1261.912029秒 |
| M3 独立作者核验器 | PASS / exit=0 | verification/VERIFY.json完整PASS；192.124090秒；原始窗口内main+verification共1463.093064秒 |
| M4 配对结算 | COMPLETE / exit=0 | analysis/analysis-v1七表；八终点、十配对、seed统计、外包结算；M4_ANALYSIS_20261012_081630.log |
| 非作者结果解释复核 | COMPLETE | RESULT_INTERPRETATION_REVIEW.md，不导入实验代码的Fraction复算吻合 |

本批全部完成，无后台实验。全部八shared配置：真深度覆盖54/54、正确KEEP54/54、负控KEEP72/72、零损伤。修复数原窄界115，窄界裁剪99，宽界无偏差0，宽界偏差raw48/clip32。8→16全部动作和外包相同，4→8有一个终点改善。

正式launch后四根科学.py及PROTOCOL哈希未变。C/D/E/F、LP、学习器、新场景均未执行；未发布GitHub/GitBook，未改部署默认。下一轮结构化视图偏差与简单背景校准只是建议，不作为本轮已测成果。

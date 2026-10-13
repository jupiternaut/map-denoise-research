# Experiment Tracker

日期2026-10-12 07:52:24（Asia/Shanghai）。原目标：保留有效几何、修复不足。

| 阶段 | 状态 | 实际证据 |
|---|---|---|
| M0 实现/单测/非确认冒烟 | PASS | 根代理37 tests、6配置smoke通过；M0_ACCEPTANCE_20261012_074657.md |
| M1 非作者源码审查 | PASS / LAUNCH_ALLOWED | 最终四文件及协议无blocker；EXPERIMENT_CODE_REVIEW.md；同族内部provisional |
| M2 一个固定正式批次 | RUNNING | 07:52:06启动；outputs/bridge-v1/RUN_STATUS.json；324新求解+108复用；deadline09:22:06 |
| M3 完整独立核验 | NOT STARTED | 必须在相同deadline前完成，不能重置窗口 |
| M4 配对结算 | PREPARED / NOT RUN | analysis/analyze_bridge.py就绪；尚无经完整核验的新正式结果 |

launch后科学源码和PROTOCOL冻结，历史目录仍只读。无GPU/API/安装或发布/部署。

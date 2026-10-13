# Experiment Tracker

日期2026-10-12 07:51:27（Asia/Shanghai）。原目标：保留有效几何、修复不足。本批只定位观测/误差界/有限传播对修复力度的影响。

| 阶段 | 状态 | 实际证据 |
|---|---|---|
| M0 实现、单测、非确认冒烟 | PASS | 根代理14+23 tests PASS；六新配置smoke exit=0；M0_ACCEPTANCE_20261012_074657.md |
| M1 非作者源码审查 | PASS / LAUNCH_ALLOWED | 最终四科学文件及协议审阅无blocker；EXPERIMENT_CODE_REVIEW.md；同族内部provisional，不是外部学术评审 |
| M2 一个固定正式批次 | READY / NOT STARTED | 324新共享求解、108历史共享复用；原5400秒总窗口尚未启动 |
| M3 完整独立核验 | READY / NOT STARTED | 另一作者核验器稳定；待主批COMPLETE7776动作；核验不重置deadline |
| M4 配对结算报告 | PREPARED / NOT RUN | 不import实验的exact分析器就绪；无新正式结果 |

根科学.py及PROTOCOL自M2启动锁定，不改算法、参数或人口。无GPU/API/安装、发布、部署；全部历史目录只读。

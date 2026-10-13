# Experiment Tracker

日期2026-10-12 08:13:18（Asia/Shanghai）。目标：保留有效几何、修复不足。

| 阶段 | 状态 | 实际证据 |
|---|---|---|
| M0 实现/测试/非确认冒烟 | PASS | 根代理37 tests、6配置smoke；M0_ACCEPTANCE_20261012_074657.md |
| M1 非作者源码审查 | PASS | 四科学文件及协议无blocker；同族内部provisional |
| M2 固定正式批次 | COMPLETE / exit=0 | 432配置、7776动作完整；324新共享求解+108复用；RUN_STATUS总1261.912029 s；日志M2_MAIN_20261012_075200.log |
| M3 完整独立核验 | RUNNING | 08:13:17左右启动verify_bridge.py；M3_VERIFY_20261012_081309.log；使用原deadline09:22:06，不重置窗口 |
| M4 配对结算 | PREPARED / NOT RUN | 待M3完整PASS工件；初步M2数字不代替独立核验 |

正式运行后没有编辑根科学.py或PROTOCOL。所有旧根仍只读，无GPU/API/安装、部署或发布。主运行结束不等于整批科学结算完成。

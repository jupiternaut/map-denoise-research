# Experiment Tracker

| Run ID | Milestone | Purpose | Status | Evidence |
|---|---|---|---|---|
| R001 | M0 | 实现、独立核验测试、四夹具集成 | COMPLETE | 44 tests OK；M0_ALL_TESTS_INTEGRATION.log |
| R002 | M1 | 非作者全源码审查 | PASS / provisional | EXPERIMENT_CODE_REVIEW_20261012_104837.md |
| R003 | M2 | 同两图像四方法 | RUNNING | 10:49:27启动；45/108图像、1620/3888动作仅进度快照 |
| R004 | M3 | 全量独立核验 | READY | 读取M2 BUDGET同绝对deadline，尚未执行 |
| R005 | M4 | 配对科学结算 | READY | analysis/analyze_structured.py，等待COMPLETE+PASS |

本轮绝对预算start_unix=1791773367.9290864，deadline_unix=1791778767.9290864，5400秒包括主批与核验。所有科学.py/PROTOCOL现已冻结；不会根据中间结果调参。进度不是科学结果，真值只在完整动作落盘后解析。

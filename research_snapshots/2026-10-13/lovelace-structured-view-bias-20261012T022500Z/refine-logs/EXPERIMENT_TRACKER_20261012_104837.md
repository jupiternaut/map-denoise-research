# Experiment Tracker

| Run ID | Milestone | Purpose | Status | Evidence |
|---|---|---|---|---|
| R001 | M0 | 实现、独立核验测试、四夹具集成 | COMPLETE | 44 tests OK；M0_ALL_TESTS_INTEGRATION.log |
| R002 | M1 | 非作者全源码审查 | PASS / provisional | EXPERIMENT_CODE_REVIEW_20261012_104837.md，放行一次冻结批次 |
| R003 | M2 | 同两图像四方法 | LAUNCH_READY | 216新求解、108复用外包、3888动作，参数不变 |
| R004 | M3 | 全量独立核验 | READY | verify_structured.py已封版，使用M2同绝对5400秒窗口 |
| R005 | M4 | 配对科学结算 | TODO | analysis/analyze_structured.py只读COMPLETE+PASS工件 |

科学.py与PROTOCOL从M2启动冻结；旧检查点不改。无GPU/API/安装/SSH/发布/部署。

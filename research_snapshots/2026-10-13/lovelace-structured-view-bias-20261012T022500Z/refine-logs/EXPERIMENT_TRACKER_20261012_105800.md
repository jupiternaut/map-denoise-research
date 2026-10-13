# Experiment Tracker

| Run ID | Milestone | Status | Evidence |
|---|---|---|---|
| R001 | M0 | COMPLETE | 44 tests OK含4夹具独立核验 |
| R002 | M1 | PASS / provisional | 非作者完整源码审查 |
| R003 | M2 | COMPLETE | exit0；108图像、3888动作、216新求解；452.208秒 |
| R004 | M3 | RUNNING | 10:57:22启动全量核验，读取原5400秒绝对deadline |
| R005 | M4 | WAITING_FOR_VERIFY | 主结果115/115 raw修复、68/94 clip修复仅待核验初步值 |

旧源与本根科学文件冻结。M2未重跑；未完成独立核验前不宣称M3 PASS，不部署。calibrated已作为共同依赖的删除对照；ablation-planner仅只读评估现有对照、不给生成器扩批。

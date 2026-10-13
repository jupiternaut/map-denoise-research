# 实验跟踪：完整预算回放运行中

| Run | Purpose | Status | Evidence |
|---|---|---|---|
| M0 | 分组传播/接口/证书单测和冒烟 | PASS | 最终33项单测通过；六臂非确认冒烟通过 |
| M1 | fresh同族代码审查 | PASS_PROVISIONAL | 无阻塞，详细限制见EXPERIMENT_CODE_REVIEW.md |
| M2 | 六臂54观测完整预算回放 | RUNNING | 06:11 UTC完成13/54组、702/2916决策 |
| M3 | 独立证书和计数核验 | NOT_STARTED | 等待全部决策及评价完成 |
| M4 | 共享维度/预算分解报告 | NOT_STARTED | 只读分析脚本已准备，未读取完整主结果 |

本表为时间截面；继续进度以outputs/ablation-v1/RUN_STATUS.json及logs_run_20261011_060400.log为准。旧八份输入与46个科学依赖不修改，新科学代码已由运行源锁冻结。

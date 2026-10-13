# Experiment Tracker：单次B70→B80干预已核验

更新：2026-10-12。协议、源与初始计划保持冻结。

| Run ID | Milestone | Purpose | Status | Evidence |
|---|---|---|---|---|
| R001 | M0 | 代码与非确认夹具 | DONE | 初始31；恢复encoded边界33通过 |
| R002 | M1 | fresh源码审查 | DONE | 首审漏检JSON边界；恢复再审PASS；完整trace保留 |
| R003 | M2 | 36图像72冷求解972动作 | DONE | recovery-1/outputs/baseline-v1/RUN_STATUS.json COMPLETE |
| R004 | M3 | 独立机械复算 | DONE | verification/VERIFY.json PASS，93源/4输入一致 |
| R005 | M4 | 深档与全组结算 | DONE | REPORT.md、analysis/METRICS.json；C1支持，C2有条件增量 |

首父启动INCOMPLETE在0求解/0动作时退出；只在recovery-1修JSON解码，不重置预算或改变科学参数。
后续消融仅PLAN，未启动任何新主批。不存在等待中的GPU作业或后台作业。

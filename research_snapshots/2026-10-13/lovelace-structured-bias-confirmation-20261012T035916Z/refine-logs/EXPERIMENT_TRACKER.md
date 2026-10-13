# Experiment Tracker：冻结确认完成

| Run ID | Milestone | Purpose | Status / evidence |
|---|---|---|---|
| R001 | M0 | wrapper、数据分离、计数/真假结果核验测试 | PASS，16/16非确认 |
| R002 | M1 | fresh非作者源码审查 | PASS，同族暂定、无blocker |
| R003 | M2 | seed61004 clip18观测/36求解/486动作 | COMPLETE，79.368秒，独一主批 |
| R004 | M3 | 全量独立机械复算 | PASS，56.994秒；含流程间隙174.455秒原窗口 |
| R005 | M4 | 配对、成本、集中与有限结论 | COMPLETE；现有cal→joint组件消融足够当前C1，额外规划不执行 |

提前启动核验FAIL保存在verification-premature-attempt，待COMPLETE后同deadline恢复；不算第二主批。共同T_cal/18union/54branch嵌套已核验；gain与覆盖不是mechanical PASS前提。

C1：joint修复23 vs cal16，MSE再降7.96%；C2保持，0初态损伤。主要增量集中第一档，真实迁移未执行。未发布、未部署。

M4附项：已有cal→joint消融结算COMPLETE；18观测/162配对机制读回COMPLETE（0新增求解）；未来表中三项均NOT RUN，不在本轮注册内。全部完成状态见 ../CHECKPOINT.md。


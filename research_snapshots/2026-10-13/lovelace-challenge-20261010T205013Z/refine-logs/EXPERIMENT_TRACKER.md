# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 算子与渲染单测 | 旧/新模型及记录、核验器 | development | tests | MUST | DONE | 82项通过，3.552秒 |
| R002 | M1 | 参数与代码复核 | 四臂 | pre-confirmation | defects | MUST | DONE | fresh-agent；两项原阻塞关闭 |
| R003 | M2 | 冻结确认 | 四臂 | new synthetic | MAE/MSE/harm/coverage | MUST | DONE | 135观测、1215任务、4860决策；72.086秒 |
| R004 | M3 | 逐条重算 | verifier | archived confirmation | differences | MUST | DONE | PASS；340组汇总、1088witness检查；2.587秒 |

小偏移：中点363/810改善、投影494/810改善，证书均无伤；强残差624/810改善但12次恶化。正确初点四臂135/135保留。所有观测真值覆盖；无空外包、无预算未决叶。
C1/C2按预定门槛通过；实际MAE不支配中点，因此下一组合候选未擅自作为第五臂加入。

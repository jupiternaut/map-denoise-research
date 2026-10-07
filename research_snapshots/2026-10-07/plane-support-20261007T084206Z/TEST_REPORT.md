# 主选择器检查

2026-10-07，liekkas。

执行 `python -B test_selector.py`：7项unittest全部通过，0.004秒。
主推理输出 `INFERENCE_SUMMARY.json`：21组票计数与理论核独立实现一致；原邻居2520个；官方诊断字段变异不改变经字段白名单清洗的输入。
推理耗时0.2484秒，评价耗时约0.18秒（不含推导、文件检查、测试与审查；不作为整体方法速度基准）。
评价 `RESULTS.json`：105候选坐标与冻结旧评价逐项完全匹配；主分母484，请求512。主验收 `primary_success=false`，并非测试失败或计算异常。

理论测试与有限合成反例分别在 `theory/TEST_REPORT.md` 和 `counterexamples/README.md`。本文件记录执行结果，不代替独立完整性审查。

随后执行 `verify_run.py`：21组决策重现一致，105候选及18现任坐标完全匹配，7臂全部聚合一致；结构化输出见 READONLY_VERIFICATION.json。独立审查另重算1123个既有坐标到原始激光的距离，最大差0，结论WARN（范围与假设限定）。

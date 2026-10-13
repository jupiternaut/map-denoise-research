# 流程更正：核验启动早于主批完成

2026-10-12 12:20:02 CST，root在求解session13549仍运行、17/18观测已完成时误触发核验。核验正确在RUN_STATUS schema/COMPLETE检查拒绝，约0.010秒，未继续解析private真值。原FAIL含错误 `VerificationError: RUN_STATUS: changed completed schema`。

之后先等待同一session退出0，确认COMPLETE/18观测/486动作/36新求解；将仅含FAIL记录的owned `verification`目录更名为 `verification-premature-attempt`，没有删除或覆盖文件；再在同一output启动冻结核验器，仍使用原BUDGET起点及deadline。核验PASS，总窗口174.454814秒。

这是启动顺序错误，不是算法负结果、核验器漏检或第二批实验。没有修改科学py或重跑求解。正式成功与原失败均保留，报告不得隐藏该过程。

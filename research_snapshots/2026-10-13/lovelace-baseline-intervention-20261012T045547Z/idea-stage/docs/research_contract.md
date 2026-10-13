# Research Contract：相机基线改变可用边缘约束
Problem Anchor：保留有效几何、修复不足；上一轮第三深度所有初态仍在外包内，0修复。
Dominant contribution待验证：从物理投影与像素积分推出缺失的纯材质边缘约束，改变相机布局恢复它；不是重新拟合评分器。
C1锁定B80新观测相对B70能否改善第三档40信息偏移；C2共同偏差依赖的额外收益及覆盖/KEEP/negative保持。
Method：公开baseline参数化历史正向几何和连续区间包络，冻结光度必要行、有限传播、三branch union、hybrid。seed61005与共同pose/noise tape新协议。
数据：受控合成，36图像324初态972动作72求解；当前未生成，未测成绩。
最高风险：改变baseline影响partial遮挡和背景校准，不只是边缘导数；完整报告，不把均值归因拆分。B70 identity测试与B80独立包络核验前置。
停止：单次预注册矩阵，不追扫baseline或seed；一seed不等于真实迁移。


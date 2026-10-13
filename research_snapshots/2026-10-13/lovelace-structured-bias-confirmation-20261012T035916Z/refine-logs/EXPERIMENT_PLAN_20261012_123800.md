# Experiment Plan：最小冻结确认

Problem：保留有效几何，修复不足。
Method thesis：同图像偏差不能由背景完全识别时，保留共同依赖能修复额外初态。

| Claim | Minimum evidence | Block |
|---|---|---|
| C1：联合增量在未见noise实现复现 | seed61004锁定方法，calibrated→joint MSE下降且新增修复>0 | B1 |
| C2：收益不以丢掉正确几何换来 | 深度/材料/偏差覆盖18，正确KEEP18，负控KEEP24，初态损伤0 | B1 |

B1 MUST：clip18观测、KEEP/calibrated/joint共486动作，36新冷求解。主表144偏移指标及配对，附按模板/深度、T_cal、集合关系与成本。单seed不报告跨seed SD，旧三seed仅开发参照。
原epsilon/beta/camera/511树/8轮/动作器冻结，无新现代学习组件。切掉raw、旧wide重复、LP、学习器、新生成器与超参数网格。

| Milestone | Gate | Budget |
|---|---|---|
| M0 | 公开/私有拆分、观测生成次序及计数、保存后评价单测 | 本机CPU |
| M1 | fresh同族非作者完整source无blocker，provisional | 不启动确认 |
| M2 | 18观测、36新求解、486完整动作 | 与M3共1800秒 |
| M3 | 不同作者逐branch/动作/评价全量核验 | 同原绝对deadline |
| M4 | 原目标→方法→结果→机制→下一步报告 | 不再加实验 |

0GPU/API/安装/SSH/发布/部署。失败保留，不通过重选seed追赢家。新主批仅一个job，非36个独立队列作业，沿用本地逐观测checkpoint执行。

## M4消融结算（完成后补充，不修改冻结协议）

calibrated→joint 是已执行的关键删除对照，有限C1成立。已有18观测/162配对的集合—动作读回已完成。见 ABLATION_ASSESSMENT.md 和 ../analysis/MECHANISM_TRACE_20261012_123000.json。未来传播轮、同CPU预算、独立成像确认只列为条件性问题，状态未执行；本轮不增加任何求解。


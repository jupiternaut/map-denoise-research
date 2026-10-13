# A9 完成检查点
language: zh

## 当前完成状态

正式批次continuous-v1已完成，STAGE_STATUS=COMPLETE、VERIFY=PASS。不再重跑同一正式批次。主机liekkas；准确新根：
/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-anchors-1

旧independent-anchors-1及更早阶段只读；A7仍为INCOMPLETE，不追认。

## 交接结论

主HULL规则，288偏移：both MAE=4561/4608 mm，MSE=33158080547/21107294208 mm²，相对KEEP MSE下降53.880955718%，122改善/166不变/0恶化。36正确现任不动；48负对照偏移不动。primary40下降61.287782602%，21改善/19不变/0恶化。

单臂改善并集93，联合122，新增29。信息性30观察的平均包络宽10.90625→2.703125 mm。全部144聚合并集单分量，所以HULL/UNION1296成对坐标相同。18共享采集实现，不是288独立场景。

## 当前保存与核验

144连续外集、432分支树、2592动作、逐行/观察/子集评价全部落盘。40手写测试通过，fresh源码ACCEPT；独立公开证书重建与全部真世界覆盖通过；fresh stdlib/Fraction算术复算2592 ROWS与136 SUMMARY零差异。

总执行681.055秒；没有后台正式任务。只运行一次，未在正式结果后改算法/注册协议/冻结源码。

## 下一阶段，不在本轮执行

针对118信息性未改善输入：
1. 用公开前沿记录区别截止/UNKNOWN保留与外松弛可行；
2. 同输入比较预注册的区间精化与共享相机/跨视图耦合界；
3. 竞争解释仍完整可行时再设计独立观测。

不自动把“不动”归因成不可辨识，不强行移到集合中心。保持覆盖、坐标改善比例与正确输入保留三项同步验收。

## 范围

本轮同族合成验证；没有真实点云迁移，没有对每个坐标边界寻找完整前向见证。无GPU/API/SSH/安装/部署/GitHub/GitBook操作，identity默认未变。
查看REPORT.md、ANALYSIS.json、COMMANDS.md、ARTIFACTS.json、FINAL_SEAL.json。


# A5 CHECKPOINT — COMPLETE/PASS

主机liekkas，绝对根 /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1，output /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/outputs/polytope-v1。2026-10-12 23:28:52 +08:00。

原始目标保留有效几何并修复不足。完成72冻结外包、5103原保留叶，648新+648旧+324KEEP=1620评价。44夹具PASS，源码fresh/same-family/provisional PASS；独立verify PASS，139源/28输入一致，main+verify267.42s。

实测37证书删叶：单行1、联合36；4396精确必要松弛见证；670 UNKNOWN原样保留。8/72 hull收窄。η1主40 cal0/joint1，均与旧相同。η0 joint两个已有成功任务更好，全144 MSE下降1.28975%；无新增改善任务数。原同leaf真深度/九材料/三偏差见证全部保留，正确初态全部KEEP。

报告REPORT.md；数据ANALYSIS.json，原输出outputs/polytope-v1。原数据/代码/报告只读；新源码正式运行后未改。没有后台科学任务、GPU/API/SSH/安装/发布；没有改部署默认。

下一建议（未执行）：主难例初态深度点值可行性与其深度盒对照，必要时把点值见证送进真实前向模型。先分辨几何区间松弛与观测相容的替代解释，不新增路由或重复δ扫描。不能把盒级见证当具体错误深度真实可行。

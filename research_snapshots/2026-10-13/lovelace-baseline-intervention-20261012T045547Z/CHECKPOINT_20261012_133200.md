# 检查点：基线成像干预完成

日期：2026-10-12；host liekkas。
Parent=/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z
Recovery=/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/recovery-1
Verified output=/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/recovery-1/outputs/baseline-v1

## 已完成
M0–M4全部DONE：身份/包络/负控/共同随机性/encoded transport测试、fresh源码审查、一次主批、独立完整核验、终点结算。
36图像324任务972动作72冷求解；mechanical PASS。
C1深档joint改善1/40→11/40，MSE3.292722→2.367334（-28.10%）。
全144偏移26→51改善，joint MSE2.552791→1.903599（-25.43% vs B70；-44.11% vs KEEP）。
保留浅档退步：0.534431→0.754683 mm²；cross-baseline joint33/121/8 better/same/worse。
C2本批支持有联合增量，非逐点支配。

## 保留状态
首次父输出INCOMPLETE，不覆盖；0完成动作，无科学成绩。
recovery-1仅修JSON解码与路径；同一data、seed、原SOURCE_LOCK/INPUTS、原1800秒deadline。
全部formal科学源码/数据/outputs自封存后不再改。报告与分析另记。
旧确认及其他历史runs只读；不重跑旧矩阵，不改默认部署。
当前没有本任务后台实验进程。无GPU/API/SSH/安装/外部发布。

## 后续仅计划
消融审稿计划独立完成后落在refine-logs/ABLATION_PLAN.md；本轮未运行。
首选固定B80禁用新增边缘推理行、保留全图校准，定位信息收益。
不使用GT挑行、挑模板、挑动作；不把宽基线当普适升级。
还缺明确成像行的机制归因及独立真实迁移。

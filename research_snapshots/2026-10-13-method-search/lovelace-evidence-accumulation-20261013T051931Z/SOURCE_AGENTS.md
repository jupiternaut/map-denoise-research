# A11：累积观测证据，不丢弃已获支持
language: zh
host: liekkas

本轮用户同意继续研究数学通路。目标仍是保留有效几何、修复不足。本轮属于“固定表示和证据、改变求解的累积方式”，不是新真实场景或新成像模型。
唯一新写根是本目录，逻辑路径 /srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z；runs 为已有符号链接，物理路径 /mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z。必须验证二者指向同一对象，不搬移或修改历史。
历史 lovelace-baseline-intervention-20261012T045547Z 及所有子阶段只读；尤其A10全部源码和封存均不改。既有open3d-019 Python -B、单线程CPU，无GPU/API/SSH/安装/发布/默认部署改变。
方案：两臂都从A10 depth已核验外集出发，统一深度容差15/1024 mm；D_then_D继续深度分支，D_then_DC同时将相机区间四分。每分支最多128分类/6秒，36观测一视同仁，同增量预算上限，实际费用另报。不读旧DC结果决定新方法；不按GT或已有改善标签挑观测、动作或预算。搜索完72集合才读取324现任，HULL输出固定。
数学：只允许精确证书删除完整深度×相机区域，未完成全部相机覆盖则保留，保留UNKNOWN/内叶/截止。新外集必须包含真实可行深度并嵌套于A10 D。HULL在一维覆盖前提下不比旧D更差；此条件性质不自动迁移到三维点集。
职责：root写协议、I/O、执行/评价/报告；point_ablation_plan只写accumulation_solver.py及其test；a10_frontier_audit只写verify_accumulation.py及其test；其他分派明确后再写。fresh审计者只读。所有手工编辑apply_patch。
正式前计划和科学源码冻结，运行一轮有界批次，所有失败保留；不得覆盖既有outputs。评价真值在全部公共预测保存、独立证书验收后解析。总批次1800秒（运行900、核验保留900），不重置。

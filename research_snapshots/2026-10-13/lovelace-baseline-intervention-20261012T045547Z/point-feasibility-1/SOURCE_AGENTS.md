# A6 点值可行性诊断
language: zh
host: liekkas

唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1。此阶段承接用户2026-10-13最新“继续”，仅在新根覆盖父阶段已结束的任务限制。此前全部源、观测、参数、树、结果只读。
原始目标：保留有效几何并修复不足。本轮诊断旧错误初态为何仍被允许，不将诊断当部署算法或新增几何修复成绩。
固定A4-V照片、324任务、两η、B80、radius、clipping、δ=1/10、ε/β、校准、原树和材料界。比较原保留深度盒和同叶内精确初态深度的必要线性约束；不新生成深度、阈值扫描、调整路由或学习器。
方法仅用public、A4-V bounded outers、A5精确判定；全部public判定保存读回完整计数后才parse旧private作分层评价。
精确证书才判不可行；精确必要松弛见证不等于完整成像可行；UNKNOWN保留未决。此阶段不修改任何已有输出位置。
root负责协议、runner、executor、统计、报告；委派作者只能编辑指定文件，reviewer只读。仅apply_patch编辑。
同host现有 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B，CPU单线程、单批主程序+独立verify共享3600秒绝对预算，无重置、无formal结果后修源。手工fixtures→fresh同族暂定源码review→冻结正式批次→独立核验→结束。
不GPU/SSH/安装/外部API/发布/部署，不修改GitBook/GitHub，不新建框架。

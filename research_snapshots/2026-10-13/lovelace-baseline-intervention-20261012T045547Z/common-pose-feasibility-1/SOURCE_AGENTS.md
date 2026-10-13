# A7 共同姿态前向见证诊断

language: zh
host: liekkas
唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1。
用户最新“继续”授权在此新根承接 A6 检查点；父阶段及全部旧结果只读，此文件仅覆盖本新阶段的旧任务限制。
目标保留有效几何、修复不足。本轮固定旧照片与公开初态深度，寻找整幅图像共用合法相机偏移的完整前向解释；不是新几何修复器，不修改默认。
仅 joint、原324任务、972分支；相机左右两标量偏移在±1/5，中间0；材料9个[0,1]、同材料跨视图差≤1/10，3个偏差服从原校准界，噪声±1/100，clip一次。不得偷读实际姿态或材料。
正判只接受有理参数完整前向复现；数值搜索失败 UNKNOWN，不得等同不可行。可沿用 A6 精确必要矛盾，须重查证书，不能称新排除。
root负责协议、runner、统计、封存；作者只能编辑指定文件；fresh reviewer只读。全部源码/文档编辑用apply_patch。
现有open3d-019 Python -B，CPU单线程；一批main+独立verify共3600秒绝对截止不重置。手工夹具→fresh同族暂定review→冻结→正式→独立核验。
方法只读public与冻结校准/必要诊断；全部结果保存读回324/972完整性后才读旧private作评价。严禁GPU/SSH/API/安装/发布/GitHub/GitBook/部署/新候选/调路由/学习器。

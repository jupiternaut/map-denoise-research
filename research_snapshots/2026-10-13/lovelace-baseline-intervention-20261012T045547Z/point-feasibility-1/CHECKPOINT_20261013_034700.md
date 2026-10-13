# A6 检查点：点值诊断已结束

2026-10-13；liekkas；唯一工作根 /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1。

## 状态

正式输出：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1/outputs/point-v1。
STAGE_STATUS=COMPLETE、verification/VERIFY=PASS；648诊断、980 local＋1944 global，完整公共记录保存读回后才进行private评价。main79.878秒，main＋verify226.462秒，单一3600秒期限，无重置。

## 结论

主η1/10深度9007/14的40错误初态/方法：cal40点值必要见证；joint39见证＋1旧来源已排除；新排除均0。主瓶颈不能全归到深度盒宽度。
两条件joint完整288偏移任务中新增点排除11：η0的5、η1/10的6；只有2任务旧盒见证→点证书，其他9旧UNKNOWN→点证书。13局部证书对应11任务，不同统计单位分开。
全部正确初态未排除；72 actual necessary witnesses存活。未新增几何位置，未部署。

## 可续方向，不是已执行项

对主39个joint错误深度使用统一相机偏移的完整前向几何；检验必要见证能否被真实共同姿态实现。固定照片、深度、噪声/材料/偏差界；合法精确前向见证是可行证据，局部优化失败不证明无解。
不继续单纯减小深度容差或调路由追分。

## 冻结边界

A4-V与A5均只读；A5已有98个索引工件逐文件一致，报告及索引封存一致。A6源/输入149/35检查一致。
保留原始三态，不用全局见证改写局部UNKNOWN记录；报告解释lifting后的数值差异。
当前没有本任务后台作业。用户本轮只授权继续实验，本轮未GitHub/GitBook发布、未GPU/SSH/API/安装。不要把已结束的3600秒原输出作为可补跑空间。

## 入口

REPORT.md：主终点与机制；ANALYSIS.json：原始记录描述性复算；PROTOCOL.md：注册范围；COMMANDS.md：命令。形式化证据与必要条件不等于完整物理soundness或真实迁移。


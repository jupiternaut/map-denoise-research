# A14：少量联合约束能否补齐快法的深度边界

language: zh
host: liekkas

目标仍是保留有效几何、修复错误几何，寻找更短的数学计算通路。本轮只研究A13快法与旧法的边界差距。所有祖先目录只读；本目录是唯一写入目标，runs现有符号链接解析到 /mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/ 下同名目录，不是另一机器。

先开发诊断，后冻结确认。开发数据限A12 data/dev；不要读取A13确认真值做设计。方法不得读取真值或整组现任。相同观测、同一深度输出、同一误差界；不能靠换目标或排除难例获胜。浮点提出排除、精确证书接受，未证实的区间保留。

现有Python：/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B。单线程CPU；无安装、GPU、外部API、发布或部署默认变更。开发总计算暂限180 CPU秒，确认暂限600 CPU秒；主Agent按诊断冻结具体矩阵。所有编辑用apply_patch，输出文件不可覆盖旧运行。fresh源码审查、fresh结果审查分别用新上下文；同家族语义审查标provisional。

分工：主Agent负责协议/执行器；诊断Agent只写diagnostic_*与DIAGNOSIS.md；构造Agent只写joint_*与JOINT_DERIVATION.md。不要扩为通用框架。

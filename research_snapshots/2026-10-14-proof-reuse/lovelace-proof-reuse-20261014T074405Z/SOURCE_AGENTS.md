# A15：纯容差、边界体检与可继承证据

language: zh
host: liekkas

本目录是本轮唯一写入目标。runs 已核验的符号链接指向同机 /mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/；所有祖先实验只读。不得发布、部署、安装或使用 GPU/API。

用户要求三项：A14 四盒逻辑和顺序不变，只把容差从 .025 改为 .001；最终边界完整四盒离线精确体检；真实调用链上独立开关热盒排序和已验证 Farkas 乘子继承。前三组历史损失子集是诊断分组，不是训练或抽样规则。旧 A14 的 18 世界 / 126 输入全量回放，非新的确认集。

精确 LP 可行只证明必要松弛可行，不能写成物理成像可行。浮点状态不是精确证书。父证书只可在同观测、同分支、同相机盒、包含关系满足且子约束严格余量重验成功后复用；失败走原认证。unknown 不得当 infeasible。

Python /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B，单线程。先实现/单测再锁定代码和协议后回放；计冷启动总成本、内部 LP、精确检查和离线诊断成本。全量运行暂限 1200 子进程 CPU 秒，诊断每盒限 3 秒；若需要扩量先报告。只用 apply_patch 编辑源码。

分工：主Agent负责执行器、纯容差版本、协议与报告；构造Agent仅写 reuse_kernel.py、reuse_search.py、reuse_tests.py、REUSE_DERIVATION.md；诊断Agent仅写 boundary_diagnostic.py、boundary_tests.py、BOUNDARY_DIAGNOSTIC.md；审查Agent只读并返回审查结果。

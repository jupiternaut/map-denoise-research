# A13：快速边界搜索与精确排除证书

language: zh
host: liekkas

仅在本目录实施用户批准的后续实验。逻辑路径是 `/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z`，沿已有 runs 符号链接解析到 `/mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/` 下同名目录。旧 A12 `/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z` 及所有祖先只读。

目标：保留 A12 自由构造的相机消元、可见面积外包围和两端搜索，检验恢复精确排除凭据后的精度—成本。不得改物理模型、用真值选择动作、调阈值以追求正结果。浮点建议、精确几何约束、证书核验分别计时；无法核验的排除应保留区间。

使用现有 `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`、单线程 CPU。无 GPU、安装、付费 API、远程机器替代、部署或发布。推理只读 photo/camera/appearance，不读任务列表或真值。开发只用 A12 dev；旧确认集仅诊断回放；新确认数据必须冻结方法后生成且全预测保存后评分。

分工：证书作者只写 cert_kernel.py、test_cert_kernel.py、CERTIFICATE_METHOD.md；独立验证器作者只写 verify_certificates.py、test_verify_certificates.py、VERIFIER_SPEC.md；主 Agent 写搜索适配器、协议与执行器。代码审查者不参与实现；最终结果审查者另起新上下文。所有编辑用 apply_patch。模型内审为 same-family/provisional，不是外部审稿接受。

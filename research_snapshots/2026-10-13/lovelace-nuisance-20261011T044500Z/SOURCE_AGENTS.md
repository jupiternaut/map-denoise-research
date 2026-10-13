# 未知外观与遮挡合同实验

host: liekkas
language: zh
唯一写入目录：/srv/slam-research/grf/map-denoise/runs/lovelace-nuisance-20261011T044500Z。
只读依赖：lovelace-hybrid-20261010T214217Z，以及lovelace-challenge-20261010T205013Z；均位于同一runs目录。
本轮不改历史、部署默认、GitHub、GitBook；不安装、不联网调用模型、不运行GPU。

Root拥有协议、合同、报告、痕迹与调度；hybrid_theory拥有nuisance_model.py/test_nuisance_model.py/THEORY.md；hybrid_runner_impl拥有run_nuisance.py/test_run_nuisance.py；lovelace_result_review拥有verification/目录的独立核验。
所有作者文件用apply_patch。不得写其他分工文件。实验源确认前冻结；输出目录必须全新。
真值仅生成观测并评价。策略接收公开相机/形状/误差范围、输入图片、带来源标签的额外外观观测、初点；不能接收实际深度、实际颜色、生成kind或隐藏随机带。
使用上一轮原样certificate_hybrid，不调阈值。颜色观测不是免费先验，单独记账。

## Pipeline Status

implementation: complete
confirmation: complete
verification: complete

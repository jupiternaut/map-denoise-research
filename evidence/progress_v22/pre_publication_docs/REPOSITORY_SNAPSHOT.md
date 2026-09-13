# GitHub 工程快照说明

目标账号：`jupiternaut`，仓库：`map-denoise-research`，默认私有。提交源为 liekkas 上当前工程本身，非替代工作树。

## 范围

包含当前 `map-denoise-dataset-pilot-v1` 全部可提交源码与文本材料：V1/V2数据接入与修复、V3–V18构造，
公开3DGS/网格模型适配实验、测试、协议、报告和既有图片。缓存和敏感文件不纳入版本控制。
为避免只有代码没有结果，另导出各轮紧凑结果表、摘要、审计、时序与清单到 `evidence/runs/`。
每个导出文件的原路径、字节数和SHA256记录于 `evidence/EXPORT_MANIFEST.json`。

`legacy_sources/` 是三个显式旧依赖项目的源码/说明副本：correlation、six-track、t2-boundary。
原件保持只读；本次没有改写其导入路径，也没有将副本运行成功冒充原目标复现。
没有收集无关的早期SLAM项目、其他独立GPU仓库或整段聊天历史。

## 不上传的内容

- 原始数据与下载资产：`/srv/slam-research/grf/map-denoise/datasets/`。
- 完整实验输入/逐点预测/中间优化记录：`/srv/slam-research/grf/map-denoise/runs/`（本工程主运行树约1.7GB）。
- Python虚拟环境、安装包、GPU模型权重、登录凭据和缓存。

因此这是**完整当前代码工程及精选证据快照**，不是整个磁盘或全部原始实验数据的异地灾备。
历史报告中的绝对路径是原始证据标识，在GitHub网页上不能直接打开；优先查看对应的 `evidence/runs/` 文件。
完整清单引用的原始NPZ/模型文件未随Git提交，不能凭清单声称远端拥有这些原文件。

## 使用与可复现性

最新点级算子的核心依赖为NumPy和SciPy。环境实测包版本见导出清单；历史分支还会按需使用Open3D、
PyMeshLab、Matplotlib或PyTorch，不将未安装的可选包伪写为已验证环境。

在项目根目录、已具备依赖的Python中可运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s exploration_v18 -p 'test_*.py' -v
```

局部算子 `exploration_v18/v18_operator.py::filter_local` 接受局部mm坐标与给定sigma。
这不需要外部原始点云下载；但完整历史实验入口包含 liekkas 主机检查、绝对运行路径及已暴露检查点依赖，
迁移到Windows/Mac或其他Linux前需要单独配置，不能盲跑下载/恢复脚本。

本次上传不新增开放源代码许可证或替第三方数据重新授权。仓库私有，公开前应另核对来源、依赖许可和数据再分发条件。

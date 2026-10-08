# LOVELACE 数学封闭实验：有限世界与连续深度

本目录发布 2026-10-08 两阶段实验的完整源码、固定几何资产、冻结方案、输入、逐决策记录、正负控制、报告、交互审阅页、独立核验和原始打包收据。

[第二阶段报告](experiment/continuous_world/REPORT.zh.md) · [连续覆盖推导](experiment/continuous_world/THEORY.md) · [第二阶段审阅页](experiment/continuous_world/review.html) · [科研图](experiment/continuous_world/results.svg) · [第一阶段报告](experiment/REPORT.zh.md)

- 第一阶段：有限九状态世界，精确有理数渲染与安全收益证书。
- 第二阶段：整个 `[540,660]` 连续深度区间及两侧相机的有界 x 位置误差；72 个参数/噪声配置、216 个初点决策。72 个正确初点全部 KEEP，144 个 ±60 mm 偏移状态全部改善，证书方法零伤害；名义相机残差网格对照误改 37 个正确初点。

这些结果只适用于声明的几何、材质、相机和误差预算。LOVELACE 模型使用来源可核对的孤立 M4 种子面片，毫米为归一化模型尺度；另含矩形解析对照。未验证完整角色、真实地图、未知旋转/材质或全局拓扑；违约控制已显示覆盖失效后会产生伤害。本支线不改变仓库默认 `identity`，也不是现有 full9 的复现或跨方法排名。

## 从仓库克隆复核

从仓库根目录执行；科学复算仅需 Python 标准库，确认环境为 Python 3.14.0：

```powershell
python -B research_snapshots/2026-10-08/lovelace-mathematical-closure/verify_publication.py
```

该命令只读核对全部发布文件的 SHA256、算法/验收器锁、两个 ZIP 和第一阶段保留情况，再从当前路径运行 33 项测试及独立区间记录验收，不覆盖归档结果。

重跑确认请另选新目录：

```powershell
Set-Location research_snapshots/2026-10-08/lovelace-mathematical-closure/experiment
python -B -m continuous_world.run_experiment --output continuous_world/outputs/my-reproduction
```

运行器拒绝覆盖已有实验。随包 JSON 几何资产足够，无需原始完整 GLB、TPU 或 GPU。审阅页与 SVG 可离线打开；可选的图/页面重建依赖已记录于 [requirements.txt](experiment/continuous_world/presentation/requirements.txt)。

## 完整下载与证据

[第二阶段完整 ZIP](Lovelace-mathematical-closure-stage2-20261008.zip) 包含两个阶段的 132 个文件；[ZIP 收据](Lovelace-mathematical-closure-stage2-20261008.receipt.json) 与 [新目录验收](Lovelace-mathematical-closure-stage2-20261008.delivery-check.json) 随附。

[第一阶段原 ZIP](Lovelace-mathematical-closure-stage1-20261008.zip) 与 [原收据](Lovelace-mathematical-closure-stage1-20261008.receipt.json) 放在实验根的父目录，使历史保留检查也能重跑。冻结源码和科学证据从本地逐字节复制，历史绝对路径仅为来源元数据，复核入口按本文件位置解析路径。

[发布清单](PUBLICATION_MANIFEST.json) · [独立分支/决策验收](experiment/continuous_world/outputs/confirmation-v1/VERIFICATION.json) · [来源与汇总审计](experiment/continuous_world/outputs/confirmation-v1/AUDIT.json) · [冻结重跑一致性](experiment/continuous_world/outputs/confirmation-v1/REPRODUCIBILITY.json)

独立区间验收器复用经数学审阅的几何内核；来源审计另行重算汇总和一个控制图，并非第二套连续几何实现。冻结重跑是同一批条件的复现，不增加独立样本数。`experiment/PACKAGE_MANIFEST.json` 是第一阶段历史收据，当前完整实验树使用 `experiment/STAGE2_PACKAGE_MANIFEST.json`。

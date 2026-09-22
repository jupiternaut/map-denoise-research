# Map Denoise Research

**多视图几何修复、收益筛选与科研决策实验。** 更新于 **2026-09-22**。

本仓库保存算法源码、冻结协议、正负结果、测试及修订记录。目前的几何主线是：利用照片与相机标定构造局部几何修正，再预测修正收益，选择移动或保留原点。

当前结果支持**有条件的偏移恢复**，不支持默认开启的通用滤波器。Windows / RTX 5080 交接包已发布，**CUDA 后端尚待实现**。

## 从这里开始

| 你想做什么 | 入口 |
|---|---|
| 在 Windows 上交给 Codex 开发 | [启动提示词](handoffs/v28-windows-5080/START_WINDOWS.md) · [完整交接 ZIP](https://github.com/jupiternaut/map-denoise-research/raw/8666e3ffee09a3a1a105fa523ed94ae2218fde01/handoffs/v28-windows-5080.zip) |
| 看工具架构与验收范围 | [架构](handoffs/v28-windows-5080/ARCHITECTURE.md) · [实施任务](handoffs/v28-windows-5080/TASK.md) · [验收标准](handoffs/v28-windows-5080/ACCEPTANCE.md) |
| 看最新几何实验 | [完整报告](research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/EXPERIMENT_REPORT.md) · [协议](research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/EXPERIMENT_PROTOCOL.md) · [执行口径补充](research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/EXECUTION_NOTE.md) |
| 复查原始指标 | [METRICS.csv](research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/evaluation/METRICS.csv) · [汇总](research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/evaluation/SUMMARY.json) |
| 阅读当前 CPU 实现 | [包说明](handoffs/v28-windows-5080/reference/package/README.md) · [运行时](handoffs/v28-windows-5080/reference/package/v28_closeout/runtime.py) · [表面假设评分](handoffs/v28-windows-5080/reference/package/v28_closeout/surfacelet.py) |
| 阅读元研究与历史探索 | [元研究导航](meta_research/README.md) · [V23–V25 / CPR / 精确搜索](PROGRESS_20260916.md) · [V19–V22](PROGRESS_V22.md) |

## 最新结果：跨场景条件恢复

固定主方法为 `post_A_keep`，模型使用旧开发场景 scan24/37 的归档样本训练。scan40 只作工程适配；scan55/65/69 为三个同来源确认场景，每场景四个 ROI。所有构造输出封存后才获取并打开独立几何参考，未依据确认结果改模型或阈值。

指标为**固定原始源点行到参考几何的最近邻 MSE，单位 mm²**，先对场景内 ROI 等权平均，再对三个场景等权平均。扰动沿逐点参考相机射线施加；这些不是整场景刚体位移。

| 输入 | 不处理 identity | 主方法 post_A_keep | 相对不处理 | 改善的 ROI |
|---|---:|---:|---|---:|
| 原始输入 | 0.753165 | 0.814326 | 恶化 8.12% | 0/12 |
| −1 mm 扰动 | 0.803265 | 0.819364 | 恶化 2.00% | 4/12 |
| +1 mm 扰动 | 1.286755 | 1.161440 | 改善 9.74% | 11/12 |
| −3 mm 扰动 | 2.867611 | 1.833010 | 改善 36.08% | 12/12 |
| +3 mm 扰动 | 4.295019 | 2.843762 | 改善 33.79% | 12/12 |

**已测到的能力：** 冻结方法在三个新场景的 ±3 mm 条件下均有收益；相对接受相同点数的随机筛选，主方法在这两种条件下的 MSE 还低约 19%。这支持修正候选及收益筛选的条件性价值。

**仍然存在的失败：** 原始输入的三个场景、12 个 ROI 全部退化，故默认仍为 `identity`。原始、正负小扰动的响应值得进一步分析灵敏度与不对称性，但本轮没有识别其因果机制。

解读边界：

- 百分比是 **MSE** 的相对变化，不是 MAE/RMSE 降幅；12 个 ROI 不是 12 个独立场景。
- 这是同 DTU/GeoSVR 来源的新场景确认，不是跨传感器普适性证明，也不是 DTU 官方整场景排行榜。
- 七个预定臂、失败和次要结果均保留；`post_AB_keep` 不能事后替代主臂。
- 参考支持范围与照片 ROI 不完全一致，完整性指标另有[支持范围诊断](research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/SUPPORT_DIAGNOSTIC.json)。该诊断不撤销原始输入的负结果，也不能证明完整性误差全由窗口造成。
- 尚未证明真实薄层身份保持或优于成熟外部方法。详见[验收判定](research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/EXPERIMENT_DECISION.json)。

## 方法与输入

```text
已有几何 + 参考照片/相机 + 四张有序来源照片/相机
  → CPU 法向与锚点
  → 多视图光度评分（20 个局部表面/支持假设 × 49 个射线偏移）
  → 候选选择与特征插值
  → 冻结收益预测器
  → KEEP / 修正点 + 决策记录
```

当前实现基于 NumPy、SciPy 和 scikit-learn，核心 API 为 `construct`、`apply_arrays`。输入需要可靠的单位、相机与图像坐标约定；**仅提供任意 PLY 不足以运行同一方法**。模型预测的是平方误差收益，不是校准后的安全概率。推理不读取评价真值。

## Windows / WSL2 / RTX 5080

1. [下载固定版本交接 ZIP](https://github.com/jupiternaut/map-denoise-research/raw/8666e3ffee09a3a1a105fa523ed94ae2218fde01/handoffs/v28-windows-5080.zip)，完整解压。
2. 在解压目录打开 Windows Codex，发送 [START_WINDOWS.md](handoffs/v28-windows-5080/START_WINDOWS.md) 中的提示词。
3. Codex 按 `AGENTS.md` 和 `TASK.md` 核对目标机器，优先在同机已有 WSL2 中实施；系统安装与驱动变更不自动授权。

包内包含架构、阶段任务、数值验收、显卡空闲调度、取消/恢复设计，以及固定提交的 CPU 源码和模型。实施顺序为：

**CPU 包装对照 → SDK / CLI → CUDA 光度评分 → 数值与决策一致性 → 5080 性能实测。**

GUI / TUI / MCP 预留共享应用 API，不是第一版前置要求。文档中的 `map-recovery` 命令是待实现接口，当前仓库没有可直接安装运行的该 CLI；GPU 提速和几何质量改善也分别验收。

ZIP SHA-256：`6c4f5c1aea8ed74960c3271f164ec75eed7a97a4a81f3a29367bdf4be7cd3139`。

## 最小检查与复现状态

在仓库根目录，有 Python 即可核对交接包文件哈希，不会加载模型或运行实验：

```bash
python -B handoffs/v28-windows-5080/verify_bundle.py
```

在独立的 Python 3.12 虚拟环境中，安装参考依赖后可运行 CPU 契约测试：

```bash
python -m pip install -r handoffs/v28-windows-5080/reference/package/requirements.txt
python -B handoffs/v28-windows-5080/reference/tests/test_runtime.py
```

这些是 CPU 包测试，不是数据集重建命令。原机打包时记录为 **5 通过、2 个依赖旧归档的回放跳过**；不能据此声称 Windows/5080 已验证。加载 joblib 仅限可信且哈希匹配的模型。固定依赖在目标平台的可安装性由实施任务核验，不自动换版本或重训模型。

| 交付 | 状态 |
|---|---|
| 冻结 CPU 算法、模型与契约测试 | 已发布；原机验证 |
| 三个同来源新场景的条件恢复实验 | 已完成；正负结果均公开 |
| Windows / WSL2 架构与 Codex 交接包 | 已发布 |
| CUDA 后端与 RTX 5080 等价/速度测试 | 待实现、待实测 |
| 对口成熟外部基线 | 待补；官方 COLMAP 本轮未运行 |
| 干净环境从数据到结果的完整复现 | 待补 |

**本轮几何确认实验已经结束。** 后两项属于研究收尾证据；GPU 移植属于工程任务。新的灵敏度实验需另立范围，不能把每个新任务都当成本轮尚未完成。

## 两条研究线与历史入口

- **几何研究**：早期多扫描关联与薄层估计，随后探索曲面修正、光度证据、表面再生、检验/投影及收益筛选。历史结果不等于当前方法能力；从 [2026-09-16 导航](PROGRESS_20260916.md) 和[最新确认发布说明](publication/CLOSEOUT_20260922.md)分别进入。
- **元研究**：研究有限解释集下如何选实验、修复冲突和比较开环/自适应策略。[Hermes 最终目录审计](meta_research/hermes/outputs/20260914T142434Z/REPORT.md)显示有限装置中 H=4 有自适应增量、H=6 开环已零误差；[有限精确搜索报告](research_snapshots/2026-09-16/pnp-oracle-lab-20260916T112018Z/REPORT.md)保留模型平局等更正。这不是 P=NP 证明，也不是通用自主科研已实现。

元研究是辅助成果，不是几何课题结题的前置条件。历史图件见 [V22 科研图册](paper_visuals/agentrx_style/README.md)。早期数据接入与原机复现命令保留在[更新前的 README](https://github.com/jupiternaut/map-denoise-research/blob/8666e3ffee09a3a1a105fa523ed94ae2218fde01/README.md)，其中“当前有效版本”仅代表当时状态。

## 仓库内容与发布边界

| 位置 | 内容 |
|---|---|
| `handoffs/v28-windows-5080/` | 当前工程交接及可独立核验的 CPU 参考包 |
| `research_snapshots/2026-09-22/` | 最新确认实验源码、协议、指标及封存记录 |
| `research_snapshots/2026-09-15/`、`2026-09-16/` | V25、CPR、后续机制和有限精确搜索快照 |
| `meta_research/` | Hermes 与受限开放模型挑战 |
| `exploration_v*/`、`reconstruction_v22/` 等 | 历史几何实验与实现 |
| `evidence/`、`publication/`、`paper_visuals/` | 结果证据、发布清单、验证记录与图件 |

这是研究与工程交接仓库，不是全部磁盘数据的备份。原始点云/照片、第三方数据包、运行环境及大部分逐点二进制输出未收录；完整历史回放仍可能需要原机材料和路径配置。冻结模型已包含在当前参考包中。

文件来源、哈希和排除范围见 [确认实验发布清单](publication/CLOSEOUT_20260922_MANIFEST.json)、[Windows 交接发布说明](publication/WINDOWS_HANDOFF_20260922.md)及[旧公开快照说明](publication/README.md)。历史封存清单可能包含未发布文件，不能把“公开子集完整”当作“所有原始运行材料齐备”。

原始协议、报告及负结果不覆盖；README 只负责导航与当前状态。第三方代码、数据和模型仍遵循其各自许可，本页不新增授权或许可。

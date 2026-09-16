# 几何滤波与元研究实验（V1–V25 / CPR / 精确搜索）

更新：**2026-09-16**。新入口：[最近进展与证据导航](PROGRESS_20260916.md)、
[公开快照与复现边界](publication/README.md)。本次保留远端已有 V22 科研图册。

- **几何线**：补齐 V23、V24、真实支持域诊断、V25 表面再生及 CPR/E0–E0-F。
  V25 和最新 E0-F 均未通过各自升级条件；没有发布新的默认滤波器。
- **元研究线**：纳入 Hermes 历轮实验、更正与开放模型挑战，以及 2026-09-16 的有限精确搜索实验。
  精确搜索是小规模穷举基准，不是 P=NP 证明，也不证明普遍的 AI 科研增益。
- 代码、协议、正负结果、指标表、图和校验记录一起发布；原始数据、逐点二进制大包、
  第三方论文全文及重复复算副本留在原机。历史绝对路径仍可能需要配置。

以下 V22 总结保留为历史阶段记录，最新状态以上述导航为准。

GitHub 更新：2026-09-13。最新入口：[V19–V22 进展与证据索引](PROGRESS_V22.md)、
[V22 跨场景报告](reconstruction_v22/REPORT.md)、[局部曲面算子](reconstruction_v22/operator.py)。

**当前结论：V20 恢复了候选搜索能力；V21 的真实网格迁移未成功；V22 减轻了旧模型损伤，
但新 scan37 上相对不处理的 MAE 仅下降约 0.60%，参考召回略降，未达到综合升级标准。**
V18 仍是受控薄板族中的内部参考，不是已经证实适用于真实地图的默认滤波器。
历史方法见 [V18 报告](exploration_v18/REPORT.md) 与 [V17 报告](exploration_v17/REPORT.md)。

### V22 论文图件与作图方法

[四页矢量 PDF 图集](paper_visuals/agentrx_style/map-denoise-visual-atlas.pdf) ·
[四类图、源码与复现说明](paper_visuals/agentrx_style/README.md) ·
[科研作图方法与书单](paper_visuals/agentrx_style/科研作图方法与书单.md)

包含 TikZ 流程图、Matplotlib 多面板数据图、LaTeX 三线表和真实源码/JSON 代码块，
并提供独立 PDF、SVG、PNG 及数据来源记录。基于已发布的 scan37 确认结果重新制图，未新增实验或改写研究结论。

- 本仓库包含当前工程源码、各轮协议/报告/测试、公开输出模型的实验适配器，以及 `evidence/` 中的结果表与校验记录。
- 原始点云、下载模型、Python 环境、逐点输出大包不在 Git 中；它们没有被删除。位置与范围见 [仓库快照说明](REPOSITORY_SNAPSHOT.md)。
- `legacy_sources/` 保留三个显式旧依赖的源码副本，不修改原始检查点。历史脚本仍可能要求原绝对路径。
- 这是研究快照，不是“克隆后无需配置即可复现全部历史”的发布包，也不宣称已取得独立真实几何净收益。
- 不默认生成导师简报；历史简报仅作为既有文件保留。

---

以下保留原始 V2 数据接入说明，**其“当前有效版本”是当时的历史状态，不代表最新研究结论**。

## 历史：多帧点云滤波数据接入与首轮预实验

## 当前有效版本：V2 修复与复评

先读 [修复报告](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/repair-v2-oyuie4pl/REPORT_V2.md)，再读历史 V1 材料。原 `PILOT_REPORT.md` / `PILOT_RESULTS.csv` 留作历史证据，其中合成评分和部分因果解释已经撤回，不是当前结论。

- 当前逐项结果：[RESULTS_V2.csv](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/repair-v2-oyuie4pl/RESULTS_V2.csv)
- 当前运行：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/repair-v2-oyuie4pl`
- 正确合成输入：上述运行下的 `synthetics/`；旧 `runs/.../synthetics/` 的局部坐标字段有已知错误，仅作历史保留。
- 算法不变；新评价器 `evaluate_v2.py` 不接受算法自报层数作为几何裁判。
- 第一次开发检查运行 `repair-v2-nmt7_by9` 也保留；它与正式修复运行使用同一组案例，不计为独立实验样本。
- 回退包：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/checkpoints/pre-repair-Y8ZRZZ/`，含修复前项目和片区/合成/输出。恢复前需另选空目录，不直接覆盖当前工作。

主机必须是 `liekkas`。大文件只写 `/srv`。旧检查点只读，不要改。

## 目录

- 代码与报告：`/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1`
- 下载与解包：`/srv/slam-research/grf/map-denoise/datasets/multiscan-pilot-v1`
- 片区、合成、预实验输出：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1`
- 旧检查点（只读）：`/home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1`

## 环境

- CPU / Open3D：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`
- 小张量 CUDA 核验：`/srv/slam-research/grf/map-denoise/envs/pathnet-v5/bin/python`

不要升级旧环境，不要改显卡驱动，不要下 TUM 全量。

## 复现（数据已在时）

V2 协议为 `REPAIR_PROTOCOL_V2.md`。每次运行都会新建唯一目录，不覆盖旧数据。不需要重提片区或重下 ETH3D。

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
"$PY" -B -m unittest discover -s tests -v
"$PY" -B repair_pilot_v2.py
```

上一条打印新 `run_dir`。将该绝对路径传给 `verify_repair_v2.py` 与 `summarize_repair_v2.py`，分别生成独立核验/测试日志和报告/图片。这两个工具拒绝覆盖已存在的同名产物。

原 `run_pilot.py` 命令入口已经禁用。`extract_patches.py` 和 `generate_synthetics.py` 如确需执行，默认也创建新目录；不要继续使用旧的直接覆盖式复现顺序。

ETH3D 若需重下（官网直链，支持续传）：

```bash
"$PY" -B download_eth3d.py
```

UCL 只会再探一次官网，不会绕过 OneDrive 登录墙：

```bash
"$PY" -B inspect_ucl.py
```

TUM 只写元数据，不会下载：

```bash
"$PY" -B inspect_tum.py
```

可选 GPU 小张量核验（`nvidia-smi` 失败不等于 CUDA 不可用）：

```bash
/srv/slam-research/grf/map-denoise/envs/pathnet-v5/bin/python -B check_gpu.py
```

## 方法名

只读导入旧 `operators.estimate`。比较 `identity`、`xyz_mixture`、`fast`、`open3d_icp_then_xyz`。最后一项是官方点到点 ICP 加自研后处理，不要写成 JRMPC 或 BALM。

## 评价边界

- 真实独立几何：本轮尚未测得。
- clean / BIM / IFC 都不是独立毫米真值。
- 合成不可辨识案例不按唯一真值打分。
- 简单端 / 桥接端 / 复杂端三条算法线都保留。

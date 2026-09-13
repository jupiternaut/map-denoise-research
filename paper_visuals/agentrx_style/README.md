# map-denoise-research：四类学术图

基于用户指定 GitHub 仓库 `https://github.com/jupiternaut/map-denoise-research`，固定提交 `a7b362e8f0e9f77a7861d202145df9bd482e9856`。制作日期：2026-09-13。

主题为 V22 的逐点局部曲面重建、多尺度连续阻尼，以及 scan37 新确认场景上的误差与覆盖权衡。借鉴 AgentRx 的论文视觉语言，所有方法内容与结果来自本项目。没有把旧 Windows 同名目录当成本次来源。

新增：[科研作图方法与书单](科研作图方法与书单.md)，详细记录工具分工、四类图的做法、PDF嵌入、复现和推荐书籍。

## 直接查看

- [四页图集 PDF（中文标题与图注）](map-denoise-visual-atlas.pdf)
- [四图总览](overview.png)
- [图1：TikZ 流程图与配置示意](fig01_pipeline.pdf) / [PNG](fig01_pipeline.png) / [SVG](fig01_pipeline.svg)
- [图2：Matplotlib 多面板数据图](fig02_confirmation.pdf) / [PNG](fig02_confirmation.png) / [SVG](fig02_confirmation.svg)
- [图3：LaTeX 三线表](fig03_table.pdf) / [PNG](fig03_table.png) / [SVG](fig03_table.svg)
- [图4：真实源码与 JSON 代码块](fig04_code.pdf) / [PNG](fig04_code.png) / [SVG](fig04_code.svg)

每类保留独立矢量 PDF、SVG、PNG。PDF 是字体嵌入的正式交付；SVG 便于后续排版，保留文本的版本需要相应字体。图1的原生可编辑源是 TikZ，图2是 Python，表格与代码块是 LaTeX。此包没有声称提供作者未公开的 AgentRx 原始绘图程序。

## 四图讲什么

1. **局部模型替换整体平行面假设。** 排除查询点自身，在32/64/128邻域各自建立PCA坐标和二次曲面；修正投影至k64法向并融合，再施加连续阻尼。配置JSON标为示意；独立DTU参考只进入评分，没有回流算法。V22没有虚构的“安全拒绝→原点回退”门控。
2. **看每个片区，也看多个指标。** 24个片区按固定编号排列，灰线连接配对观测而非时间。主候选和同位移RMS常数阻尼共用蓝/橙编码；下方全景保留全部9方法，包括Frozen V18，右侧明确显示局部放大范围。
3. **完整公开九方法数值。** 216条原始记录重新汇总，表格显示MAE、1mm参考召回、F-score、位移RMS。主候选标蓝不代表每列最优。
4. **把机制和研究判断连起来。** 代码原样摘录 `reconstruction_v22/operator.py:119–138`，包括共享观测的方差说明和alpha更新；JSON直接摘自确认比较，`passed=false`指未通过预设综合升级标准，不是运行失败。

## 数据和结论边界

- 源：`evidence/progress_v22/runs/reconstruction-v22-qayc8gft/confirmation_RESULTS.json`、`SUMMARY.json`、`FINAL_AUDIT.json`，以及`reconstruction_v22`源码与协议。哈希见 `PROVENANCE.json`。
- scan37为**一个**新确认场景，24个不共享源顶点的片区，每片区输入1024点，共24576个源点（约原网格3.13%）。这不是24个独立场景。
- MAE的固定观测区域掩码合计评分17124点，各片区591–937点；不把输入1024点与实际评分点数混为一谈。原记录里的局部参考点计数可能跨区域重复。
- 每个方法的汇总为24个片区的**等权均值**，不是按点数加权均值，也不是官方全场景DTU总分。
- 主候选比identity的MAE低约0.598%，但召回低约0.333个百分点；未优于同位移RMS常数阻尼的平均MAE。因此不写成已经建立普遍真实几何净收益。
- 本次读取已发布指标制图，没有下载原始点云、重跑去噪算法或重新计算几何距离。原项目APSS历史逐点重放的验证差异仍保留，不能把本次制图核验通过当作所有历史验证通过。

## 复现

在完整仓库内运行，或先按 `PROVENANCE.json` 放回对应输入文件。Windows测试环境为Python3.14、NumPy、Matplotlib3.11.1、PyMuPDF、Pillow，以及Tectonic0.17.0。仅绘图不需要Open3D、SciPy、GPU或论文中的原始运行环境。

字体：Times New Roman（正文与图表）、Consolas（代码）、SimSun（图集中文说明）。首次Tectonic运行联网获取所需TeX资源。字体安装位置不同可修改两个`.tex`和`build_artifacts.py`里的字体配置。

```powershell
.\build_all.ps1 -Tectonic 'C:\Users\gengr\Documents\Paper-Figure-Toolkit\runtime\tectonic-0.17.0\tectonic.exe'
```

Tectonic也可放入PATH。脚本依次：重算数据与绘图→编译TikZ→编译booktabs与代码块→生成矢量图集→验证输入哈希、数值与PDF结构。

- `plot_results.py`：数据图和CSV。
- `fig01_pipeline.tex`：可编辑TikZ流程图。
- `build_artifacts.py`：从证据自动生成三线表与原文代码面板；编辑源生成器后重建，避免只修改会被覆盖的`.tex`。
- `verify_artifacts.py`：原始输入哈希、独立数值交叉核对、原样摘录、PDF位图/越界/缺字检查。
- `data/`：配对值、逐项记录、均值、实际JSON和源码片段。
- `VALIDATION.json`、`MANIFEST.json`：验证结果与文件清单；`build-logs/`和`.log`保留编译诊断。

便携Tectonic来自官方发布 `https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0`，下载的Windows MSVC压缩包SHA256为`f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`。运行时放在图包之外，不修改系统PATH。

## 视觉规范与验证

图内英文便于用于论文；图集使用中文标题与图注。170mm左右双栏宽、白底、深灰文字、蓝`#3778A8`和橙`#CB7736`。颜色以外还使用实心圆、空心方块、虚线和直接标签。无生成式实验图片、虚构曲线或补造结果。

已按输出尺寸检查PNG，修复数据图轴边截断、流程图说明压线和表格底部间距。各独立PDF和图集均保留矢量内容；PNG仅为预览。最终可机器复核的结果见 `VALIDATION.json`，但机器检查不能替代读图判断。

使用本机 `data-analytics:visualize-data` 技能进行图型、数据粒度、配色、图例和导出检查。图件和文档作为新增目录交付，原有科研源码、实验结果及旧工作目录保持原样。

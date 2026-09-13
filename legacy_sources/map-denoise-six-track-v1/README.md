# 六线首轮实验交付

主机：liekkas；2026-09-11。源码与报告全部在本目录。

- [导师汇报：可直接阅读／转述](/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/导师汇报.md)
- [完整报告：含方法、对照、失败与图](/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/REPORT.md)
- [运行前计划](/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/PLAN.md)
- [检查点与交付包](/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/CHECKPOINT.md)

原始输入、输出、逐例指标、日志在：
`/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600`。

## 六线与入口

| 内容 | 源文件／报告目录 | 对应原始运行 |
|---|---|---|
| T1/T2 理论与联合去偏 | `tracks/t1_t2/` | `t1_t2/run-initial-01`、`run-resources-02`、`root-newseed-check-01`、`real-smoke-01` |
| T3/T6 分层与 CAD-like | `tracks/t3_t6/` | `t3_t6/development-0bx4br9p`、`external-extension-tei9ntbh`、`matched-oracles-pmqknc_0`、`scale-ablation-0fs7n0gm`、`root-postfreeze-01` |
| T4 评价与 Oxford 接入 | `tracks/t4/` | `t4/run-selwrbu7` |
| T5 等输出优化 | `tracks/t5/` | `t5/` |

各轨报告写明命令和输入协议。复跑应使用明确的新结果目录，原运行拒绝覆盖。
T5 当前脚本包含固定 RUN 路径，不应直接在原位置重复生成；`verify_saved.py` 可只读复核。

既有 Python：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`。
设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`。
PathNet 使用既有专用环境和冻结权重；完整加载信息见其原始 diagnostics。
本轮未安装新环境或下载数据。旧代码和外部基线是只读依赖，归档不是可脱离本机运行的容器镜像。

## 保留与复算

- `BASELINE_SOURCES.json`：201 个旧项目/论文依赖的原始哈希。
- `audit_sources.py check BASELINE_SOURCES.json`：只读校验旧依赖。
- `INDEPENDENT_AUDIT.json`：主线程复算 T2 误差、T5 输出与分支。
- `FINAL_VALIDATION.json`：最后一次测试与源文件核查记录。
- `root_geometry_check.py`、`root_scale_check.py`：冻结候选的新参数／种子重复和后续尺度控制，所有结果仍属探索证据。
- `make_figures.py`：从已保存输出生成报告图，按固定首例作图，不搜索最好结果。

本轮只产生局部点几何。没有生成三角网格拓扑证明，没有运行真实 3DGS 去噪，没有宣布独立确认集胜利。

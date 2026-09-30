# 点云误差关联：分治探索检查点

主机 liekkas，2026-09-11。先读 [结果报告](/home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1/REPORT.md)。

本轮冻结已有方法，以相同误差多重集合的配对数据检查帧关联作用。正式完成 162 份合成输入、1,944 个合成输出，以及 Oxford 六片区的 78 份干预输入、234 个输出。不是新算法或真实精度领先报告。

## 文件索引

| 路径 | 内容 |
|---|---|
| `PROTOCOL.md` | 运行前冻结协议 |
| `generate_cases.py`, `data/` | 合成生成器、合法输入及分离评价数据 |
| `operators.py`, `METHODS.md` | 六个冻结方法的适配与来源 |
| `run_experiment.py`, `synthetic_results/` | 正式执行及逐输出几何、信息和 hash |
| `analyze_results.py`, `analysis/` | 配对结果和独立复算 |
| `figures/` | 已检查的两张结果图及 PDF |
| `real_probe.py`, `real_results/` | 真实输入的配对干预与逐点输出 |
| `REAL_REPORT.md` | Oxford 响应结果及纯帧并列解反例 |
| `INDEPENDENT_DESIGN_REVIEW.md`, `METHODS_AUDIT.md` | 并行独立审查 |

所有算法输入仅含坐标、帧和提供的 σ。GT 不传入估计器；合成 GT 只用于构造控制和输出保存后的评价。旧算子只读导入；生成器中的射线求交逻辑按来源注释复用到新文件，旧检查点未改动。

## 当前机器复现

依赖现有两个旧源码目录和 Oxford 原文件，详见 `METHODS.md`、`REAL_REPORT.md`；本目录不是脱离这些依赖的独立安装包。环境为 Python 3.12.13、NumPy 2.2.6、SciPy 1.15.3、Open3D 0.19.0；绘图使用 Matplotlib 3.11.1。

执行以下命令会在本目录创建唯一的新结果子目录，**不会覆盖本轮正式结果或旧检查点**：

```bash
cd /home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
CORR_PY=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
CORR_REPRO=$(mktemp -d /home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1/repro-XXXXXX)
"$CORR_PY" -B -m unittest test_generation test_operators test_analysis -v
"$CORR_PY" -B run_experiment.py --data /home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1/data --out "$CORR_REPRO/synthetic_results"
"$CORR_PY" -B analyze_results.py --data /home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1/data --run "$CORR_REPRO/synthetic_results" --out "$CORR_REPRO/analysis"
"$CORR_PY" -B plot_results.py --analysis "$CORR_REPRO/analysis" --out "$CORR_REPRO/figures"
"$CORR_PY" -B real_probe.py
```

`real_probe.py` 自行在 `real_results/` 创建新的唯一执行目录。正式真实执行目录为 `real_results/execution-o60mtr0x/`，追加只读反例诊断代码与结果也保存在其中。

上面的合成复跑使用本轮冻结数据。`generate_cases.py` 在当前 `data/` 已存在时拒绝覆写；11 项生成器测试读取现有冻结数据并核验关键生成性质。`audit_geometry.py` 是正式结果的一次追加审计脚本，其输出以排他模式创建；结果已经存在，不需要重复执行覆盖。

## 当前判断与下一步

已确认的机制信号：对能消去帧截距的方法，共同误差可能比同值打散误差更容易；把总误差当作残差 σ 会抹掉真实双层。另发现冻结 fast 对数学上并列的纯帧层归属存在数值不稳定。

下一轮集中研究“从观测中分开估计帧内残差与共同偏差，再进行结构保持滤波”。平局修复单独记录，不用修复后的稳定性冒充真实几何正确性。新的同行基准与独立真实几何评价尚未完成。

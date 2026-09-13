# V16 复现

在主机liekkas、本项目目录执行。无需安装环境、下载数据、修改旧检查点或使用GPU。

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
V16_PY=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
"$V16_PY" -m unittest discover -s exploration_v16 -p 'test_*.py' -v
"$V16_PY" exploration_v16/run.py --workers 4
```

run.py打印新建的 `mechanism-v16-*` 绝对路径。将其填入下面V16_RUN，后处理必须对应那一次运行。

```bash
V16_RUN=/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/mechanism-v16-hky6n1fv
"$V16_PY" exploration_v16/audit.py "$V16_RUN" audit_new
"$V16_PY" exploration_v16/readout.py "$V16_RUN"
"$V16_PY" exploration_v16/supplement.py "$V16_RUN"
```

上面的路径是本次已完成运行；不要对其重复readout/supplement，因为它们采用独占创建，保护已有结果。
若只查本次结果，直接打开REPORT.md、运行目录的CSV与readout/SUMMARY.json。
审计可用新的audit目录名重复执行，不覆盖旧核查记录。

首轮560个模型生成后，评分dict.update误用被修复，使用resume_scoring.py完成剩余流程。
当前run.py已修复，新复现不需要resume_scoring.py。首轮原源码及差异完整保存在source/和SCORING_REPAIR.json。

文件职责：

- models.py：原自由斜率函数的适配，以及新固定斜率oracle求解；不导入生成器/评价器。
- generator.py：固定有限支撑与边缘分布的标签置换实验。
- metrics.py：模型面、归属、投影误差的分解；只在评分阶段读取GT。
- run.py：生成、封存、四进程拟合、评分与V15重放。
- audit.py：独立重算新评分与重放诊断、核对哈希与点级一致性。
- readout.py：配对区间、切片、模型选择变化和预先固定示例。
- supplement.py：不处理对照、统一纵轴与事后明确标注的机制示例，不增加拟合。

数据约束见PROTOCOL.md；结果解释见REPORT.md；数学恒等式见DERIVATION.md。

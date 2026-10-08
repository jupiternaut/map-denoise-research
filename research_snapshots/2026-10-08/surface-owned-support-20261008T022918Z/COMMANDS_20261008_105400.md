# 执行与核验命令

目标主机：liekkas。工作目录：`/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z`。
使用已有 Python：`/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python`；无需安装、GPU 或 API key。

## 可直接重跑的检查

```bash
cd /srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B -m unittest -v test_support
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B audit/independent_replay.py
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B audit/check_boundaries.py
```

独立回放会读取照片与两份完整激光 PLY，需要相应原始数据仍位于封存路径；核验程序只向终端输出。其空间、单位和评价范围见 AUDIT.md。哈希不匹配时应停止，不修改锁文件来使检查通过。

## 本轮实际执行过的生成顺序（历史记录）

以下步骤产生已封存文件，**不要在原目录中重复覆盖**。若新实验改变实现或协议，应另开运行目录、记录来源、重新锁定。

```bash
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_experiment.py lock
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_experiment.py extract
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_experiment.py infer
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_experiment.py evaluate
```

机制装置顺序为同一 Python 执行 `mechanism/run_mechanism.py render`、`score`、`evaluate`。理论测试为 `theory/test_theory.py`。后验诊断为 `diagnose_evidence.py`、`mechanism/footprint_diagnostics.py`；它们写独立诊断工件，不改变主方法预测。

本轮没有操作 GitHub、GitBook、部署或历史数据。执行状态和结果以 REPORT.md、CHECKPOINT.md 及各封存文件为准。

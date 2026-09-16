# V25 执行检查点

- 状态：WINDOW_COMPLETE_DEV；算法已实现并在开发集上跑完。假说未成立。
- Host：liekkas
- Workdir：`/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration`
- 开始：2026-09-14T16:53:14Z；本检查点：2026-09-14T17:19:29Z
- 运行进程：无本任务后台进程。
- GPU：全程忙（LightRAG PID 601178），未启动 GPU 作业。
- Python：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`（只读执行）
- 科学结论：V25 替换式再生未胜过 identity；默认锁定 identity。官方 COLMAP 未跑。无新确认场景。
- 封存：`runs/2026-09-14T165314Z/`（128 个已评分输出，VERIFY 最大差 0）

## 已完成

Q00–Q03、Q05–Q09、Q11、Q13、Q14。Q04/Q10/Q12 BLOCKED。

## 下一条已存在命令（复现，不是未跑实验）

```bash
cd /home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
V25_PYTHON=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
$V25_PYTHON evaluation/run_verify.py
```

若要重跑单个已有 ROI（会覆盖该 ROI 新写文件，不删旧 v1 封存副本）：

```bash
$V25_PYTHON -m src.v25.cli run-roi --roi scan24_window_left
$V25_PYTHON -m src.v25.cli evaluate --roi scan24_window_left
```

官方 COLMAP 仍未安装。GPU 空闲后也**不要**把未跑的 CUDA COLMAP 写成已经测过。

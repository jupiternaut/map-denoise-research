# V25 实际命令

解释器（只读已装库，不向该环境 pip install）：

```bash
export V25_PYTHON=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration
```

以下只记录已经执行过的命令；未跑的不写造成功。

## 2026-09-14T16:52Z 预检

```bash
python3 preflight.py
```

exit 0，`INPUT_PATHS_READY`。colmap 不在 PATH。GPU 被 PID 601178 (LightRAG) 占用。

## 2026-09-14T17:00Z 单测与准备

```bash
export V25_PYTHON=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$V25_PYTHON -m unittest tests.test_eval tests.test_isolation tests.test_synthetic tests.test_cameras
$V25_PYTHON -m src.v25.cli prepare
$V25_PYTHON baselines/colmap_probe.py
```

- unittest：eval/isolation/cameras 通过；synthetic 在改世界纹理后通过。
- prepare exit 0：scan24 重投影 0.356 / 1.794；scan37 0.441 / 1.696。
- colmap_probe：`NOT_FOUND`，官方 PatchMatch 基线 BLOCKED。

## 2026-09-14T17:05Z 父 ROI 矩阵（CPU）

```bash
$V25_PYTHON -m src.v25.cli run-roi --roi scan24_window_left   # exit 0, 11.01s, rss ~524 MiB
$V25_PYTHON -m src.v25.cli evaluate --roi scan24_window_left
for roi in scan24_gable_center scan24_turret_join scan24_wall_control \
           scan37_scissor_cross scan37_clamp_jaw scan37_driver_handle scan37_stone_control; do
  $V25_PYTHON -m src.v25.cli run-roi --roi "$roi"
  $V25_PYTHON -m src.v25.cli evaluate --roi "$roi"
done
$V25_PYTHON -m src.v25.cli seal
$V25_PYTHON -m src.v25.cli report
```

全部 exit 0。产物在 `runs/2026-09-14T165314Z/<roi>/*.ply`。

## 2026-09-14T17:10Z 同 bundle 新算子

```bash
cp runs/2026-09-14T165314Z/SEALED.json runs/2026-09-14T165314Z/SEALED_v1.json
for roi in scan24_window_left scan24_gable_center scan24_turret_join scan24_wall_control \
           scan37_scissor_cross scan37_clamp_jaw scan37_driver_handle scan37_stone_control; do
  $V25_PYTHON -m src.v25.cli regenerate --roi "$roi" --extra
  $V25_PYTHON -m src.v25.cli evaluate --roi "$roi"
done
```

exit 0。新增 `v25_wta_atlas`、`v25_depth_cc`、`v25_gated_move`。

## 2026-09-14T17:12Z 全体 core 收缩（重建 evidence）

```bash
for roi in scan24_window_left scan24_gable_center scan24_turret_join scan24_wall_control \
           scan37_scissor_cross scan37_clamp_jaw scan37_driver_handle scan37_stone_control; do
  $V25_PYTHON -m src.v25.cli core --roi "$roi"
done
$V25_PYTHON -m src.v25.cli seal
$V25_PYTHON -m src.v25.cli report
```

exit 0。

## 2026-09-14T17:16Z 独立核验与图

```bash
$V25_PYTHON evaluation/run_verify.py   # exit 0, 128 行, max_abs_delta 0.0
$V25_PYTHON -m src.v25.figures         # exit 0
$V25_PYTHON -m src.v25.cli report
$V25_PYTHON -m unittest tests.test_eval tests.test_isolation tests.test_synthetic tests.test_cameras
```

9/9 测试通过。旧 `calibration.py` 哈希仍为 `81834b4e...`。


# V13 多尺度表面图册原型

[REPORT.md](REPORT.md)含完整结果。改变归属、方向、位置的共享尺度；不是已验证的通用地图滤波器。
旧源码和输出不覆盖。主机liekkas，项目根目录 `/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1`。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v13/run_v13.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v13/run_v13.py --fresh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v13/run_v13.py --curved
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v13/audit_v13.py
```

各命令创建新运行目录。`--fresh`复用本轮固定平板确认种子，重跑不是新增未知测试。
`--curved`是参数化弯曲机制测试，不是真实扫描模拟；不要与 `--fresh` 同时使用。
API：`api.estimate(xyz_world_m, scan_id, sigma_mm, mode, neighbors)`，返回新XYZ、诊断、支持与归属。
mode可选global_hard/atlas_hard/atlas_tied/atlas_relax；完整调用旧合法输入上游，不需提供GT或缓存答案。

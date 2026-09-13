# V12 局部窗口与官方外部基线

结果见 [REPORT.md](REPORT.md)，外部库适配与参数见 [EXTERNAL_IMPLEMENTATION.md](EXTERNAL_IMPLEMENTATION.md)。
运行主机liekkas，项目根目录 `/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1`。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v12/run_v12.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v12/run_v12.py --fresh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v12/audit_v12.py /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/local-v12-confirmation-1ehbo99b
```

命令创建新运行目录，不覆盖历史结果。`--fresh`使用本轮固定确认种子；重跑这些种子不构成新的盲测。
`local_filter.estimate_frozen(state, sigma_mm, k, backend)`为窗口原型；
`external.estimate_frozen(state, kind, scale, raw)`为外部后端适配器。
两者使用现有合法输入上游状态，返回新XYZ、诊断和支持信息。失败窗口版本不推荐自动替换现有地图。
PyMeshLab已安装在独立目录，通过external.py中的明确路径导入，不需修改旧环境。

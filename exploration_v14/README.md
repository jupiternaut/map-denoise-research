# V14 效果实验

主机liekkas；项目`/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1`。
这是扩展实验，不是论文。先看[REPORT.md](REPORT.md)与事前[PROTOCOL.md](PROTOCOL.md)。

在项目根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -m unittest discover -s exploration_v14 -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -u exploration_v14/experiment.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -u exploration_v14/audit.py /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/effect-v14-taip2cld
```

每次创建新运行目录。审计别次运行时传入相应绝对路径。
重放使用相同种子不是新的未见测试。实验使用给定sigma，不包含独立真实几何验证。
旧代码与环境不改动；运行快照可用于追溯。初次审计的外部重放数值容差修正见报告。

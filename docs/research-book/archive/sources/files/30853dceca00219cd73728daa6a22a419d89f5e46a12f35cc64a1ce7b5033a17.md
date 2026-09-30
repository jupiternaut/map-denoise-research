# V10 频谱滤波实验

结论见 [REPORT.md](REPORT.md)。实际主机 `liekkas`，未修改旧检查点或 Skill。

在项目根目录运行，均创建新结果目录，不覆盖旧运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v10/run_v10.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v10/run_joint.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v10/audit_v10.py
```

`spectral_filter.estimate(xyz_world_m, scan_id, sigma_mm, method, iterations)` 是首轮完整输入API；
批量实验复用旧上游状态，算法看不到评价GT。`joint_decoder.filter_frozen` 是第二轮联合偏差后端。
两次历史运行各自冻结当时源码；后续报告、测试和第二轮源码的新增不改写首轮源码快照。

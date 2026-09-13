# 独立参考真实迁移实验 V21

先读 [REPORT.md](REPORT.md) 和冻结的 [PROTOCOL.md](PROTOCOL.md)。结论：固定 V18 适配器没有通过本次真实迁移验收；不是正收益发布。

在现有 liekkas 项目根目录运行：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s real_closure_v21 -p 'test_*.py' -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B real_closure_v21/run.py
```

用程序打印的新唯一目录核验：

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B real_closure_v21/verify.py /绝对路径/新运行目录
```

输出 `.npy` 是毫米世界坐标的 N×3 点云，原始行号保存在 `inputs/*_ids.npy`；没有替换原网格。

`camera_probe.py` 已完成从原作者包提取相机。不要重复执行：它为排他写入，仅适用于尚无本轮相机文件的环境；已保存文件与旧公开文件逐字节一致，无需再次传输约 1.5GB。

依赖沿用原项目 Open3D 环境和已有隔离 PyMeshLab 接口；没有新建环境或修改驱动。

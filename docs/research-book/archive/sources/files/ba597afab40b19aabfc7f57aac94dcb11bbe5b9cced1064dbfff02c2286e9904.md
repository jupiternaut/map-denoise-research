# V4 并行构造检查点

主机 liekkas；代码、数据及原 V3 检查点不迁移。先读 [完整报告](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v4/REPORT.md)。

已完成：同面聚合三臂、同目标运输热启动、真实扰动分解；507 个开发输出、312 个冻结确认输出、144 个真实干预输出。61 项测试通过。独立真实几何效果、GPU 加速与外部联合配准基线未完成。

## 调用算子

在本目录，使用当前固定环境：

```python
from surface_pooling import estimate
# xyz_world_m: N×3，世界米制坐标；scan_id: N个整数扫描编号。
# sigma_mm 是点级测量噪声尺度，不应混入扫描整体偏差。
xyz_filtered_m, diagnostics = estimate(
    xyz_world_m, scan_id, sigma_mm=1.0, variant="compatible")
```

输出保持原点数、顺序和米制坐标；返回新数组，不修改输入。sigma 尚非自动盲估。模块依赖同一项目内的冻结 V3 源码，不能单独拷走一个文件就声称可复现。

## 复现测试与读取结果

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v4
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -m unittest discover -s . -p 'test_*.py' -v
env PYTHONDONTWRITEBYTECODE=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python summarize_v4.py /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v4-confirm-1_0s40f5
```

`summarize_v4.py` 默认只读；不要对已有 AGGREGATES.json 重复加 `--save`。

## 重跑实验

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python run_v4.py
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python run_v4.py --confirm-from /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v4-dev-e0c_n4hk
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python real_components.py
```

每次新建唯一结果目录，不覆盖历史结果。第二条只允许源码与协议哈希一致；现在这些确认种子已经暴露，重跑只能称复现，不能称新确认。`verify_v4.py <新结果目录> --tests` 可以重新复算并测试；它也拒绝覆盖已有核验文件。

## 不可误读的主要结果

- 保留种子确认：相容聚合对独立拟合，平均表面 MAE 0.237390 → 0.175109 mm，22/24 改善；层距诊断存在小幅退化。
- 热启动 balanced 两轮与零初值六轮的确认 MAE 0.273058 / 0.273300 mm；记录均时 401 / 722 ms，尚非受控性能基准。
- 真实分解是已知扰动恢复，不是真值精度；本轮真实分解使用旧图/运输算法定位机制，不是新相容聚合的真实效果评测。

本目录 REPORT/README/汇总脚本是确认结束后的报告工具，不纳入事先冻结的算法集合；算法及协议本身保持冻结。

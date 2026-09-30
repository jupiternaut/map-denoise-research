# V5：参数共享消融与真实迁移（不替换 V4）

主机 liekkas。用户仍审阅的精确分区搜索和多机GPU方案未实施。

**状态：新全组共享斜率构造未通过实验。继续使用既有 V4 compatible 作为候选，不能把本目录默认实验变体当作已验证的新滤波器。**

[完整报告](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v5/REPORT.md)

## 已完成

- slope_pooling：V4精确复制、组间共享斜率、节点独立截距三个同输入/同上游状态的变体。
- registration_baselines：官方Open3D GICP + 明示的输入锚站/质心平移规范，无自研点级后处理。
- real_transfer：与V4逐文件相同的24真实输入，八种方法重新运行。
- 351开发输出、216新种子确认输出、192真实输出；79项新旧测试通过。
- 旧V4及更早文件保持只读，源码/结果哈希检查通过。
- 两个公开双层样例的[机制诊断](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/diagnostic-v5-91id9auq/DIAGNOSTICS.json)：混层节点的斜坡解释被共享参数传播，独立消元与SVD结果一致。可运行 `diagnose_shared_slope.py` 在唯一新目录复现，不修改算法。

## 读取已存结果

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v5
env PYTHONDONTWRITEBYTECODE=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python summarize_v5.py /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v5-confirmation-p0oymcit
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python real_transfer.py --verify-only /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-transfer-v5-zrtui_f5
```

已有AGGREGATES.json不要重复加`--save`；已有VERIFICATION.json不覆盖。

## 重跑复现

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -m unittest discover -s . -p 'test_*.py' -v
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python run_v5.py
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python run_v5.py --confirm-from /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v5-development-vdbkdouk
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python real_transfer.py
```

实验创建唯一新目录；确认入口要求算法与开发源哈希一致。本轮912601/912613/912627已暴露，之后重跑只能称复现。

## 算子实验接口

```python
from slope_pooling import estimate
# measured_xyz_m: N×3当前测量米制世界坐标，scan_id: N个整数，sigma_mm:点级噪声毫米。
out, diagnostics = estimate(measured_xyz_m, scan_id, sigma_mm,
                            variant="v4_compatible")
# 另两种实验变体 shared_group_slope / node_intercepts 不推荐自动替代V4。
```

返回同点数、同顺序的新数组。冻结状态fingerprint、原序unsupported_point_indices、秩和条件数在diagnostics中。依赖同项目V4/V3，不能只复制一个文件就声称环境完整。

## 范围

合成真值只在评价侧，真实参考仍是未扰动测量。没有证明独立真实精度，没有通用去噪或新最优性结论。报告及事后诊断不改变冻结算法；负结果和旧输出均保留。

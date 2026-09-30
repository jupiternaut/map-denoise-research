# 局部多帧分层点云滤波

本轮按专用滤波器推进，形式验证和穷尽边界不是交付前提。任务是：在局部近似平行
表面中，去掉帧间共同偏差及层内噪声，同时保留两层结构。不是全地图重建系统。

## 交付

- [结果报告](REPORT.md)：算法、共同比较、复测与下一步选择。
- [导师简报](导师简报.md)：可直接用于汇报的项目摘要。
- [机制推导](MECHANISM.md)：截距消元与有序分层。
- `filter_patch.py`：可运行入口；默认 `fast`，不自动按真值挑方法。
- `development-summary.json` / `replication-summary.json`：由保存输出汇总。
- `real_smoke_report.md`：六个 Oxford 片区的接入结果，不宣称已证明真实精度收益。

## 输入与运行

NPZ 必需字段：`xyz_mm` (N,3)、`frame` (N)、正数 `sigma_mm`。
可选 `roi` 默认全0（目标），`bias_bound_mm` 默认8σ。Z是输入提供的局部近似法向；
所有坐标单位毫米。`fast` 和 `fixed-map` 固定此法向，`tilted-map` 会联合估计法向。
sigma是输入假设，不是算法自动推断的真实传感器精度。不得把任意PLY点云当作已具备帧信息。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B \
/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/filter_patch.py \
/absolute/input.npz /absolute/new-output.npz --mode fast
```

保留点数；不覆盖输入或已有输出。输出几何及相邻 JSON 记录模式、估计结果、位移与来源。
`fixed-map` 是更慢的逐帧比例联合模型；`tilted-map` 是实验性法向扩展，不是默认升级。

## 本轮证据

开发集171个输入×19方法=3249个输出；33个新种子/非网格参数复测输入×12方法=396个输出；
真实样片6×3=18个输出。辅助开发试跑另存，不混入正式汇总。12项单元检查通过。

结果根目录：`/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704`。
旧 `map-denoise-six-track-v1/tracks/t1_t2/experiment.py` 作为只读依赖。
旧两份检查点校验通过，本轮未回写旧算法。

`proposal-methods` 保留一次导入路径错误的启动目录；成功补跑为 `proposal-methods-02`。
runner分模块加载时，proposal需要把本目录的 `candidate` 加入 `PYTHONPATH`；CLI已处理。

所有“改善”均注明输入族、指标和对照。专用方法可以有适用范围，但不能把变平滑等同于变正确。

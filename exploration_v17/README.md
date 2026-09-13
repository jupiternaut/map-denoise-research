# V17 自适应空间关联实验

主机：liekkas。范围：条件局部点地图表面估计，不含真实扫描真值、3DGS、GPU 或完整 SLAM。

- `v17_operator.py`：GT-free 估计器及常数/线性/RBF 共享候选池。
- `v17_apply.py`：可复用的 `estimate` / `filter_local` 入口，不运行旧对照；BIC 不运行三折 CV。
- `v17_run.py`：42 开发、140 新统计输入、144 新首回波输入，七种输出；预测封存后评分。
- `test_v17.py`：梯度、嵌套、留出折隔离、非零斜率及候选保留测试。
- `v17_audit.py`：独立公式重算几何误差、候选评分、CV 预测和选择，复核旧哈希。
- `v17_summary.py`：全表与种子配对差异。
- `PROTOCOL.md`：预先写下的开发/确认约定。
- `DERIVATION.md`：数学表达、选择器意义及其限制。

本轮运行目录：
`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/adaptive-v17-mklh_h2j`

解释器：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`。
运行前设置 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。
从项目根目录执行：

```bash
python exploration_v17/test_v17.py
python exploration_v17/v17_run.py development --workers 4
# 使用上一步打印的新目录，不能重跑写入已有封存结果：
python exploration_v17/v17_run.py confirmation --dest NEW_RUN --workers 4
python exploration_v17/v17_run.py bridge --dest NEW_RUN --workers 4
python exploration_v17/v17_audit.py NEW_RUN
python exploration_v17/v17_summary.py NEW_RUN
```

`python` 应替换为上述解释器绝对路径。开发使用已暴露的 V16 输入，依赖其旧运行目录。
保存的模型保留原点顺序；首回波模型数组位于冻结排序顺序，须用 `order` 映射回原始点。
选择器只看观测负对数似然；给定 sigma 和固定上游是明确条件，不是假装盲估计。
`shared_case_seconds` 包含共享候选池、三折重训、旧对照和保存，不能相加成各方法独立耗时。

实际调用：`filter_local(local_xyz_mm, sigma_mm)` 返回局部点、模型和选择分数。
要求调用者提供局部坐标系；只改变局部 Z，不负责世界坐标点云分块、法向估计或跨站配准。
命令行输入 NPZ 的键为 `local_xyz_mm`，输出以独占创建避免覆盖：

```bash
python exploration_v17/v17_apply.py INPUT.npz OUTPUT.npz --sigma-mm 1
```

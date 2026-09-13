# V7 地图滤波构造

主机 liekkas。沿用原项目环境、输入、验证器；旧V2–V6与原计划只读。

- [报告](REPORT.md)：首版重关联的负结果、真实局部表示诊断和数学解释。
- [协议](PROTOCOL.md)：公开开发条件与输出/评价隔离。
- [算法](algorithm/reassociation.py)：`freeze`、`fit_frozen`、`estimate`。
- [数学说明](math/MATH_NOTE.md)：测度、条件单调、表示细化、输出损失。
- [真实诊断](real_geometry/REPORT.md)：同一面距离换算与训练/留出拟合。
- [输出决策算子](decision/soft_map.py)：相同MAP关联直接投影soft-M面，去掉硬重拟合。
- [噪声选择偏差诊断](math/SEARCH_DIAGNOSIS.md)：保存阶段的局部证据与反例。

运行（每次新建唯一目录，不覆盖结果）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/run_v7.py
```

`--primary-only`只跑3公开主例；不传则24例。`--budgets 1 3 6`为本次实际预算档。
新运行目录从标准输出取得后，传给同目录下 `verify_v7.py` 和 `summarize_v7.py`。
两者新建审计/汇总记录，已存在时拒绝覆盖。重放历史结果请使用保存的源码版本，
不能将更改后的代码运行伪装成原版本复现。

主比较是自有方法的机制与成本控制，不是外部方法强基准；未运行新种子确认，
没有建立真实去噪精度提升或论文新颖性。数学性质通过与几何效果更好是两件事。

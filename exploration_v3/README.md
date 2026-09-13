# 关系图与表面分布：可回退探索检查点

主机：liekkas。所有代码新增在本目录，旧 V2、数据、旧算子未修改。地图去噪研究技能用于同输入消融、几何评分与历史留档；形式验证不是算法的前置门槛。

## 看成果

- [算法构造与两个有限数学命题](CONSTRUCTION.md)
- [主实验完整报告](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v3-lnx049yd/REPORT.md)
- [同坐标输出图](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v3-lnx049yd/figures/world_step_4mm.png)
- [570 行完整结果](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v3-lnx049yd/RESULTS.csv)
- [核验记录](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v3-lnx049yd/VERIFICATION.json)：570 输出、360 个表面评分重算；13 项新测试和 30 项原测试通过。
- 小样本开发运行仍保留：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v3-lfm1nwgv`。

## API

```python
from graph_surface import estimate
output_world_m, info = estimate(xyz_world_m, scan_id, sigma_mm=1.0, variant="graph")

from measure_surface import estimate
corrected_world_m, info = estimate(xyz_world_m, scan_id, sigma_mm=1.0, variant="unbalanced")
```

两者都保留点数、顺序及世界坐标系，只接 XYZ、扫描 ID、提供的 sigma。第一端是局部表面滤波；第二端单独是整扫描共同法向纠偏，运行器另提供其接相同 local_only 后置滤波的管线。不是可直接读取 3DGS 参数并完成去噪的工具。

## 重跑

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python \
  /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v3/run_exploration.py
```

每次创建 `/srv` 下唯一运行目录；不会覆盖之前结果。评估协议见 PROTOCOL.md。当前仍是开发数据上的构造；没有预留种子确认、独立真实几何验证、对口 JRMPC/BALM 比较或 CUDA 性能结果。

## 独立预算诊断

见 TRANSPORT_BUDGET_PROTOCOL.md 和 transport_budget_probe.py。只改变运输外层迭代 3/6/12、固定其他配置，区分求解预算和关联机制。
独立运行目录：`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/transport-budget-v3-mxi1kprj`。它不替换主实验表；更长迭代要计入额外成本。

补测已完成：[预算结果说明](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/transport-budget-v3-mxi1kprj/READOUT.md)。96输出全成功；四个完整输入上UOT+local从3轮0.6828 mm降到6轮0.3006 mm，同预算平衡为0.2981 mm（四例平均数）。六轮完整管线约0.71 s。部分重叠和结构反例保留。

## 当前判断

图端值得继续：24 例表面误差中位数 0.6561 -> 0.2823 mm，但单线程约 10 -> 150 ms，且无偏差单面有退化。
运输端不按三轮预算判死；追加实验已确认求解预算有作用，但 UOT 仍未胜过同预算平衡版本。
真实片区少改写不等于更准确，其恢复误差仍接近 identity。下一步分别处理真实局部适用性和原型计算预算，而不是宣称通用最优。

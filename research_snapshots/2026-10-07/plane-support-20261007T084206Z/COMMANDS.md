# 复算入口

运行主机：liekkas；目录 `/srv/slam-research/grf/map-denoise/runs/plane-support-20261007T084206Z`。

```bash
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B test_selector.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B theory/test_kernel.py
```

已执行且封存的主流程：

```bash
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B experiment.py freeze
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B experiment.py infer
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B experiment.py evaluate
```

输出采用独占创建，直接再次执行写入阶段会拒绝覆盖。应在明确的新运行版本中改变 ROOT 并重新锁定输入，不能删除本次封存文件来重跑。检查现有封存、重现决策和重新计算全部聚合可只读执行：

```bash
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B verify_run.py
```

此脚本由执行者提供，不是独立审查的替代品。

真实数据先验范围：2个旧场景，不是新场景确认。没有重新跑COLMAP、GPU或采集；官方对照是同一旧请求上的冻结输出。合成子实验有单独协议，不能合并其成功计数为真实实验数量。

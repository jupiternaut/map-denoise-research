# V6：两个受控实验，不替换 V4

[完整结果报告](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v6/REPORT.md)

状态：关联拆分反事实支持进一步研究合法关联；PCA方向替换没有建立真实恢复收益。
oracle 只能诊断，不能作为生产接口给未知数据提供答案。此前检查点继续只读。

## 复现两线

在目标主机liekkas执行，下列入口都创建唯一新运行目录，不覆盖已有结果。

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v6/association
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python run_association.py

cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v6/direction
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python run_direction.py
```

这些是公开数据复现，不会成为新确认。两线各自PROTOCOL和source快照记录算法与输入。
direction依赖原V3/V4/V5，只改变每次新载入私有实例的方向估计；不能单独复制一个文件
后声称完整独立环境。association原始API无GT参数，oracle API显式接受层标签且仅拆组。

## 已存输出

- [关联运行](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/association-v6-bgcq8zff/RESULTS.csv)
- [方向运行](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/direction-v6-edpxxdsb/RESULTS.csv)
- [独立关联审计](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/session-v6-pt14hi1q/ASSOCIATION_AUDIT.json)
- [独立方向审计](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/session-v6-pt14hi1q/DIRECTION_AUDIT.json)
- [DA局部支持诊断](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/direction-support-diag-v6-smuzp3nl/READOUT.md)

输出NPZ保存同序XYZ和来源ID、实际支持mask；关联线另外保存冻结状态与各臂分组。
mask不同于“点有没有移动”，支持率与几何分数必须同时读。

所有脚本均未连接Windows/Mac、未开发GPU核、未下载数据、未推送外部仓库。

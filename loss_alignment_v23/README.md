# V23：由损失选择修正动作的几何算子

入口：[完整报告](REPORT.md)、[事前协议及评分前输入修正](PROTOCOL.md)、[算子](loss_operator.py)。

`actions(points_mm)` 产生12个同点数候选与冻结局部曲面目标：不处理、五种V22输出、六种正反法向偏移。
`photo_loss(points, images, matrices, support)` 接收实际照片与固定可见支持，计算跨视图RGB方差。
`select(losses)` 在可用候选中选择最小损失，平局保留identity。它不接受独立扫描参考。

没有新增复杂优化器，也没有训练网络。本轮测试目标函数对实际几何收益的指向性。
照片不是深度；同一重建的输入照片不是独立重建验证集；几何参考仅在封存全部输出后用于评分。

原机liekkas、项目根目录、已有环境：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s loss_alignment_v23 -p 'test_*.py' -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B loss_alignment_v23/run.py
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B loss_alignment_v23/verify.py /新运行路径
```

数据已下载时不要重跑prepare脚本；它们使用防覆盖写入，而非通用续传器。
run每次创建新目录；verify、posthoc、plot_decisions防止覆盖已有记录。历史源锁对应运行时文件，
本报告等后续新增文件不冒充事前冻结文件。原始参考、网格、照片不写入Git。

正式运行：`/srv/slam-research/grf/map-denoise/runs/loss-alignment-v23-9g6_i29x`。
本轮不自动提交或推送GitHub。

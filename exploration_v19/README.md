# V19：两条路线并行，而非预设赢家

先读 [总报告](REPORT.md)。

- A：[确定点输出算子](track_a/operator.py)、[方法](track_a/METHOD.md)、[结果](track_a/RESULTS.md)。
- B：[后验几何导出](track_b/operator.py)、[方法](track_b/METHOD.md)、[结果](track_b/RESULTS.md)。
- 共同：[协议](PROTOCOL.md)、[生成与评分](common.py)、[运行](run.py)、[审计汇总](review.py)。

本轮结果在原主机liekkas，旧V18只读。没有自动更新GitHub仓库或修改V18默认行为。

## 原环境复现（创建新运行目录，不覆盖旧运行）

在项目根目录，使用已有Python环境并限制BLAS线程：

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export V19_PYTHON=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
"$V19_PYTHON" exploration_v19/run.py create
# 将输出的精确新目录填入V19_RUN；不要填既有的封存目录。
export V19_RUN=/absolute/new/run/path
"$V19_PYTHON" exploration_v19/run.py prepare --dest "$V19_RUN" --phase development --workers 4
"$V19_PYTHON" exploration_v19/run.py run --dest "$V19_RUN" --phase development --track a --workers 2
"$V19_PYTHON" exploration_v19/run.py run --dest "$V19_RUN" --phase development --track b --workers 2
"$V19_PYTHON" exploration_v19/run.py lock --dest "$V19_RUN"
"$V19_PYTHON" exploration_v19/run.py prepare --dest "$V19_RUN" --phase confirmation --workers 4
"$V19_PYTHON" exploration_v19/run.py run --dest "$V19_RUN" --phase confirmation --track a --workers 2
"$V19_PYTHON" exploration_v19/run.py run --dest "$V19_RUN" --phase confirmation --track b --workers 2
"$V19_PYTHON" exploration_v19/check_tests.py --dest "$V19_RUN"
"$V19_PYTHON" exploration_v19/review.py --dest "$V19_RUN"
```

同一phase准备后，两条run命令可以在两个终端并发；估计器不读取评价文件。
历史V18数据路径及早期上游依赖仍要求本机已有文件；这不是无数据依赖的通用安装包。

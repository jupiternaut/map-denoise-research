# V20 运行与复核

结论见 [REPORT.md](REPORT.md)，预先冻结的设计见 [PROTOCOL.md](PROTOCOL.md)。

在 liekkas 的项目根目录运行；需要原项目的 V11/V14–V19 模块、指定旧缓存与 Open3D 环境。没有下载安装新依赖。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s exploration_v20 -p 'test_*.py' -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B exploration_v20/run.py --workers 4
```

运行器创建新的唯一目录，不覆盖旧结果。将输出路径传给下面两个复核程序：

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B exploration_v20/review.py /绝对路径/新运行目录
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B exploration_v20/complete_oracle.py /绝对路径/新运行目录
```

二者均使用排他创建，已有验收文件时不会覆盖。`complete_oracle.py` 补上原始 R 拟合输出的事后 oracle 范围，不拟合新参数。详见报告中的诊断修正记录。

本次已完成运行：
`/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/candidate-repair-v20-qndibmbc`

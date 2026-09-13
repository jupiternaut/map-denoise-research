# V15 强对照挑战

先读 [REPORT.md](REPORT.md)。旧研究目录只读；本目录是新增实验，不替换默认算法。

在主机 liekkas 的项目根目录执行：

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -m unittest discover -s exploration_v15 -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -u exploration_v15/run.py
```

程序创建新唯一运行目录并打印绝对路径。它依赖原 V14 输入和缓存结果、原上游估计器以及已安装官方 PyMeshLab 环境。
旧文件不会被覆盖；重跑相同种子是复现，不是再次获得未见确认。

对打印的新运行目录运行以下命令（将参数换成新路径）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v15/audit.py /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/challenge-v15-tchfksi_
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python exploration_v15/readout.py /srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/challenge-v15-tchfksi_
```

示例原运行已经有 audit/readout，脚本再次写入会拒绝覆盖。需要复现全部核查时先运行 run.py 得到新目录，不删除原审计。

文件职责：

- `constant_solver.py`：同表示的64初值恒定混合连续求解挑战，输入不含GT。
- `run.py`：复用旧输入/缓存、并行输出封存、开发选择、新确认。
- `test_challenge.py`：数学梯度、结果保留和评价逻辑单元测试。
- `audit.py`：所有几何评分独立复算、哈希/选择/重放核查。
- `readout.py`：冻结后诊断与图片，不修改算法或配置。

使用限制：固定平面生成族、给定sigma、共享旧纠偏与支持；不是独立真实地图或3DGS结果。

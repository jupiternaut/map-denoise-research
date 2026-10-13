# A14 可复现命令

唯一目标主机liekkas；目录 /srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z。
既有CPU环境，不创建环境或安装包。先`hostname`及`pwd -P`核对；不可静默换机器或目录副本。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd /srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest test_adapter -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B joint_tests.py
```

以下是本轮执行顺序，不是让读者覆盖旧结果。输出排他创建，已完成后不要原目录重跑。可在新的明确授权实验目录中按新身份配置复现。

```bash
set -e
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B experiment.py smoke
# fresh源码审查完成后：
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B experiment.py freeze
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B experiment.py generate
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B experiment.py predict
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B experiment.py verify
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B experiment.py score
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B analyze.py
```

原始公共观测和真值分别保存在data/confirm/public.json与private/truth.json；工作进程仅接收公开观测与分派现任。独立核验器只读公开观测/状态。评分脚本在全部预测及核验成功后才读取真值。方法与原始结果由SOURCE_FREEZE及结果记录关联，但不宣称操作系统级访问隔离。

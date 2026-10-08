# 运行与复核

已验证主机liekkas，工作目录：
`/srv/slam-research/grf/map-denoise/runs/footprint-support-20261008T025757Z`。

Python：`/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python`。无新增安装、GPU、API服务。

## 可复跑单测

```bash
cd /srv/slam-research/grf/map-denoise/runs/footprint-support-20261008T025757Z
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B -m unittest -v test_kernel test_estimator test_oracle
```

独立复算入口在audit/replay.py、audit/check_evaluation.py；审计说明见audit/AUDIT.md。它们读取封存工件，审计输出只能写audit目录。

## 已执行顺序，仅作来源记录

```bash
python -B run.py prepare
python -B run.py score
python -B run.py oracle
python -B run.py infer
python -B evaluate_v2.py
python -B diagnose.py
```

实际使用上述绝对路径Python，score/oracle额外设OPENBLAS_NUM_THREADS=1和OMP_NUM_THREADS=1。两个评分进程可并行，其余按先后执行。

prepare、score、oracle、infer、evaluate_v2和diagnose使用排他创建，不能在已有工件上直接重跑。新实验另建目录并重新封存，不删除旧输出、不篡改LOCK来绕过检查。

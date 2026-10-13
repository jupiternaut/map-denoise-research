# A9 命令
language: zh

准确根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-anchors-1；liekkas，单线程CPU，无GPU。

手写测试：
```
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest -v test_continuous_solver test_verify_continuous test_continuous_pipeline
```

正式仅一次（源码审查门禁通过后）：
```
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B execute_continuous.py
```

运行器共享3600秒窗口；自动先公开推理、再独立检查与评价。不手工单独重置main或audit窗口。


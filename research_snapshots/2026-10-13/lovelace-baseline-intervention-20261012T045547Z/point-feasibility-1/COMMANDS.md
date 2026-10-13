# A6 执行命令

精确目标：liekkas；工作根 /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1；现有CPU环境 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python。

## 已执行手工测试

```bash
cd /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest -v test_point_solver test_point_pipeline test_verify_point
```

45手工夹具PASS。不存在正式输出时，fresh源码审查通过且当前字节匹配后，原执行器可执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B execute_point.py --output /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1/outputs/point-v1
```

本条由execute_point.py创建唯一3600秒绝对期限并依次执行公共诊断、后验评价、独立核验。正式输出不可覆盖，不直接重跑旧路径，不用新期限单独补验伪装原批次。正式完成状态见STAGE_STATUS.json及verification/VERIFY.json，不由本说明承诺。


# A6 手工测试封存（正式结果前）

2026-10-13；liekkas；根 /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1。

命令：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest -v test_point_solver test_point_pipeline test_verify_point`。

45项PASS，退出0，unittest Ran 45 tests in 2.260s。全部使用手工夹具；未运行正式矩阵、未parse private。

16求解器接口、9管线、20独立核验器；包括点/盒分歧、闭端点、union gap、全部3分支、0/1 clipping、signed links、逐叶 lifting、证据变造、648/2924门禁、实际runner/executor源码门禁和同一绝对预算。测试通过只支持实现检查，不是科学结论。

初步阶段的冻结源路径检查曾错误预期父根baseline_geometry.py；根据实际A4导入身份纠正为recovery-1/baseline_geometry.py。详见TEST_SUITE_PRELIMINARY.md。正式执行尚未开始。


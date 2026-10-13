# A5 命令与工件

精确主机liekkas；工作根本文件所在joint-feasibility-1。运行时：/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python，SciPy1.15.3。程序设置CPU单线程，无GPU。

夹具复跑：

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s . -p 'test_*.py' -v
```

正式已启动命令：

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B execute_polytope.py
```

execute为当前阶段唯一正式入口；创建outputs/polytope-v1后不允许同路径重跑覆盖，main与verify一个3600秒窗口。科学不成功不等于实现失败；INCOMPLETE也不能作为完整负结果。

结果核心：outputs/polytope-v1/{STAGE_STATUS.json,verification/VERIFY.json,outers.json,decisions.json,keep/decisions.json,results.csv,coverage.json,summary.json,TIMING.json}。

不重跑旧阶段，不更改δ、深度叶或hybrid。private在所有动作保存读回门禁后仅供评价，独立核验也保留同一顺序。

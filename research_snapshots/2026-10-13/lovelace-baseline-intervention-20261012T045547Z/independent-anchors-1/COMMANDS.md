# A8 执行与复算命令

准确主机：liekkas。所有下列命令均在 /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/independent-anchors-1 执行。指定 Python 为 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python，启用 -B；OMP/OPENBLAS/MKL/NUMEXPR 都为单线程。

## 已执行：冻结前测试

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest test_anchor_solver test_verify_anchors test_anchor_pipeline -v
```

36 项 PASS，2.025 秒；最终测试原文在 TEST_SUITE_20261013_074400.log。源码审查的同族复用背景见 ASSURANCE.md 与 .aris/traces/；它不替代正式机械核验。

## 已执行：一次正式批次

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B execute_anchors.py
```

依次 generation → main → verification，共享一个 3600 秒 epoch 窗口。实际生成 0.0639 秒、main 441.5983 秒、核验 140.6140 秒，执行器总计 582.2784 秒。STAGE_STATUS=COMPLETE，VERIFY=PASS。全部记录先保存并核验，再做真实世界覆盖及评价。历史输出不删除、正式批次不重跑。

精确逐次命令、时间和退出码：outputs/anchor-v1/STAGE_STATUS.json；预算：BUDGET.json；源码和输入：SOURCE_LOCK.json、INPUTS.json。独立核验检查 162 个源码哈希、17 个输入哈希、1296 个任务—臂、3888 个分支、144 个真世界—臂。

## 已执行：事后描述性复算

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B refine-logs/analyze_anchors.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B refine-logs/compare_anchor_sets.py
```

两个脚本仅在正式 COMPLETE/PASS 后读取保存记录，分别生成 ANALYSIS.json 与 PAIRED_SETS.json，同时先生成带时间戳快照。它们不调用求解器，不修改科学结果。输出使用独占创建；不要在封存目录里重复运行。

## 收尾清单

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B refine-logs/seal_anchors.py
```

seal 重新比对当前源码/输入和旧 A6/A7 索引哈希，并生成本目录全文件清单、工件索引及最终封存回执。只执行一次；成功后不得再修改本目录或生成额外文件。

本轮未推送 GitHub/GitBook，未改变任何部署默认。


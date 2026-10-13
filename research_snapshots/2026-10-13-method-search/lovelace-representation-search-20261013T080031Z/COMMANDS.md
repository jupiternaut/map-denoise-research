# A12 命令与数据流

目标主机liekkas，逻辑根`/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z`，经已有runs链接访问同一物理目录。所有命令在此根运行。

解释器：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`。
环境：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。

顺序：
```
python -B data_builder.py --split dev
python -B -m unittest test_data_builder test_harness -v
python -B run_experiment.py freeze
python -B data_builder.py --split holdout --freeze SOURCE_FREEZE.json
python -B run_experiment.py predict
python -B run_experiment.py supplement
python -B run_experiment.py score
python -B analyze_results.py
```

生成与正式输出采用新建独占文件；这些命令不是要求在已完成目录直接重复覆盖。独立复算使用审计脚本，不重新挑选方法。开发构造源/命令/CPU成本在arms/free、arms/library/METHOD.md；选择器仅一项开发后修订。

正式PREDICTIONS_READY固定全部108次调用/756条预测，评分之前检查完整性；后验分析不反向修改方法。本轮未运行真实照片/点云、GPU、GitHub推送或GitBook更新。

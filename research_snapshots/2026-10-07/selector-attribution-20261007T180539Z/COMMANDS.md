# 执行记录

准确主机liekkas；工作目录本文件所在目录。Python为`/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python`。

按以下顺序实际成功执行（-B禁旧目录bytecode写入）：

```sh
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B -m unittest -v test_policies.py
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_ablation.py prepare
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_ablation.py freeze
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_ablation.py infer
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_ablation.py evaluate
```

28单测通过。真实输入首次推理成功、评价成功，无结果后修源码。脚本使用exclusive-create，禁止原位重跑覆盖。需重新生成时显式选新实验目录，并核对冻结历史路径，不得假装新目录即原运行。

主流程无GPU、安装、Git写入、网络上传。缓存GT只在evaluate阶段读取，未重建激光评价参照。

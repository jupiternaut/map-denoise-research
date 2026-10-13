# A13 复现命令

工作目录：`/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z`，主机 liekkas。
使用现有 Python：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`。
所有命令设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。

按顺序运行：

```bash
python -B -m unittest test_verify_certificates test_harness -v
python -B test_cert_kernel.py
python -B run_experiment.py smoke
python -B run_experiment.py freeze
python -B run_experiment.py generate
python -B run_experiment.py predict
python -B run_experiment.py diagnose
python -B run_experiment.py verify
python -B run_experiment.py score
python -B analyze_costs.py
```

这里 `python` 指上述绝对路径；不安装、不切换环境。结果使用独占创建，不覆盖现有结果；已完成的命令不要原地重跑。报告中的耗时是冷进程启动至退出，不包含主控创建输入 JSON 的时间；验证器重放费用另报。RAM 只记录序列化前观测到的进程峰值。

方法只接收一份观测，解码只接收单个现任。生成器使用冻结物理模型精确生成有理数读数，再 `float(exact)` 转成二进制64位数并经 JSON 往返；精确方法将读数解释为闭合舍入原像区间，而非当作无误差精确十进制。

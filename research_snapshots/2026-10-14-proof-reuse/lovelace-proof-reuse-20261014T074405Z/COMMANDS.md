# A15 复现命令

所有命令在本目录执行，Python 为 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -B experiment.py freeze tolerance
python -B experiment.py predict tolerance
python -B experiment.py verify tolerance
python -B boundary_diagnostic.py --batch outputs/tolerance --arms original pure --output outputs/boundary-tolerance.json
python -B experiment.py score tolerance
python -B switch_runner.py freeze
python -B switch_runner.py predict
python -B switch_runner.py state_audit
python -B experiment.py verify switches
python -B experiment.py score switches
python -B switch_runner.py profile
```

以上生成命令使用 exclusive-create，不覆盖已有结果；要重跑需新实验目录及明确修订根路径。不能在旧目录反复执行覆盖原证据。只读复核可调用相应模块的 check 与读取结果，不能重新执行 freeze。

review 后追加 switch_runner，保留原 experiment.py 未变。其全局资源账按已存在的任务回执和诊断回执计数；父进程证书验证成本另记。

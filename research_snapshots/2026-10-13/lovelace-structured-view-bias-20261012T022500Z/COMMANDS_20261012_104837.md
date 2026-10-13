# 本机复现命令

唯一目标host liekkas；工作目录：

```bash
cd /srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z
env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_structured_solver test_run_structured test_verify_structured
env PYTHONDONTWRITEBYTECODE=1 python3 run_structured.py --smoke
env PYTHONDONTWRITEBYTECODE=1 python3 run_structured.py --output /srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z/outputs/structured-v1
env PYTHONDONTWRITEBYTECODE=1 python3 verify_structured.py --output /srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z/outputs/structured-v1
env PYTHONDONTWRITEBYTECODE=1 python3 analysis/analyze_structured.py
```

正式输出必须新目录；上述structured-v1只运行一次，存在即拒绝覆盖。核验器读BUDGET的原绝对deadline，不重新给5400秒。分析只有COMPLETE+PASS才运行。测试/夹具不作为确认数据，不重跑历史求解。
日志：refine-logs/M2_RUN.log、M3_VERIFY.log、M4_ANALYSIS.log。源码与历史完整输入身份在输出SOURCE_LOCK.json/INPUTS.json；具体复用原图/外包/动作canonical hashes在REUSE.json。

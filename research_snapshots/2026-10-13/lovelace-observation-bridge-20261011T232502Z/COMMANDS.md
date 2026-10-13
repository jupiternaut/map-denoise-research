# 已执行命令与复现入口

确切host liekkas；本根`/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z`。现有Python为`/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python`，CPU单线程，不安装依赖。以下是实际已执行记录，不是待执行计划。

## M0测试与冒烟

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B -m unittest -v test_run_bridge
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B -m unittest -v test_verify_bridge
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_bridge.py --smoke
```

根代理日志：`refine-logs/M0_RUNNER_20261012_074135.log`（14 tests）、`M0_VERIFIER_20261012_074646.log`（23 tests）、`M0_SMOKE_20261012_074435.log`（六新配置smoke）。均exit=0。

## 正式批次，仅执行一次

M1非作者审查放行后，于2026-10-12 07:52:06 Asia/Shanghai启动：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B run_bridge.py --deadline-unix 1791768126 --output outputs/bridge-v1
```

日志`refine-logs/M2_MAIN_20261012_075200.log`，exit=0；RUN_STATUS=COMPLETE、7776动作、1261.912029秒。BUDGET实际start=1791762726.6309001，deadline=1791768126（09:22:06北京时间）。

## 完整核验，已执行

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B verify_bridge.py --output outputs/bridge-v1
```

日志`refine-logs/M3_VERIFY_20261012_081309.log`，exit=0。工件`outputs/bridge-v1/verification/VERIFY.json`的status/mechanical_verification均PASS；核验192.124090秒，至核验完成主运行+核验共1463.093064秒。使用原始同一截止时间，没有重置预算。

## 核验通过后配对分析，已执行

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B analysis/analyze_bridge.py --output outputs/bridge-v1 --destination analysis/analysis-v1
```

日志`refine-logs/M4_ANALYSIS_20261012_081630.log`，exit=0；不导入runner/solver，Fraction结算并检查输入哈希前后不变。另有非作者结果复核`analysis/RESULT_INTERPRETATION_REVIEW.md`。

## 复现限制

本批没有后台实验。所有正式输出和历史输入不可覆盖；四根科学.py及PROTOCOL正式launch后冻结。SOURCE_LOCK锁63源，INPUTS锁26历史工件；核验PASS不是物理前向模型正确性或操作系统信息隔离的证明。

今后重跑须另建经授权输出版本，并另立真实预算；不能照抄本记录的绝对截止时间来续期当前批次。本文保存实际命令，不擅自启动新批次。未执行C/D/E/F、GPU、安装、API、GitHub/GitBook发布或部署操作。

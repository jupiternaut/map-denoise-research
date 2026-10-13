# 运行与复核

主机为 liekkas。新工作目录：
`/srv/slam-research/grf/map-denoise/runs/lovelace-hybrid-20261010T214217Z`。
只读依赖：`/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z`，包含声明的 GitHub source_snapshot。
这不是一个无依赖的分发包；迁移机器时必须显式迁移两个目录并调整依赖路径，不能声称只复制新目录就能复现。

## 作者单测及冒烟

```bash
cd /srv/slam-research/grf/map-denoise/runs/lovelace-hybrid-20261010T214217Z
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B run_hybrid.py --smoke
```

独立核验器测试是否纳入 discover 以其目录结构为准；也可显式运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest verification.test_verify_results -v
```

## 一次冻结确认

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B run_hybrid.py --output outputs/confirmation-v1
PYTHONDONTWRITEBYTECODE=1 python -B verification/verify_results.py --output outputs/confirmation-v1
```

输出目录和 VERIFY.json 必须是新的；已有结果不覆盖。重跑应选新的输出路径，不能修改冻结旧结果。
运行过程产生 JSON/CSV/NPZ 结果及计时；确认决策先保存，随后用生成真值评价。
参数固定于 PROTOCOL.md，不提供按确认结果调参的命令行开关。
本轮不需要显卡、不启动服务、不上传任何仓库。

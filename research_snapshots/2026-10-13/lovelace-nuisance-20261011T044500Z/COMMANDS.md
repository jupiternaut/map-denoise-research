# 本轮运行入口

主机：`liekkas`。所有命令在以下精确目录执行：

```bash
cd /srv/slam-research/grf/map-denoise/runs/lovelace-nuisance-20261011T044500Z
```

作者测试（仅非确认测试夹具）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -v
```

独立源审查通过后，冒烟使用600 mm、99001种子，不写确认数据：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B run_nuisance.py --smoke
```

一次正式确认；默认目录必须不存在，程序拒绝覆盖任何已有结果：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B run_nuisance.py --output outputs/confirmation-v1
```

五臂共用观测与旧hybrid动作；只有两臂使用额外的三个有来源材料样本。
54观测配置、486初态、2430决策。一个CPU线程，GPU=0，单次确认硬上限1800秒。
旧两个运行目录只读；不修改真实点云默认，不发布仓库。

独立确定性核验（正式确认完成后；独占写入VERIFY.json）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B verification/verify_results.py --output outputs/confirmation-v1
```

由保存的评价行生成描述统计，不生成任何新决策：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B analysis/summarize.py
```

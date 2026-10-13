# 复现入口

本目录对应 liekkas 上的纯CPU连续深度挑战；不修改旧实验或运行默认。不需要API key或GPU。

进入目录：

```bash
cd /srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z
```

确认前单测与独立核验器单测：

```bash
PYTHONPATH=source_snapshot python -B -m unittest continuous_world.test_interval_model continuous_world.test_interval_solver test_layered_model test_projection_policy test_challenge_baseline test_run_recording verification.test_verify_results -v
```

最小冒烟（不是确认集）：

```bash
python -B run_challenge.py --smoke
```

固定协议运行一次，并使用从未存在的输出路径：

```bash
python -B run_challenge.py --output outputs/confirmation-v1
```

禁止覆盖已有结果。重新复现请换全新的输出目录，例如 `outputs/reproduction-1`，不要改原确认目录。Linux runner采用1800秒SIGALRM截止，每9个完成观测留部分记录；部分运行标记INCOMPLETE，不成为通过的确认。

核验归档结果：

```bash
python -B verification/verify_results.py --output outputs/confirmation-v1
```

核验器使用独立的动作与汇总计算；连续成像数学内核共享，不称第二套独立物理模型。生成器与评价器同进程，策略未收到真值，不称密封盲评。
已有核验报告会拒绝覆盖。本目录确认已核验，直接读取 outputs/confirmation-v1/verification/VERIFY.json；重新执行时核验自己的新复现目录，不删除旧核验报告。

范围与验收见PROTOCOL.md；数学依据见THEORY_CHALLENGE.md、LAYERED_THEORY.md。结论以REPORT.md及其中链接的原始JSON/CSV为准。

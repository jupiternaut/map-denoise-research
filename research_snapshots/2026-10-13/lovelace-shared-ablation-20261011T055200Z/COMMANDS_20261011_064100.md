# 执行、核验与复现入口

确切主机：liekkas。
确切目录：/srv/slam-research/grf/map-denoise/runs/lovelace-shared-ablation-20261011T055200Z。
状态：已完成；无本批后台实验进程。

## 本轮实际执行

以下入口均在确切目录执行：

```bash
python -B -m unittest test_run_ablation test_grouped_solver test_verify_ablation -v
python -B run_ablation.py --smoke
python -B run_ablation.py --output outputs/ablation-v1
python -B verify_ablation.py --output outputs/ablation-v1
python -B analysis/summarize.py
```

最终 33 项测试 PASS，六臂冒烟 PASS，主运行 COMPLETE，独立 VERIFY PASS，分析完成。测试包含预期的故意 INCOMPLETE 反例输出，不是正式实验失败。早期测试期望错误的日志保留，不改写成算法负结果。

证据日志：

- logs_tests_20261011_060100.log
- logs_smoke_20261011_055900.log
- logs_run_20261011_060400.log
- logs_verify_20261011_063000.log
- logs_analysis_20261011_063630.log

主运行单 CPU，GPU/API/安装均为 0；SIGALRM 真期限 5400 秒。旧处理器异常文字含 1800 秒，已记录为诊断文字缺陷，实际运行期限没有改。

## 不覆盖式复现

不要直接重复上述封存输出命令。run_ablation 和 verifier 均拒绝覆盖。仅在需要复算时使用全新路径，例如：

```bash
python -B run_ablation.py --output outputs/ablation-reproduction-02
python -B verify_ablation.py --output outputs/ablation-reproduction-02
```

这两个命令只是复现说明，本轮未执行。现有 analysis/summarize.py 固定读 ablation-v1 并写 analysis-v1，也拒绝覆盖；不可把它误当作支持任意 --output 的入口。

报告：REPORT.md；状态与源码锁：outputs/ablation-v1/RUN_STATUS.json、SOURCE_LOCK.json；机械验收：outputs/ablation-v1/verification/VERIFY.json。


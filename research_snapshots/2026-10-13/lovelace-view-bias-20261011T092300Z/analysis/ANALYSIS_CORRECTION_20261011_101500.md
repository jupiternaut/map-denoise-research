# 汇总字段口径更正

2026-10-11 10:15 UTC。主批源码、决策及核验未改。

首版只读汇总 `outputs/analysis-v1` 的 `correct_kept`、`negative_offset_kept` 实际统计坐标未变，因而把 strict 的 INCOMPATIBLE 回退也写成 kept。正式核验的状态统计没有这个缺陷：strict 每β正确 KEEP=0/54、真深度覆盖=0/54、486项 INCOMPATIBLE。

只修改根目录科学锁之外的分析脚本；原脚本保存在 `analysis/summarize_20261011_092900.py`，原分析输出保留。修订版输出到新目录 `outputs/analysis-v2`，区分 KEEP、坐标未变、INCOMPATIBLE。两项汇总回归测试要求相同坐标误差下状态仍可区分。

此更正不影响合法方法的改善数、MAE、覆盖或本轮主张。报告只引用 v2 与正式 VERIFY。

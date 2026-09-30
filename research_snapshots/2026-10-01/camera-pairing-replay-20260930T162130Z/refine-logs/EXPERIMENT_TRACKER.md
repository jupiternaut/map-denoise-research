# 实验执行表

日期：2026-10-01 01:12:35（Asia/Shanghai）。状态：M0–M3完成；旧计划及旧run只读。

| Run ID | Milestone | Status | 完成证据 |
|---|---|---|---|
| R000 | M0 映射 | DONE | 26视图、43来源、24极线对；MAPPING_REPORT.md |
| R001 | M1 冒烟 | DONE | 两U首ROI；U0严格复现，U1有效120/128；11单元测试 |
| R002 | M2 固定旧候选 | DONE | 12 U0案例×2 W；旧输出/见证逐值一致 |
| R003 | M2 上游修复 | DONE | 12 U1案例×2 W；497/512，15质量失败 |
| R004 | M3 评价 | DONE | 384项MSE独立复算差0；REPORT.md与figures |
| R005 | fresh语义审稿 | UNAVAILABLE | 新代理创建返回agent thread limit reached；未伪报PASS |

C1支持本次开发回放：native共同集初始MSE453.456→150.121mm²，四ROI均改善。
C2仅条件收益：完整U1W1的C见证通过既定联合验收，H不通过；W1不是每条件MSE都优于W0。
数值核验existing代理/same-family/provisional。部署默认未改、未Git push、未发布GitBook。

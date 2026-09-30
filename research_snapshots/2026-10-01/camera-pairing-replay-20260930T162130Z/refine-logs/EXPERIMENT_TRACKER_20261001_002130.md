# 实验执行表

日期：2026-10-01（Asia/Shanghai）。状态：计划完成，实验未启动。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | 照片和相机映射来源核对 | K/R/C + resize | scan24/37 | 来源、二维极线残差 | MUST | TODO | 不用GT拟合 |
| R001 | M1 | 最小冒烟和计时 | U0W1、U1 | 单ROI | pixel_id、候选恒等性、同支持 | MUST | TODO | 获取完整运行耗时依据 |
| R002 | M2 | 评分映射效应 | U0W0/U0W1 | 固定旧12案例 | MSE、移动账、选择损失 | MUST | TODO | U0W0复用封存输出 |
| R003 | M2 | 上游修复及交互 | U1W0/U1W1 | 同像素12案例 | 构造覆盖、KEEP/Oracle/输出 | MUST | TODO | context变化单独登记 |
| R004 | M3 | 评价与独立复算 | 四单元 | 已曝光开发回放 | 主终点与损伤账 | MUST | TODO | 新预测封存后评价 |

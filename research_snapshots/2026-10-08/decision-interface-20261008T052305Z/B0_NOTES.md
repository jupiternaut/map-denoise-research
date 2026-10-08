# B0 决策实现与确定性检查

日期：2026-10-08。主机：liekkas。此文件只记录实现检查；没有启动 B1 标定、B2 批量回放或 B3 确认，没有新的科学实验结果。

## 实现接口

`decisions.py` 不导入数据、真值、旧预测器或运行器，只依赖 NumPy 与标准库。公共函数保持 `INTERFACE.md` 的签名：

- `acceptance(grid, loss, sigma2)`：原 9/4 接受节点，平坦性由 `decide` 独立检查。尺度不可用或普通模型曲线非有限时，返回全 False。
- `combine_scale(sigma_values, pixel_counts, sigma_cal2=None)`：有效局部 sigma≥1 保留；只替换缺失折。返回合并方差与每折 `local` / `calibration` / `unavailable`。缺一折且无回退时方差为 NaN，不删除该折。像素分母非正或非有限视为合同错误。
- `distribution(...)`：R/U 都只给同一 A 内节点质量，使用完整原始网格上的物理深度 Voronoi 单元宽度，端点截在公开深度域。外部 full9 有效域之外的 NaN 节点质量为零，但不删除网格节点或改变相邻单元宽度。返回值是分数权重，未声称已校准后验。
- `crps(grid, q, truth)`：离散节点质量的精确 CRPS，用 O(n) 的排序和式计算；校验质量非负且总和为 1。`truth` 仅由独立标定调用方供给，`decide` 不接受真值。
- `decide(...)`：实现 P/M/Mraw/R/U，返回 JSON-safe 决策字典。

统一字段为 `selected_depth, reason, move, scale_valid, accepted_count, raw_valid, flat, rule, candidate_count, finite_score_count`。P 追加 `intervals, support_mean, estimated_squared_gain`；M/Mraw 追加 `finite_candidate_count, best_loss, best_count`；R/U 追加 `risks, selected_risk, best_count, kappa, temperature`（成功到达相应决策分支时）。

可能的原因包括：`ok_move, keep_incumbent, raw_invalid, nonfinite_curve, scale_unavailable, flat_curve, threshold_empty, tied_best, no_finite_candidate, no_positive_gain`。结构错误、漏算动作节点或非法参数抛出 `ValueError`，不会将程序合同错误伪装成正常科学拒绝。

## 保持历史行为与 full9 有效域

P 保留传入候选顺序及 `argmin` 的首索引平局行为。`historical_intervals` 是已经封存的准入输出，P 精确重放它，在检查 `raw_valid`、整体非有限或新平坦规则之前执行；空历史列表仍 KEEP。该分支不重新宣称输入评分或尺度有效。

未提供历史区间时，P 使用接受物理单元加 1 mm 填充，裁剪至完整网格端点后合并。均匀网格上与旧 half-step+padding 构造一致；不均匀网格按已澄清的物理单元规则处理。

所有新规则将动作去重并排序，并自动包含 KEEP。M/Mraw 必须在网格上找到每个动作的精确评分节点，禁止最近格点替代。M/Mraw 平局 KEEP；R/U 也要求唯一且严格负的最小相对风险，否则 KEEP。κ>1 只增加逐可能真值的正误差增量代价，不改变负增量。

`accepted` 外部布尔向量代表 full9 的自身准入及有限评分域。P/M 可使用它而无需灰度残差尺度，R/U 仍需要正的分数尺度。Mraw 绕过该接受门槛，但继续使用有限的候选评分。新规则的平坦性在有限评分域内检查；全域无有限值则 KEEP。普通模型没有外部 mask 时继续要求完整曲线有限。

## 检查与结论

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B -m unittest -v test_decisions
```

结果：35 项确定性单元检查通过，测试框架计时 0.034 s。检查数不是独立场景数量。

覆盖的主要合同：

- 只读旧 w002 封存曲线；P 的 `[486,782] → 634 → 660` 完整复现；M 在 540/600/660 三现任下均选择已有最低残差候选 600。测试前后该 NPZ 的 SHA-256 不变。
- 历史 P 候选顺序平局、空区间、NaN/flat/raw flag 下的历史区间重放；均匀与不均匀网格的填充及端点裁剪。
- 新规则候选乱序/重复、精确平局、近似平局、平坦曲线、非有限值、外部 mask、空 A、无局部尺度、Mraw 绕过尺度/A、精确动作评分节点。
- 尺度回退不替换已有有效局部值、不删除缺失折；通常分离曲线在同时缩放 L 和 sigma² 后保持接受与决策。
- R/U 共享 A 与单元宽度；逆深度均匀采样转换回深度后仍按 mm 测度积分；解析指数曲线的 2→1→0.5 mm 积分误差递减；CRPS 与直接双重求和一致。
- 两个分离谷的风险反例：R 可以选择不在 A 中的中间动作，而残差 argmin 因两个同分最低谷 KEEP；κ=5 可以将 κ=1 的移动改为 KEEP。这些反例保留，不把 R 预设为更好方法。
- 条件分离引理：在相同 RMS 范数、真候选在 C、预测误差≤η且每个错误候选距真候选预测大于 2η 时，残差 argmin 唯一恢复真候选。等号处可出现平局。该检查没有把 MAD 当作 η，也没有新增经验保证。
- w002 支持凸包包含三个现任，严格最坏平方收益在这三个初态均选择 KEEP；这可以阻止所示伤害，但不能修复两个错误初态。

## 数值保证的边界

新规则严格使用预登记 `abs_tol=1e-10, rel_tol=1e-10`；P 保持旧 `1e-12` 平坦门槛。固定绝对容差意味着“任意缩放 L 和 sigma² 后，所有边界情形的平局判定都不变”在数学上不成立。测试明确构造了这个边界敏感例，并检查远离容差边界的缩放不变量；没有擅自用归一化分数替换原始 M 分数来隐藏它。

本检查只验证声明公式与代码是否一致。旧 w002 已暴露、两谷和分离算例均为确定性构造；它们不计入新的校准或确认样本，不代表相对 full9 的新成像增量。

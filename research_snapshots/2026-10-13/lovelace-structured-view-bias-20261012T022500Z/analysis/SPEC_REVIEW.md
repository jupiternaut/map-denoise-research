# 非作者规格与输入设计审查

日期：2026-10-12；实际本机 `hostname=liekkas`。核对的唯一工作根为 `/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z`，未使用 SSH 或替代环境。

审查类型：experiment-bridge 所要求的同模型族内部、非作者、provisional 审查；不构成独立科学复现或新颖性保证。本阶段完整读取本根 `AGENTS.md`、`PROTOCOL.md`、`refine-logs/EXPERIMENT_PLAN.md` 和当前 425 行 `run_structured.py`。solver 与独立核验器仍在实现，尚未完成全源码审查，也未运行正式批次。

结论：**SPEC_PASS；LAUNCH 尚未放行。** 当前规格与已读输入接口没有阻断性设计问题。最终放行必须等待作者宣布实现完成，再审阅六个科学源码/测试文件及相关冻结数学接口。

## BLOCKING

当前规格审查未发现必须修改的设计缺陷。实现尚未完成不是科学缺陷，但 M1 全源码审查、M0 测试和 smoke 尚未完成，因此此文不授权 launch。

## 正确的设计与当前可见证据

1. **C1/C2 与控制变量。** raw/clip 使用前批同图像、同 54 基础观测、同初态；keep、精确复用的 wide、calibrated、joint 四法清楚区分。calibrated 已是机制删除对照，不需要额外扩网格。原动作器、三分支、depth `[540,660]`、每分支 511 盒、8 轮和 `15/32 mm` 不变。预算相同只指树预算，协议没有把它误写为 CPU 成本相同。
2. **共享误差与观测行。** 对 `d=1-c`、`t=beta-b`，未裁剪行 `sum(A*d)+t ∈ [1+beta-y-epsilon, 1+beta-y+epsilon]` 正确。clip 的 `y=0` 只保留下界 `1+beta-epsilon`，`y=1` 只保留上界 `beta+epsilon`，正确保留饱和读数的单侧语义。每个 view 共用一个 t，逐观测 bounded noise 仍可独立取值；两者没有混淆。
3. **必要外包。** 非负材料与系数条件下，上界约束使用几何系数下界 L、下界约束使用 U 是必要条件。calibrated 上界减 `t_lo`、下界减 `t_hi` 正确；joint 保存共同变量依赖。有限轮传播只称必要外包，未声称精确可行性。排除需直接矛盾或变量区间交叉，并计划保留可重放证书。
4. **公共背景及公平初始化。** mask 要求在全 depth、全部 branch、允许 camera 误差内可见面积上界均为零；这足以推出前向值为白背景，不需要 truth 选像素。由同图像 mask 行得到的同一个 `T_cal` 同时用于 calibrated 和 joint；它是公开观测推导，不是额外标定样本。clip 白端只给 t 上界，正偏差不能强行点校正。空校准交集必须计为 INCOMPATIBLE。
5. **接口不接私有生成变量。** 当前 `execute_new()` 只向 solver 传观测、camera radius、method、clip 标识和注册超参数；不传 case、seed、真实 depth、材料、实际 bias 或 incumbent。配置中的 `bias_pattern_generation_only` 是生成说明，未传入科学求解接口。最终还需核对 solver 与其依赖没有旁路读取 truth。
6. **落盘后评价。** 当前 `_run()` 在读私有 truth 前检查 3888 动作、3888 唯一 `(task_id,method)`、已保存 `decisions.json` 读回相等，以及各方法 972 条和 reused 标识；之后才调用 `private_truths()`。实际误差相对于 dataset truth 计算，非相对于另一方法输出。最终需核对导入依赖和核验器的同样顺序。
7. **复用与成本。** `prepare_reuse()` 要求指定前批 COMPLETE、指定 host/output 的 VERIFY PASS，并重放历史 shared 外包所对应动作；input/science lock 检查已有来源不变。计数 216 新 solver、108 历史外包、1944 新方法动作、972 wide 复用动作、972 keep 动作相互一致。当前计时字段将新冷 solver、selector 和历史 solver 时间分开，未把历史求解冒充新确认。
8. **结论范围。** 协议明确是同源合成机制回放，相关初态不当独立场景；主终点、覆盖、正确/负控 KEEP、INCOMPATIBLE、seed sample SD、成本和逐例损伤均已规定。机械通过与 C1/C2 科学正效应分开，零增量时优先简单校准的判断合理。

## NON-BLOCKING

- 协议矩阵下方的“1944 新非KEEP动作”应在最终报告中写作“1944 新求解方法的动作记录”（或沿用代码 `expected_new_method_actions`）；calibrated/joint 的实际动作状态可能为 KEEP，不能把方法名称与动作状态混用。这是术语提醒，不影响当前矩阵或执行逻辑。
- 本阶段并未核实初步 48 pixel mask 数字，也未把该数字当作通过依据；最终应由全域几何推导与核验器独立计算得到。
- 有限传播可能导致零收益，clip 饱和可能使背景无法识别正偏差；这是已声明的研究问题，不是需要额外实验补救的 blocker。

## 最终源码审查待核对项

完整读取 `structured_solver.py`、`test_structured_solver.py`、`verify_structured.py`、`test_verify_structured.py`、`run_structured.py`、`test_run_structured.py` 以及冻结几何/union/action/生成/评价接口。重点核对共同 `T_cal`、单侧 clip、每 view 共享与逐测量 noise、全域 mask、Fraction 必要传播和矛盾证书、wide 完整身份、3888 保存读回之后才解析 truth、计数/实际新增成本，以及核验器不导入新 solver 数学。仅在该阶段无 blocker 后，另写 `refine-logs/EXPERIMENT_CODE_REVIEW.md` 对一个冻结主批给出明确 PASS/LAUNCH_ALLOWED。

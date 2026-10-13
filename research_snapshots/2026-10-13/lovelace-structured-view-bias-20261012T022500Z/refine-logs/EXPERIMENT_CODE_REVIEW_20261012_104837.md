# 非作者完整源码审查

日期：2026-10-12，Asia/Shanghai。实际本机 `hostname=liekkas`，实际目录 `/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z`。未使用 SSH、替代目录、GPU、API、安装、发布或部署；审查者没有运行正式批次，没有修改科学源码或旧根。

**结论：PASS / LAUNCH_ALLOWED。** 允许按本根冻结协议运行一个固定主批及其机械核验，M2+M3 共用原始绝对 5400 秒窗口。该结论对应下列最终源码版本；未发现需要在 launch 前修复的 blocker。正式结果仍不存在，故此结论不是正式批次的 VERIFY PASS，也不是 C1/C2 的正结果。

本审查按 experiment-bridge 非作者审查要求执行，属于同模型族内部 provisional 审查，不是独立科学复现或新颖性保证。核验器的“独立”指不同作者实现的新约束与传播、树、动作和评价机械重建；其冻结几何核仍被复用。

## BLOCKING

无。当前实现与固定协议一致，已读测试和只读交叉检查没有显示阻止一个冻结批次执行的问题。

## 审阅范围与最终源码身份

完整读取本根 `AGENTS.md`、最新 `PROTOCOL.md`、`refine-logs/EXPERIMENT_PLAN.md`；完整逐段读取下面六个源码和测试文件。最终 `test_run_structured.py` 已包含第 10 项四夹具集成测试，根代理通知的最终 hash 与实际文件一致。

| 本根路径 | 行数 | SHA-256 |
|---|---:|---|
| `PROTOCOL.md` | — | `cf45f6c9719f30d8a0e9a696171f3fb7d5b228f7d36fecdf2dd27ceda33bf291` |
| `structured_solver.py` | 360 | `775a52a81e1451d8c95a003aa1bb188c30b4ab38365a55eac578e74f3a24cbf9` |
| `test_structured_solver.py` | 225 | `e6e2db47cf5bf719fb65e59270e7312bdb757098ae1e73f06a27e3fca70b1cd7` |
| `verify_structured.py` | 876 | `7fdd2473574522b03d79107e580d2526c7ed81e5173f81089938e23d809f89ea` |
| `test_verify_structured.py` | 256 | `dafc3abc732d48c89c3b52661781fcad5ac1c9d9e6a4a7211718bb3e33c5205e` |
| `run_structured.py` | 453 | `da90b3d760adc437e130ef6235c3dc8337f9c762b0b14f79313658064687c538` |
| `test_run_structured.py` | 144 | `a882161eca7a081c5d36a1be5282901e7bac3ea53d51d54d0c9a80d9b0dcb66c` |

还直接检查了以下冻结接口的源码，而非只读作者摘要。下述相对路径均位于 `/srv/slam-research/grf/map-denoise/runs/`：

- 完整 `lovelace-shared-ablation-20261011T055200Z/grouped_solver.py`：原 BFS、union、精确数值接口及冷缓存清理。hash `1019ecb22d4951274014a36fa0325b61f8585990209fd95a975b150a84310260`。
- 完整 `lovelace-nuisance-20261011T044500Z/nuisance_model.py`：全域 camera/depth 几何面积上下界、point geometry、三材料/三视图 layout。hash `e6701907d43113cc7ebe3cde38c6d75f4a9f45f648a2cc2e648ee5e336e3aa5d`。
- `lovelace-challenge-20261010T205013Z/layered_model.py` 的矩形投影、inner/outer、面积及遮挡差集；其 `source_snapshot/continuous_world/interval_model.py` 的精确 Interval、depth、camera、grid；`interval_solver.py` 的 Fraction 编码、BFS 与端点 gain；`challenge.py` 的 projection。
- `lovelace-nuisance-20261011T044500Z/run_nuisance.py` 的生成、identity、select_action、聚合；`lovelace-hybrid-20261010T214217Z/hybrid.py` 的完整 midpoint→projection 动作；`lovelace-shared-ablation-20261011T055200Z/run_ablation.py` 的 identity、aggregate、covered、cache 接口。
- `lovelace-observation-bridge-20261011T232502Z/run_bridge.py` 的图像变换、旧数据和 source lock、复用参数校验，hash `7926aba59cc59c13016aec9eb436e46e3d80314561217241d9458fc8655e85d4`；同根 `verify_bridge.py` 的独立变换、旧 wide 分支/union、来源清单与 canonical JSON hash，hash `d3a1404814abf39384101e355dd92904eff3edac7ab66f1f960bc7f7c50c773a`。
- `lovelace-view-bias-20261011T092300Z/verify_view_bias.py`、`lovelace-shared-ablation-20261011T055200Z/verify_ablation.py`、`lovelace-shared-appearance-20261011T052300Z/shared_verification.py` 的导入链、公开生成/人口、私有事实解析、重建评价与聚合；`lovelace-nuisance-20261011T044500Z/verification/arithmetic.py` 的 partition、closed union 和独立动作端点计算。

## 数学与执行核对

1. **观测、共享依赖和饱和语义正确。** `structured_solver.py:59` 从每 RGB channel 的必要半空间取交集；raw 可在 `[0,1]` 外。clip 0 只留 lower、clip 1 只留 upper，内部值保留两侧。`necessary_rows()` 将一 view 的所有 pixel 连接到同一 t，三个材料在所有视图共享；噪声没有错误地共享成单个变量。故意把两个像素拆开可行而共同 t 矛盾的测试实际覆盖共享机制。
2. **几何必要外包正确。** 非负 darkness 下，upper 使用面积 L、lower 使用 U；calibrated 分别减 `t_lo`/`t_hi`。历史几何通过整段 depth 的投影 Interval 和允许 camera 区间得到面积外包，rear visible upper 用 rear outer 减 front inner。新 solver 没有用生成 branch 或采样端点代替全域界。
3. **公开 mask 和共同 T_cal 正确。** `background_mask()` 在三种 branch 的完整 `[540,660]` 上检查三个面积 upper 均零；与允许 camera 误差的旧几何相接。两新方法走同一函数，校准只用这些像素及 t 先验。`_solve_branch()` 的 joint 初态恰为三材料先验加 T_cal，calibrated 固定同一 T_cal。正偏差 clip 白端保持不可辨识区间；空交集产生三分支 excluded_calibration，动作器返回 INCOMPATIBLE。核验器独立重建并显式比较两个方法的共同初始化。
4. **Fraction contractor 和证书正确。** `propagate()` 的 upper 行用当前 lower 计算其余变量最小和，lower 行用 upper 计算最大和；更新的是相反端点，因此当前行的 total 不会被同一行内更新失效。直接行矛盾、初始交叉和更新后交叉均有明确处理。排除保存初态、确定顺序更新及首个矛盾；保留节点保存最终区间与计数，核验器从初态独立重算完整传播后逐项比较。有限轮保留没有被写成精确 joint feasibility。
5. **树、分支和原动作器保持。** 新 BFS 对三个全场 branch 各自二分，以闭区间 union 汇总，原 midpoint/projection 端点 gain 动作不变。注册域宽 120，容差 `15/32`，完整深度 8 的二叉树是 256 叶、511 节点；因此本次满预算下 verifier 拒绝 unresolved_budget 是正确的，不会错误拒绝合法的注册树。测试中较小预算的 solver 仍保守保留未决盒；正式 verifier 只接受注册 511 预算。
6. **没有真实偏差或其他私有输入进入新求解。** `run_structured.py:186` 仅传 image、radius、method、clipped 与注册参数。实际 bias 只在 `private_truths()` 为评价加入；生成 branch/material/depth/seed/offset 不传入科学接口，incumbent 只进入冻结动作器。导入旧模块未在顶层解析 truth。旧 truth 文件在 input lock 中只被按字节校验；正式私有解析发生在动作栅栏之后。
7. **3888 动作完整保存/读回后才评价。** `run_structured.py:374` 检查总数、唯一 `(task_id,method)` 和保存文件读回相等；376 行检查四方法各 972 与 reused 身份，378 行才调用 `private_truths()`。核验器先读完整唯一 saved action population，再调用私有生成核验。其 AST 审计与源码阅读一致；这是本地流程顺序保证，不是操作系统隔离证明。
8. **wide 是精确历史复用。** 公开人口来自 RBdelta/CBdelta，与同初态绑定；历史外包完整 body、time、原动作 body 均比较，原图/外包/动作采用已有 canonical JSON hash 记录。参数要求 epsilon+beta=.02、8 轮、511 盒、相同 tolerance，前批指定 host/output 的 COMPLETE/VERIFY PASS 被校验。复用外包不计成新求解或新确认。
9. **矩阵、评价、成本和窗口一致。** 108 图像×2 新方法=216 新求解；972 tasks×4=3888 动作；1944 新方法动作、972 wide 复用、972 identity 分开。评价使用实际 depth，逐初态配对方向与 AE/MSE gain 正确，覆盖用保留集合而非仅 hull；生成分支和 latent 材料/t 覆盖只在评价计算。模板、seed、正确/负控、INCOMPATIBLE 分母均在独立核验重建。冷 solver 时间含背景准备，历史 solver 时间另列；同树预算未冒充同 CPU 时间。核验器读原 BUDGET.deadline，SIGALRM 只使用剩余时间，不重置 5400 秒。

## 测试与审查者只读检查

已完整查看测试源码和以下实际日志：

- `refine-logs/M0_AUTHOR_TESTS.log`：23 tests，OK，1.627 秒。
- `refine-logs/M0_VERIFIER_TESTS.log`：20 tests，OK，0.122 秒。包括错误 mask、clip、row/update/prior/crossing、树父子/计数/orphan、来源 hash、missing action 和预算的拒绝测试。
- `refine-logs/M0_SMOKE.log`：保留的初次失败是字符串 prefix 被位置参数误传为 radius；最新 runner 已改为 `prefix=`。
- `refine-logs/M0_SMOKE_V2.log`：seed99125 的 raw/clip×calibrated/joint 四个非确认 smoke 完成，均保留 depth600 且正确动作 KEEP；mask 均48，四次求解约0.67–0.92秒。
- `refine-logs/M0_ALL_TESTS_INTEGRATION.log`：最终 44 tests，OK，6.669 秒。新增四夹具测试实际逐分支运行独立 `verify_arm_public`、`verify_branch`、`verify_union`，再检查 depth600 保留。

审查者另执行一个 `python -B` 只读、内存检查，未运行正式 solver 批次、未解析私有 truth 文件、未写测试工件：

- 单独导入 `verify_structured` 后确认 `structured_solver` 和 `run_structured` 均不在 `sys.modules`：PASS。
- runner `configuration()` 与独立 `registration()` 完全一致：PASS。
- runner/核验器的 70 个科学源文件清单及当前 hashes 一致，37 个历史输入清单及 hashes 一致：PASS。
- 旧 INPUTS/源锁/指定前批验证记录检查：PASS。
- 独立 source parse-order audit：动作读回 gate374、origin gate376、private parse378：PASS。
- 正式 `outputs/structured-v1` 不存在；本次审查没有将 smoke 当作主批结果。

## NON-BLOCKING 与结论范围

- `verify_structured.py:822` 的已有输出字段名 `new_nonkeep_actions` 实际指 1944 条 calibrated/joint 方法动作记录，其中状态可为 KEEP；最终报告应按此解释，不把1944写成实际 MOVE 数。最新协议已修正对应文字；无需为该字段名改科学源码或阻止 launch。
- C1 的 raw/clip 对 wide 比较同时包含公开校准与所注册观测语义的影响；纯共享依赖的有限增量应据 C2 的同 T_cal、同图像 calibrated→joint 配对来描述。不能把有限 synthetic 回放写成新场景迁移或独立真实相机证明。
- 机械 PASS 不要求正收益、全覆盖或零损伤这一科学结果预先成立；实际安全/负控/收益必须完整汇报。有损伤或 joint 无增量时按协议结算，不事后改阈值。已有 by_case_seed 数据足以在 M4 给出模板/seed 分组结果，无需扩批。
- 本审查和确定性核验均复用冻结几何；没有独立证明该几何的物理充分性。录制时间只能核对归属和求和，不能由日志独立复现 wall time。M2/M3 的正式完整核验仍须在原窗口内完成。

以上不构成待修 blocker。**最终判定保持 PASS / LAUNCH_ALLOWED：仅针对本次已列源码和协议的一个冻结主批。**

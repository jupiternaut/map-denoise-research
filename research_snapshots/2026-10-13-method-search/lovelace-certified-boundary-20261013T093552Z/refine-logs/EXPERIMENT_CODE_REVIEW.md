# A13 实现审查

日期：2026-10-13。审查路线：fresh agent，按 experiment-bridge 要求使用 gpt-6-astra / xhigh；same-family / provisional。此为实现审查，不是外部审稿接受，也不是正式结果验收。

## 结论

当前实现未发现阻止冻结和按已注册协议运行的剩余 BLOCKING 问题。数学上的排除链条在已检查的冻结模型内成立；正式批次的每个证书、分区、预测和指标仍须经过执行后的独立验收。本审查没有生成新确认集，没有运行或查看确认预测／真值，没有修改实验源码或祖先文件。

## 目标与实际读取范围

已核对主机 liekkas。唯一目标逻辑根为 `/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z`，现有 runs 符号链接解析到 `/mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z`，不是替代副本。

完整读取 experiment-bridge 技能、A13 AGENTS、EXPERIMENT_PLAN、research_contract、search_method.py、worker.py、run_experiment.py、cert_kernel.py、verify_certificates.py、三个测试文件、CERTIFICATE_METHOD、VERIFIER_SPEC、COMMANDS；补充读取新增 analyze_costs.py。只读核对 A12 的 TASK、forward、free/method、library/methods、data_builder，以及 A11 accumulation_solver、A9 continuous_solver、A10 tightening_solver、原有理数照片／anchor 生成与核验路径、baseline_geometry。开发测试读取获准的 A12 六世界 dev 和现有 A13 smoke；没有读取旧确认集。

## 数学与信息隔离核对

- 相机消元正确：中相机规范固定为 0，先由中视图约束共享 k，再构造侧相机必要区间；丢弃剩余相关性只会放宽可行域。
- 舍入处理正确：生成器 public_packet/float_tree 对有理数执行 float，再经 JSON binary64 往返；A13 将读数解释为相邻浮点中点给出的闭原像包围。clip 后与 [0,1] 相交，饱和方向只使用可靠单边约束。物理常数采用任务指定的有理数，搜索区间端点采用实际 binary64 的精确值。
- 几何包围正确：正深度上的投影边缘极值可由深度与相机端点组合取得。左右材料使用内／外矩形面积；可见后景使用 `Rinner \\ Fouter ⊆ Ractual \\ Factual ⊆ Router \\ Finner`。同一个后景矩形用于面积差两项，方向与遮挡次序正确。未把浮点 renderer 系数直接视作精确几何，也没有引入旧基线的 Lipschitz 收紧。
- 15 变量必要条件正确：d=1-c 非负；照片 b 与独立外观 ba 分开；coupon 的 transfer+noise 总界为 0.01；reference 界为 0.005；同材料跨视图差界为 0.1。照片下读数使用面积下界，上读数使用面积上界，符号正确。
- 证书判定正确：Phase-I LP 和其对偶量只提出正乘子；有理数核精确计算 `min_box(sum yA)x - sum yb`，只有严格正 margin 才排除。成立不依赖浮点最优性或浮点容差。独立核验器自行重建必要行，不依赖作者提供的矩阵，并校验 wrapper 的分支、端点和 margin。
- 搜索保留 A12 的两端次序、倒深度二分、0.025 mm 停止精度与 hull 投影。拒绝／超时证书转成保留终端单元；待处理单元保留。verify_state 按精确端点核验三个分支的闭覆盖、正长度重叠与 hull；显式全域回退不被冒充成正常无重叠分区。
- prepare 只收到 observation；decode 每次只收到一个 incumbent。worker 不导入生成器或评价模块；新确认生成在 freeze 检查后，score 在完整 756 预测及独立核验之后才加载确认真值。最新 diagnose/verify 读取公开 dev 文件，不再把含 dev truths 的 task_bundle/dev.json 整体载入。

## 实验忠实度、预算与统计

三臂确实是冻结 numeric、同搜索加精确确认的 certified、冷启动 A9→A10→A11 legacy。固定 18 世界、每世界 7 输入、6/24 秒两截面、三臂轮换顺序，得到 108 次调用／756 预测；评分按实际真值，正确输入、扰动、负控和世界数均单列。失败 identity 仍纳入预测分母；不以证书不可得推断原浮点排除错误。

认证上下文、浮点提案和精确认证时间在 prepare 总预算内。HiGHS 显式单线程及每次至多 0.20 秒；协作式截止后还由 worker 的 budget+2 秒硬限保护，父进程有限超时保护冷进程。2400 秒批次门限保留 INCOMPLETE 状态。process_seconds 从启动到退出，含导入、序列化与写出；创建输入 JSON 不含在内且 COMMANDS 已披露。独立重放费用单列，证书构造／精确检查分项及原始失败记录可追溯。新增 analyze_costs 只在 score 后读取结果，不影响推理或动作选择。

本次没有执行正式批次，因此不声称 108 次调用已完成、180 CPU 秒开发总账已最终结清、正式覆盖已通过，或 certified 已获得成本／精度优势。

## 本轮发现及已复核修正

1. 初读版本的文档序列缺少独立 verify 动作，score 可直接读取真值。主协调者已补充 verify，逐一核验 36 个 certified 保存状态和 dev 同提案重建状态，生成 INDEPENDENT_VERIFICATION.json；score 的 all_valid 门位于确认真值读取之前。当前源码已复核。
2. 初读 COMMANDS 用 unittest 导入 test_cert_kernel，却不能发现其函数式 run 测试。实际 discovery_count=0。最新 COMMANDS 已改为直接执行 test_cert_kernel.py；本审查也直接运行了这些性质与真实 dev witness 检查。
3. 保存状态的证书正确不能单独证明保存预测正确。最新 verify 已逐项核对 selected 等于保存 hull 的 clip 投影，或在 task failure／state=None 时等于 incumbent，并将 decoder_valid 合并到 valid。已检查当前表达式和现有 smoke 的 14 个预测。
4. verify 的 truth_loaded:false 原先伴随载入含 dev truths 的文件；现已使用 data/dev/public.json，与元数据和信息边界一致。

这些修正仅涉及执行／核验协议与文档，没有参与或更改数学方法实现。

## NON-BLOCKING 限制与报告要求

legacy 仍使用冻结 A12 library/methods.py 的 `Q(str(x))` 将收到的浮点读数十进制化；它没有 A13 的舍入原像包围。因此应称为“历史精确有理数求解器在十进制化读数上的基线”，不能声称三臂都有相同的输入传输至物理几何认证链。保留该历史实现符合既定对照；此限制不取消实测成本／误差比较，但报告须明确披露。主协调者已确认会保留此限定。

认证的是冻结合成物理模型内的必要条件排除；保留集合不证明存在真实可行世界，也不支持跨机制或真实图像推广。按计划报告全部差额，不补造非劣阈值或把 same-family 审查说成外部接受。

## 实际验证证据

使用指定 Python -B，四项线程环境变量均为 1；测试进程设置 25 CPU 秒上限。本审查两次开发检查的完整进程 user+sys 合计约 **1.60 CPU 秒**，低于分配的 30 CPU 秒；没有额外 holdout。

- kernel：180 个精确面积包围样本、六个真实 dev 可行 witness、三个远端正排除证书、独立 b/ba、饱和／全白、零预算等全部通过；测试体 0.839 CPU 秒。
- verifier/harness：12 项 unittest 全部通过，含篡改、可行点、一个 ULP 的覆盖缺口／重叠、强制拒证不剪枝、零预算、冻结 numeric 等价；报告墙钟 0.417 秒。
- 保存的 certified smoke：独立 verify_state 通过 **14 个排除、10 个保留单元**，三个分支闭覆盖与 hull 一致；该次核验 0.00791 CPU 秒。
- 当前 Python 源码 AST 解析通过；numeric 和 certified smoke 的 **14 个保存预测**均与其保存 hull／失败 identity 规则一致。

审查者仅写本文件及指定私有 trace 目录；MANIFEST 和总开发账由主协调者汇总。最终正式结果审查应另起 fresh context。

## 追加审查：冻结清单导入冲突修补

2026-10-13，same-family / provisional。结论：**接受当前 freeze 接线修补，可重试冻结；无剩余阻塞项。** 这不提前验收新确认实验或正式结果。

重新核对 liekkas 与同一逻辑／物理目录后，读取了当前 run_experiment.py 第 41–66 行、DEVELOPMENT_CHECKS.md、search_method/worker 的冷加载入口及此前核对的祖先路径约束。原故障来自生成器和 legacy 在同一解释器中加载不同文件但共享 baseline_geometry 模块名。当前修补让生成器依赖留在父进程，legacy 依赖在指定 Python -B 的独立冷子进程中枚举；子进程同样先 import search_method，再 warm_load('legacy')，与正式 worker 的方法加载次序一致。旧求解器的 exact-path guard 仍执行，没有替换或修改祖先几何。父进程合并两边实际文件路径后再写冻结清单；子进程失败会由 check=True 直接阻止写出，且有 30 秒超时。

本次直接执行了完整当前 freeze 函数，**仅将最后的 save 替换为内存捕获**；没有改实验源码、没有写 SOURCE_FREEZE、没有调用生成或预测。捕获清单含 40 个源文件，逐项重算与清单现有摘要一致，同时包含生成器使用的根 baseline_geometry.py 和 legacy 使用的 recovery-1/baseline_geometry.py，并覆盖 data_builder、free/method、library/methods、cert_kernel 和当前 run_experiment。实际导入成功，旧路径 guard 没有触发。检查结束时 SOURCE_FREEZE.json 与 data/confirm 均不存在。

只读预检完整进程费用：real 1.40 秒，user 0.86 秒，sys 0.06 秒，包含冷子进程；其中函数内记录父 CPU 0.4392 秒、子 CPU 0.3902 秒。连同上一轮 reviewer 检查约 **2.52 CPU 秒**，仍低于分配的 30 CPU 秒。40 文件清单是本次内存试运行的实测结果；下一步正式冻结的实际输出仍应由主协调者核对并记录。

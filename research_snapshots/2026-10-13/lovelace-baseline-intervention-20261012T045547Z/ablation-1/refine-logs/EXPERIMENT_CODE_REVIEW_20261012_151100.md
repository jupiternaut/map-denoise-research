审查结论：**当前源码未发现阻塞性缺陷；可进入精确路径测试通过后的冻结与正式执行。** 这是复用上下文审查，不是 fresh/external review，也不是科学结果验收。我未修改代码或运行测试、实验；完整测试套件的执行结果由 root 单独记录。

审查目标已核实为 `liekkas` 上：

`/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1`

**BLOCKING**

无剩余已发现的源码阻塞项。正式结果仍须通过本阶段独立核验；本结论不预告消融方向或收益。

**已确认正确**

- **掩码与行身份**：[ablation_solver.py:56](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/ablation_solver.py:56) 固定原像素编号；第70行起按 frozen 必要行相同的 upper→lower 顺序记录裁剪后实际存在的侧；第115–119行先生成完整必要行再过滤。图像、几何和 view-bias 索引没有压缩或错位。证书行号正确对应过滤后的 contractor 行序。
- **校准保持**：同文件第165–173行始终使用完整432标量图像与公开背景掩码计算校准，随后才把删行约束送入分支求解。runner 检查校准与封存 full 完全相同；verifier 独立重建并再次比较。该操作确实是“关闭推理行”，不是减少采集。
- **动作与私有评价顺序**：[run_ablation.py:137](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/run_ablation.py:137) 检查完整唯一矩阵；第263行起产生1944个 selector replay，第271行起产生144次冷求解及1296个 masked 动作。两矩阵保存、读回并检查后才调用 `evaluate`；私有标签解析仅用于此后评价。预先读取私有文件字节计算哈希不等同于解析标签。
- **选择器身份**：runner 第106–125行的端点增益、严格 MOVE 条件、midpoint-first 与 projection fallback，与历史独立算术一致。[verify_ablation.py:605](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/verify_ablation.py:605) 独立重算三个选择器，并逐项比较648个 hybrid replay 与封存动作语义。
- **独立核验范围**：verifier 第84行起独立生成掩码不等式；第135行起重算每节点、传播更新、排除证书和完整树；第221行起检查校准、union/hull 与成本记录。第617行之后才解析私有真值；第642–651行核对 masked、selector、reference 的全部 JSON/CSV、coverage、汇总值和分母。未发现将科学收益或负结果作为机械 PASS 条件。
- **源文件、输入与时间**：verifier 第279–314行使用既有独立源图与封存输入检查；新阶段源图包含本目录 Python 文件。runner 第250–253行已改为只启动绝对截止前的剩余时间，消除了初版“哈希后再计3600秒”的偏移；verifier 继续使用相同截止时间。
- **测试目标修复**：[test_suite.py:11](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/test_suite.py:11) 先将三个精确源文件绑定到 canonical `sys.modules`，再以独立模块名加载精确测试文件，并检查 `__file__`。这修复了历史 `sys.path` 导致同名旧测试被加载的实际问题。
- **篡改夹具修复**：[test_verify_ablation.py:180](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/test_verify_ablation.py:180) 现在追加伪造传播更新，确保发生实际变更；原来把已经为空的 `steps` 赋为 `[]` 是测试无操作，修正合理，未改变科学算法。

**NONBLOCKING：最终报告必须保留的区别**

1. runner 第153行与 verifier `make_result` 的 `covered` 表示 **hull 包含**；`coverage.json` 的 `covered` 表示 **retained union 包含**。当前汇总不使用前者推导 coverage，因此未形成数值错误。最终真值/分支覆盖结论应引用 `coverage.json`。
2. 当前 verifier 核验了足以计算 A1/A5 的逐任务数据和分层汇总，但没有输出最终 paired better/same/worse、有限 outer 嵌套及 A1/A5 因果差分表。这些应在结果解释阶段明确计算，不应把机械 PASS 当作这些科学结论已经成立。
3. `matched_drop` 匹配视图、行和像素数量，协议正确声明没有匹配信息量。不能把其结果解释成排除了所有“任意删行”的替代解释。

**科学解释边界**

A1 的有效对照是固定完整校准、其余算法和动作规则后的指定行条件贡献；A5 是同基线 cal/joint 增量的交互，均不能自动分摊为物理原因比例。

A7 是对**新 cal/joint 嵌套效果**的历史选择器重放控制。历史 [test_projection_policy.py:88](/srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z/test_projection_policy.py:88) 已明确展示 midpoint 可以比 projection 更准确。这里可检验的数学性质是：truth 被覆盖且 \(H_{\rm joint}\subseteq H_{\rm cal}\) 时，projection-only 的 cal→joint 误差不增；它不保证 projection 优于 hybrid，也不适用于未证明嵌套的 B70/B80 比较。当前审查没有观察到任何新实验改善。

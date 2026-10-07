# 实验完整性审计

- 审计目标：`/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z`，实际主机 `liekkas`。
- 审计者：fresh agent `/root/selector_attribution_audit`，GPT-6-Astra / ultra。
- 保存时间：2026-10-07T18:59:44Z。
- 审计路径：`experiment-audit`；`review_independence: same-family`；`acceptance_status: provisional`。
- **总体结论：WARN。** 已执行的确定性复算全部通过；未发现伪造GT、自输出归一化、缺失结果或未执行的政策臂。警告针对报告的一处证据范围省略，以及两旧场景、有限支持、指标与物理层含义不同等实质性解释边界。这不是外部家族验收。

## 独立复核与范围

完整读取锁定策略、推理/评价代码和所列主要文档；完整解析全部输入、决策、指标及其所用历史数据，不依赖作者对结果的口头总结。新增验证代码不导入作者选择器、收益核、评价器或PLY读取器；采用对每个区间端点直接计算精确有理数三维平方距离差的方法，独立复算选择和保存的上下界。

已核验：

| 检查 | 结果 |
|---|---|
| 当前四份输入/预测/评价清单 | 26/26条哈希绑定匹配，21个不同路径 |
| 递归历史及评价输入绑定 | 77/77匹配，47个不同被绑定路径，24份JSON；无缺失、冲突或无法解析字段 |
| 政策输出 | 21请求×16臂＝336个选择全部一致；保存的新臂收益上下界、目标、eligibility也一致 |
| 历史复现 | old photo 21/21；star_full/star/cycle的R 63/63 |
| 坐标 | 105 proposal＋18 incumbent，浮点二进制表示逐坐标一致 |
| 完整逐点与转移表 | 17×512＝8704行逐项一致；68条改变ID的转移一致 |
| 指标与归因 | 全部17臂、每ROI/场景、分母、损伤/大错/缺失计数和5组算术分解一致 |
| 原始激光重新查询 | 614坐标行（509 incumbent＋105 proposal）；与缓存距离最大差0 mm |
| 独立穷举距离 | 全部123 active对象对每个原始激光顶点穷举；最大差0 mm |
| 冻结作者单测重执行 | 28/28通过 |

激光顶点数分别为scan118的3,086,735和scan122的2,731,197。按ROI的原始CPU有效数为119、116、123、126，总计484；CPU缺失28另列；每臂有限输出509，其中缺失层25。另直接读取历史CPU NPZ逐项核对全部512个primary标记。

复算工件：[recompute.py](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/recompute.py:1)、[RECOMPUTED.json](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/RECOMPUTED.json:649)、[HASH_BINDINGS.json](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/HASH_BINDINGS.json:1)、[SYNTHETIC_TESTS.txt](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/SYNTHETIC_TESTS.txt:31)。

## A. GT来源：PASS（限定评价含义）

原始参考来自历史下载清单记录的DTU官方Points.zip成员stl118_total.ply / stl122_total.ply，非本模型预测或其他baseline输出。文件SHA-256与清单一致；本审计重新读取两份完整PLY并复算距离。原评价确实读原始PLY、构造最近邻树、对每个冻结位置查询，当前评价则按exact coordinate复用这些结果。

证据：

- [参考清单](/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z/references/MANIFEST.json:4) 与 [第二场景来源](/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z/references/MANIFEST.json:13)。
- [原始评价读GT与查询](/srv/slam-research/grf/map-denoise/runs/surface-evidence-20261001T140058Z/evaluate.py:295)；[候选查询](/srv/slam-research/grf/map-denoise/runs/surface-evidence-20261001T140058Z/evaluate.py:323)。
- [当前精确坐标核对](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:189)；[距离来源选择](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:207)。
- [审计原始GT复算](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/RECOMPUTED.json:1398)。

这里的`real_gt`只表示参考数据来源。评价量是“到原始激光顶点的最近距离”，不等于官方全点云benchmark，不包含完整性评价，也不能识别照片匹配是否属于正确物理表面。作者已经在[结果声明](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/RESULTS.json:871)和[报告](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/REPORT_20261007T184027Z.md:21)说明这些区别。未重新从互联网下载数据；官方来源的历史认证依赖保存的下载清单，当前审计验证其字节身份与实际计算链。

## B. 归一化与分母：PASS

主指标为四个ROI的MSE等权平均，ROI内部各自按冻结的原始CPU有效人口取均值；不是把484点全部混成一个不加区分的均值，也不是只评价发生移动的点。MAE同样ROI等权。原始mm/mm²值完整提供；百分比使用共同冻结photo基线23.95525131520087 mm²作分母，没有按各新臂自身最大值或分布重新缩放。

证据：[协议主次指标](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/PROTOCOL.json:24)、[指标代码](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:159)、[百分比公式](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:223)、[photo逐ROI分母及原值](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/RESULTS.json:5)。

所有臂主分母484、有限输出509、缺失层有限25/28均保持一致，491个非active请求保持photo的选择/距离。全部三缺失active请求保留null。改善/恶化计数遵守1e-9 mm容差；因此old_raw中的若干极小数值变化也会计数，不能直接解释为有实际尺寸意义的修复。例如[old_raw微小变化](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/TRANSITIONS.csv:65)只有约1e-7 mm量级。

## C. 结果存在性、完整性与封存：PASS（时间证据有限）

全部16个政策臂以及photo基线存在，全部8704行、68条转移、17臂全指标和5组归因都已从原始行及祖先选择独立重建。报告表格显示精度内的数值、两种百分比差值、点计数和案例距离与工件相符。没有未报告的预定格子；star_full的Q/QG不是预定格子。

证据：[矩阵](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/PROTOCOL.json:8)、[所有臂执行](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:133)、[评价全部臂](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:200)、[报告数表](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/REPORT_20261007T184027Z.md:27)、[跟踪状态](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/EXPERIMENT_TRACKER.md:5)。审计/报告行仍为IN_PROGRESS，符合这份报告明示“待审计”的状态，不是把尚未完成的审计谎报为完成。

当前记录的UTC时序为2026-10-07：

| 事件 | created_at |
|---|---|
| 输入保存 | 18:38:53.142479 |
| RUN_LOCK | 18:39:31.688576 |
| DECISIONS | 18:39:31.831865 |
| 预测seal | 18:39:31.839868 |
| RESULTS | 18:39:50.175628 |
| 评价seal | 18:39:50.194634 |

mtime同序；[锁](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/RUN_LOCK.json:2)、[预测seal](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/PREDICTIONS_SEALED.json:2)、[评价seal](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/SEALED.json:2)的绑定全部通过。代码先锁验证再推理，预测落盘后再seal，再由独立evaluate入口读取GT缓存；没有发现冻结代码变化。已绑定的历史文件与保存哈希相符，审计前后复算所读文件也未变；这不等价于证明整个历史目录从未发生任何未绑定修改。

这些都是本地主张和可重算哈希，不是不可篡改外部时间戳；无法仅凭它们证明作者此前没有见过成绩、原始单测在锁前的精确执行时刻或任何其他进程绝无GT访问。该设计本身已声明是暴露后的回溯实验；不应把技术封存包装成盲测证据。

## D. 死代码、政策忠实性与GT隔离：PASS

P/GP/R/Q/QG和三个旧规则对照在实际推理循环中均被调用。独立复算覆盖273个新策略选择、63个旧策略选择；空支持、缺失current、KEEP、tie、候选范围、Q轨迹选择和投影、正收益门槛均符合协议。

- [P/GP/R与Q/QG执行定义](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/policies.py:119)、[门槛和排名分离](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/policies.py:160)、[旧门槛及删除](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/policies.py:204)。
- [旧真实choose规则](/srv/slam-research/grf/map-denoise/runs/surface-evidence-20261001T140058Z/surface_kernel.py:336)；[U11到旧choose的适配](/srv/slam-research/grf/map-denoise/runs/official-mechanism-20261001T183124Z/patch_kernel.py:73)。
- [测试区分P/GP/R](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/test_policies.py:81)、[Q/QG区别](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/test_policies.py:117)、[旧门槛边界](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/test_policies.py:137)、[实际测试输出](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/SYNTHETIC_TESTS.txt:31)。

GP与R在这批所有请求上同ID是实测相同，不是实现空壳；合成例能让两者选出不同候选。QG和R在star的#75也确实不同。

输入allowlist已对每行和每个嵌套对象独立验证：row仅scene/roi/query/primary/current_id/center/ray/intervals/objects/original_id/tracks；object仅candidate_id/xyz_mm/depth_mm/old_score；track仅id/reference_depth_mm/ncc_reference_sources/intervals_mm。全部都可逐项追溯至冻结GT-free祖先文件。primary是原始CPU有效标记，不是GT误差标签，并且不参与策略排序。历史复现目标只用于输出断言，不进入选择器。

证据：[输入提取](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:66)、[复现断言](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:141)、[GT读取阻断器](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:57)。审计还单独实测该阻断器拒绝当前POINT_METRICS、RESULTS以及两份原始PLY的打开请求。它是进程内路径阻断器，并不是敌手不可绕过的系统沙箱；实际选择器静态数据流与完整allowlist没有显示GT输入。已暴露数据上的设计影响仍属于E项限制。

## E. 研究范围与归因解释：WARN

1. **需修正文案的证据范围。** [报告第62行](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/REPORT_20261007T184027Z.md:62)“两次大修复…P、GP、R、Q、QG都选择同一个历史候选”缺少star限定。这个叙述对star系列成立；star_full的P/GP/R也成立。cycle在#5和#95没有支持，所有cycle策略均KEEP，绝没有完成这两次修复。证据：[star两点转移](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/TRANSITIONS.csv:18)、[star Q/QG](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/TRANSITIONS.csv:41)、[cycle全部转移](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/TRANSITIONS.csv:34)、[query5 cycle空支持](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/INPUTS.json:45)。建议在新版本报告中明确补上证据系列，保留原稿不覆盖。类似#60/#75案例讨论也应统一明确star。
2. **只支持这批回溯条件对比。** 两个已暴露场景、四ROI、每格一次确定性执行；真正有非空支持的star_full/star/cycle只有6/6/3点。没有独立确认、跨场景泛化、随机重复不确定性或部署验证。总计512请求不是512个独立的有信息处理效应。证据：[协议范围和限制](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/PROTOCOL.json:3)、[支持人口](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/RESULTS.json:706)、[报告回放边界](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/REPORT_20261007T184027Z.md:11)。
3. **算术分解成立，但不是因果模块百分比分账。** star/P路径为7.233013651037616 − 0.13464136086766132 + 0 = 7.098372290169955 mm²；star/Q路径为6.951108134019179 + 0.00005505569908947905 + 0.14720910045168623 = 7.098372290169955 mm²。参照改变时增量变化。五组恒等式均复算通过。证据：[ATTRIBUTION.json](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/ATTRIBUTION.json:13)、[Q路径](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/ATTRIBUTION.json:22)、[解释声明](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/run_ablation.py:248)。报告展示到6位小数的等式应理解为近似，可使用“≈”避免舍入后相差1e-6 mm²的字面等号问题。
4. **旧/新接口不构成干净二因素设计。** 新臂从photo开始且可选择全部坐标，包括原incumbent；旧删除门槛臂从原始incumbent开始，只收有限旧NCC分数候选，同时存在缺失current例外。patch/视图预算/评分也不同。协议已明确这些差异：[PROTOCOL.json:22](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/PROTOCOL.json:22)。旧门槛删除变差不能单独证明“新观测的因果贡献”。
5. **目标“保留正确几何”尚未实现。** star/P与star/R均把#48从0.508移到2.734 mm、#75从0.985移到6.203 mm；新伤害各2。star/R在scan122的MSE从7.517608150133937上升到7.6993937103058。gate的假定射线集合平方收益不是最近激光误差保证，更不是正确表面归属。证据：[伤害转移](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/TRANSITIONS.csv:21)、[R伤害](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/TRANSITIONS.csv:32)、[R场景MSE](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/RESULTS.json:113)、[定理前提](/srv/slam-research/grf/map-denoise/runs/track-discrimination-20261007T160716Z/theory/kernel.py:51)。

报告已经明确多数上述边界，正确揭示P超过R以及门槛可能挡好点。应保留这些限制，不能据本次结果把P升级为默认部署策略，也不能宣称数学构造普遍无效。可支持的决策是：本轮固定证据/候选上，复杂排序没有比P提供正增量；先结束这次消融，不追加事后有利政策。

## F. 评价类型：PASS

规范分类为 **`real_gt`**，子类型为 **复用exact-coordinate的原DTU激光最近顶点距离**。结果文件使用的更详细标签`real_gt_reused_exact_coordinates`可映射到此分类。

照片轨迹、阈值支持集合和正gain是无GT构造的选择证据与模型条件结论，不是新ground truth；它们不能取代真实激光误差或物理层标签。未发现把模型生成参考当真实GT评分的情况。合成单测属于实现验证，而不是性能实验或额外real_gt样本。

证据：[结果类型](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/evaluation/RESULTS.json:871)、[策略kernel条件](/srv/slam-research/grf/map-denoise/runs/track-discrimination-20261007T160716Z/theory/kernel.py:3)、[纯合成测试声明](/srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/test_policies.py:1)。

## 必要修正与结论边界

- 新报告补全第62行及附近案例的star限定；不改锁定协议、预测或原评价。
- 保留4 ROI等权、484主分母、28缺失分列、6/6/3支持人口、两场景暴露回放、实际伤害、非官方benchmark和非因果分账等限制。
- 区分“当前冻结文件和计算可重现”与“历史执行先后可被独立外部证明”。旧目录无改动的机器可核验范围仅限有历史哈希绑定的文件。
- 此审计没有提出新政策、结果后调参、追加臂、安装或GPU任务。写入仅限audit/和授权私有trace目录。
- 全部确定性核验PASS可以作为此次冻结回放计算正确的依据；语义审计仍为same-family/provisional，不能标作外部或跨家族接受。

## 复核命令与本审计执行记录

```sh
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B /srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/recompute.py
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B /srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z/audit/hash_bindings.py
```

脚本采用exclusive-create保存独立结果；现有结果存在时不会覆盖。recompute首次运行在独立PLY解析器处理CRLF头部时中止，发生在写任何复算结果前；仅修正audit内解析器后成功完成，未触及原实验代码或结果。此故障不被计为原实验失败。

完整原始审计请求、最终回复和调用元数据位于私有`.aris/traces/experiment-audit/2026-10-07_run01/`。原始调用时刻未记录，trace按实际保存时间标记，不编造调用持续时间。



## 执行者落实

审计原稿原文保留。范围省略已在REPORT_20261007T190028Z.md/REPORT.md补全，四舍五入等式改近似号，补充old_raw微小计数解释。未修改冻结计算与指标。审计仍为WARN、same-family/provisional，未由执行者升级判决。

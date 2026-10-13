# 确认前代码复核（同模型族，provisional）

Fresh code review 已完成：总判定 PASS。BLOCKING: None。NON-BLOCKING: None（未发现需要改代码才能执行已冻结确认的缺陷）。这是确认前source review，不能标成已完成135/1215/6075全确认独立验证。

身份/边界：hostname 实测 liekkas；realpath 核实新目录 /srv/slam-research/grf/map-denoise/runs/lovelace-hybrid-20261010T214217Z 与只读依赖 /srv/slam-research/grf/map-denoise/runs/lovelace-challenge-20261010T205013Z。实际 import 的 challenge、run_challenge、layered_model、continuous_world.interval_model/interval_solver 路径全部在该旧目录。最后小型算术检查输出 confirmation_exists=False。全程未编辑文件、未运行confirmation、smoke或GPU；所有Python使用-B/PYTHONDONTWRITEBYTECODE。

证据（以下N=新实验绝对目录，D=只读依赖绝对目录）：
1. N/hybrid.py:19-24 正确先调用midpoint，仅MOVE时原样复制全部字段再添加branch；否则投影fallback。D/source_snapshot/continuous_world/interval_solver.py:81-95 严格gain>0、空外包INCOMPATIBLE及KEEP零收益正确；D/challenge.py:27-35 正确投影。N/THEORY.md:49-78 正确区分无条件的输出保留/MOVE集合等价与t∈H条件下逐任务AE/SE不增，并给出不支配projection反例。
2. N/run_hybrid.py:29-38、89-95、210-233 与协议全部参数一致：五模板，6217/11、7864/13、8441/13，半径0/1/5/1，41001/2/3，hybrid-v1，未截断1/100噪声与五档权重；135配置、1215初态、6075动作。D/layered_model.py:31后深度720；D/interval_solver.py:26默认域540..660、15/32、511且未决盒保留。新预测在新观测前重算，没有读取历史outputs/forecast缓存。
3. N/run_hybrid.py:98-111、238-261使用相同观测/预算，一次outer用于三证书。D/challenge.py:15、38-92确为540..660步1/2加初点、左右五档、l∞残差，参考相机固定0；分相机最小化等价25组合搜索；1e-12优先KEEP，否则最近初点/较小深度tie规则。源码明确有限网格而非连续全局最优。
4. N/run_hybrid.py:98-105选择器无truth/seed/state_group参数，循环只读观测和初点；155-174以真实truth算AE/gain与state_group，269-272在全部6075已写盘并读回后才评价。没有按state_group选动作或用真值调参数路径。
5. N/run_hybrid.py:114-151、181-192及D/run_challenge.py:76-101正确计算paired、六类汇总、information strata、improved/harm/moved、MAE/MSE。独立配置算术结果：tasks1215、correct135、small810、large270、quarter_offset270、informative972、hidden243、inside_prior1080、outside_prior135。PROTOCOL:17/34明确种子是重复噪声而非独立场景；:42-43正确规定C2 small严格多修复且MAE严格更低，并保留负结果。隐藏层实现D/layered_model.py:212及理论均为几何导出的深度无信息；N/verification/verify_results.py:314-317检查正确证书KEEP和hidden inside_prior KEEP。
6. N/run_hybrid.py:5-8限制CPU数值库线程；281-295以1800秒SIGALRM包住整次运行、超时/RuntimeError标记INCOMPLETE并重抛；D/run_challenge.py:30-55按整观测提交记录避免半份JSON。新输出默认N/outputs/confirmation-v1，79-86拒绝已存在输出/旧目录。265-267成本记录包含共享outer、新forecast compile、grid score及逐初点渲染选择；不能只报轻量动作时间。
7. N/verification/verify_results.py独立调用旧算术helper，不导入选择器/runner；252-357检查完整人口、关联、全部几何评价行、完整trace分割/每个记录排除证据的噪声冲突、135真值保留叶覆盖、全部证书动作/gain/branch、paired及所有汇总；CSV/JSON一致性在405-409。source清单runner与verifier独立枚举相等。现有verification tests确实为14项，不是缺文件。

实际执行：
A. 新目录 unittest：test_hybrid全部15 + verification.test_verify_results全部14 + test_run_hybrid中registered_new_configuration、random_tape_uses_new_prefix_and_exact_registered_weights、source_lock_is_absolute_and_covers_all_imported_old_sources、old_helpers_keep_their_original_root、selector_has_no_evaluation_inputs_and_integrates_all_hybrid_branches共5 = 34项，0.293s，全部OK。
B. 旧目录只读已有unit tests：test_projection_policy、test_challenge_baseline、test_layered_model、continuous_world.test_interval_solver、continuous_world.test_interval_model共64项，2.473s，全部OK。
总计本人实际98项通过。其余4项runner tests会创建临时文件，按本次严格只读要求仅完整静态审查，未把根代理的38项运行报告当作本人执行。

完整阅读覆盖：N的AGENTS.md、PROTOCOL.md、refine-logs/EXPERIMENT_PLAN.md、idea-stage/docs/research_contract.md、hybrid.py、run_hybrid.py、THEORY.md、test_hybrid.py、test_run_hybrid.py、verification/verify_results.py、verification/test_verify_results.py；D的challenge.py、run_challenge.py、layered_model.py、source_snapshot/continuous_world/{interval_solver.py,interval_model.py,THEORY.md}、THEORY_CHALLENGE.md、LAYERED_THEORY.md，以及上述5组旧tests和旧verification中被复用的算术/trace/聚合/model helpers。

必须保留的能力边界（非缺陷）：新verifier:388-391已披露，残差检查网格归属、KEEP tie一致性与真实几何结果，不重新证明全部有限网格最优；所有记录witness检查噪声冲突，但只抽样重算共享enclosure；成像模型合同没有独立重证；runtime只有有限非负性检查、不是独立benchmark。一次只读Git元数据探测发现/srv/slam-research/grf/map-denoise非Git checkout，所以来源commit按给定身份及本地来源声明核对，未宣称从Git对象重新认证snapshot。当前C1经验检查与C2结果仍pending，不能用本review替代确认后VERIFY。

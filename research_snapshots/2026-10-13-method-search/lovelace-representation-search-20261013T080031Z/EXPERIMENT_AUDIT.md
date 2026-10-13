# A12 独立实验审查

日期：2026-10-13。审查者：gpt-6-astra / ultra，fresh fork none，任务 /root/a12_integrity_review。
目标主机：liekkas。逻辑根：/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z。
物理根为既有 runs 符号链接下的同一目录 /mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z，未使用替代工作区。

## 总体判定：WARN；已保存结果的算术核验：PASS

未发现虚构真值、自归一化、丢弃失败分母、已保存预测与评分不一致或方法读取确认真值的证据。结果支持本成像族内的误差—成本比较。它不支持两个预算截面上的全面精度胜利，也不能把全部速度差异归因于一种新的数学表示。

审查类别为 same-family，acceptance_status 为 provisional；确定性算术核验单独记为 PASS。这不是跨模型家族的独立认证。

## 直接复算结果

主实验包含18个世界、126个唯一现任任务、108次冷进程调用和756个保存预测。每个臂与预算包含108个扰动现任、18个正确现任。下表误差仅针对108个扰动现任；耗时为该组18次进程总和。

| 方法 | prepare预算/世界 | MAE mm | MSE mm² | 改善/不变/恶化 | 进程墙钟总秒 | 进程CPU总秒 |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 6 | 1.7605777721 | 5.7882201032 | 75 / 33 / 0 | 81.084260 | 81.058917 |
| library | 6 | 1.7342497845 | 5.6980581233 | 76 / 32 / 0 | 62.296475 | 62.275493 |
| free | 6 | 1.7219088542 | 5.6686799167 | 76 / 32 / 0 | 5.230546 | 5.212767 |
| fixed | 24 | 1.7006849261 | 5.5799895712 | 77 / 31 / 0 | 140.316764 | 140.284101 |
| library | 24 | 1.7263152142 | 5.6626899257 | 76 / 32 / 0 | 66.011165 | 65.969991 |
| free | 24 | 1.7219088542 | 5.6686799167 | 76 / 32 / 0 | 5.288890 | 5.281386 |

共同identity的扰动现任 MAE=4.3001305556、MSE=23.8201476131。六组均保留全部18个正确现任，最大正确输入偏移为0；18/18真实深度均在相应保存hull内。没有进程失败、预测失败、缺失prepare CPU记录、超过预算0.25秒的prepare、free内部fallback或free LP失败。

6秒截面free相对fixed的扰动MSE下降约2.065%；24秒截面则上升约1.589%。free与library在24秒截面存在不同指标的取舍：free MAE稍低，MSE稍高。因此不能称free在两个截面上全面更准确。每个世界7个现任共享一次观测与prepare；756个输出不是756个独立场景。

独立脚本从receipt/result/job与私有模拟真值重新计算全部SUMMARY字段、COSTS字段、METRICS.csv单元格、配对比较、hull解码及覆盖率，与保存结果匹配。未导入原评价函数，也未重跑任何方法。证据：independent_audit.py，以及 .aris/traces/experiment-audit/2026-10-13_run01/arithmetic-v2.json。首次复算保留在同目录arithmetic.json；v2增加了成本组成、legacy原子分类耗时和补充输入一致性检查。

## A. 真值来源与信息边界：PASS

真值由生成器在预测之前抽样确定，不来自任一方法输出。data_builder.py:119–165从新有理数深度、材料、相机偏移和独立误差构造世界；141–157同时核验独立精确照片、公共浮点前向与anchor方程。GENERATION记录生成时间08:26:33.126889 UTC，晚于SOURCE_FREEZE的08:26:32.037403 UTC；最大前向/观测差异为6.661338147750939e-16。

公共观测只含photo、camera、appearance；全部18份公共记录和每个job的观测已逐项匹配。data_builder.py:89–91定义白名单，worker.py:35只把观测与预算交给prepare，worker.py:46–55逐次只把一个incumbent和状态副本交给decode。task_id及成组incumbent存在于执行器job中，但已审查的方法调用未把它们交给prepare或decode。库适配器使用固定内部ID，library/methods.py:25–29不提供真实世界ID。

照片bias与appearance bias独立。free的15变量约束、饱和单边界、跨视图材料范围、相机标靶误差消元与给定物理模型一致；没有加入额外观测、隐藏材料标签或深度目录。库臂的全局depth选择由开发集冻结，arms/library/method.py:20–36不作确认结果选优。

run_experiment.py:95–121只生成预测，156–164在完整108次/756个预测门槛及已存结果校验后才读取truth。审查者也只在协调者明确确认主实验、补充预测和评分完成后首次读取私有truth。信息隔离为代码与角色约定，不是OS沙箱；本审查不把它描述为访问控制证明。

## B. 分数归一化：PASS

run_experiment.py:123–134计算原始坐标MAE/MSE。百分比只以共同identity的初始MSE为分母；没有以模型自身预测的最大值、最小值或均值构造夸大分数。原始误差、正确现任损伤、全部126个输出和108个扰动任务均分别保存。独立复算一致。

## C. 文件存在、分母与冻结：PASS；交付记账：WARN

主实验108个receipt/result/job及补充3个receipt/result/job均存在；主实验每组18个世界、126个任务完整且唯一。保存输出与任务、公共观测逐项相符，失败未被删去。本次没有实际失败，不能据此声称故障路径得到实测覆盖。

SOURCE_FREEZE列出的14个文件全部匹配，且主实验ready和GENERATION引用同一冻结文件摘要。只核验已有冻结/receipt不变量，没有新增冻结方案。此结论明确限于清单列出的14个本轮文件；从绝对路径导入的历史算法不在这14项中。审查已直接检查相关历史导入路径保护和算法源码，但没有宣称本轮清单覆盖了全部传递依赖。

截至09:46 UTC，refine-logs/EXPERIMENT_TRACKER.md:7–13仍保留RUNNING/TODO及“尚无正式结果”，与已完成原始产物不一致；REPORT.md和CHECKPOINT.md尚不存在。应在最终交付时同步状态；这不改变已保存数值的真实性。本审查没有修改这些非授权文件。

## D. 指标调用与死代码：PASS

主评价aggregate在run_experiment.py:187–193用于全部、扰动、正确、负控、分支、对比度和seed分层；均有保存输出。analyze_results.py:20–72生成配对、覆盖及补充结果，所有主比较与补充条目已复算。原有未用于本轮的历史求解器工具函数不构成本轮指标缺失。

## E. 比较范围、预期与因果解释：WARN

预注册只覆盖一个已知物理族：3个seed block × 3个遮挡分支 × 2个对比度层，共18个新参数世界。它不是跨物理机制泛化、真实点云验证，也不是对多次独立构造episode的随机试验。library作者有历史背景而free作者fresh；该差异已在EXPERIMENT_PLAN.md:15及arms/library/METHOD.md:7披露。因此不能估计“自由构造Agent”相对“选择Agent”的普遍因果效应。

METHOD.md中的改进理由属于构造前假设，不是经过概率校准的预测。library先约定6世界开发集的全局MSE选择规则，再冻结depth；free的期望是通过可行外集投影减少不兼容现任。两档正式预算均预先确定，没有依据确认成绩只选6秒赢家。开发预期与确认事实必须分开描述。

full/lowcontrast是同色大背板负控，每个seed block一个。它提供模型内不可辨识例子，不表示其他所有低对比世界也不可辨识。

## F. 评价类型：simulation_only

模拟真值来自独立于方法输出的生成世界，故不属于模型输出自参考的synthetic_proxy；但没有真实场景GT，所有性能陈述必须保留“在此模拟物理族与本批世界中”的限定。

## 三臂到底改变了什么

| 层面 | fixed | library | free |
|---|---|---|---|
| 主要数学对象 | 分支标记的连续深度可行外松弛、区间并 | 同一个方法族，冻结选择depth | 分支标记的连续深度外松弛、区间并 |
| 决策 | HULL投影 | HULL投影 | HULL投影 |
| 约束/变量 | 材料暗度、照片偏差、独立appearance偏差，共15变量；相机外区间 | 同fixed所用的既有内核 | 同类15变量必要LP；消去共享标靶误差后保留边际相机区间 |
| 几何计算 | 有理数矩形内/外界与额外Lipschitz界，最终增加相机分区 | coarse后depth端点细化，不运行最终accumulated阶段 | NumPy浮点矩形内/外界与可见后片集合差，未作相机分区 |
| 搜索 | A9广度优先→A10端点→A11相机tile；三分支分阶段分配预算 | A9→A10；同预算分两阶段 | 全部分支共同选择当前hull两端，按逆深度二分 |
| 排除凭据 | 浮点LP提案后精确有理数矛盾证书重验 | 同fixed | 接受HiGHS浮点infeasible状态；有padding与slack，无精确证书 |

直接依据：library/methods.py:49–86；arms/free/method.py:21–153、156–246；历史continuous_solver.py:92–181、206–280；tightening_solver.py:114–257；polytope_filter.py:253–387。历史文件位于既有lovelace-baseline-intervention-20261012T045547Z工作区，A11包装位于lovelace-evidence-accumulation-20261013T051931Z/accumulation_solver.py。

三臂共同物理信息与损失成立，但比较同时改变数值算术、约束构建、细化顺序、容差、证书工作及序列化状态大小。free的较小实际耗时是这整个实现包的事实，不能分解成“新表示”的独立贡献。独立构造再次得到外集+HULL是有意义的构造结果，但没有发现新表示族的证据。

free删除LP不可行cell的代码在arms/free/method.py:191–194；浮点求解器状态不是精确证明。18/18覆盖与零正确输入损伤是本批经验结果。fixed/library的精确证书保证也以相应输入与外松弛模型覆盖为条件；公共浮点观测经Q(str(x))转换，不应把数值输入适配说成对原始实数测量的无条件定理。本审查阅读了证书执行路径并检查保存分类记录，未重新逐个执行所有有理数证书。

## 成本口径与补充消融

worker.py:22–68分别记录import、prepare、decode；74–83单独记录一次实际JSON序列化与文件写入。run_experiment.py:59–69从启动子进程前到退出后测量进程墙钟与子进程CPU。该“冷进程端到端”包括输入读取、import、prepare、decode、序列化、写入和退出开销，但不包括此前driver创建job文件的时间。不得把prepare时间直接称为完整端到端时间。

主批次墙钟360.491577秒；108个子进程墙钟总和360.228099秒。free的18次prepare总时间仅1.300931/1.314250秒，而对应完整进程总时间为5.230546/5.288890秒。审计等待与交互时间不计入求解加速。开发成本在各METHOD.md中分别报告：fixed历史研发未知；library实测比较与测试CPU54.757096秒；free已测区间3.987947秒且明确未精确计入全部import/检查，不能把后者当严格完整成本或把历史成本当0。

worker.py:66在序列化之前读取ru_maxrss。因此max_rss_kib应标作“序列化前采样得到的进程峰值RSS”，不是包含后续序列化及退出阶段的完整进程峰值。

保存的fixed/library分类共6512次：4889个FEASIBLE_WITNESS、1607个CERTIFIED_INFEASIBLE、16个UNKNOWN；最大分类耗时0.096569秒，没有超过0.25秒的已记录分类。这些状态描述外松弛及其证书，不是6512个独立物理实验。

补充消融仅移除跨视图材料范围约束（arms/free/method.py:236–238）。预定公共索引0/2/4确实对应首个seed block的三种branch contrast世界。其21个预测与完整free逐项完全一致，最大差0。三世界全/消融MSE分别为0.8156349619、1.2348551930、5.6872166665。因此本消融没有提供该耦合约束有效的正证据；更不能区分几何表示、搜索调度与精确证书成本的贡献。

## 主张影响与行动项

- C1：支持“本批6秒截面free以较少实际进程成本达到略低MSE，零观测到的正确输入损伤”；24秒截面应报告速度与精度取舍，不能写成两档全面胜出。
- C2：本次无归因支持。三案例耦合消融为零差异，不能解释主要速度来源。
- “自主构造了可执行方案”：在披露上下文及开发过程的范围内有支持。
- “发现新数学表示/普遍发现能力/实景可靠性/精确非退化证明”：本次证据不支持。
- 最终交付同步tracker及报告缺口，沿用上述端到端与RSS口径。无需为完成本审查增加训练、方法修改或实验。

## 审查独立性与记录

审查者先直接阅读本轮协议、生成器、方法、库、worker、runner与冻结清单，没有收到执行者的源代码解释。随后为查询agent身份调用list_agents，工具同时暴露了已完成的其他审查者及library作者摘要；当时尚未读取holdout真值，且主要源码检查已完成。此暴露如实记录，不能宣传为完全无其他审查信息的复盲审查。

数值独立复算完成后，协调者又给出了拟向用户采用的解释，供事实冲突检查；它没有参与最初源码判定或算术复算。该解释与已复算事实相容。审查结论继续以直接文件证据为依据。

完整证据目录：/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z/.aris/traces/experiment-audit/2026-10-13_run01/。
可执行复核：/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z/independent_audit.py。
所有审查新增文件均为该脚本、本报告/JSON和私有trace；没有改动方法、生成器、评价器、冻结清单或历史实验。

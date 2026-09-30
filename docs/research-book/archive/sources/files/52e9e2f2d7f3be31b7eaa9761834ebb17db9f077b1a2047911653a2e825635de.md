# 给 FBALE5：统一点云实验与元推理实验

日期：2026-09-15。状态：统一建模与研究交接，不是新算法已经通过实验。

**使用顺序已调整：本文是第二阶段的历史证据与候选框架，不再作为独立探索的首个提示词。** 首轮请在新会话只提供 [START_FRESH_FBALE5.md](/home/grf/Documents/Codex/2026-09-15/START_FRESH_FBALE5.md)，保存独立构造后再提供本文。不要把两份同时附上并要求模型“忽略后半部分”；这不构成信息隔离。已经读过本文的会话只能称为知情再构造。

材料入口：历史 GitHub 仓库见附录 A；Hermes 阶段成果、后续更正及原始报告索引见附录 B。本文中的统一模型、三类动作、2×2 实验和下一步方向都是作者候选方案，允许被替代；不是必须遵守的科学答案。历史实测、原因解释与路线决定须分别核查。

## 0. 任务身份

你的研究对象是：**当观测、表示、代理目标、求解程序都可能不够好时，如何依据实验结果选择并验证下一次修订。**

点云项目和元推理项目是两个具体载体，不是让你二选一。不要把任务缩回“继续调点云滤波器”，也不要只在完整目录上再比赛一次选点策略。允许大胆提出新的表示、观测动作、候选生成方式及修复机制；每项构造必须说明它改变了哪个对象、预测什么可观察后果。

收到本文后，先将已保存的独立构造与历史逐项对照，再决定保留或修订哪些机制，不要直接把自己的构造改写成本文的框架。本轮交付统一模型或有依据的替代抽象、两个实例化、可检验命题和初始实验协议。本文本身不授权修改旧项目、下载大数据、启动长实验或对外提交。若随后得到执行授权，在单独的新工作区实施；旧材料只读。

## 1. 统一的层次，而不是统一成同一个损失函数

内层问题：根据已有证据，构造对真实对象有用的输出。

外层问题：选择获取证据、检验假设、修改表示、修复程序、选择输出和停止的行动，使内层最终结果改善。

| 共同对象 | 点云载体 | 元推理实验的载体 |
|---|---|---|
| 隐藏世界 | 表面、扫描变换、遮挡、测量误差 | 真实动力学、隐藏记忆、观测漂移或污染机制 |
| 允许观测 | 点、帧来源、相机、图像、公共参照 | 输入—响应轨迹、预驱动历史、复测结果 |
| 科学表示 | 单/多曲面、关联、位姿/误差模型 | 状态维数、动力学项、记忆状态、观测映射 |
| 实际求解 | 匹配、优化、搜索、模型选择、几何导出 | 参数拟合、结构搜索、状态估计、预测导出 |
| 外部成功标准 | 独立参考下几何改善且结构/覆盖可接受 | 未参与拟合的干预预测改善，必要时另测结构辨识 |
| 外层决策 | 下一步补观测还是改表示/实现/导出 | 下一步复测、隔离、扩展解释还是改变实验 |

两列的领域求解器不必相同。希望迁移的是**如何提出、检验和修复解释的决策机制**。几何 MAE 与动力学 NMSE 不能直接相加成为“科研总分”。

## 2. 最小数学模型

领域记为 d，隐藏世界为 w，研究阶段为 t。令：

\[
H_t=\{(a_j,o_j,\text{来源},\text{版本},\text{成本})\}_{j<t}
\]

为只追加的证据历史。复测产生新记录，不覆写旧测量；采信/隔离状态单独保存。

\[
M_t=(\mathcal R_t,\mathcal O_t,\ell_t,Q_t,U_t)
\]

为当前科学工件：表示空间、观测/关联模型、代理目标、实际求解程序、选择与输出规则。拟合参数、候选池、测试结果和程序哈希另存于计算状态 Z_t。

研究策略执行：

\[
a_t=\pi(H_t,M_t,Z_t,B_t).
\]

这里 B_t 是剩余资源；策略不能读取隐藏答案或封存确认分数。

对于获取外部证据的动作：

\[
o_{t+1}\sim K_d(\cdot\mid w,H_t,a_t).
\]

对于计算/修订动作：

\[
(M_{t+1},Z_{t+1})=\Gamma(M_t,Z_t,H_{t+1},a_t).
\]

真实观测规律 K_d 与工件内假设的观测模型 \(\mathcal O_t\) 不相同。修改后者不能追溯性改变数据生成过程。

终点输出为 \(\hat z_\tau=\operatorname{Export}(M_\tau,Z_\tau)\)。单领域目标写作：

\[
\min_\pi\;\mathbb E[L_d^*(\hat z_\tau,w)]
\quad\text{s.t.}\quad
\sum_{t<\tau}\mathbf c(a_t)\le\mathbf B,
\quad \text{满足预定结构/安全约束}.
\]

若尚未给出任务分布，不宣称已定义这个期望或找到了最优策略。可以先报告锁定任务集上的逐任务损失及配对差异。

测量次数、内部拟合次数、计算时间及 token 可分别计费；不要把一次调用内枚举几十个候选算成“只拟合一次”，也不要未经说明把一分钟计算等同一次物理测量。充足算力允许更强构造，不取消对照的资源口径。

外部目的 \(L_d^*\) 的含义固定；代理目标 \(\ell_t\) 可以修订。评价器有 bug 时可以修复，但须留版本、对同一批输出统一重评分，再锁定新确认协议，不能靠换评分让新版本获胜。

## 3. 研究动作必须同时覆盖三类变化

1. **获取新的世界证据**：新视角、公共参照、独立复测、预驱动历史、改变激励。
2. **重新利用已有证据**：搜索更充分、修求解器、恢复被丢掉的输入字段、调整输出、检查实现。
3. **修订科学表达**：引入隐藏状态、改观测模型、由平面换曲面、改关联对象、扩展可组合的程序语法。

第 2 类并非没有价值：有限计算者可以从旧数据中提取之前未用到的关系。但不能把同一数据的反复拟合当成新的独立观测。LLM 既有知识也是先验信息来源；比较不同策略时要说明是否共享。

“诊断”首先产生可检验假设，不自动等于识别出了唯一原因。多个失配可以同时存在。

## 4. 三个具体的数学支点

### A. 当前有歧义，不等于任何实验都无解

若两个世界给出相同的当前输入，而目标不同，当前估计有歧义。只有当所有允许的取证策略也无法区分它们，才形成实验后的不可辨识性。

对目标输出相距 \(\delta\) 且可见记录同分布的两个世界，在度量损失下，任何策略至少在其中一个世界的期望误差不低于 \(\delta/2\)。这是标准两点论证，不包装成原创定理。

点云例：两站各见一层，层差与站偏差互相替代；公共参照可能打破歧义。动力学例：当前位置/速度相同但初始记忆不同；可观测预驱动可能打破歧义。新增信息有用与新求解器优越，是两种结论。

### B. 表示、搜索、选择可以分账，但不能凭均值猜分账

固定证据、输出语义和一个外部风险 R。令可表示输出为 \(\mathcal M\)，实际生成的非空候选集为 \(C\subseteq\mathcal M\)，选中输出 \(\hat z\in C\)，所有允许输出为 \(\mathcal Z\)。则：

\[
R(\hat z)-\inf_{\mathcal Z}R
=\underbrace{\inf_{\mathcal M}R-\inf_{\mathcal Z}R}_{\text{表示差距}}
+\underbrace{\min_C R-\inf_{\mathcal M}R}_{\text{搜索差距}}
+\underbrace{R(\hat z)-\min_C R}_{\text{选择差距}}.
\]

这是同一风险下的代数分解，不是已测得的因果贡献百分比。有限候选 oracle 只估计候选选择空间，不能冒充整个表示族最优；不知道真值时也不能直接运行上述 oracle。导出规则变化必须先映射到同一个输出契约。

### C. 修复能力的收益与反馈的收益分开

对锁定的能力集合 \(\mathcal A\) 和预算 B，定义：

\[
\Delta_{\rm feedback}(\mathcal A,B)
=R(\pi_{\rm preplan};\mathcal A,B)
-R(\pi_{\rm adaptive};\mathcal A,B).
\]

预先计划允许看初始历史后制定，随后不能根据新结果改变被比较的动作；内层拟合仍可使用得到的数据。若两者真是同一模型下的精确最优策略、且自适应策略集合包含预先计划，则期望最优风险不劣是包含关系的结果，不是发现新科研规律。

该差异是否严格为正、是否跨任务保持，以及近似策略能否实现，才需要实验。更换能力集合后的收益另报，不能算到反馈头上。

## 5. 现有证据如何放进模型（只读核查，不是本轮重跑）

核查主机：liekkas。

**点云 V25**：[/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration/REPORT.md](/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration/REPORT.md)。

- 8 个父 ROI 上 atlas 均未胜过 identity：是这套证据—求解—导出组合的负结果，不足以单独认定表示理论无效。
- [evidence.py](/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration/src/v25/evidence.py) 使用离散深度赢家；[regenerate.py](/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration/src/v25/regenerate.py) 的某些移动将多个输入直接吸附到同一证据点。应分开检查深度证据、几何拟合和输出覆盖，不能把三者统一叫“模型不好”。
- PLY 数量不是独立场景数量；不部署失败候选与长程研究任务是否充分执行也是两个问题。

**元推理当前代码**：[/home/grf/Documents/meta-research-open-world-challenge](/home/grf/Documents/meta-research-open-world-challenge)。

- 旧 construction 报告的“四族、初始记忆未接历史”不能原样当成当前状态。
- 当前 [host/artifact.py](/home/grf/Documents/meta-research-open-world-challenge/host/artifact.py) 枚举 16 种结构组合；带记忆结构还搜索 4 个衰减值。它实现了受限原语组合，不等于无限开放的科学发现。
- 当前 [host/score.py](/home/grf/Documents/meta-research-open-world-challenge/host/score.py) 已提供公开预驱动并估计记忆初态。
- 新 [SUMMARY.json](/home/grf/Documents/meta-research-open-world-challenge/outputs/capability_v1/SUMMARY.json) 记录 T2/library 的 NMSE 无历史 0.396749 → 有历史 0.001740；T4 为 0.327016 → 0.002213。是本任务族中的信息利用收益，非普遍收益。
- [policies/capability.py](/home/grf/Documents/meta-research-open-world-challenge/policies/capability.py) 两种能力策略共用覆盖采样，所以 proposer 与 library 的差异测试的是候选搜索/选择，不是自适应实验调度。观测漂移分支仍有明显误差。

旧有限目录实验 H=4 有严格自适应收益、H=6 开环已零误差，是用户提供并已审计过的历史结论；不能自动扩展到不完整解释集。本文件没有重新运行那组 DP。

## 6. 下一步最小但有力度的实验

首先用旧案例开发诊断，不把已反复看过的 V25 ROI 或记忆系统当新盲评。第一问是：**同样观察到失败时，策略能否用一个有区分力的行动，决定补信息还是改程序/表示？**

构造两种初始症状相近、最佳修复不同的任务，再加入两种原因同时存在的任务。原因只由评估宿主持有。候选动作的效果必须实际运行验证，不能由生成器直接回传“正确修复标签”。

随后用同样的公共输入和计费比较：

| | 依据初始历史锁定取证计划 | 根据新增结果调整取证计划 |
|---|---|---|
| 原有共享修复工具 | A | B |
| 扩展后的共享修复工具 | C | D |

- C−A：增加这组修复能力的系统收益。
- D−C：扩展能力固定后的反馈增量。
- D−A：整个新系统的收益，不能独归于“元推理”。
- 此矩阵只识别取证调度增量；若要研究“何时改表示/求解器”，须把该选择也纳入被比较的动作并明确冻结边界，不能悄悄让一边更聪明。
- 所有单元内部采用同等的候选生成/拟合规则；不得给自适应端独占新修复器或更准先验。
- 如果指定了可调用的新动作，预先计划基线也可以调用。不得只保留弱覆盖基线。

大胆构造可以来自新表示，但它必须作为公开可调用的修复工件，或与同样拥有该生成能力的策略比较。输出至少包含结构说明、可执行预测、输入需求、使用的证据、失败预期。新增数学名词不是新增能力。

分别报告两领域的主误差、结构/覆盖损伤、有效修复比例、拒绝比例、所有内部求解成本及逐步轨迹。按独立系统/场景统计。不得依靠改归一化或更换终点把负结果变正。

跨领域主张需要冻结同一外层策略及其决策规则，用公开的领域适配器去新载体确认。只把两种任务都写进这套符号，不构成迁移证据。

## 7. FBALE5 读过历史后的交付与边界

交付一份统一研究说明，包含：

1. 对照已经保存的独立构造与第 2 节候选模型；可以替换后者，标明事实、假设、已知理论、待证命题。
2. 两个可逐步执行的研究回合：一个几何，一个动力学；每回合至少有一次能改变修复选择的反事实实验。
3. 一项非平凡构造，明确是新信息入口、新表示、修复器，还是外层策略；给出它可能失败的配对案例。
4. 提出适合自己构造的验证协议；第 6 节只是可选设计，不要求套用。说明每个对照能识别什么、不能识别什么，以及什么结果值得实施下一阶段。
5. 精炼的相关工作对照：哪些内容已存在，真正候选贡献剩在哪里。

可以否定本统一模型，也可以提出更好的抽象，但必须指出具体接口或反例。不要为了遵守本文件而维护一个错误框架，也不要为了显得创新而刻意反对可靠证据。不要用泛泛“都是 POMDP/状态机/图优化”代替上述交付。保留首轮构造原文，另列哪些修改由历史证据促成、哪些没有受到直接反驳；不得把读后结论追写成读前预测。

TLA+ 仅在需要检查有限契约时使用，例如日志不覆写、成本不漏计、预测先于揭示答案、权限边界；它不能证明观测足够、程序代表了真实规律，或实验结论具有新颖性。不先造运行时、GUI、MCP 平台或通用研究 DSL。

## 8. 理论定位：相邻理论，不是已经严格等价

- 有限计算者如何选择计算，以改善最终行动：对应 [Russell 的 rational metareasoning / bounded optimality 研究](https://aima.eecs.berkeley.edu/~russell/research-bo.html)。这里必须保留计算状态，不能一面假设完美推断、一面又无解释地宣称重复计算增加了世界信息。
- 若给定世界空间、先验、转移/观测规律及动作，可建立部分可观测决策实例；参见 [Kaelbling、Littman、Cassandra，Planning and Acting in Partially Observable Stochastic Domains](https://www.cassandra.org/arc/papers/aij98.pdf)。有限时间不自动使状态有限或精确规划可行。
- 当候选表达本身需要扩展，不能只在旧模型的后验里重新分配概率；参见 [Gelman、Shalizi，Philosophy and the Practice of Bayesian Statistics（作者稿）](https://sites.stat.columbia.edu/gelman/research/unpublished/philosophy.pdf)。这为模型检查与修订提供理论参照，不直接给出可执行的自动科学发现算法。

工作名：**部分观测下的科学模型与求解程序联合修订**。这是本任务的描述性命名，不声称已经提出新的学科、定理或成功方法。

## 附录 A：GitHub 与本地材料入口

用户此前指定的点云研究 GitHub 仓库：**[jupiternaut/map-denoise-research](https://github.com/jupiternaut/map-denoise-research)**。

- 这是历史代码与研究记录的网络入口。本次只补入已知地址，没有联网核对其当前提交、公开性或完整目录。
- 本地 V25 和元研究工作区均未被本次 `git rev-parse` 识别为 Git 工作树；不能据此宣称它们已同步到上述仓库。未找到或核验 Hermes 成果的独立 GitHub 地址，不编造链接。
- 点云当前本地工作区：[V25](/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration)；[报告](/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration/REPORT.md)；[检查点](/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration/CHECKPOINT.md)。
- 元研究当前本地工作区：[open-world challenge](/home/grf/Documents/meta-research-open-world-challenge)；[任务定义](/home/grf/Documents/meta-research-open-world-challenge/TASK.md)；最新可见结果入口是附录 B 的 capability_v1，而不只是 construction 旧报告。

本地路径均属于主机 liekkas。跨机器或网页模型不能靠这些路径读取文件。本文件内嵌的阶段摘要可以随文件传递；原始代码、JSON、报告尚未复制或打包进本文，也未上传 GitHub。若接收端没有该主机访问权，必须另行提供材料包；不能把“有链接”当成“已阅读原文”。

## 附录 B：Hermes 元研究成果与更正链

以下历史目录在 liekkas 上存在，本轮只读核对报告和检查点，未重新运行实验。日期均为 2026-09-14，时间戳为 UTC。保留前后变化，避免把一次实现缺陷误当成研究方向的定论。

### B1. 从冲突处理到完整目录的研究链

| 检查点 | 做出的工作和结果 | 现在应保留的结论 |
|---|---|---|
| 08:48：修复器/策略分开 | 同一冻结历史下，新修复器可逆隔离污染记录；报告中旧 NN 兜底约 3.40–3.64 的污染误差降到 0。共享新修复器后，预算 16 上三种策略均为 0。 | 这组收益首先属于修复器；不证明更聪明的调度。先前“没有区分实验”的解释被纠正：配对世界同点复测同值，但新位置可不同。 |
| 11:10：小目录精确 DP | N=8、18 世界目录，报告过目录内收益和 holdout 退化，并建议终止策略搜索。 | **该归因后来被更正**，不可单独引用为“精确调度没有迁移能力”的证据。 |
| 12:46：修复 DP 验收 | 初始记录与复测历史共同筛选；空支持触发声明过的回退；H=4/6 分别求解。目录期望风险 DP 不劣于固定策略；H=4 为 0，对照约 0.917/1.042。 | 前次混淆包括终点最优策略的中途截面、空目录停机、只按当前标签筛选。修正实现后，策略研究不能沿用旧的关闭理由。 |
| 13:26：2340 世界完整目录 | 公开规则枚举 780 组、2340 世界；比较初始观测条件化的最优预先计划和自适应 DP。H=4 目录反馈增量约 0.0103；H=6 两者均为 0。 | 分开“拥有目录并预先规划”的收益与“获得新观测后改变动作”的增量。 |
| 14:24：初始设计条件化审计 | 438 种初始历史中，按生成位置条件化使 36 组计划改变，涉及 108 世界；H=4 目录误差仍为 0.0102564，72 个残差世界不在改变组内。 | **最终保留**：H=4 有严格但有限的自适应收益；H=6 在 2340 世界内开环=自适应=0。没有测出最小必要预算，也不是跨机制迁移。 |

原文入口：

- 08:48：[REPORT](/home/grf/.hermes/attachments/outputs/20260914T084840Z/REPORT.md)；[CHECKPOINT](/home/grf/.hermes/attachments/outputs/20260914T084840Z/CHECKPOINT.md)。
- 11:10：[REPORT，需结合后续更正](/home/grf/.hermes/attachments/outputs/20260914T111013Z/REPORT.md)；[CHECKPOINT](/home/grf/.hermes/attachments/outputs/20260914T111013Z/CHECKPOINT.md)。
- 12:46：[REPORT](/home/grf/.hermes/attachments/outputs/20260914T124618Z/REPORT.md)；[CHECKPOINT](/home/grf/.hermes/attachments/outputs/20260914T124618Z/CHECKPOINT.md)。
- 13:26：[REPORT](/home/grf/.hermes/attachments/outputs/20260914T132630Z/REPORT.md)；[CHECKPOINT](/home/grf/.hermes/attachments/outputs/20260914T132630Z/CHECKPOINT.md)。
- 14:24：[最终审计 REPORT](/home/grf/.hermes/attachments/outputs/20260914T142434Z/REPORT.md)；[CHECKPOINT](/home/grf/.hermes/attachments/outputs/20260914T142434Z/CHECKPOINT.md)。

最终分支例子：初始历史 `{2:1, 5:-1}` 下，开环查 `3` 再查 `6`；自适应查 `3`，若返回 `-1` 则改查 `0`，否则查 `6`。这是新增标签实际改变了有用行动，不只是换日志名称。目录中 72/2340 个世界留下开环残差，均属 model 类；自适应达到 0。确认样本来自同一已枚举族，不能称为开放世界确认。

这一阶段留下的下一问题是：**解释集不完整时，什么证据支持扩展解释集？** 不继续以 H=6 零误差装置作为更复杂调度器的主要竞争场。

### B2. 动力学 construction：有实现，但旧结果的能力边界明确

原文：[construction REPORT](/home/grf/Documents/meta-research-open-world-challenge/outputs/construction/REPORT.md)；[CHECKPOINT](/home/grf/Documents/meta-research-open-world-challenge/outputs/construction/CHECKPOINT.md)；[逐次结果 ROWS](/home/grf/Documents/meta-research-open-world-challenge/outputs/construction/ROWS.json)。

- 完成 6 个构造系统、5 种策略、30 个策略回合；记录墙钟 24.7 秒。是构造实验，不是 12 开发 + 24 盲评的完成记录。
- C2 记忆系统：扩大模型能力后 NMSE 约 0.723 → 0.233；C4 改初态后却约 1.236 → 1.308，错误结构选择保留。
- 同构造器下 D−C 在 C0–C4 近零，C5 反而增加约 0.054；不能据此声称顺序科研策略胜出。
- 当时记忆分支缺少对应公开历史、候选从 m0=0 滚动预测。这个历史缺口已由下一阶段的评分接口修订，不能继续当成当前未修复状态。
- 旧检查点列出的未完成项包括 12 系统开发、24 系统冻结盲评及官方基线；本轮看到的 outputs 只有 construction 和 capability_v1，没有看到这些完整验收的新增工件。不要把“尚未见证据”写成已经完成。

### B3. capability_v1：可组合表示与历史输入的新结果

原文：[SUMMARY](/home/grf/Documents/meta-research-open-world-challenge/outputs/capability_v1/SUMMARY.json)；[ROWS](/home/grf/Documents/meta-research-open-world-challenge/outputs/capability_v1/ROWS.json)；[模型代码](/home/grf/Documents/meta-research-open-world-challenge/host/artifact.py)；[评分代码](/home/grf/Documents/meta-research-open-world-challenge/host/score.py)；[策略代码](/home/grf/Documents/meta-research-open-world-challenge/policies/capability.py)。

当前 artifact.py 的 SHA256 为 `88737a1b3a2c52c4fe790e9206bc3ce2764879e0f4c77f1d93a2f6128dd0097c`，与该 SUMMARY 的记录一致；只说明这个代码文件的对应关系，不是全环境复现证明。

| 系统 | library 无历史 NMSE | library 有历史 NMSE | 应怎样理解 |
|---|---:|---:|---|
| T2_memory | 0.396749 | 0.001740 | 可见历史帮助估计隐藏初态，明显改善该条件预测。 |
| T4_mem_x3 | 0.327016 | 0.002213 | 相同机制在另一构造系统上有收益。 |
| T3_drift | 0.248521 | 0.254662 | 加历史不包治百病；仍选到记忆/identity 观测解释，漂移问题未解决。 |

可组合代码现为 16 种结构组合，记忆结构再比较 4 个衰减参数值。propose 与 library 共用原语和覆盖计划；前者没有脱离这套有限组合空间。在 T5，propose 的有历史 NMSE 为 0.001679，library 为 0.002005；这是一项局部搜索/选择差异，不是实验调度收益。

这仍是受信算法实验，非具有 OS 隔离的闭卷 LLM 科研挑战。上述三个阶段共同提供了统一建模的案例，但尚未证明一般科研效率提高或点云上的策略迁移。

### B4. FBALE5 应当继承什么

继承证据与修订链，而不是继承每一轮的路线判断：

1. 输入历史/支持条件不足，可以伪装成算法或策略失败。
2. 修复器能力、候选搜索、实验反馈是不同的收益来源。
3. 精确最优只针对给定目录、目标、预算和执行语义；实现错误不能包装成理论负结果。
4. 受限任务上的零误差可以结束该任务的精度竞赛，但不能证明解释集完整。
5. 新统一研究应检验这些诊断与修订机制能否跨载体工作，而不是重新将上述旧案例当作未见确认集。

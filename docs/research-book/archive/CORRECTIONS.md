# 更正与撤回链

更正不是删除失败，而是限制旧结论的有效范围。以下从实验记录自动聚合；完整上下文和原版本仍可阅读。

## [EG002 · 早期 parallel-V5：薄层关联与切向覆盖是两种增量](experiments/EG002.md)

- 此V5属于早期map-denoise-v0/parallel_geometry_v5，与项目exploration_v5不是同版。

## [EG003 · 早期 parallel-V5 Oxford：来源适配不是去噪成功](experiments/EG003.md)

- 保留首轮精确体素边界差异及修复，不将重评包装成新数据。

## [EG005 · 项目 V1：保留输出，撤回旧几何结论](experiments/EG005.md)

- 由EG006/EG007记录V2对评分、帧截距和native改写的更正；不删除V1原文。

## [EG006 · 项目 V2：改为世界坐标和实际XYZ评分](experiments/EG006.md)

- 撤回V1由模型元数据推导的几何/结构排名。

## [EG007 · 项目 V2：已有纠偏能力与native改写必须分账](experiments/EG007.md)

- 撤回V1“没有帧截距”和“5 mm偏差导致4 cm错误”的解释；标量反例不升级为完整3D不可能定理。

## [EG009 · 项目 V3预算补测：求解不足不能当表示不足](experiments/EG009.md)

- 首版固定3轮的负结果被补测收窄，不等于运输族无效。

## [EG011 · 项目 V5：冻结关联后的共享斜率失败](experiments/EG011.md)

- 与EG002的parallel-V5分开；与V6比较须区分新确认和公开开发，以及compatible/pool_independent基线。

## [EG012 · 项目 V6关联oracle：共享效果取决于分组](experiments/EG012.md)

- V6原组/独立为V4 compatible；不能与V5 pool_independent或不同种子胜负机械拼接。

## [EG015 · 项目 V7真实支线：厘米高度残差可能是坐标条件数](experiments/EG015.md)

- 修正把旧轴厘米残差直接解释成sigma或物理厚度的倾向。

## [EG016 · 项目 V8：截断补偿减少点误差却加重结构偏差](experiments/EG016.md)

- 其风险解释由V9进一步收窄：即使真模型、独立积分也会缩层，见EG017。

## [EG017 · 项目 V9真模型反例：条件无偏不等于潜在结构恢复](experiments/EG017.md)

- 更正V8把风险主要归于模型估错/数据复用的解释边界，不撤销V8数值。

## [EG019 · 项目 V10：层距参数变准，实际XYZ仍更差](experiments/EG019.md)

- 不再将失败单独归为冻结旧偏差；两次单面误拆来自同seed配对，不是独立证据。

## [EG020 · 项目 V11：空间关联的合成正结果与真实迁移失败](experiments/EG020.md)

- 谱初始化标签方向事后修正未改变负面判断，不算另一独立确认。

## [EG023 · 项目 V14：扩展条件与外部对照支持窄族收益](experiments/EG023.md)

- 官方工具配置失败及数值容差调整仍需随原报告保留；不把通过的子集写成全部成功。

## [EG024 · 项目 V15：强求解缩小差距，但结构代价仍在](experiments/EG024.md)

- 来源间距没缩至一半不是零误合层证明，后续EG025收窄该代理解释。

## [EG025 · 项目 V16：空间混合和斜率/选层交互，更正间距代理](experiments/EG025.md)

- 收窄V15来源间距指标的结构解释，不撤销原数值；K=1本身也不是整个输出拓扑的充分判据。

## [EG027 · 项目 V17遗漏：已有更优可行解被新搜索丢失](experiments/EG027.md)

- 纠正该5例的错误归因；并不否认其他实验已实际出现目标错序。

## [EG028 · 项目 V18：候选保留修复了旧例，没有新确认增益](experiments/EG028.md)

- 修复V17已定位遗漏，不覆盖V17原始输出和负结果。

## [EG030 · 项目 V19：crossfit有搜索混杂，quota无法补回漏层](experiments/EG030.md)

- EG031后续恢复同样本能力；本轮原负数值保留，不再作为纯拆折效应。

## [EG031 · 项目 V20：修复搜索后crossfit仍差，池内oracle余量有限](experiments/EG031.md)

- 初版oracle以旧解替换R而未评分原R；后续从保存参数补齐，属于事后诊断，不是新盲确认。

## [EG032 · 项目 V21：独立参考真实迁移未通过](experiments/EG032.md)

- 汇总failed=6表示不可评分，不是6次算法崩溃；不能删去后宣称全协议通过。

## [EG033 · 项目 V22：MLS重构减轻损伤，阻尼没有额外收益](experiments/EG033.md)

- 原verify.py并非全部通过，不能把后续审计写成所有历史输出逐点一致。

## [EG034 · 项目 V23：照片LOSS下降仍损伤独立几何](experiments/EG034.md)

- 旧cameras.npz camera_mat是归一化屏幕变换而非像素内参；首次试运行在候选/GT评分前中止，后来改用实际COLMAP内参。

## [EG035 · 项目 V24：同位移预算区分修正场与alpha分配](experiments/EG035.md)

- 等预算归一alpha有时放大场，不是V22原始输出；不是全部几何模型无用或XYZ信息已耗尽的证明。

## [ES001 · 早期 local_operator_v1：GEPA实际搜索、边界保护失败与评分勘误](experiments/ES001.md)

- 统一评分勘误保留原代码、原分数和旧输出；仅更正identity分数，不重写NO-GO历史。
- 本记录补回EG001之前的实际搜索，不把它与主项目V1混为同一版本。

## [ES002 · 早期 observation_direction_v3：射线方向投影未增加最终收益](experiments/ES002.md)

- 初始ru_maxrss可能继承父进程高水位，后改用exec后VmHWM复测；原记录保留，输出哈希不变。
- 7次次分支提交不归功于射线约束，真值V门通过率也不当作普遍提交上限。

## [ES003 · 早期 parallel_geometry_v4：直接输出去噪与外部基线首轮](experiments/ES003.md)

- 旧Oxford范围不匹配由后续ES006/EG003修正，但V4原数值保留。
- 开发冻结检查修复后重跑不是新增样本或独立复现。

## [ES004 · 六线 T1/T2：帧占比混淆、联合去偏与观察等价](experiments/ES004.md)

- 这里已有联合帧偏差能力，是后来EG007纠正“没有帧截距”解释的前史。

## [ES005 · 六线 T3/T6：复杂慢参考的增益主要由邻域尺度解释](experiments/ES005.md)

- 初版association oracle重取同类邻域，改变空间足迹/更新，降格为非匹配诊断；后来固定邻域2×2才提供较干净归因。
- 外部PCL小半径是记录在案的开发扩展，不是预先独立确认。

## [ES006 · Oxford支持链补充：已知空间不是完整可见性，稀疏不等于无局部交叉支持](experiments/ES006.md)

- 精确体素边界除法/倒数乘法差异已修，27份评分不变；两次运行不计54独立实验。
- 原范围/双向已知空间的不同数值是评价范围变化，不是算法输出改善。

## [ES007 · 六线 T5：C++融合EM已完成CPU加速](experiments/ES007.md)

- 本条补回已完成的CPU优化；不能继续笼统写整个早期项目从未做性能工程。

## [ES008 · T2-boundary-v1：有序截距消元滤波器、法向扩展与真实接入](experiments/ES008.md)

- baseline目录中的profile是自研独立参考，不是官方工业算法。
- 优化器success、APPLY和K=2不提供真实几何正确证书；真实位移数值不是误差下降。

## [ES009 · GPU-v1：已分组局部法向滤波的CUDA/Triton与Graph runtime](experiments/ES009.md)

- 首次Graph捕获失败目录保留，AST变换后补跑成功。
- 开发时端点同分引起大坐标差，改为复现NumPy float64求和/cumsum语义；204例相符仍非所有输入逐位等价。

## [ES010 · correlation真实支线：纯帧端点数值平局与层身份翻转](experiments/ES010.md)

- 更正将目标/层距不变等同最终XYZ不变的推断；变化不是仅坐标gauge或全局平移。
- 415是可能重叠片区记录和，不是415个独立地图点；234输出不是234真实场景。

## [ET001 · Windows/WSL2 RTX5080：交接包就绪，目标机未验证](experiments/ET001.md)

- PACKAGING_VALIDATION 不是 Windows/5080 执行结果；START_WINDOWS 是未来任务提示，不是执行日志。

## [ET002 · 应用架构与CPU/CUDA分工：明确标注的设计](experiments/ET002.md)

- 图中的共享能力层是应用分层，不是内核/用户态或权限沙箱。
- 接口命令示意没有可执行完成状态，不能用文档代码块当运行记录。

## [ET004 · 硕士论文第一稿：六章叙事与证据核读](experiments/ET004.md)

- 33,300字、70页、31图、15表属于原建议规模，不是第一稿完成量。
- 将浏览器阅读版当学校格式定稿，或把文献阅读清单当已运行基线，均会扩大证据。

## [ET005 · 作者精修路线：哪些研究缺口重要，哪些工程不是前置](experiments/ET005.md)

- 此处9月23日尚待的几何外部基线/新确认，后于LG026已部分完成；不把旧路线当9月29日最终状态。

## [ET006 · 论文第二稿：扩写、综述图与继承审查边界](experiments/ET006.md)

- 字数包含标题/表格/图题，不含标点/英文/参考文献/附录；不能当学校口径页数。
- 旧审查报告/截图没有自动升级为第二稿复审；HTML不是Prism编译结果。

## [ET007 · 第二稿本地LaTeX/PDF导出：已编译，非Prism或学校定稿](experiments/ET007.md)

- 不能继续笼统写第二稿未生成PDF；实际本地XeLaTeX导出已完成。
- 本地成功不等于Prism成功，Windows可使用TeX Live的说明也不等于Windows实测。

## [LG001 · V25 多面再生：实现完成，替换式输出失败](experiments/LG001.md)

- 报告日期为 9月14日，归档目录位于 9月15日；不按目录名改写实验日期。
- 自研 fusion_wta 不是官方 COLMAP。

## [LG002 · V26 逐源证据、双视图支持与网格遮挡](experiments/LG002.md)

- 不合格现任被置为最低分会制造移动优势；这是具体契约错误，不是浮点随机平局。

## [LG003 · V26 守卫补测：撤销缺现任证据的移动](experiments/LG003.md)

- 旧首轮保留；这是事后开发修正，不是新场景确认。
- 沿用首轮 Oracle 不能评价守卫后可达最优。

## [LG004 · V27 实际位置重评分与几何联合修正](experiments/LG004.md)

- 注入运行前获知 native 不利的定性消息，不能称盲测。
- 缺失候选代价规格差异的中性重放未改变主结果；局部能量下降不保证几何下降。

## [LG005 · V28 多方向半窗候选与候选后选择](experiments/LG005.md)

- 学习器使用另一开发场景 GT 标签，不能称无监督。
- 历史已存在三选一路由，后续接路由不能再称首次。

## [LG006 · 9月22日三场景冻结确认与评价足迹补充](experiments/LG006.md)

- 准备版 CHECKPOINT/README 不代表最终状态；EXECUTION_CHECKPOINT 明确完成。
- 补充只缩小 E_sym/recall 解释，不改成绩。

## [LG007 · 9月26日相对收益、双头及校准工作点](experiments/LG007.md)

- 额外损伤减少 84.65% 不是几何精度改善 84.65%。

## [LG008 · 保留视图成对证据：恢复增量与原生损伤](experiments/LG008.md)

- 旧缓存 heldout 对 A_all 并非真保留，A_all 已用全部四图。

## [LG009 · 候选 A/B × 保留视图交叉与互补收益](experiments/LG009.md)

- 候选增加后的 Oracle 弱单调是集合包含性质，实测贡献是增量幅度而非该性质。

## [LG010 · 共享比较器与候选专属支持](experiments/LG010.md)

- 本轮新的是共享竞争与专属支持，不能称首次路由。

## [LG011 · 源视图可见性特征 × 学习容量](experiments/LG011.md)

- 全部 KEEP 的原生优先回退不算修复；事后各列最高分不可拼成一个方法。

## [LG012 · 连续步长 Oracle 与固定路由照片搜索](experiments/LG012.md)

- 74.69% 是真值诊断上界，不是已取得的实际算法恢复率。

## [LG013 · 局部多曲面场、层责任、Z-buffer 与图](experiments/LG013.md)

- ±3 网格与注入幅度重合，不能预支非网格迁移。
- 早期图自环实现错误在正式评分前修复，smoke 保留。

## [LG014 · 多曲面场专属 Oracle：拟合损失与选择损失并存](experiments/LG014.md)

- 场专属 Oracle 没有 fresh-view 有效掩码，差距也含强制 KEEP 与准入；非纯排序损失。

## [LG015 · 多视图共同解释：方向、来源与旧状态更正](experiments/LG015.md)

- 旧 REPORT/CHECKPOINT 的 P2 running 是时间切片，已被后续 replay 完成状态补充。

## [LG016 · 修订理论：有限信息界与真实逐源冲突](experiments/LG016.md)

- R-only/W-fixed 检查为首个 GT 探针后的事后补充，不是预注册独立确认。
- 汇总顶层计数有宏平均值，不是 pooled 总数；不以行数充样本。

## [LG017 · 统一理论历史矩阵与合成相机—关联桥梁](experiments/LG017.md)

- 历史阅读不是本轮重跑；早期已有帧偏差/V10 联合纠偏/V28 相机干预，不能称首次发现相机因素。

## [LG018 · 真实照片相机校正 × 共享稀疏轨迹](experiments/LG018.md)

- 小匹配子集 +3 宏改善 93.19% 不能替代全支持 0.94%；单点 ROI 过高权重可扭曲聚合。

## [LG019 · 稠密多峰、真实反向搜索与共享深度](experiments/LG019.md)

- 运行前将完全平坦曲线改为 unknown；首次评价重复字典键修复只涉及评价流程，旧失败目录保留。

## [LG020 · 共享 K1/K2 表面与普通平滑对照](experiments/LG020.md)

- 旧求解目标 11.443 与修后 4.600 均保留；修一例不等于任意实例全局最优。

## [LG021 · 有限纹理表面元：动态遮挡的真实渲染闭环](experiments/LG021.md)

- 不能说所有损伤是把坏几何藏起来：主法损伤中两拟合视图均不可见贡献仅约 3.09%/3.78%/8.13%。

## [LG022 · 106维渲染证据接回旧几何收益学习器](experiments/LG022.md)

- 特征来源是原照片/相机，并未引入前轮修正相机；不要把多个分支自动拼成已运行系统。

## [LG023 · 渲染特征 × 容量及错误邻居压制机制](experiments/LG023.md)

- uint8 KEEP 索引下溢与阈值命名空间在诊断中修正；空失败目录保留，未改模型参数。

## [LG024 · 同候选同证据：四类学习器与 pairwise 排序](experiments/LG024.md)

- natural/ balanced 分数语义不同，不能拼两个工作点赢家。

## [LG025 · 固定像素邻域同臂反事实：已执行并训练](experiments/LG025.md)

- 后来 benefit-harm 把此实验写成未完成是错误；coherent-context 明确更正。

## [LG026 · 六个新场景与 RIMLS/PathNet 外部确认](experiments/LG026.md)

- 目录时间戳始于 UTC 9月28日；报告/后续本地记录为 9月29日。
- 旧 group 报告的独立确认待办被本记录后续完成，不能继续复制旧状态。

## [LG027 · 收益/损伤双头：同树预算条件排序增量](experiments/LG027.md)

- protected 是开发联合条件不可行后的预注册回退，不是达标配置。
- 报告中单点/邻域实验未完成说法由 LG028 更正；LG025 已完成。

## [LG028 · 统一位移邻域与实际邻居几何分账](experiments/LG028.md)

- CORRECTION 明确同臂邻域并非首次执行，撤销旧未完成状态。
- 本报告泛称下一步首可见/竞争面、颜色遮挡分账又覆盖既有工作，LG029 再更正。

## [LG029 · 归属历史核查与纹理来源×邻居小型干预](experiments/LG029.md)

- 问题未解决不等于未研究归属；owner/track/局部layer 不等于物理面身份。
- 同一确定性装置相关配对不是独立场景，不能作显著性或真实迁移主张。

## [LG030 · 真实照片纹理来源：排序有增量，几何未闭合](experiments/LG030.md)

- 单 pilot 的替换比例不能代表整批；一致性回退不是可靠纹理证书。

## [LG031 · 纹理证据学习：局部选择增量与原生失败](experiments/LG031.md)

- protected 比旧少损伤不等于原生净改善；全 KEEP 不能称零伤害修复成功。
- 原生支持移动 45→30，固定点数诊断存在局部增量，但未严格匹配位移 RMS。
- 开发校准按案例相对 MSE、主评价按绝对 MSE 宏平均，二者权重不同。

## [MC001 · 9/13研究契约审计：修正命中集、一步信息价值与玩具向真实停题的外推](experiments/MC001.md)

- 审计保留报告错误与代码正确的区别；不得把手设玩具后验及Stop决定当真实主工程停题依据。

## [MC002 · Markdown任务入口转成TLA+：完成语义映射，未执行模型检查或任务实验](experiments/MC002.md)

- 翻译不是机械照抄旧诊断框架；明确允许构造新假说和表示，并撤去诊断框架的普遍必要性。

## [MC003 · Hermes METHOD：单回合CEGIS玩具已运行，空工作区不是上下文消融](experiments/MC003.md)

- 空目录/空工作区不等于新会话清除旧上下文；允许表示升级也不等于预算内已到达充分表示。

## [MC004 · CPR首轮构造：置信集修订与MDL是候选方案，保留既有V25暴露声明](experiments/MC004.md)

- 本项记载旧构造及其暴露，不采纳其中已被MC005撤回或被CPR002/CPR006进一步收窄的保证。

## [MC005 · CPR第二次构造：KEEP+ACQUIRE、保证对象分账与E0先行，仍无实测](experiments/MC005.md)

- 替代MC004的完整性不变、KEEP阻断研究动作、元层逐行同算子等表述；复合容差原假设等余下统计问题仍未证明。

## [MC006 · E0-D原诊断报告与正式技术报告属于同一证据批次，不另计确认](experiments/MC006.md)

- 与CPR002–CPR006是同批次不同报告层级；本记录不增加独立实验计数，348/448口径采用348。

## [META001 · Hermes mismatch：分歧采样没有稳定胜过覆盖式失配发现](experiments/META001.md)

- 隔离与随机性归因随后由META002修正；原数值保留。

## [META002 · Hermes phase-1审计：收益发生在H0被否定之后，隔离测试需要真实双环境](experiments/META002.md)

- 收窄META001的早期查询收益解释，纠正原隔离测试。

## [META003 · Hermes conflict：复测、扩展与诊断在预算和污染机制上有不同表现](experiments/META003.md)

- 原报告的无可区分动作、H1折衷拟合解释被META004取代；数值不撤销。

## [META004 · Hermes纠错：同点复测等价不等于所有新位置查询等价](experiments/META004.md)

- 明确替代META003原报告的no available action separates与H1 compromise解释。

## [META005 · Hermes fitter-first：固定历史后可逆隔离恢复持续污染任务](experiments/META005.md)

- 该轮append-only实现及候选保留细节仍由META007后续清理。

## [META006 · Hermes共享新拟合器：lookahead改了轨迹却未改善预算16终点](experiments/META006.md)

- 后续META007修正lookahead概率及候选保留实现，不能将其作为精确最优基线。

## [META007 · Hermes早期ExactDP：18世界报告保留，终点和空支持解释已被替代](experiments/META007.md)

- 复测覆写、只留首候选与非概率lookahead权重的清理记在本轮；本轮DP比较又被META008纠正。

## [META008 · Hermes DP修复：按终点分别求解、完整历史条件化和空支持回退](experiments/META008.md)

- 替代META007中budget4精确DP较差和holdout退化独归于先验迁移的解释。

## [META009 · Hermes完整支持：2340世界与共用能力的开环/自适应对照](experiments/META009.md)

- 修正小目录初始化即失去支持的比较前提。

## [META010 · Hermes最终终点：H4严格自适应增量，H6开环已为零](experiments/META010.md)

- 纠正fixed等于non-adaptive的混称：retest_then_cover已有反馈；openloop可用初始观测。

## [META011 · Hermes初始设计审计：位置生成机制改变部分计划但不改变H4残余](experiments/META011.md)

- 补充锁定filter仅按值、不按初始位置生成机制条件化的假设。

## [META012 · Open-world construction：四个预定义构造器跑通，未达盲评目标](experiments/META012.md)

- 旧README同时保留任务包未启动与六系统已运行文字，以运行报告限定阶段；当前能力由META014更新。

## [META013 · Open-world旧记忆接口缺口：有状态表示仍从m0=0预测](experiments/META013.md)

- 当前实现不能继续按旧m0=0限制描述，见META014。

## [META014 · Open-world capability_v1：16结构库存与历史初态修复已存在](experiments/META014.md)

- 更新META012/META013中只对construction成立的四命名构造与m0=0状态。

## [META015 · Open-world残余：漂移未解决，library/propose共享覆盖采样](experiments/META015.md)

- 收窄将capability收益统称为反馈/开放研究成功的解释。

## [META016 · 9/15统一交接：三类研究动作与能力/反馈2×2是候选框架](experiments/META016.md)

- 统一框架不能被追写为已完成跨领域迁移，也不能把旧项目负结果合并成单一原因。

## [META017 · 9/16有限精确搜索：搜索收益、信息缺口与同分策略更正](experiments/META017.md)

- 替代早期过程消息将131对统称为模型优化更好却真实更差的解释。

## [CPR001 · E0：旧守卫零报警是可触发性问题，不能充当安全证据](experiments/CPR001.md)

- 旧E0的BH最小p条件由CPR002修正；守卫计数本身未撤回。

## [CPR002 · E0-D检验器：BH晚排序触发修正，扩校准未稳定恢复检出](experiments/CPR002.md)

- 取代旧E0将最小p样本数量作为必要条件的解释；更正不撤销原拒绝数。

## [CPR003 · E0-D投影：同掩码24/24胜ARGMIN，只有3/24胜identity](experiments/CPR003.md)

- 连续边界后续由CPR004修复；对identity的3/24计数未改变。

## [CPR004 · E0-D集合一致性修复：连续覆盖与网格投影错配造成348次覆盖内损伤](experiments/CPR004.md)

- 替代覆盖与投影同集合的错误实现假设；早期过程将348误写448已由正式报告更正。

## [CPR005 · E0-D新守卫：阻止错误合层，也阻止相同输入下的正确去重影](experiments/CPR005.md)

- 纠正把可触发守卫等同真层身份识别的解释。

## [CPR006 · E0-D辨识与理论边界：相同当前接口不等于全部取证无解](experiments/CPR006.md)

- 限制任何从当前输入歧义推广至所有允许实验歧义的表述。

## [CPR007 · E0-E实际构造：TEST与alpha_target不同，ACQUIRE仅记录未执行](experiments/CPR007.md)

- 纠正将E0-E TEST当原BH或把ACQUIRE标签当实际采集收益的混称。

## [CPR008 · E0-E选择性收益：削掉背层损伤的同时丢失约一半有效修复](experiments/CPR008.md)

- 不把相对无否决改善写成相对identity全面成功。

## [CPR009 · E0-E反例与门槛：次级谷不是物理层身份，E1/E2/E3未执行](experiments/CPR009.md)

- ACQUIRE不是新增证据；E1/E2/E3计划与实际执行分开。

## [CPR010 · E0-F局部守卫：恢复修复能力，但低错误比例与身份损伤仍失败](experiments/CPR010.md)

- 防止把正修复保留比例、背层损伤比例和有害点率混成同一概率。

## [CPR011 · E0-F噪声反事实：定位一组误拒绝，但不能分摊全部因果贡献](experiments/CPR011.md)

- 收窄把E0-E所有损伤归因噪声或单一估计模块的说法。

## [CPR012 · E0-F真实接入盘点：均值ZNCC不能恢复逐视图层支持](experiments/CPR012.md)

- 防止将有效投影计数解释成已认证可见源，或将真实接口盘点解释为真实修复成绩。

## [OR001 · QCE 分叉交接：候选暴露与提交的前史](experiments/OR001.md)

- 所谓共同939样本实际STOCK仍保留1000，不能将旧消融解释视为完成结论
- 期限内首次成功风险条件不能直接作为修复后lifetime安全保证

## [OR002 · 道路重定位：密集地图天花板与41张时序重访](experiments/OR002.md)

- 29个无有效PnP和1个无正确位姿不能全部归因于检索或不可观测


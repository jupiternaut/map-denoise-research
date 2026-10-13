# A6：精确初态深度 vs 所在深度盒
2026-10-13，Asia/Shanghai；host liekkas。原始目标：保留有效几何并修复不足。

## 要区分的命题
H_interval：深度盒允许某错误初态借用盒内其他深度的面积/遮挡支持。点值必要约束将出现新精确矛盾。
H_remaining：点值必要约束仍允许替代材料解释。joint见证同时给出9材料与逐视图共享的3偏差；calibrated见证只有9材料，T_cal已边际化，不证明某组共同偏差可行。两者都仍松弛姿态/面积关联，不能宣布真实前向可行或物理不可辨识。
UNKNOWN不能充当任一命题证据。

## 输入与完整分母
冻结A4-V数据/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1/data、旧bounded outers与A5封存/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/outputs/polytope-v1。
36旧公开图像×2方法×9初态=648诊断；18同源配对来源×2η，不是324独立场景。
全部原A4 retained_ids中满足lo≤z≤hi的闭区间叶：calibrated494＋joint486=980点—叶配对。共享端点的两叶均纳入。原区间判定取A5对应5103叶记录中这980项，不重跑LP。
每诊断全部3分支global点值检查，共1944。连无原保留叶的18cal/24joint任务也跑global，不缩分母。总新增2924检查。整个科学矩阵是一项串行CPU作业，不是2924独立调参作业。
同照片、radius、B80、单次clip、ε=β=1/100、δ=1/10、9材料、3 joint bias、T_cal、同depthbranch、同LP证据算法。无深度分裂、传播、新候选、新参数扫描、新路由、训练或几何输出。

## 唯一干预
同叶主对照：旧geometry_bounds(branch,radius,lo,hi,B80)→geometry_bounds(branch,radius,z,z,B80)，材料和t界用原A4该叶界；cal行仍边际化原arm T_cal。
global一致性对照：同点值行，9原[0,1]材料（joint追加原arm T_cal）。每分支都执行，不因已有见证/矛盾早停。
所有浮点LP只提候选，沿用A5 polytope_filter：精确有理加权严格矛盾才CERTIFIED_INFEASIBLE；所有行/盒界满足才FEASIBLE_WITNESS；否则UNKNOWN。
点值与盒值的姿态radius相同。z固定不等于相机姿态固定，见证没有姿态坐标；本轮不把它当真实照片解释，不另作姿态搜索或名义姿态失败外推。

## 聚合与一致性
同叶集合空记SOURCE_EXCLUDED（历史排除来源），不得记新点值矛盾。任一见证→FEASIBLE_WITNESS；非空全证书→CERTIFIED_INFEASIBLE；其他→UNKNOWN。global三分支同规则。
源hull包含与源retained union包含分别记；不拿hull填空隙。
检验点面积区间嵌于同叶面积区间。由于d≥0，点行蕴含盒行；原传播界为其后果，global点见证应同时满足全部包含该z的同branch原叶界/盒行。精确global见证与同branch local矛盾共存必须机械失败。UNKNOWN与见证/证书差异可以是数值条件性，不硬改为答案。
保留980 interval→point三态转移，single-row与组合矛盾分账，UNKNOWN原因与成本。不拼接不同分支/叶的见证。

## 评价与主终点
先保存、重读648诊断和全部1944global/980local条，唯一身份完整校验；才读取旧private评价。
主终点仍η=1/10、真实depth9007/14、非负控、8错误初态×5来源=40/方法。数：原证据保留而点值被精确排除多少，仍有必要见证多少，UNKNOWN多少；这是诊断不是新修复数。报告全部η×method的162、正确18、偏移144、负控24、每深度非负控40。
实际生成depth/9材料（joint3bias）用于后验覆盖核验：72全局点行和所有正确初态同叶见证满足。任何真实初态被CERTIFIED_INFEASIBLE判除是机械失败；新求解UNKNOWN不以真值反写。
原A4与A5hybrid状态/输出只作冻结上下文，不改变KEEP/MOVE或评分成绩。科学正负不作为机械PASS条件。

## 执行与验收
[ ] 手工fixtures：点/盒分歧、区间嵌套、共享端点、全分支、empty/gap、0/1clip、signedlinks、同叶global冗余、伪证书和错配、完整记录先于private。
[ ] fresh源码审查：same-family/provisional，非外部物理证明。
[ ] 源/输入锁定，正式单批所有2924检查；main＋独立verify共用3600秒绝对deadline，不重置，超时INCOMPLETE。
[ ] 独立verify不导入新point_solver/runner，重建geometry/necessary rows，核对精确证书/见证、population、聚合、覆盖与CSV/JSON/summary。
[ ] 中文报告与检查点，旧阶段只读，矩阵结束即停。
同host现有open3d-019/bin/python -B、CPU线程1。不得GPU/SSH/API/安装/发布/部署。

## 技能实施
experiment-bridge要求fixture及fresh审查再正式执行；ablation-planner的fresh设计完整保存在私有trace。无可用research-contract模板则按同语义手写研究契约。此处用户“继续”授权该既定诊断，不引入新数据。

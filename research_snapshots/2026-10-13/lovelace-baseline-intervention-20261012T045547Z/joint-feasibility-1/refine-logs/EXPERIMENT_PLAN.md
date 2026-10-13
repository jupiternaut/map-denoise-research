# A5 消融执行计划

# A5：冻结深度盒上的联合可行性过滤

日期2026-10-12；host liekkas。唯一新写根/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1；/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1及所有历史目录只读。

## 原始目标与唯一问题
保留有效几何并修复不足。A4-V有界关系保住真参数却未增加主40修复；本轮问：原逐变量区间外包是否因丢失材料/视图偏差相关性，留下不能同时满足必要行的深度盒？
本轮是已有必要线性松弛的联合可行性检查，不声称完成真实非线性成像模型辨识。

## 冻结与分母
沿用A4-V已生成的36观测/324任务（18旧来源×η0/1/10），bounded同δ1/10，calibrated和joint。共72原外包，5103原保留叶；只遍历完整原retained_ids，不读真值选择叶。不再生成图像。
保留原姿态/噪声/校准/最终clip模型、三分支、深度端点、8轮结果、511盒、15/32mm、旧hybrid。无新增行/分裂/δ扫描/变量极值收窄。
基线box=冻结旧bounded；polytope=仅删除持精确不可行证书的原叶。648基线动作＋648新动作＋324KEEP=1620评价行。
主终点仍为η1/10深度9007/14mm的40信息偏移，每方法分别报。另全144偏移、18正确、原negative24、其他深度/模板。324任务不是独立场景。
所有叶、所有任务保留分母，无成功挑选。

## 叶级约束与证书
retained_tolerance：原depth区间重建旧geometry和necessary_rows，bounds取旧最终九材料界及joint三偏差界（cal使用原校准边界）。unresolved_budget：相同重建，bounds取旧记录初始材料/校准界；不追加传播，单独计成本。本矩阵实际budget叶为0，但实现不删除该路径。
upper原样；lower取负为Ay≤b。逐行计算min(A_i·box)>b_i是否已有单行矛盾，避免把补做单行归因于联合相关性。
浮点phase-I LP只产生候选：min s，Ay−s·1≤b，l≤y≤u，s≥0。仅在success且finite时取λ=-ineqlin.marginals，非负权重有理化上限1e9。实际精度由实现预先固定，不用数据选择。
独立重算a=λA、B=λb和m=Σ[a_j≥0?a_j l_j:a_j u_j]。仅精确m>B才可CERTIFIED_INFEASIBLE并删叶，不需要a=0。单行直接证书是特例λ_i=1。
FEASIBLE_WITNESS必须有精确有理y同时通过全部行与bounds；它只证明整个深度盒的必要线性松弛非空，既不证明真实成像可行，也不证明盒内指定初态可行。
无法精确核验、浮点失败/NaN/超时、零差额均UNKNOWN保留；UNKNOWN不能当可行或不可行。
不改变留下叶的bounds/trace/depth区间；仅更新各branch retained_ids/hull与union outer。空集明确KEEP/INCOMPATIBLE，无真空认证。

## 信息顺序与独立核验
方法只读旧公共图像/任务与sealed bounded外包；private不入选择。所有648新、648旧、324KEEP动作写出读回，唯一计数后才parse旧private评价。
独立核验器不导入新filter/runner：独立重建各叶geometry/必要行/box，逐证书/见证精确检验，UNKNOWN原样保留，核对无遗漏/无新叶、旧trace不改，outer/hybrid/actions/CSV/JSON/指标独立复算。
同一生成叶中真深度、九材料以及joint三个偏差见证仍须覆盖；正确初态KEEP。机械PASS不要求科学增量为正。
真实参数覆盖不是仅检查总体hull：须保留对应生成分支和同时见证。

## 预注册诊断链与验收
依次报告：三状态与single-row量→原叶删除/内部删除与凸包端点变化→主40错误初态仍在凸包数→新修复、相同、损伤/MSE/MAE。
分别报cal/joint、η0/1/10。真正相关性删除是原已传播叶、无单行矛盾而联合证书拒绝。
若只删内部叶、hull不改，则冻结hybrid可能不变；不得改成用非凸interval决策来追成绩。
若UNKNOWN多，零增量不能当联合松弛无帮助；若精确见证多而无行动增量，只能评价冻结盒级松弛。

## 运行边界
现有/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python /SciPy1.15.3，CPU单线程，无安装/GPU/API/SSH/发布/部署。
手工非确认夹具测试/计时→fresh源码审查→冻结→一次正式main+verify共3600秒绝对窗口，不重置；每LP默认最多5秒且受剩余预算约束。超时INCOMPLETE、留已完成工件、不缩分母或覆盖。
当前矩阵结束即停止；不添加真实新数据、其他δ或其他候选。
语义审查same-family/provisional；精确有限证书检查机械PASS范围明确。

## 方法说明来源
SciPy1.15.3官方linprog-highs接口、marginals符号与有限求解状态：
https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html
加权矛盾检验由本实验直接精确算术验证，不相信浮点infeasible标签。


## 执行顺序
- M0：手工signed rows微小矛盾环、可行相关盒、边界点、错误权重/变造证书、零差额、失败/UNKNOWN、预算叶路径与信息隔离测试。
- M1：fresh同族源语义审查，阻断项修复后复核；绑定最后源码版本与测试记录。
- M2：完整72case、5103原leaf后处理，648新/648旧/324KEEP全写后评价，计LP及geometry成本。
- M3：同一预算内完整独立核验并汇总主/次指标。
- M4：目标驱动报告、检查点、manifest；不扩大任务。

停止条件是完整矩阵和核验完成，或者共同预算耗尽；不是“成绩终于为正”。


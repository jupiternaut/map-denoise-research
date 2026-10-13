# A5 联合可行性：在冻结外包上排除相关性伪解
language: zh
host: liekkas
唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1。本阶段承接用户最新“继续”，覆盖父阶段已结束的具体任务限制；此前所有代码、输入、结果只读。
原始目标保留有效几何、修复不足。前阶段/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1 COMPLETE/PASS。本轮只固定该数据、bounded δ=1/10、全部几何/校准/行/传播/深度树与hybrid，在封存保留叶上增加联合必要线性约束可行性检查。不换数据/材料界/生成器/阈值/学习器/真值选点/部署。
比较完整两η×18来源×cal/joint：72原外包后处理，每72×9=648新动作＋648冻结baseline＋324KEEP，1620评价行。主终点保持非零η第三深度40任务/方法。每个原retained leaf都检查，保留其原深度区间，不新增深度分裂；处理未评估budget叶时重建相同必要行和其公开变量界，计为新增检查成本。
LP浮点仅产生候选证书/见证；只有精确有理加权矛盾证书独立核对后才能删叶。浮点不成功、证书不成立或无法找到精确可行点时保留并记UNKNOWN，不能当空集。精确可行点只证明必要线性松弛可行，不证明真实成像模型可行。记录FEASIBLE_WITNESS/CERTIFIED_INFEASIBLE/UNKNOWN。
旧几何候选位置不改。校准与全部clip/噪声/材料差行仍冻结。新非空outer仍按旧hybrid行动；空集明确KEEP/INCOMPATIBLE。
方法只读公开观测/任务和旧bounded outers，不接private。所有648新/648旧/324KEEP动作保存读回唯一计数后才读取旧private评价，参照不入选择。
root：protocol/runner/evaluator/tests/report。lp_certificate_author只写polytope_filter.py/test_polytope_filter.py。polytope_verifier_author只写verify_polytope.py/test_verify_polytope.py。plan与reviewer只读且不能读private。
main+独立核验一次3600秒绝对窗口，不重置；完整矩阵超时INCOMPLETE，不事后缩分母。CPU单线程。允许使用同host现有 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python（SciPy1.15.3），不安装依赖、不GPU/API/SSH/发布/后台循环。
只apply_patch编辑。手工非确认fixture测试与计时→fresh代码审查→冻结→正式执行。科学结果正负不作为机械PASS条件。此矩阵完成即停止。


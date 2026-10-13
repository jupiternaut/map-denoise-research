# 中点精修与区间投影补修：新样本确认协议

预注册时间：2026-10-11 05:42:17 Asia/Shanghai；确认尚未运行。
目标仍是保留有效几何并修复不足，不增加新评分器、材质拟合或未知拓扑自由度。

## 两个固定主张

C1（条件命题）：真深度在外包凸包 H 内时，midpoint-first hybrid 保留所有被认证的中点输出；在其余任务仅执行已认证的区间投影。每个任务的绝对误差及平方误差都不大于 midpoint，MOVE 数等于 projection。它不保证逐任务优于 projection。
C2（新样本经验）：是否能在新的深度与噪声下，严格增加小偏移安全修复数并降低 midpoint 的 MAE，同时保持正确初点零损伤。与 projection 和同信息残差网格的精度差异实测决定。

## 模型与固定参数

复用上一轮只读的五个成像模板、面积遮挡内核与连续深度外包求解器。模板为 lovelace_triangle、rectangle、layered_contrast、layered_same、layered_hidden。形状、颜色、尺度、后层深度720和白背景已知；只估计前层标量深度。三台平行相机，参考相机误差为0，两侧横向误差连续有界。
相机误差半径0、1/5、1 mm；通道非截断噪声硬界1/100。
本轮新深度 **6217/11、7864/13、8441/13 mm**，新种子 **41001、41002、41003**。
随机磁带前缀 hybrid-v1；相机误差及噪声五档倍数仍为 -1,-1/2,0,1/2,1。
135参数/噪声配置（5模板×3半径×3深度×3种子），不是135独立物体。
各配置共享同一观测，初态偏移0、±1/4、±3/4、±2、±60 mm；共1215初态。
外包域[540,660]，容差15/32 mm，最大511盒，未决盒保留。
残差基线深度网格540..660步长1/2，加初点；左右误差各五档，l∞光度残差。tie阈值1e-12；所有预测从相同精确成像模型生成。此基线是有限网格，不是连续全局最优。

## 五臂与核心动作

KEEP；certificate_midpoint；certificate_projection；certificate_hybrid；pose_profiled_residual。
hybrid先计算midpoint：通过严格正最坏收益证书就原样采用；否则计算projection，能MOVE才执行，否则KEEP或INCOMPATIBLE。保存branch=midpoint/projection/keep/incompatible。
所有选择器只能使用公开模型、观测、自己的初点和共享外包，不接收truth或state_group。
公共残差预测在生成确认观测前计算，不复用旧轮缓存；计入其成本。
所有决策写盘后才评分。旧四臂与新一臂在同一批新观测中比较，不把旧均值与新均值当配对实验。

## 主终点与预先分账

主要：small（±1/4,±3/4,±2，810任务）的改善数、损伤数、MAE/MSE；正确135初态是否全部保留。
paired检查：hybrid每任务误差≤midpoint，旧midpoint MOVE输出一字不改，hybrid MOVE集合=projection。
分开报告informative四模板与hidden；inside_prior/outside_prior；±1/4；三种子为重复噪声，不能扩大独立场景数。
hidden inside_prior不得产生所谓照片深度修复；outside_prior收益只来自先验范围。
计算成本包括共享外包求解，不能把轻量动作函数计时当端到端时间。

## 验收与停止

作者测试、fresh-agent代码检查在确认之前；记录完整审查痕迹。
全确认只运行一次，不看真值调参数；算术独立核验全部决策与汇总。
若135/135真值覆盖、hybrid零损伤、上述paired关系成立，则C1实验检查通过；C2需要small严格更多修复且MAE严格低于midpoint，不预设对projection/残差的赢家。
若C2不成立，保留负结果，不追加搜索。删除midpoint/删除projection补修已由两条单臂完成组件消融。
一个CPU确认作业，硬上限1800秒；GPU0；不得修改部署默认或推送仓库。

# 冻结共享偏差模型的未见种子确认

注册日期2026-10-12；host liekkas；唯一新根同AGENTS。用户“继续”承接上轮报告提出的未使用seed确认。本文在任何seed61004生成之前写入；已在现有runs的py/PROTOCOL/CHECKPOINT中搜索该整数，未匹配。没有试跑后再选择seed。

## 两项主张

C1：上轮clip联合模型相对同T_cal的简单校准，其新增修复及平均MSE收益能否在未使用噪声实现上复现？C2：真深度/真实隐变量覆盖、正确初态保留与负控能否同时保持？两项允许失败；不事后改参数或动作追正数。

## 冻结方法与源

只读上轮根`/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z`：使用原structured_solver.py与其依赖、原hybrid动作和独立verify_structured.py的必要行/传播/树/union检查函数。不得修改或copy后改算法。
生成器沿用`lovelace-shared-appearance-20261011T052300Z/run_shared.py`中的CASES/TRUTHS/OFFSETS及其原generate_observation，prefix保持`shared-appearance-v1`，唯一改变seed61004。旧三个seed61001/2/3结果仅为开发参照，不并作新确认样本。
新图像由未裁剪基础RGB直接加生成端偏差(-.01,0,+.01)，再裁剪一次到[0,1]。不得先clip基础图。生成器返回的额外calibration采样全部从公共输入中删除，不交给方法；同图像公开保证白背景仍可用。

## 固定矩阵

六模板×三深度×一个seed=18观测；每观测九原初态，共162任务，其中18正确、144偏移，偏移含24等色全遮挡负控。方法KEEP、calibrated、joint，每方法162动作，共486；两拟合方法各18冷求解，共36。没有历史外包或动作复用，没有raw单元或额外seed。
epsilon=beta=.01；camera radius=.2 mm；三个遮挡分支、depth[540,660]、511盒/branch、8轮、15/32 mm；两新法同图像T_cal，原动作器不变。每次求解清几何cache并计入背景准备。全部方法同图像/信息/树预算；不声称相同CPU时间。

## 流程和信息边界

prepare独立进程生成data/observations.json、tasks.json和private/truth.json。公共观测只含observation_id/unit_id/clipped/radius/observation，无case/seed/colors/实际pose/bias/额外calibration。任务仅含ID和incumbent；solver不接初态。冻结data作为当前生成输出，不冒充旧输入。
runner读取公共输入、产生所有方法的外包与动作。全486动作完整保存读回及唯一性校验后，才读取data/private/truth.json；评价只与生成真值比较。源码顺序检查不等于OS级隔离。
核验器不导入新runner/冻结solver作为计算答案；可复用上轮不同作者核验器和历史几何核，独立再生seed61004观测/真值、逐branch精确重算更新和证书、树/union、动作与全部JSON/CSV指标。历史物理核soundness仍不是外部独立证明。

## 主终点和判断

每方法144偏移MAE/MSE、改善/同分/损伤；calibrated→joint逐任务更准/同分/更差、新增/丢失修复；按六模板、三深度报告。一个seed不报告seed标准差或总体显著性。
18真深度/生成分支覆盖、18正确KEEP、24负控KEEP、材料与joint偏差覆盖、INCOMPATIBLE、预算未决、冷求解秒、平均外包宽度。集合包含与点估计误差分开；配对退步不得删掉。
有限确认正向条件：joint偏移MSE更低、新增修复>0，且覆盖/正确保留/负控保持、无初态损伤。若校准追平，简化；若失败，不改变方法重跑。本次只确认同生成族未见noise/pose实现，不称真实场景迁移。

## 五阶段与预算

M0准备代码与非确认测试；M1 fresh同族非作者完整源码审查；M2固定一个CPU主作业36求解；M3不同作者机械核验；M4结算报告。主批+核验共享一次1800秒绝对窗口，超时INCOMPLETE，不重置或缩样本。预计数分钟求解加核验，0GPU/API/安装/SSH。未见数据准备在M1通过后才首次运行。
不重跑A+B/C+D，不扩偏差排列或连续噪声，不加LP/学习器/新动作，不发布GitHub/GitBook或修改部署。

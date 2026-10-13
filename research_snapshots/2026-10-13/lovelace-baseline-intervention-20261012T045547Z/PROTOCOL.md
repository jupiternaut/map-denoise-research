# 基线70→80的单次成像干预

首次seed61005标签生成前注册；host liekkas；新根同AGENTS，旧确认根只读。
用户“继续”承接冻结确认，先诊断深档不动，不重跑旧确认或调算法。

## 两项主张
C1：改变观测几何能够使第三深度的信息模板产生更多有效修正、更低MSE，而不是依靠改变评分或动作。
C2：同条件的联合偏差依赖增量、真深度覆盖、正确KEEP与负控保持是否仍成立。C2的joint增量允许消失（说明新观测让简单法足够）。

## 理论预测（不是成绩）
针孔u=(X-c)/z。B70右侧远前缘=-90/z在z600越过u=-.15；B80远前缘=-100/z的阈值2000/3>660。因此深档多出rear-free的纯front-left边缘像素，破坏材质—深度缩放歧义。B80近前缘另有z600格线切换，不能声称第一档必获益。
none分支的单个边缘行面积A_f=11250(100/z²-.15/z)，reference纯左patch A_r=225000/z²；比值=5-3z/400。partial参考可能混rear，该比值不直接适用。full等色负控全域F subset hidden rear，仍对front depth不敏感。

## 矩阵与公共信息
原6CASES×3TRUTHS，seed61005。先一次生成原tape shared-appearance-v1:case:1/5:truth:61005 的pose两抽样及432噪声；B70与B80严格共用这些值。每条件18图像、162初态；总36/324/972动作/72冷求解。
同一场景/深度/偏移跨baseline共享pair_id。公共字段仅observation_id/pair_id/unit_id/baseline/radius/clipped/observation；private含case/truth/colors/branch/pose/noise/bias/seed。公共不含额外calibration采样。
基础无噪图+epsilon噪声+viewbias(-.01,0,.01)，仅最后裁剪[0,1]一次。
两方法从自身条件图像与public geometry算white-background mask和T_cal；同条件cal/joint必须一致，不要求两baseline的T_cal相同。baseline是公开相机元数据，不是GT。
冻：radius=.2、epsilon=beta=.01、depth[540,660]、3branch、511盒/branch、8轮、15/32mm；冻结necessary_rows、propagate、union与hybrid。新实现仅把相机位置作为显式geometry参数，不改变光度不等式或传播。
冷计时清所有新/旧geometry cache，计入背景准备。数据生成与求解分进程；全部动作存/读回后评价私有真值。

## 终点及规则
主终点：第三档5信息模板×8偏移=40任务每方法，B80相对B70的改善数、MSE及配对better/same/worse；该档8个负控另列。
全三深度、六模板、原正确初态与全部偏移均报告，不删除首档恶化或联合配对退步。
C1正向：joint第三档信息MSE下降且改善数增加，并且全图truth/branch覆盖、正确KEEP/负控保持及0初态损伤。若B80没有新修正，不改B再跑；若改善来自cal而非joint，采纳更简单解释。
C2同条件cal→joint增量另算；集合嵌套与误差不等价；覆盖/隐变量/INCOMPATIBLE/预算未决/冷CPU单列。
新相机条件+一个noise/pose实现是受控合成干预，非真实迁移。不作总体显著性检验、未见族泛化或普遍无损结论。

## 五阶段与预算
M0 B70身份、B80点几何被区间包络、full负控全域包含、共同随机性、指标负结果保留的非确认测试。
M1 fresh同族非作者完整源码审查，PASS后首次生成新数据。
M2唯一72冷求解/972动作主批；M3不同作者独立复算公共几何/约束/树/动作/指标；M4配对结算与机制解释。
主批+核验共1800秒原窗口，0GPU/安装/API/SSH。不扫描其他baseline，不加seed/raw/LP/学习器/轮数。未发布或部署。


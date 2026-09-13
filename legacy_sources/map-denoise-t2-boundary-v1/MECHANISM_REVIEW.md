# 专用滤波机制短审阅

2026-09-11。只读检查 MECHANISM.md、baseline/scalar_reference.py、baseline/plane_profile.py，并核对 common.py 的评价坐标。未修改冻结算法，未新增实验。

## 结论

**截距消元、排序切分和三维投影方向通过代码对应检查；没有发现需要改动冻结几何算子的符号错误。交付前有两项解释／评价口径应修正。**

## 已核对的核心

- `a*=mean(y)-d*mean(h)` 代回得到 `S-2d*C+d²*Q`，与 `_profile` 的 `c=row.sum()-prefix-hi*mean`、`q=lo*hi/n` 一致。d 非负时，高层取排序后的高端点；固定高层点数，交叉标签不能降低平方误差。熵项代码是 `n H(p)`，乘 `2 sigma²` 对应完整数据分类负对数似然的同尺度形式。
- `plane_profile` 将标准化 XY 斜率除以 scale，得到物理斜率；法向 `n=(-beta,-gamma,1)`，观测平面坐标 `s=z-(xy-center)·slope`。更新 `p'=p+(predicted-s)n/(n·n)` 满足 `n·(p'-center)=predicted`，正负号与归一化正确。
- 该更新同时包含估计的帧偏差去除和层内噪声投影，不是一个独立的完整位姿配准结果。当前最终几何评价先将输出旋回真值坐标，分别计算法向和三维误差，**没有仅用自家投影残差冒充几何准确度**。

## 应修正 1：消元公式适用的是哪一个实现阶段

MECHANISM.md 中“一维搜索、不需要同时猜全部帧位置”的说法，应限定为 `_profile` 和无参照 `scalar_profile_hard` 的硬分类候选。

`scalar_profile_lbfgs` 和 `plane_profile_map` 把 profile 用作初始化，随后 L-BFGS 的变量中仍显式包含每个 `a_f`，优化的是观测混合似然而非该分类熵目标；plane 还联合优化共同斜率。因此不能把完整 plane 候选描述成“解析消去全部帧偏差后只解一维问题”。

建议补一句：**“上述消元用于便宜的硬关联参考和初始化；最终似然／倾斜平面版本从该初始化出发，联合细化帧截距、比例和几何参数。”**

## 应修正 2：bias_rmse_mm 的方向定义

`common.generate` 先沿真值法向加入标量偏差 b，再旋转输入；`common.evaluate` 却直接将各方法返回的标量 bias 与原 b 相减。

- scalar 方法返回的是输入 z 方向截距变化，隐含位移向量 `b_est * e_z`。
- plane 方法返回的是沿估计单位法向的距离，隐含位移向量 `b_est * n_est`。

法向有旋转或估计偏差时，两种返回量不能不换方向就称同一物理偏差 RMSE。若要比较真法向偏差，应在评估器中将估计位移向量投影到输入坐标中的真值法向：`b_normal = b_est * dot(direction_est, direction_true)`，或分别记录完整偏差向量误差。方向只在评估使用真值，不能传给算法。

**不必因此重跑或修改滤波器。** 可以从保存的 bias、info.normal（plane）和输入旋转重算；若本轮不需要参数指标，也可不以跨方法 bias_rmse 排名。已正确计算的最终法向误差、层距与三维点误差不受这一标量口径问题影响。

除此之外，本次短审阅未发现需要纠正的消元／投影方向错误。

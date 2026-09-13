# 较强同信息 scalar 参考：profile 初始化与直接似然求解

## 交付与冻结

`scalar_reference.py` 已可调用，导出：

```python
METHODS = ('scalar_profile_hard', 'scalar_profile_lbfgs')
xyz_mm, bias_mm, info = estimate(inp, method)
```

冻结 SHA256：`ebc82a17ed431ee10056ef3c64113e8897f10821c673c4c0c99e469ef1773a9a`。
使用旧 Input；法向坐标为输入第三列，提供 sigma，普通方法不接受真值。
所有结果为开发探针，不是独立真实验证。

## 官方实现查找

在 `/srv/slam-research/grf` 和 `/home/grf/Documents/Codex` 的只读文件检索中，
没有定位到可直接调用的 JRMPC/BALM 源码。`matlab`、`octave`、`roscore`、`catkin_make`
均未在当前 PATH 找到；这不是机器全部磁盘均不存在的证明。

[BALM 官方仓库](https://github.com/hku-mars/BALM) 明确依赖 ROS、PCL、Eigen 和 catkin 构建。
其说明也指出初始位姿较差时关联/平面提取可能需要粗到细调整。
本轮没有安装 ROS 或下载一整套系统，因此**没有运行 BALM**。

JRMPC 的 [Inria 项目入口](https://team.inria.fr/perception/research/jrmpc/) 本轮浏览失败；
找到了 [公开 MATLAB 实现仓库](https://github.com/dkounadis/jrmpc) 与
[NumPy/PyTorch 移植](https://github.com/pcrresearch/JRMPC-PyTorch-Numpy)，
但没有将未审计、未运行的移植标为官方复现成绩。后者是后续可接入选项，不声称部署必然困难。

本轮可运行产物是下面明确命名的**自制强参考**，不是 JRMPC/BALM 替身。

## 算法

对每帧法向观测排序。给定共同间距 d 和分割位置 c，两个局部均值写为 a_f 与 a_f+d。
去掉 frame 截距后，残差平方和为

\[
\mathrm{SSE}_f(c,d)=S_f-2dC_f(c)+d^2Q_f(c),\qquad Q_f=n_0n_1/n_f.
\]

加上最大似然分类比例对应的熵项，在每个 d 上对各帧切分位置枚举并取最小，
再通过固定 33 点网格和四个局部标量搜索寻找共同 d。没有逐点真值标签。
这给出各帧截距、共同间距和每帧比例的 profile 初始化；不是独立拟合后简单平均各帧间距。

`scalar_profile_hard`：无参照时用上述完整数据分类目标和复杂度惩罚选择单/双层，
输出 MAP 层位置；有稳定参照时使用下面的联合数值求解，再输出 MAP。
故它是明确的混合参考，不是全场景纯闭式 profile 解。

`scalar_profile_lbfgs`：从 profile 派生三个初值，存在稳定参照时另加一个参照初值，
用 SciPy L-BFGS-B 直接优化

\[
-\sum_i\log\{(1-\pi_{f_i})\phi_\sigma(y_i-a_{f_i})+
\pi_{f_i}\phi_\sigma(y_i-a_{f_i}-d)\},
\]

有参照则添加相同 frame 截距下的参照似然。实现包含解析梯度，**不是 EM 更新**；
单层亦有同数值求解对照，最后用 BIC-style 复杂度分数选层数，输出后验平均层位置。
偏差由 frame 截距去整体平移规范得到。

约定与限制：共同法向、已提供 sigma、每帧比例自由；忽略 Input.bias_bound_mm，
没有严格 bias 盒约束。BIC-style 分数只是固定比较准则，不是校准概率、全局最优或可辨识性证书。
没有 WAIT 保护；同输入的单面重影与隔离双层仍会得到相同输出，相关损失保留。

## 最小实际运行

旧九个几何/观测族×两个种子 `91113,91117`×五方法，共 **90 个保存输出**。
输出先保存再交评估器；运行目录：

`/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/baseline/`

下表是两种子的法向 MAE 均值，mm：

| 条件 | 逐帧均值 | Open3D ICP＋同后滤波 | 旧 joint | profile-hard | profile-LBFGS |
|---|---:|---:|---:|---:|---:|
| 平衡交叉 | 0.02602 | 0.11477 | 0.01618 | 0.01718 | 0.01622 |
| 不平衡交叉 | 1.20568 | 1.47236 | 0.02060 | **0.01868** | **0.01926** |
| 低信噪比 | 1.18689 | 1.17267 | 1.18451 | 1.99929 | 1.23893 |
| 空间偏差失配 | 0.49818 | **0.14095** | 0.53157 | 0.55639 | 0.54237 |
| 离群失配 | **1.04584** | 1.05016 | 1.07963 | 1.08109 | 1.07944 |
| 隔离双层无参照 | 4.00000 | 4.00000 | 4.00000 | 4.00000 | 4.00000 |

所有常规返回有限；所选 L-BFGS 解均报告成功。单/双层加参照目标的有限差分梯度 L2 差
分别约 `7.34e-6`、`7.37e-6`（差分步长 1e-6，固定小测试），支持基本实现检查，
不是进行形式完备性验证。

18 个案例算子调用的中位时间：profile-hard **6.61 ms**，profile-LBFGS **20.25 ms**；
旧 joint **36.03 ms**。CPU/BLAS 1 线程，全部小于 2400 点；此处不是整地图速度。

## 对研究判断的意义

这个参考明显强于“逐帧去均值”在不平衡交叉样本上的表现，并且达到旧 joint 的同一数量级。
因此，新候选不能仅凭击败逐帧均值宣称独特增量；应与直接似然求解和便宜 profile 参考比较。
反过来，它验证了 observed gap 信号不完全依赖特定 EM 代码。

低信噪比和非标量偏差仍有明显缺陷。较好的求解器没有改变所需几何证据，也没有使受限模型
自动适用于真实地图。代码已冻结交给根线程统一扫描，不再依据后续结果改参数。

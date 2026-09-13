# V7 的两个数学构造：关联测度与几何决策

日期：2026-09-12。这里只证明有限模型内的小命题，并用独立 NumPy 玩具核验。
这些数值不是 V7 数据集效果、真实几何正确性或新颖性证据。主算法仍按已冻结计划
使用各点候选集合上的均匀先验；以下没有悄悄增加一个新的主实验先验策略。

## 1. 把关联写成测度分解，而不是给未知表面分配固定点数

令观测经验测度为 μ=Σ_i a_i δ_(x_i,z_i)，其中 a_i=w_i/Σ_j w_j。
取关联质量 γ_ig=a_i r_ig，并约束 Σ_g γ_ig=a_i。
这保留每个输入点的质量，但表面的总质量 Σ_i γ_ig 由证据决定；不强行令各面等大。
这里是单边质量约束的熵正则关联，不应直接命名成具有两个指定边际的平衡最优运输。
有关质量边际和熵正则的标准背景见 [Peyré–Cuturi 原作者专著](https://optimaltransport.github.io/pdf/ComputationalOT.pdf)。

固定合法候选集 C_i、候选先验 π_ig>0（g∈C_i，行和为1）、σ>0、非负点权重，定义

\[
c_{ig}(β)=\frac{(\tilde z_i-x_i^Tβ_g)^2}{2σ^2},\quad
F(r,β)=\sum_i w_i\sum_{g\in C_i}r_{ig}\left[c_{ig}(β)+\log\frac{r_{ig}}{π_{ig}}\right].
\]

### 命题：同一软目标的两步交替不会增加 F

记 Z_i=Σ_g π_ig exp(−c_ig)，q_ig=π_ig exp(−c_ig)/Z_i，直接代入得

\[
F(r,β)=\sum_iw_i\,\mathrm{KL}(r_i\Vert q_i)-\sum_iw_i\log Z_i.
\]

所以固定 β 时 r=q 最小化 F。固定 r 后，所有熵项不再依赖 β，按 w_i r_ig
对**原始去偏测量**做加权最小二乘也最小化相应部分。两次精确更新均不增 F。
这是标准 EM 变分视角在本任务中的直接应用，不是新定理；原始参考为
[Neal–Hinton 的 EM 论文](https://www.cs.toronto.edu/~hinton/absps/em.htm)。

秩亏只意味着参数可能不唯一，任一确实最小化该加权残差的解仍有上述性质。
但数值截断后必须检查实际目标；不以数值求解器名称代替检查。空分量保留旧参数即可。
改变候选、先验、权重、σ或观测，必须重新说明目标。最终硬化标签、共享斜率再拟合
属于后续决策/其他模型，不包含在这个软目标单调承诺中。

### 表示细化反例：复制一个名字不是获得一条新证据

两个相同代价的面，先验 (1/2,1/2)，责任度也为 (1/2,1/2)。仅把第一个面写成两个
几何完全相同的候选，再用均匀先验 (1/3,1/3,1/3)，两类几何的聚合责任度变成
(2/3,1/3)。这是**更改了先验测度**，不是 softmax 的实现错误。

若只是同一测度换一种表示，应将原先验质量拆分，例如 (1/4,1/4,1/2)。精确复制面
的代价相同，指数项可相加，因此聚合责任度仍是 (1/2,1/2)。非等代价的其他候选也
不影响这个证明。允许两个副本独立变形则扩大了模型类，已不是单纯表示细化。

对 V7 的可执行用途：记录候选覆盖与数量；用精确复制测试检查“改变表示”与
“改变先验/模型容量”的区别。不为追求这个不变性合并几何接近的面，否则可能误并
真实薄层。这是元数学视角在这里的具体落点：区分模型的写法与它表达的假设。

## 2. 后验估计与几何输出应使用明确的损失函数

假设候选平面在法向坐标 −2 mm、+2 mm，某点的模型内责任度恰好各为1/2。

| 动作 | 输出 | 到候选面并集距离 | 模型内真实对应坐标的期望平方误差 |
|---|---:|---:|---:|
| 后验均值 | 0 mm | 2 mm | 4 mm² |
| 确定性硬选其中一个面 | −2 mm | 0 mm | 8 mm² |

因此，均值是平方损失的合理动作，却可能把不确定的两面写成空腔中的伪面；硬选择
留在一个候选面上，却不能保证对应点平方误差更好。最大后验标签在模型内0–1标签
损失下是最优动作，这与几何平方损失不同。以上等后验反例只说明损失不等价，
不证明真实场景中哪种输出总体最好。

V7 先采用计划中的硬关联输出，并记录责任度熵、前两名差值和实际位移。它们是
模型内歧义指标，未经独立校准不能作为误删率、误关联率或安全风险保证。
字典缺了正确面时，一个责任度为1、熵为0的模型仍可完全错误。故不新增一个利用
低熵直接通行或高熵直接拒绝的门来改变固定输出支持。事后可检查这些分数是否真的
预测关联错误；这项检查与几何收益分别汇报。

## 可复现核验

- [代数验证程序](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/math/verify_math.py)
- [六项单元测试](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/math/test_verify_math.py)
- [数值记录](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/math/VERIFICATION.json)

运行环境沿用既有 Open3D 环境中的 NumPy，不安装新依赖；验证程序拒绝覆盖已有
JSON。固定种子913107只用于这里独立生成的160点代数玩具，不计为主数据确认种子。

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -m unittest discover -s /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/math -p 'test_*.py' -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/math/verify_math.py --output /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/math/VERIFICATION.json
```

第二条命令仅首次生成时使用；再次运行可省略 `--output` 比较标准输出。数学检查的
作用是排除实现或解释混淆，是否改善地图仍由相同数据、预算和几何指标的主实验决定。

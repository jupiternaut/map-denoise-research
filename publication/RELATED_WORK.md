# 精确搜索实验的相关工作与新颖性边界

2026-09-16 发布说明。新写了实现，不等于发现了新理论。

| 本实验部分 | 已有研究参照 | 不能借用的结论 |
| --- | --- | --- |
| 有限语法内搜索程序 | [Syntax-Guided Synthesis, 2013](https://escholarship.org/uc/item/1g67m7hp) | 穷举不是新的通用合成算法。 |
| 计算能力、样本与泛化分离 | [Valiant, A Theory of the Learnable, 1984](https://web.mit.edu/6.435/www/Valiant84.pdf) | 本实验不证明新的可学习性定理。 |
| 查询结果改变后续动作 | [Golovin & Krause, Adaptive Submodularity](https://arxiv.org/abs/1003.3967) | 未证明相关结构条件，不能移植其近似保证。 |
| 同一代理目标下部署行为不同 | [D'Amour et al., Underspecification, JMLR 2022](https://jmlr.org/beta/papers/v23/20-1335.html) | 本实验平局案例不是首次发现这种现象。 |

本次材料的定位：**可复现的有限实例与错误归因诊断**。131 个逆转全部发生于代理目标平局，
因此不能描述成“严格改善目标导致真实表现变差”。可复现性和反例有复用价值，外部学术创新性仍待证明。

P=NP 是动机中的假设，不是实验结果。穷举有限实例既不实现未知的多项式算法，
也不能消除不可辨识性、解释集缺失或真实场景信息不足。

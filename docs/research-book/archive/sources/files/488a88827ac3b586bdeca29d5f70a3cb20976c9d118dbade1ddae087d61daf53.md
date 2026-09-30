# 两端构造 V3：原型与开发实验报告

## 结论

已得到两个可运行构造及配对消融。24 个可评分原合成案例的表面误差中位数：旧 fast 0.656137 mm，共享偏差表面图 0.282258 mm；两个中位数之比对应下降 56.98%。这是公开开发案例上的信号，不是独立确认，也不是逐案例改善率的平均。

关系图端更值得继续投入；不平衡运输在本轮固定预算下没有优于平衡运输。真实片区中旧方法的大幅改写明显减少，但新方法恢复到原测量云的误差仍接近 identity，尚未建立真实几何精度收益。

## 实际交付与范围

- 完整运行 570/570 个输出；27 原合成案例（24 唯一真值可评分、3 标量歧义）及其 12 个子采样/坐标变换诊断；2 个真实场景、6 片区×3输入。570 输出不是570场景。
- 新估计器只接收 XYZ、扫描 ID、提供的 sigma；无 GT 法向、关联、逆扰动或 clean 参考。此轮未使用射线、盲估 sigma、全六自由度位姿或 CUDA。
- wall time 71.190 s，方法累计 67.218 s；Python 进程峰值 RSS 591712 KiB（包含 Open3D 导入，不是 GPU 显存）。报告、图和独立审查不在该 wall time 内。
- 1919 个受保护历史文件哈希未变；预留种子仍未使用。

## 算法不是同一公式加参数

A：同帧近邻差分估计共同方向；局部单元可有 1/2 个表面；不同单元通过同一扫描的共同偏差连接，交替估计局部关联、表面和扫描偏差。local_only 使用相同法向、局部拟合和支持条件，仅关闭共享偏差。

B：把扫描表示为实测位置、局部方向和采样密度代理质量；balanced/UOT 求跨站软匹配。匹配只驱动整扫描共同法向平移，不用运输重心替换点坐标。这样纠偏阶段严格保持同一扫描内任意点对相对向量。之后再独立测试相同 local_only 后置滤波。它仍是表面测度启发的原型，不是完整 varifold 配准或新的层论算法。

完整方程与有限命题：[CONSTRUCTION.md](/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v3/CONSTRUCTION.md)。

## 1. 原合成全条件：表面误差

每格三种子的中位数，单位 mm，越小越好。sigma=1 mm；bias 是逐扫描共同法向偏差 RMS。

| 真层距 | 偏差 RMS | Old fast | Open3D ICP + XYZ | Shared-bias surface graph | Balanced + local | Unbalanced + local |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.0428 | 0.0480 | 0.1602 | 0.1619 | 0.1608 |
| 0 | 4 | 0.2163 | 0.2228 | 0.1602 | 0.3488 | 0.5895 |
| 2 | 0 | 0.4339 | 0.3978 | 0.2519 | 0.2517 | 0.2519 |
| 2 | 4 | 0.4548 | 0.4052 | 0.2520 | 0.3999 | 0.5851 |
| 4 | 0 | 0.9458 | 0.6221 | 0.3203 | 0.3198 | 0.3200 |
| 4 | 4 | 0.8466 | 0.6265 | 0.4945 | 0.5495 | 0.7378 |
| 8 | 0 | 1.1101 | 0.9937 | 0.4451 | 0.4408 | 0.4371 |
| 8 | 4 | 1.1163 | 0.9711 | 0.4798 | 0.6405 | 1.0316 |

**必须保留的反例：** 无偏差单平面旧 fast 约0.043 mm，新图端约0.160 mm；局部化牺牲了简单场景中的全局平均收益。不能写成全面优于旧方法。

![全部条件](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v3-lnx049yd/figures/condition_errors.png)

## 2. 几何和成本一起看

下表对原24例分别取中位数。sample coverage 依赖原采样，不是连续表面完整度；法向角与同XY拟合间距使用GT来源组，仅作评估。

| 方法 | 表面 MAE mm | 来源对应 RMS mm | 1 mm采样覆盖 | 来源表面偏转° | 单例 ms |
| --- | --- | --- | --- | --- | --- |
| Input | 1.8914 | 2.5602 | 0.493 | 0.380 | 0.03 |
| Old XYZ | 1.7496 | 2.1938 | 0.369 | 1.078 | 19.90 |
| Old fast | 0.6561 | 0.8589 | 0.817 | 1.350 | 9.97 |
| Open3D ICP + XYZ | 0.5046 | 3.0566 | 0.260 | 1.119 | 26.08 |
| Local surface only | 1.6616 | 2.2108 | 0.474 | 0.539 | 36.75 |
| Shared-bias surface graph | 0.2823 | 0.4348 | 0.957 | 0.401 | 149.91 |
| Balanced transport | 0.8828 | 1.1483 | 0.648 | 0.224 | 344.27 |
| Unbalanced transport | 0.9393 | 1.2141 | 0.611 | 0.245 | 345.19 |
| Balanced + local | 0.3550 | 0.5424 | 0.915 | 0.331 | 380.27 |
| Unbalanced + local | 0.4487 | 0.7243 | 0.844 | 0.269 | 380.39 |

每例一次 warm-call、固定单线程 CPU 的探索计时；不同方法预处理和工作量不同，不是优化后性能下界。Open3D ICP 加自研后处理仅是明确对照，不能代表 JRMPC/BALM 或整个学术前沿。

## 3. 同输入消融与观测变化

| 输入 | 方法 | 案例数 | 表面 MAE 中位数 mm |
| --- | --- | --- | --- |
| full | Local surface only | 12 | 3.2903 |
| full | Shared-bias surface graph | 12 | 0.2876 |
| full | Balanced + local | 12 | 0.5345 |
| full | Unbalanced + local | 12 | 0.7445 |
| density | Local surface only | 4 | 3.5706 |
| density | Shared-bias surface graph | 4 | 0.3766 |
| density | Balanced + local | 4 | 0.4318 |
| density | Unbalanced + local | 4 | 0.6196 |
| partial_overlap | Local surface only | 4 | 3.4971 |
| partial_overlap | Shared-bias surface graph | 4 | 0.3818 |
| partial_overlap | Balanced + local | 4 | 0.5454 |
| partial_overlap | Unbalanced + local | 4 | 0.8036 |

density/partial_overlap 只来自一个已曝光种子的四种场景；不等价于四个新数据集。只按当前观测横向位置和点序子采样，同步保留来源与评价数据。

平衡/不平衡都使用24次内层、3次外层预算。平衡列归一残差接近零不代表行边缘也收敛；应以 metadata 里的双边残差判断。不平衡质量更少不意味着自动选对物理表面。

## 4. 真实输入：减少改写，尚非真实去噪成功

| 方法 | 零注入改写 mm | 平移后对原测量 RMS mm | 平移+旋转后 RMS mm |
| --- | --- | --- | --- |
| Input | 0.0000 | 3.5355 | 3.6947 |
| Old XYZ | 42.3024 | 42.4520 | 42.4085 |
| Old fast | 42.4432 | 42.5917 | 42.5080 |
| Open3D ICP + XYZ | 42.6576 | 42.8443 | 42.8014 |
| Local surface only | 0.8041 | 3.6425 | 3.7089 |
| Shared-bias surface graph | 0.8430 | 3.5775 | 3.6955 |
| Balanced transport | 0.2233 | 3.4746 | 3.6818 |
| Unbalanced transport | 0.2045 | 3.4883 | 3.6515 |
| Balanced + local | 0.8673 | 3.6198 | 3.7512 |
| Unbalanced + local | 0.8650 | 3.5899 | 3.7247 |

这里原测量云不是独立真值，真实片区也不是毫米级真薄层标注。新图端偏差注入后表现接近 identity，说明尚未证明它能恢复真实扰动。零注入少改写只排除了旧模型的大范围几何压平，不证明去噪收益。

## 5. 台阶是否变成斜坡

![所有方法同坐标输出](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/exploration-v3-lnx049yd/figures/world_step_4mm.png)

例子固定为 gap4 / bias4 / seed912101，所有点和所有方法使用同一坐标范围。虚线只在评估图表示已知合成表面，不提供给算法。图端仍有未校正/未支持点；不能称完全恢复。

GT、塌缩、斜坡、整体偏移100 mm负控见 NEGATIVE_CONTROLS.json；自报K变化不改变几何分数。所有旋转诊断只逆一次共同刚体换坐标，不逐层/逐帧对齐。

## 研究判断

1. 可以继续投入：同帧关系估方向、局部表面与共享扫描偏差的组合产生了开发证据。跨片区共享量有作用，不只是换名。
2. 不能从中推出：所有指标改善、真实几何改善、比对口联合配准方法更好。正常单面回归和计算成本已经可见。
3. 运输端暂不淘汰：三轮外层有总位移预算上限，必须通过质量—计算预算对照区分求解不足和关联构造不足；若增加预算也不改善，再改关联。
4. 下一次独立确认前应冻结完整方法及所有选择规则；不要用这批开发结果当未见证据。真实场景后续需要局部输入/参考对齐与独立几何证据。

## 复现

源码目录：`/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v3`。冻结快照在本运行 `source/`，CSV与NPZ均保留。可用项目当前入口重新跑一个新目录：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python \
  /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v3/run_exploration.py
```

运行使用原机 liekkas 和既有绝对数据/算子路径。源码快照是审计记录，不是脱离现有数据环境即可执行的便携包。

## 追加：运输预算补测已完成（不替换上面的冻结结果）

新增独立96份输出，仅改外层3/6/12轮，仍使用公开开发seed912101。四个完整输入上，不平衡运输+local的平均表面MAE为0.6828 / 0.3006 / 0.2994 mm；同预算平衡为0.3713 / 0.2981 / 0.2981 mm。六轮完整管线平均耗时约0.70–0.71 s，是三轮约0.37 s的额外成本。

这支持“首版复杂端存在求解预算不足”，不支持“不平衡运输已优于平衡”。该4例平均数不能与本报告24例中位数直接混合；部分重叠及结构指标反例仍需同时看。完整[预算补测与限定](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/transport-budget-v3-mxi1kprj/READOUT.md)。

实现核验：13项新测试、30项原测试通过；570输出检查、360个合成表面评分独立重算与CSV完全一致。核验不等于真实几何或创新性证明。

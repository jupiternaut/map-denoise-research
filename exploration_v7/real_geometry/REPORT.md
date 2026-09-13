# V7 真实支线：轴向残差不等于几何噪声

日期：2026-09-12；主机：liekkas。此支线只做真实表示诊断，**没有生成去噪候选输出，
没有独立真实几何 GT，不能把以下亚毫米残差叫作恢复精度**。

主运行：[/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-x5ximvmn](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-x5ximvmn)。

## 1. 明确的新判断

原来“PCA 方向下 DA 局部残差 18–38 mm，所以所有单元被拒绝”并不意味着这些表面
真的有厘米级噪声。至少存在两种不同情形：

- **da_thin：同一个旧拟合面其实很贴近观测，但固定轴几乎与面平行。**
- **da_wall / da_junction：旧受约束图表没有拟合到贴近观测的局部面；不受该轴图表限制的
  正交平面，可以解释很多留出观测。**

这给下一种构造提供了具体方向：片区平面使用单位法向和偏移量表示，纠偏方向不要被
一个全片区 PCA 轴永久锁死。它还不是“把轴换掉就一定去噪成功”的结论。

## 2. 对同一几何面的精确换算

旧单元在归一化坐标下拟合 `z = c_k + beta_u*x_u + beta_v*x_v`。先以实际 mm 尺度还原
`a=beta_u/scale_u, b=beta_v/scale_v`，再有

\[
 d_\perp=|r_z|/\sqrt{1+a^2+b^2},\qquad
 \kappa_{axis}=\sqrt{1+a^2+b^2}=1/|n^T e_z|.
\]

这只是同一面的等价坐标表达，可以逐点核验；它没有假设真实噪声分布。
其“元”层价值在于区分**坐标表达的事实**和**真实噪声的假说**，没有调用元数学定理。
若未来用正交似然替换轴向似然，还需变换测量协方差；只换残差却保持同一个 σ，
不是同一个概率模型，不能直接比较旧、新训练目标值。

复用六片区，各 zero / normal_translation 共 12 个保存状态，重放 84 个局部拟合，
全部实际支持掩码精确复现。未修改门限、σ、模型或输出。

仅列 zero 条件：

| 片区 | 同一旧拟合的高度中位残差范围 mm | 换算正交中位残差范围 mm | 轴条件数范围 |
|---|---:|---:|---:|
| da_thin | 18.958–26.923 | 0.241–0.547 | 49.08–85.63 |
| da_junction | 31.677–36.462 | 28.472–32.621 | 1.10–1.26 |
| da_wall | 22.495–38.395 | 21.216–34.650 | 1.00–1.81 |

zero 全部 42 个单元中，31 个旧高度残差门失败；其中 5 个（全部 da_thin）换算后的
正交残差中位数 ≤5 mm。**这是计量解释改变，不是五个单元已经安全获准修改。**

完整逐单元记录见 [PHYSICAL_REPLAY.csv](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-x5ximvmn/PHYSICAL_REPLAY.csv)。
这些旧残差条件于保存的全输入方向和估计帧偏差，不是独立留出。

## 3. 小规模训练／留出检查

六片区各选三个空间分布锚单元，共 18 个。锚点按输入坐标选择，不按残差选择。
每站沿其单元内主方向划分八个经验分位空间条带，第 1、5 条作为留出；其余训练。
原尺度与半尺度都评价同一个半尺度核心留出集，共 562 个不同留出点。

- 上游全输入 PCA/cells 仅用于固定空间设计及轴条件诊断。因此是**给定空间设计的后级
  留出**，不是完整管线的未见场景验证。
- 拟合面仅使用训练 XYZ；不使用全点偏差、不使用留出点拟合、更新责任度或权重。
- 固定 K=1 或 K=2，以正交 TLS 拟合；K=2 使用四个输入派生起点、最多 12 次分配／重拟合。
  留出点只对已经冻结的训练面取最近距离，不再拟合。
- 跨站联合为总共 K 面（3K 几何自由度）；分站独立两站最多 2K 面（6K 自由度）。
  两者**总容量不相同**。这不是“新增关联模块”的公平强基准，也不是新方法发明。
- 半尺度可能缺少训练点或秩不足：显式记为 UNFIT，不能把这些留出点悄悄删掉后宣称提升。

下表仅列**原尺度、跨站联合**；统计量是“可评分的站×单元中位残差”的中位数，
不是池化全部点的平均、不是墙体定位精度。K=1/2 在这些行使用相同留出点。

| 片区 | 留出点数 | 单面正交残差 mm | 双面正交残差 mm | 单面沿旧轴所需移动 mm |
|---|---:|---:|---:|---:|
| da_wall | 237 | 0.721 | 0.430 | 234.997 |
| da_junction | 151 | 0.257 | 0.162 | 138.715 |
| da_thin | 82 | 0.238 | 0.165 | 31.759 |
| cy_wall | 43 | 2.018 | 2.585 | 4.469 |
| cy_junction | 29 | 21.803 | 1.357 | 79.648 |
| cy_thin | 20 | 0.581 | 0.326 | 0.581 |

读法：

1. DA 三片区训练出来的简单局部面已能较好解释不少留出观测，但这些面常几乎平行于
   旧全片区 PCA 移动轴。继续在该轴上放大步长，可能只放大不适定性。
2. cy_junction 的双面解释明显优于单面；它提示表面混合/几何容量问题，而不只是 σ。
   两面模型容量更大，最近面距离也不惩罚多余面，所以仍不能据此证明真实面数为 2。
3. 半尺度并不普遍更好。例如 cy_wall 双面的四个可配对站×单元全部改善，
   但 cy_thin 双面只有 1/4 改善；不能提出“永远缩小邻域”的规则。

所有 272 条结果都保留：228 条可评分，12 条 UNFIT，32 条没有核心留出点。
分站二面在小尺度更容易无法拟合，不能与联合方法不同覆盖下的平均直接比较。
详见 [HOLDOUT.csv](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-x5ximvmn/HOLDOUT.csv)
及 [SPLIT_DESIGN.json](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-x5ximvmn/SPLIT_DESIGN.json)。

## 4. 开发过程和验证

首版统一 8×8 网格留出受扫描密度不均影响，部分站几乎没有训练点。原始运行完整保留于
[/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-rlp6r6j5](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-rlp6r6j5)。
第二版换成预定义分站空间条带以缓解划分失衡，不按几何残差选点。这些数据和两次划分
都已公开用于开发；第二版不是未见确认。两版源码分别保存在各自运行目录。

- 几何与划分单元测试 7/7；单位归一化还原、平行轴不伪造有限位移、平面拟合不改输入等。
- [AUDIT.json](/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/real-geometry-v7-x5ximvmn/AUDIT.json)：
  使用另一种平面矩阵公式重算 228 条评分、1140 个数值，核查 272 次训练／留出不相交。
- 417 个输入、V6 保存结果及复用源码前后哈希不变；旧文件未修改。
- 主诊断墙钟 0.555 秒（含读入/哈希/输出），本地址空间 VmHWM 75,936 KiB。
  `ru_maxrss` 同时记了 636,160 KiB，可能包含执行前进程生命周期峰值，不能与 VmHWM 混称。
  这不是完整去噪算法耗时，也没有测 GPU。

## 5. 对主线的具体影响

保留 V7 合成重关联主线；真实支线给后续接口一个明确要求：

**候选平面提供单位法向、偏移及残差协方差，关联用物理距离或对应测量似然；
动作空间应考虑局部面法向／扫描位姿，而不是默认沿一根全局轴移动。**

先在相同输入上实现这一局部表示对照，再谈恢复收益。当前证据不支持调整 σ 后就能
恢复，也不支持全局 PCA 或轴向模型必须永久禁用。低残差面之间仍可能存在偏差、
真实多层与重影混淆；这项诊断没有解决它们。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/real_geometry/run_diagnostic.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/exploration_v7/real_geometry -p 'test_*.py' -v
```

每次诊断在 `/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1/` 下新建唯一目录，
不覆盖已有结果。`audit.py` 接该新目录为参数，并新建 `AUDIT.json`；已审计的目录不覆盖。

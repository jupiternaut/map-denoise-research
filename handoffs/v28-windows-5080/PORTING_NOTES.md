# GPU 迁移笔记：哪些必须等价，哪些值得优化

源码基线：GitHub `2cf73a80f5e41a5d126ebdc8321c0cd6c61f24f9`，本包 reference/package/v28_closeout。路径/行号均针对该固定版本；BUNDLE_MANIFEST 记录实际导出字节。

## 1. 实际调用链

| 位置 | 已核实的行为 | 第一版处理 |
|---|---|---|
| runtime.py:95–118 | 输入、射线、PCA法向、anchor、score/solve | CPU准备；只替换score调用 |
| surfacelet.py:36–159 | 20假设×49偏移×4来源评分 | GPU主目标 |
| direct_evidence.py:30–100 | 相机规范化、双线性采样、ZNCC | 相机CPU；采样/ZNCC GPU |
| surfacelet.py:162–185 | 分数聚合、字典序选择 | 原CPU实现 |
| surfacelet.py:188–303 | 深度特征、fit/all、缺测标记 | 原CPU实现 |
| graph_field.py:7–17、120–133 | PCA和插值 | CPU |
| runtime.py:124–150 | 特征、插值、候选坐标 | CPU |
| runtime.py:34–80 | 模型哈希、预测、路由、随机对照 | CPU |

**不要走错路径**：`graph_field.build_graph/solve_field` 和 `direct_evidence.score_patches/aggregate_scores` 虽然存在，但不在当前 `construct` 主路径。第一版不是“把图优化器改CUDA”。

## 2. 固定算法账本

- 单位mm；更新 `p + offset * unit(p-reference_center)`，不是沿法向，也不是只改camera-z。
- normal bank：PCA k8/k24/k64 + camera-fronto。normal最大绝对坐标分量取正，用原PCA约定。
- supports顺序：full/left/right/top/bottom，`dx<=0`、`dx>=0`、`dy<=0`、`dy>=0`，中线重叠而非互斥。
- hypothesis顺序 normal-major；A固定index5（k24-full），B搜索0..19。
- 7×7 patch；全片49像素、半片28像素，至少ceil(80%)即40/23个共享有限像素；总体标准差严格>1e-4。
- offsets 49个：[-6,6]步长0.25，含精确零；每次20假设，不按GPU快慢缩减。
- voxel1.5mm、每体素原输入首点、最多1000 anchors，超过后seed20260922无放回抽样并排序。
- 全点PCA、anchor选择、插值按完整输入做，不能逐空间块重新定义邻域；score批大小是性能参数。
- 四来源顺序不变。all候选取最佳三源、至少两有效；fit只读源0/1决定提案，然后读源2/3计算保留视图特征。
- all主臂的来源2/3已参与提案，不能把其特征叫独立验证证据。
- primary `post_A_keep`；secondary `post_AB_keep`。正增益预测才MOVE；预测是mm²收益，不是概率。
- KEEP行坐标精确取原数组；不得经过投影后再“还原”。随机臂在所有输入行匹配接受数量，不是匹配实际移动距离。

## 3. 最容易悄悄改变算法的细节

### 相机和像素

`_camera` 对P左块按第三行长度归一化并规范符号；中心需与P吻合（rtol1e-9, atol1e-6）。半分辨率图像使用Pillow bilinear，RGB /255后灰度权重[.299,.587,.114]；投影左乘包含半像素项的resize矩阵，不能只缩放fx/fy。

`_bilinear` 接收(x,y)，SciPy读取[y,x]；有效区[0,W-1]×[0,H-1]，只允许1e-9数值越界后clip。其他越界/无效留NaN，不是黑色0，也不是边缘复制。手写torch双线性采样需专门处理最后一行/列、整数像素和NaN邻点，先对照SciPy；若使用grid_sample，要明确align_corners与归一化，并复刻显式掩码。

### NaN和统计

ZNCC只用reference/source同时有限的像素，mean、variance和covariance用同一集合；默认torch.std的样本方差不是原算法。除法仅在supported时发生。原实现评分中间FP64，scores写成FP32，`solve_surfacelets`再升FP64。

invalid不能变成0 correlation、0 cost或极优分数；all invalid最终回精确零。聚合有效来源不足两张即invalid，不改变分母让一张视图过关。

### 同分、特征与模型

`_choose`: 最小cost（绝对容差1e-12，rtol0）-> 最小|offset| -> 最低hypothesis -> 最低depth index；不能普通argmin替代。局部峰谷/curvature会受微小分数变化影响，不能只检查最终点云均值。

post特征按POST_KEYS顺序展开，加原missing标志；原sentinel和最终FP32转换不变。模型joblib/特征schema每次校验哈希。AB路由max(0,A,B)，同分KEEP>A>B。

### 并行与分块

同一 normal 的49射线采样可供五supports复用；可以按anchor/source分块但需还原原轴顺序。不要修改全局随机状态来改变anchor。保留source轴全部信息供CPU最终处理。

先传入CPU规范化camera；GPU输出score/ref_variance/ref_fraction/ref_valid，其他元数据保持同义。只把必要中间数组拷回CPU，不能在每个patch调用.item()导致同步风暴。

## 4. 性能策略

先测score耗时占比。CUDA第一版可能受FP64吞吐或小batch launch开销限制；不预报十倍/百倍。首选原生torch，不为追求“底层”立即开发自定义ISA/MLIR。

可比较batch=1/7/32/64/128，具体上限服从内存；严格版和优化版分开列。全scores约4×1000×20×49个float32，但显存还包含图像、坐标广播、临时张量与allocator保留，不能只按scores估算。

如需CPU复核临界候选，作为显式hybrid配置，记录触发比例和时间；不得把部分CPU重算藏在“纯GPU”速度里。第一版可先不做自适应回退，集中实现等价评分。

## 5. 软件包装不是新科学结论

一次成功GPU回放说明计算实现成立，不说明native质量改善。测试真实ROI也不能以GT挑batch/精度获得更好指标。若浮点变化导致指标偶然提高，不把它作为算法创新。

该移植最有价值的后续用途是更快扫小扰动响应：0、正负幅度、纹理/标定/遮挡。但那是另外冻结协议的灵敏度实验，本任务不自动启动。

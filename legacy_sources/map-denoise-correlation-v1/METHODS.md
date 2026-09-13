# 冻结方法适配

目标主机：liekkas。适配器直接导入以下只读源码，不复制或更改算法：

- `/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py`
- `/home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1/baseline/scalar_reference.py`

统一接口为 `estimate(method, xyz_mm, frame, sigma_mm) -> (output_xyz_mm, info)`；`METHODS` 是六项名称元组。输入仅包含三维坐标、点对应的帧号和提供的 sigma。估计函数没有数据集路径、EVAL、GT、真实层标签或干预幅度参数，也不读入数据文件。冻结模块内的生成与评价函数不会调用。

所有点的 ROI 固定为零，不提供已知稳定单面锚点。所有方法都使用同一 `sigma_mm`。`bias_bound_mm = max(8.0, 8.0 * sigma_mm)` 是预先固定的先验规则，不是实际施加干预的上界或真值。六个方法中，只有 `joint_forced` 使用这个边界；其他方法接收同一 Input 但不使用该字段。fast 原始信息中的 `bias_bound_used=False` 被保留。

所有标量方法把 `xyz_mm[:, 2]` 视为已提供的共同法向坐标，不估计逐点法向。若调用方对真实片区做共同法向坐标变换，应在估计前统一变换并在输出后统一变回；该变换不属于这六个算子的学习或优化。

| 正式名称 | 冻结调用 | 定位 |
| --- | --- | --- |
| `identity` | `experiment.estimate(inp, "identity")` | 原样输入控制 |
| `xyz_mixture` | `experiment.estimate(inp, "xyz_mixture")` | 自研固定噪声标量一/二层混合；BIC 选择；软后验输出 |
| `frame_center_then_xyz` | 同名旧调用 | 自研先去除各帧法向均值相对全局均值的偏差，再用同一个标量混合后端 |
| `joint_forced` | 同名旧调用 | 自研潜在层/共同帧偏差 EM；强制输出，未使用歧义停止策略 |
| `fast` | `scalar_reference.estimate(inp, "scalar_profile_hard")` | 自研排序硬分配 profile；单层候选仍通过旧 L-BFGS-B 调用；不是官方 JRMPC/BALM |
| `open3d_icp_then_xyz` | 同名旧调用 | 官方 Open3D 点到点 ICP，再接自研共享标量后端；不是官方 JRMPC/BALM，也不是联合 BA |

fast 的无锚点双层候选使用带惩罚的 complete-data classification likelihood；不可把它和其他方法称为完全相同的边际 BIC。输出使用后验阈值 0.5 的硬二层分配。冻结 profile 的离散搜索、局部优化、选择准则、迭代上限均未修改。

## 从源码核对的 Open3D 设置

使用 `registration_icp`、`TransformationEstimationPointToPoint()`、单位变换初始化、最大对应距离 **0.025 m**、`ICPConvergenceCriteria(max_iteration=30)`。除迭代上限外，收敛条件使用冻结代码所调用 Open3D 的默认设置。

在当前全零 ROI 下，以帧 0 的全部点为参考。所有帧（包括帧 0）依次对参考运行 ICP，各自估计的刚体变换应用于该帧所有点。源坐标先从 mm 转为 m。全部帧变换后统一加上 `原输入三维均值 - 已变换输出三维均值`，消除任意共同平移；旋转保留。之后转回 mm，记录逐帧法向均值差作为 bias，再接共享标量滤波。这个三维均值校正在标量后处理之前，不能保证后处理后的最终均值完全相同。

## 接口约定与来源

旧算法要求连续、从零开始的帧索引。适配器按排序后的原始帧 ID 做稠密重编号，不改变点序或帧归属；`info.frame_ids` 保存对应关系，`info.bias_mm` 按这个顺序排列。原始连续编号保持不变；任意编号时 ICP 参考是原编号最小的帧，由 `info.icp_reference_frame_id` 明确记录。

输入先复制成独立数组，返回输出也是独立数组；适配器检查非空 N×3 输入、整数帧号、正且有限 sigma、输出点数和有限性。`info.backend_info` 原样保留冻结方法返回的信息，其常用字段同时保留在顶层；附加正式方法名、原始调用名称、来源、先验角色与帧偏差。

`source_hashes()` 返回适配器自身和上述两个冻结源文件的绝对路径到 SHA256 的映射。应在计时前保存该映射。导入 `operators` 时已加载 NumPy；`warmup()` 单独加载 SciPy 依赖、两个冻结模块和 Open3D，随后用不读文件的固定 8 帧×12 点样本分别调用所选方法。它返回函数内部依赖导入时间、各方法预热时间、总预热时间、软件版本、解释器和线程环境；传入 `run_estimates=False` 可只做依赖导入。调用者应在逐方法计时前执行它；小样本不声称预热了每一种数值分支。实际方法耗时仍包含适配复制、帧重编号与输出检查。

规定环境为 `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`。运行前设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`；适配器记录环境而不静默修改调用进程线程设置。

## 低成本检查

`test_operators.py` 使用固定种子 911073、8 帧×24 点的随机小样本，逐一比较六个适配输出与独立导入后的原函数直接调用，核对原 metadata、帧偏差、输入不变、点数及有限性。另检查只读数组、稀疏帧 ID 的语义保持、非法输入与 SHA256。该测试仅核对适配和输入契约，不构成算法精度或机制证据。

运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B test_operators.py
```

2026-09-11 在 liekkas 执行：4 项测试全部通过，测试主体 0.935 秒（包含 `setUpClass` 的导入与小样本预热；不含 Python 启动和 NumPy 顶层导入）。六个输出与直接调用按 `rtol=1e-12, atol=1e-10 mm` 比较，原信息字典完全相同。环境：Python 3.12.13、NumPy 2.2.6、SciPy 1.15.3、Open3D 0.19.0，四个线程环境变量均为 1。

核对时冻结 `experiment.py` SHA256 为 `40df63459fe620788acb714f44bd794757f8197542eb03411e685e33ad384f57`；冻结 `scalar_reference.py` 为 `ebc82a17ed431ee10056ef3c64113e8897f10821c673c4c0c99e469ef1773a9a`。本次未执行整套合成或真实片区实验。

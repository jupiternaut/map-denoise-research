# 多表面场实验：独立实现与数值审计

状态：**PASS（实现、口径与可复算性）；主方法没有超过冻结 prior_recovery。**

本审计没有修改方法、推理输出、历史结果或部署默认。对象是主机 `liekkas` 上本目录及 `/srv/slam-research/grf/map-denoise/runs/multisurface-field-20260926T102842Z`。数据仍是已暴露的 3 场景 × 4 ROI × 5 状态回放，不能改称独立确认。

## 1. 执行与独立核算

- 全部 3 场景推理封存后，才打开激光参考；420 份 actual PLY 均先于评价生成。
- 评价完成 60 个 case × 12 个 arm = **720 唯一行**，已封存。
- 独立用 CSV 重算 660 个汇总数值，最大差 **3.552713678800501e-15**。
- 原评价记录 240 条历史基线核验、120 条 Open3D 独立最近邻核验，最大差均 **0**。每个新方法/新 oracle 抽查 64 个固定支持行。
- 另从 `scan55_roi0__minus3` 的 12 项原始 POOL 坐标重建 64 行候选最近邻表，采用 Open3D 而不是评价的 SciPy 查询；保存 oracle 的距离与逐行最小值最大差 **9.9643e-15 mm**，到候选坐标集合最大差 **1.4648e-14 mm**。
- 6 项 evaluator 单元测试通过。尝试 pytest 时环境未安装 pytest；未安装软件，改用文件原生的 unittest 入口通过。该工具调用失败不属于实验失败。
- scene69 曾因总 worker 限额暂停，`scan69_roi0__minus1` 的 70.45 秒包含暂停；不得将其作为纯计算耗时比较。其他评价和数值结果不受影响。

## 2. 指标与支持口径

所有方法使用冻结 native 行定义的 ROI/ObsMask 支持；不因修正后点的位置改变入选行。激光参考按同一 ROI/ObsMask 裁切，再作 0.8 mm Open3D 体素下采样。支持逐项核对历史保存掩码。

先对 ROI 等权，再对三个场景等权；本实验每场景恰有四个 ROI，等同 12 ROI 等权，但不等同全点数加权。改善率用汇总 MSE 的比值：`100 × (1 − MSE_arm / MSE_identity)`。不同扰动分别报告，不能按条件选赢家拼成一种方法。

`improved_fraction` / `harmed_fraction` 的判定是参考最近邻距离下降 / 增加超过 0.1 mm，分母是固定支持行。位移 RMS 也是固定支持内 RMS；不是全 PLY 行 RMS，也不是仅已移动点 RMS。

反向 MAE 约 15 mm，是参考 ROI 面积大于当前 source 足迹的结果。它测“裁切参考到支持内输出”的距离，不是官方全场完整性或薄层身份/拓扑的证明。

## 3. 所有锁定臂的结果

表中为相对 identity 的 source-MSE 改善百分比；负值为恶化。主臂始终为 `multi_field`，未按结果切换。

| arm | native | −1 mm | +1 mm | −3 mm | +3 mm |
|---|---:|---:|---:|---:|---:|
| identity | 0 | 0 | 0 | 0 | 0 |
| prior_recovery | −7.673 | 1.164 | 10.036 | 46.994 | 39.022 |
| point_wta | −106.268 | −82.892 | −44.543 | 33.679 | 27.726 |
| single_field | −70.216 | −92.361 | −41.588 | 12.211 | 10.132 |
| multi_field | −83.352 | −84.097 | −41.000 | 23.887 | 21.948 |
| multi_field_visibility | −94.434 | −99.309 | −51.325 | 19.059 | 15.940 |
| multi_field_graph | −80.117 | −88.663 | −43.733 | 21.033 | 18.795 |
| oracle_discrete | 32.602 | 38.873 | 43.897 | 69.943 | 63.267 |
| oracle_continuous | 38.970 | 49.613 | 49.463 | 74.691 | 64.985 |
| oracle_proposals | 64.221 | 65.854 | 73.316 | 90.665 | 92.574 |
| oracle_all_layers | 65.358 | 67.407 | 74.703 | 90.945 | 92.751 |
| oracle_expanded | 67.637 | 70.643 | 76.318 | 91.944 | 93.103 |

### 相对冻结 prior_recovery 的配对 ROI 胜/平/负

| arm | native | −1 mm | +1 mm | −3 mm | +3 mm |
|---|---|---|---|---|---|
| multi_field | 1/0/11 | 2/0/10 | 1/0/11 | 3/0/9 | 3/0/9 |
| multi_field_visibility | 1/0/11 | 2/0/10 | 1/0/11 | 3/0/9 | 3/0/9 |
| multi_field_graph | 1/0/11 | 2/0/10 | 1/0/11 | 3/0/9 | 3/0/9 |

相对 identity，主臂 native 为 0/12 改善，−3 为 7/12，+3 为 8/12。±3 有恢复能力，不能因此改写为超过旧方法。

## 4. Oracle 不能混称

- discrete：每点在旧 `KEEP/A/B` 中用参考选最好。
- continuous：冻结的精确旧线段并集 oracle，而非粗网格近似。
- proposals：仅新 POOL 最初九项，包括 ±3/±6 mm 和半步，不含场预测。
- all_layers：旧候选、全部实际输出以及 POOL 所有项。它不是 K2 实际选择器所能选择的候选全集。
- expanded：all_layers 加旧 continuous 坐标；只作评价上界。

`EVALUATION_ADDENDUM.md` 在打开本轮 GT 结果前增加了 proposals 对照。−3/+3 的 broad proposals 已有 90.665%/92.574% 上界；新增 field 坐标使它只升至 90.945%/92.751%。因此主要新上界来自宽搜索，不能归因给多表面场。所有 oracle 都是 GT 诊断；部分候选没有通过观测证据有效性掩码，不能把该上界称为可部署收益。

## 5. 封存后的受限场容量诊断

经新授权，主评价封存后增加 `posthoc_field_capacity.py`。结果单独封存在 `OUT/posthoc_field_capacity`，不改原主张、不回写参数。只评 `KEEP/K1` 和 `KEEP/K2a/K2b`，完整复用固定支持与参考预处理。

| 事后诊断 oracle | native | −1 mm | +1 mm | −3 mm | +3 mm |
|---|---:|---:|---:|---:|---:|
| KEEP/K1 | 26.381 | 25.728 | 29.121 | 45.815 | 47.421 |
| KEEP/K2a/K2b | 44.491 | 43.516 | 44.153 | 62.691 | 62.657 |

60 case × 2 = 120 行；独立重算 70 个汇总数值，最大差 8.8818e-16。120 条实际场输出属于对应候选坐标集合的检验最大差 0。耗时 19.44 秒、单 worker。

−3 状态，actual K2 MSE=2.182636，K2-only oracle=1.069867，all-layers oracle=0.259657 mm²。

+3 状态，相应为 3.352352、1.603877、0.311356 mm²。

这将两项差距分开：actual→K2-only 是在已有场候选中的选择差距；K2-only→all-layers 是受限场候选相对更宽候选池的容量差距。两者都存在，不能只归因于选择器。**K1/K2-only oracle 也没有使用 fresh common-view 有效性掩码，而是直接评价导出的几何候选；所以它不是保留当前 hard-KEEP 规则后可达到的上界。** 第一项“选择差距”同时包含观测无效时被迫 KEEP 的部分，不能解释成纯评分排序损失。该诊断不是改进方法，更不是已实现 62.7% 恢复。

## 6. 代码审计要点与已修缺陷

- 初期 graph kNN 用 `near[:,1:]` 假定首项一定是自身，重复 UV 会残留 self-edge。独立小例复现了假 label-change inertia；主执行前改为明确剔除自身 index、保留最多六个其他邻居，回归测试通过。旧 smoke 保留并在实现日志说明，60-case 正式推理使用修复后代码。
- field K1/K2/visibility 共用 fresh KEEP/K1/K2a/K2b 的 source-view mask。相机投影尺度归一化后再以真实 camera-Z 构造 inverse depth；输出为物理坐标。
- 无有效证据或拟合支持的点回退 KEEP。graph 的不支持点虽然固定 KEEP，仍通过边影响邻点；这是额外平滑先验，不是独立视图证据。
- graph 保留迭代中最低能量结果；不是全局最优算法。单视角自渲染可见性来自当前多面输出，也不是外部真可见性。
- point_wta 共用九候选的 view mask；场选择共用四候选的 fresh mask，因此 point_wta→single_field 不是只改变正则项的纯消融。single→multi→visibility→graph 的共用证据口径更直接。
- 父任务的独立 representation audit 已报告较高比例的 ±6 mm clipping 与 K2 层间距触界。本审计没有把那些数字当作自己重算；结合 K2-only 容量结果，应把局部窗口/拟合压缩和选择两方面都保留为待解释因素。

## 7. 结论

本轮正确实现并测量了一个新表示分支，但没有产生默认方法升级。K2 候选比 K1-only 更有几何潜力；实际 K2 没有稳定利用该潜力，且宽搜索中可用的好坐标被场表示遗漏了一部分。可据此继续研究“保留候选再选择”与“压缩成局部场”的取舍；不能把高 oracle 数值或 graph 复杂度当成已实现的进步。

主评价 seal SHA-256：`15dd0f7688ab730162df19af76d1d28795eefe0066827a837037cce00c3189ff`。

事后诊断 seal SHA-256：`33e25b3f8fa87773886c91eb3981caebc3f468fe0377347fa2b5155a67603b97`。

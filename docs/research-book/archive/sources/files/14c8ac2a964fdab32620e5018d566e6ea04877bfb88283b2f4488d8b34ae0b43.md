# V25：实现了局部表面再生，但同观测下没有稳定胜过 identity

2026-09-14，主机 **liekkas**。已实际构造、运行、封存并独立复核。这是开发集结果，不是新场景确认。

## 1. 本轮回答了什么

假说是：在旧重建局部支持或关联错误时，用同样的多视图证据做多表面、可变支持域再估计，能比普通局部融合和只移动旧点更好恢复结构。

本轮**没有得到支持**。

1. **做出了可复查的新能力。** 共享 evidence bundle、identity、自研 CPU ZNCC plane-sweep 融合、单面限制、旧点移动、V25 多面 atlas，以及后来的 WTA-only atlas、深度跳跃连通域、门控位移，都写出了真实 `.ply`。
2. **官方 COLMAP PatchMatch+fusion 基线未运行。** PATH 与有限目录中没有 `colmap`。自研融合的名字是 `fusion_wta` / `cpu_zncc_plane_sweep_wta`，不是官方 MVS。
3. **V25 多面再生没有额外收益。** 在 8 个照片定义的父 ROI 上，`v25_atlas` 的对称距离全部差于 identity。去掉第二峰值、改深度连通域、缩小到网格中心 40% 支撑后，替换式输出仍然更差。
4. **门控位移是伤害最小的 V25 变体，但仍不是升级。** 它大体保住 identity，只在 `scan24_wall_control` 和 `scan37_stone_control` 上略好或持平；结构区没有 20% 收益。
5. **不把 scan37 改名为确认，也不部署 V25。** 默认输出锁定为 identity。见 `runs/2026-09-14T165314Z/METHOD_LOCK.json`。

20% 是投入目标，不是测得结果。本轮不宣称毫米级物理精度已经分出新方法胜负。

## 2. 输入、坐标和 ROI

相机复用 V23 的 COLMAP 针孔解析（副本 `third_party/calibration_v23.py`，哈希未改）。物理坐标：`p_phys = scale_mat * p_norm_h`，投影 `uv ~ P_colmap * inv(scale) * p_phys_h`。

稀疏重投影（与 V23 同口径）：

| 场景 | 中位 px | P95 px |
|---|---:|---:|
| scan24 | 0.356 | 1.794 |
| scan37 | 0.441 | 1.696 |

ROI 由 `0022.png` 上的照片框加输入网格反投影得到，**不是按激光误差挑选**。每场景 3 个结构区 + 1 个普通面控制。随后用“距质心最近 40% 输入网格点”做了一次全体一致的 core 收缩，仍不读参考。

激光 STL 与官方 ObsMask 只进入 `evaluation/`。构造器签名与源码不含这些路径。

## 3. 实现了什么

共享证据：参考图 256 像素裁块、80 个沿相机 Z 的深度面、7×7 掩膜 ZNCC、3–5 个可见视图。深度范围来自输入网格加 28 mm 余量。bundle 内无 GT。

| 臂 | 含义 |
|---|---|
| identity | ROI 内网格顶点，0.8 mm 体素去重 |
| fusion_wta | 同一 bundle 的 WTA 深度反投影（自研 CPU，非官方 COLMAP） |
| restricted_single | 同一证据上只拟合 1 个平面并重采样 |
| restricted_point_move | 旧点拉到最近 WTA 证据 |
| v25_atlas | WTA+第二峰值，最多 3 个平面，按观测支撑采样 |
| v25_wta_atlas | 同上，但不用第二峰值 |
| v25_depth_cc | 按深度跳跃分裂 WTA，保留连通域 |
| v25_gated_move | 仅当 WTA 与旧点相差 3–12 mm 时才移动 |

评价：固定空间 AABB ∩ 官方 mask，0.8 mm 体素，`E_sym = 0.5*(acc+comp)`，空输出罚 1000 mm。独立 `evaluation/verify.py` 不导入生产评分函数。

## 4. 父 ROI 主结果（开发，单位 mm）

128 个已封存输出的独立重算最大 `|ΔE_sym| = 0`。

| ROI | 类 | identity | fusion_wta | restricted_single | v25_atlas | v25_gated_move |
|---|---|---:|---:|---:|---:|---:|
| scan24_window_left | 结构 | 1.016 | 3.225 | 21.026 | 2.942 | 1.269 |
| scan24_gable_center | 结构 | 0.540 | 4.690 | 9.695 | 6.549 | 0.941 |
| scan24_turret_join | 结构 | 0.452 | 5.751 | 26.189 | 7.738 | 1.014 |
| scan24_wall_control | 控制 | 2.494 | 2.297 | 15.779 | 2.840 | 2.418 |
| scan37_scissor_cross | 结构 | 4.477 | 4.134 | 11.523 | 8.627 | 4.535 |
| scan37_clamp_jaw | 结构 | 0.829 | 7.089 | 18.740 | 10.755 | 1.799 |
| scan37_driver_handle | 结构 | 0.821 | 3.491 | 7.665 | 5.903 | 1.223 |
| scan37_stone_control | 控制 | 0.583 | 1.874 | 3.520 | 2.617 | 0.582 |

fusion_wta 在 wall_control 与 scissor_cross 略优于 identity；这是标准光度融合信号。v25_atlas 在这 8 项上都更差。restricted_single 因整块 ROI 被压成一张平面，覆盖崩溃（completeness 升到几十毫米）。

`v25_wta_atlas` 在 wall_control 上到 2.202，低于 identity 的 2.494，但仍是单个控制区，不能当通用升级。去掉第二峰值没有把结构区救回来，说明失败不单是砖墙假峰。

## 5. 机制修改留下的预测对照

| id | 预测 | 改动 | 结果 | 决策 |
|---|---|---|---|---|
| v1_atlas | 多峰+多面能恢复窗框/工具层 | 初始 atlas | 全面差于 identity | 保留失败，不部署 |
| v2_no_alias | 第二峰值是纹理假峰 | 只用 WTA 做 atlas | 结构区仍差 | 假峰不是唯一主因 |
| v2_depth_cc | 深度跳跃分裂比全局 3 平面好 | 连通域 | 仍差于 identity | 表示层不是唯一主因 |
| v2_gated | 只改证据强烈不一致的旧点 | 3–12 mm 门控 | 接近 identity，无 20% 收益 | 最小伤害，不升级 |
| v3_core | 大 AABB 使匹配过粗 | 全体 ROI 缩到最近 40% 网格点 | identity 往往更好；替换式仍差 | 支撑域缩小不够 |

因此当前坏例更像是：**这套 ZNCC 深度本身不够准，替换会丢掉已经较好的输入网格。** 不是“差 0.001 mm 再调 alpha”。

## 6. 密度、覆盖、控制

- 主分在体素去重后计算。单元测试：点重复一倍不提高分数；空输出得 1000 mm 而不是 0。
- identity 在多数结构区 1 mm 召回已经很高（gable 0.92、turret 0.97）。V25 替换降低召回。
- 控制：stone 上 atlas/fusion 变差；wall 上 fusion 与 wta_atlas 变好。平均“有时更好”不能盖住 turret/clamp 的大幅损伤。
- 覆盖：restricted_single 的 laser 支撑仍在，但输出撑不住参考，completeness 惩罚很大。这是公平评价，不是删难像素。

## 7. 未测与阻塞

- **官方 COLMAP/OpenMVS：BLOCKED。** 见 `baselines/colmap_probe.py`。
- **GPU：未用。** RTX 3060 Ti 全程被其他进程占用（LightRAG 约 4 GiB）。全部结果是 CPU。
- **新确认场景：未取得。** ETH3D 本地只有激光 7z；UCL 官方盘要登录；Oxford zip 有照片和 PCD，但没有本任务的 COLMAP/`scale_mat` 契约。未打开 Oxford 评价端雷达来选场景。
- 没有把开发分装成确认，也没有用 GT 选主候选。

## 8. 交付位置

代码：`src/v25/`，`evaluation/`，`tests/`，`configs/`

运行：`runs/2026-09-14T165314Z/`

- `SEALED.json` / `SEALED_v1.json`：输出哈希
- `RESULTS.csv` / `RESULTS.json`：128 行
- `VERIFY.json`：独立重算，最大差 0
- `METHOD_LOCK.json`：不升级
- 各 ROI 目录：`evidence.npz`、各臂 `.ply` / `.json`

图：`figures/esym_parent_rois.png`，`figures/esym_by_roi.png`，`figures/overlays/`

命令：`COMMANDS.md`

## 9. 研究决策

**关闭本次 V25 多面再生作为默认可部署方法。** 保留实现与全部失败工件。若继续，应先改匹配/可见性（更可靠的深度），或先复现官方 MVS，而不是再加一张面。

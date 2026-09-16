# E0：CPR-2（置信集投影修正算子）预注册合成机制实验协议

本文件是 E0 实验的预注册协议。所有数值、规则与预测在运行任何代码之前写定；本文件的 sha256 记录于 `COMMANDS.md`。运行结果不得反过来修改本文件中的任何参数、阈值或预测。

除特别说明外，全文长度单位为 mm。

工作目录限定为 `/home/grf/Documents/Codex/2026-09-15/e0/`；不修改、不创建该目录之外的任何文件；不读取其他项目目录；不安装任何包；仅使用 CPU。

解释器（只读使用；含 numpy 2.2.6、scipy 1.15.3、matplotlib 3.11.1）：
`PY=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`
环境变量：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg`

---

## 1. 生成模型（Generative model；每次运行独立设种子）

### 1.1 几何与真值定义

- 组件位于二维网格上：Nx=200 × Ny=100，共 N=20000 个组件，网格间距 h=0.8。
- 在位点（incumbent point）p_i=(x_i, y_i, z0_i)，法向为 +z，默认 z0_i=0。
- 未知真值偏移 t*_i 沿 +z 方向；真值点为 (x_i, y_i, z0_i + t*_i)。
- 输出 θ̂_i 的坐标误差为 |θ̂_i − t*_i|。
- 真值定义固定不变：t*_i 是从在位点到该组件自身所属真实面片（own true surface sheet；见下文双面片区）的偏移。全程不得切换定义。

### 1.2 证据用偏移网格

- t ∈ [−12, 12]，步长 0.05，共 481 个采样点。

### 1.3 子集标签（互斥）

- `base`：t* ~ N(0, 0.2²)。
- `big`：占全部单元的比例 f_big ∈ {0.01, 0.10}，以成簇的圆盘状斑块分布（半径 3–6 个单元，放置于双面片区之外，尽量互不重叠）。斑块内 t* = s·U(2, 8) + N(0, 0.2²)，同一斑块共用一个符号 s。（调整斑块数量以近似达到目标比例；报告实际实现的比例。）
- `twosheet_back` / `twosheet_front`：一个矩形区域，20 列 × 全部 100 行（2000 个单元，占 10%），棋盘格排布：一半单元为 BACK（背面片，z0=0，t* ~ N(0, 0.2²)），一半单元为 FRONT（前面片，z0=6.0，t* ~ N(0, 0.2²)）。
- `misfit`：双面片区之外 3% 的单元：t* ~ N(0, 0.2²)，但证据是垃圾（见下文）。
- 多峰标志 mm=True：在 `base` 与 `big` 组件中各占 15%（twosheet 与 misfit 组件永不置 mm）。

### 1.4 证据代价曲线

c_i(t) = M_i + A_i·(1 − B_i(t)) + ε_i(t)，其中：

- B_i(t) = max( exp(−(t−μ_i)²/(2w²)), r_i·exp(−(t−μ_i−Δ_i)²/(2w²))·[mm_i] )，w = 1.0。
- 对比度 A_i 来自生成层（generative stratum）混合，概率为 (0.30, 0.35, 0.35)：flat 层 A ~ U(0.02, 0.08)；medium 层 A ~ U(0.08, 0.25)；sharp 层 A ~ U(0.25, 0.80)。存储生成层标签 gen_stratum_i。
- μ_i = t*_i + b_i，b_i ~ N(0, 0.15²)，适用于 base / big / twosheet_front。
- mm 组件：Δ_i = sign·U(4, 8)（符号随机），r_i ~ U(0.7, 1.05)。
- twosheet_back：跨面片捕获（cross-sheet capture）：B_i(t) = max( exp(−(t−(6.0+b_i))²/(2w²)), 0.3·exp(−(t−t*_i)²/(2w²)) )，即深谷位于 FRONT 面片处（+6），真值处的谷很弱（0.3）。
- misfit：基线 M_i ~ U(0.3, 0.6)，μ_i ~ U(−10, 10)（垃圾）；其余所有组件 M_i = 0。
- ε_i(t)：平滑噪声：在 t 网格上的白高斯噪声与 σ=0.4（t 单位）的高斯核卷积，重新归一化为单位标准差，再乘以 σ_ε = 0.03。各组件独立。

### 1.5 分数与可观测分层

- 相对分数 s_i(t) = c_i(t) − min_t c_i(t)。绝对分数 m_i = min_t c_i(t)。t_argmin_i = argmin_t c_i(t)（网格上）。
- 可观测分层：Â_i = max_t c_i − min_t c_i（对比度）。分层 = 在测试集 I 上 Â 的三分位点（tertiles）；同一组切分点应用于 K 与 K'。标签 obs_stratum ∈ {flat, medium, sharp}。

---

## 2. 校准（Calibration）

- 校准集 K 与检查集 K'（各 2000 个组件，独立抽取），两种变体：
  - `exch`：从与 I 相同的混合分布抽取（相同的子集比例，包括 twosheet / misfit / mm；twosheet 组件的真值是各组件自身的 t*）。
  - `sfm`：仅从 gen_stratum='sharp' 且 mm=False 的 `base` 组件中抽取（类 SfM：容易、纹理丰富的点）。
- 校准用真值处分数：s_k(t*_k)，通过网格上的线性插值得到。
- 按可观测层 σ 计算（若 n_σ < 50 则回退为使用合并的 K，并记录此事）：
  - q_test[σ] = {s_k(t*_k): k∈K_σ} 中第 ⌈(1−α_test)(n_σ+1)⌉ 小的值
  - q_tgt[σ] = 同一集合中第 ⌈(1−α_tgt)(n_σ+1)⌉ 小的值
  - q_abs[σ] = {m_k: k∈K_σ} 中第 ⌈(1−α_abs)(n_σ+1)⌉ 小的值
  - （若序号超过 n_σ，取 +inf。）
- 参数：α_test=0.10；α_tgt ∈ {0.20, 0.50}；α_abs=0.05；BH q=0.10；ε_tol=0.5；τ ∈ {1.0, 2.0, 3.0}；防护近邻阈值 d_g=0.4；集合度量体素尺寸 0.8。

---

## 3. 决策（CPR-2；仅单轮；不执行采集）

- Conformal p 值：p_i = (1 + #{k∈K_σ(i): s_k(t*_k) ≥ s_i(0)}) / (n_σ(i)+1)。
- 对全部 i ∈ I 在水平 q 下做 BH → 拒绝集 Rej。
- 水平集 L_i = {网格 t : s_i(t) ≤ q_tgt[σ(i)]}；连通分量 = 连续网格索引的极大连续段。n_comp_i = 分量数。w_i = 距 0 最近的分量的宽度。
- 输出决策 out_i：
  - 若 m_i > q_abs[σ(i)]：KEEP
  - 否则若 i ∉ Rej：KEEP
  - 否则若 n_comp_i == 1：MOVE，θ̂_i = L_i 中距 0 最近的端点（投影）。（合理性检查：必须有 0 ∉ L_i；若出现 0 ∈ L_i，则置 θ̂=0 并计为一次异常。）
  - 否则：KEEP
- 研究决策 res_i（独立于 out_i）：
  - 若 m_i > q_abs[σ(i)]：RESPECIFY
  - 否则若 n_comp_i ≥ 2：ACQUIRE
  - 否则：NONE
- 采集优先级（仅报告；它不是信息价值 VOI）：在 L_i 上的权重 ω(t) ∝ exp(−s_i(t)/τ_g)，τ_g = max(q_tgt[σ(i)]/2, 1e-6)。令 comp0 = 包含 0 的分量（若存在），否则为距 0 最近的分量。π_i = comp0 之外的权重质量 / 总质量；g_i = 其余分量上 dist(0, 分量) 的加权均值。prio_i = π_i·g_i。按子集（mm 与非 mm）报告其分布。

---

## 4. 各臂（Arms；每个组件的 θ̂_i；所有臂共用同一组曲线）

- IDENTITY：0。
- ARGMIN：t_argmin_i。
- GATED(τ)：若 |t_argmin_i| > τ 则取 t_argmin_i，否则取 0；τ ∈ {1, 2, 3}。
- TEST_ARGMIN：恰好对 CPR-2 下 out_i==MOVE 的那些组件（相同的可移动集合）取 θ̂ = t_argmin_i；其余为 0。
- CPR2：按上文决策。
- 防护变体 CPR2_G 与 TEST_ARGMIN_G：构建三维输出 (x, y, z0+θ̂)；对每个被移动（MOVED）的组件，计算其到所有其他输出点的最近邻距离（scipy cKDTree）；若 NN < d_g=0.4 → 回退 θ̂=0 并置 collapse_i=True。防护误报率 = 双面片区之外的组件中，被标记数 / 被移动数。

---

## 5. 度量（Metrics；每个臂；总体、按 obs_stratum、按子集标签）

- e_before=|t*|，e_after=|θ̂−t*|。损伤 Dmg_i = e_after > e_before + 1e-9。移动 M_i = (θ̂≠0)。
- 相对 N 的损伤率（#Dmg/N）；相对移动数的损伤率（#Dmg/#M）；损伤幅度 d=(e_after−e_before)₊：在移动组件上的均值、p95、最大值；总损伤 Σd。
- e_after 的均值、RMSE、中位数；在子集 `big` 上：e_after 均值与修复覆盖率 = 满足 e_after < 0.5·e_before 的比例。
- 覆盖满足的移动（coverage-satisfied moved）：在 MOVED 组件中，按 t*_i ∈ L_i(α_tgt) 是否成立划分（插值：s_i(t*) ≤ q_tgt）。对每个臂报告每个划分中的损伤计数与损伤率。（定理：CPR2 在覆盖满足划分中的损伤计数必须恰为 0。）
- CPR2 残差检查：在 MOVED 且覆盖满足的组件中：比值 e_after/(w_i/2) 的中位数、IQR，以及 Spearman corr(e_after, w_i/2)。
- 覆盖率：在 I 上 L_i(α_tgt) 的真实覆盖率与 {s_i ≤ q_test} 的真实覆盖率，按 obs_stratum 与按子集；在 K' 上的代理覆盖率（proxy coverage）按 obs_stratum（使用相同分位数）。给出 95% Wilson 区间。
- FDP_ε：在 Rej 中，|t*| ≤ ε_tol 的比例（标注："经验量，点原假设 p 值"）。
- 决策图：KEEP/MOVE 比例；NONE/ACQUIRE/RESPECIFY 比例。ACQUIRE 敏感度 = P(ACQUIRE | mm)，误报率 = P(ACQUIRE | 非 mm 且非 twosheet)，按 obs_stratum。RESPECIFY 在 misfit 上的敏感度，在非 misfit 上的误报率。
- 集合度量：输出点云 O（体素去重 0.8：每个被占据体素一个点，取体素均值），真值点云 T（全部真值点，同样去重）。accuracy = O→T 的最近邻距离均值；completeness = T→O 的最近邻距离均值；均给出总体与限定在双面片区（按 x 范围）的结果。同时报告去重后的点数。
- 面片身份（双面片区）：BACK 组件的翻转率 = 输出 z 距 6 比距 0 更近的比例；FRONT 组件的翻转率 = 距 0 比距 6 更近的比例。
- CPR2 与 TEST_ARGMIN（相同可移动集合）的实用等效边界，在总体与每个 obs_stratum 上评估：|Δ(#Dmg/N)| ≤ 0.002；|Δ 平均损伤幅度| ≤ 0.05；|Δ p95 损伤| ≤ 0.2；|Δ 平均 e_after| ≤ 0.02；|Δ 修复覆盖率| ≤ 0.02。每个维度报告带符号的差值（哪个臂更优）。只有在所有场景与所有分层中五个维度全部落在边界内时，才可以写"未检出保留增量"；否则列出哪个维度偏向哪个臂。未检出差异不是等效的证明。

---

## 6. 场景（Scenarios）

f_big ∈ {0.01, 0.10} × 校准 ∈ {exch, sfm} × α_tgt ∈ {0.20, 0.50} = 8 个场景；种子 {1, 2, 3} → 共 24 次运行。跨种子聚合为均值 ± 标准差；同时保留每个种子的 JSON。对每个场景的种子 1 保存 decisions.csv（每个组件：标签、t*、A、obs_stratum、p、out、res、各臂的 θ̂、collapse 标志；可 gzip）。

---

## 7. 预注册预测（Pre-registered predictions；逐条检查；写 PASS/FAIL/PARTIAL 并附数值）

- P1（exch）：每个 obs_stratum 上 L(α_tgt) 的真实覆盖率 ≥ 1−α_tgt 减去 Wilson 容差；CPR2 的 #Dmg/N ≤ α_tgt；CPR2 在覆盖满足的移动划分中的损伤计数恰为 0。
- P2（sfm）：K' 代理覆盖率在所有分层通过；I 上的真实覆盖率在 flat 和/或 medium 层低于 1−α_tgt−tol；CPR2 在这些分层的损伤率高于 exch 对应值。
- P3（flat 层，exch）：CPR2 的 KEEP 率 ≥ 0.95 且 #Dmg/N_flat ≤ α_tgt；ARGMIN 在 flat 层的损伤率 ≥ 0.5；每个 GATED(τ) 在 flat 层的损伤率 > CPR2。
- P4（投影 vs argmin，相同可移动集合）：报告五个维度；预期（不要求通过）：TEST_ARGMIN 在 sharp 层的 `big` 子集上 e_after 均值更低；CPR2 在覆盖满足划分中损伤为 0 而 TEST_ARGMIN > 0；总损伤幅度的比较是开放问题。决策规则：除非所有场景与所有分层中全部边界均被满足，否则保留投影。
- P5（双面片）：无防护时，CPR2 与 TEST_ARGMIN 使 ≥ 50% 的 BACK 组件翻转，且区域 completeness 相对 IDENTITY 变差；有防护时，翻转率 < 10% 且集合度量 ≈ IDENTITY（0.1 mm 以内）；防护误报率 ≤ 1%。
- P6（残差 ≈ w/2）：CPR2 覆盖满足移动的比值 e_after/(w_i/2) 的中位数位于 (0.5, 1.5)。
- P7（检测）：mm 组件的 ACQUIRE 敏感度在 medium+sharp 层 ≥ 0.6；RESPECIFY 在 misfit 上的敏感度 ≥ 0.8 且误报率 ≤ α_abs + Wilson 容差。

---

## 8. 已采纳的修正（Corrections adopted）

(a) 等损伤率不是删投影的归因规则；比较五个维度并只在预定实用等效范围内全部无增量时才考虑删去投影；未检出差异不等于等效。

(b) 容差原假设（|θ⁰−t*| ≤ ε）与点原假设的 conformal p 值不匹配；FDP_ε 只作为经验量报告，不宣称 BH 在容差原假设下有效。本 E0 只跑单轮；多轮覆盖（R·α 联合界）标为待验证，不在本实验声称。

(c) π_i·g_i 称为"采集优先级"，不是 VOI；它缺少具体采集动作的区分能力因子；本 E0 不执行采集，只报告分布。

---

## 9. 禁止事项（Prohibitions）

- 真值 t* 只能用于：K/K' 上的校准分数（按设计它们是校准通道），以及评估度量。它不得用于对 I 的任何决策，也不得用于选择任何参数。
- 看到结果后不得调参。不得依结果决定停止。
- 若全量运行后发现 bug：修复、记录该 bug 并重跑，保留两份日志。绝不为使预测通过而修改参数或阈值。

---

## 10. 图（e0/figures/）

- fig1：场景 exch、f_big=0.10、α_tgt=0.5，跨种子平均：每个臂在每个 obs_stratum 上的损伤率（#Dmg/N）与 e_after 均值。
- fig2：exch 与 sfm（α_tgt=0.5）下，每个 obs_stratum 的真实覆盖率 vs 代理覆盖率。
- fig3：双面片区：IDENTITY、CPR2、CPR2_G、TEST_ARGMIN、TEST_ARGMIN_G 的 accuracy / completeness 与 BACK 翻转率。
- fig4：CPR2 覆盖满足移动的 e_after vs w_i/2 散点图（单次运行）。

---

## 11. 工作顺序与产出

1. 先写本文件；在运行任何代码前将其 sha256 记录于 `COMMANDS.md`。
2. 实现 `e0_mechanism_test.py`（单文件或附一个小型辅助模块）。
3. 先用小 N 做冒烟测试（记录日志），再将完整预注册网格运行一次。
4. 写 `RESULTS_RAW.md`：直接来自数据的表格，以及对每条预注册预测的 PASS/FAIL/PARTIAL 行（附精确数值），不做超出数值的解释。`COMMANDS.md` 记录每条实际运行的命令、退出码与耗时。
5. 图保存至 `figures/`。

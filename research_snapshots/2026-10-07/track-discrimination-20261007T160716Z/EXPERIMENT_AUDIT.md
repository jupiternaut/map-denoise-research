# Experiment Integrity Audit

- Date: 2026-10-08 (Asia/Shanghai).
- Audited host: `liekkas`; absolute root: `/srv/slam-research/grf/map-denoise/runs/track-discrimination-20261007T160716Z`.
- Reviewer: fresh Codex reviewer `/root/track_discrimination_audit`; independent mathematics subreview `/root/track_discrimination_audit/independent_math`.
- Review independence: **same-family**. Acceptance status: **provisional**. No cross-family acceptance claimed.
- Overall verdict: **WARN**. Integrity status: **warn**.
- Review scope: the originally listed protocol, preparation, observation, inference, evaluation, report, tracker and theory artifacts, plus the historical evaluation ancestry. Added read-only verifier and posthoc observation diagnostic were inspected in the supplemental review below.
- All paths below are relative to the audited root unless a full path is given. Exact input hashes are embedded in `audit/AUDIT.json` and `audit/RECOMPUTE.json`.

## Overall judgment

没有发现伪造真值、预测自身归一化、虚构主结果、候选坐标被改写或核心收益公式错误。所有七臂固定总体指标均可从逐点文件独立复算；原始激光距离、63个选择决策及原图轨迹也获得核验。当前报告正确承认主方法 cycle 未通过预定验收，不能将 star 对照事后改称主方法。

WARN 来源是可修正的记录/命名问题，以及明确的证据边界。它不是“主实验成功”的认证，也不是对自然场景泛化、物理层身份或预注册历史的保证。

## A. Ground-truth provenance, identity and leakage — PASS with qualifications

真实评价使用 DTU 激光顶点，不使用模型生成参考。当前脚本复用旧距离，但先核对105个候选及18个现任的精确坐标（`experiment.py:142`、`experiment.py:149`、`experiment.py:154`）。距离祖先在读取封存预测后，从原始 PLY 创建 KD-tree，并对候选调用距离函数（`/srv/slam-research/grf/map-denoise/runs/surface-evidence-20261001T140058Z/evaluate.py:289`、`evaluate.py:296`、`evaluate.py:308`、`evaluate.py:317`、`evaluate.py:328`）。

本审计重新校验了本地下载清单中的 PLY SHA-256，并直接从3,086,735个 scan118 激光顶点及2,731,197个 scan122 激光顶点计算614个有限历史现任/候选距离，最大差为0 mm。下载清单是 `/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z/references/MANIFEST.json:2`，指向 DTU Points.zip 成员。审计没有重新从网络取得原始压缩包；这是可验证的本地来源链，而非独立网络取证。

请求像素、场景、相机、射线、旧候选及当前 photo 输出均逐项对应。准备器只向观测接口投影像素/相机/图像元数据（`prepare.py:29`、`prepare.py:45`）。观测代码在原图域搜索（`observation/extract.py:230`、`observation/extract.py:270`），对源—源搜索固定实测 source1 像素，未用旧候选坐标造循环。读保护是 Python allowlist（`observation/extract.py:36`），不是 OS 沙箱；推理的禁读钩子见 `experiment.py:71`。静态代码及保存的访问记录未显示本次匹配/决策读取激光或评价表。

当前 observation/RUN_LOCK/预测/evaluation 封存及继承评价封存均匹配。记录的顺序为观测封存16:18:07Z → RUN_LOCK16:20:50.793Z → 决策16:20:50.908Z → 预测封存16:20:50.910Z → 评价16:20:59Z。文件时间和自记录时间一致，但同一可写目录内的时间戳及哈希不能独立证明历史上从未提前查看评价。两场景、照片和残余请求已经暴露；这是被明确标注的开发回放，不是盲确认（`PROTOCOL.json:10`、`REPORT.md:14`、`REPORT.md:36`）。

重要身份边界：最近激光顶点不是参考像素的已知同层真值。当前评价不能检验匹配是否属于同一物理目标，报告已正确说明（`REPORT.md:55`、`REPORT.md:59`）。

## B. Score normalization and denominators — PASS

主 MSE/MAE 是四个 ROI 各自均值再等权平均；每臂固定484个原 CPU 有效请求，ROI分母为 upper_fold116、base_ridge119、feather126、book_edge123。512总体中的28个原缺失请求另报，不进入主误差分母。改善/恶化/不变及好点损伤是484点上的计数。实现见 `experiment.py:124` 至 `experiment.py:139`。

没有将评价误差除以本方法自身最大值、最小值或均值。报告百分比是相对固定 photo_U11 MSE 的相对变化，且同时给出原始 mm²，未隐藏绝对误差。NCC 中按 patch 标准差归一化是匹配定义，不是评价分数的事后缩放。

| Arm | ROI-equal MSE mm² | ROI-equal MAE mm | >5 mm | Improved / worsened / unchanged | New harm vs photo | Missing finite /28 |
|---|---:|---:|---:|---|---:|---:|
| incumbent | 29.818499866 | 1.063462938 | 14 | 1 / 3 / 480 | 2 | 25 |
| photo_U11 | 23.955251315 | 0.955204942 | 12 | 0 / 0 / 484 | 0 | 25 |
| joint_interval | 23.416432108 | 0.916811716 | 10 | 2 / 1 / 481 | 1 | 26 |
| official_double | 22.400310398 | 0.905289407 | 12 | 3 / 2 / 479 | 2 | 25 |
| star_full | 16.856879025 | 0.801693935 | 11 | 3 / 2 / 479 | 2 | 25 |
| star | 16.856879025 | 0.801693935 | 11 | 3 / 2 / 479 | 2 | 25 |
| cycle | 23.027710912 | 0.947873657 | 12 | 1 / 1 / 482 | 1 | 25 |

以上全部与 `evaluation/RESULTS.json:3` 及 `evaluation/POINT_METRICS.csv` 一致。star相对降幅29.631800547%，cycle为3.871971080%；逐点等权MSE分别为16.460218327和22.371647388，区别来自事先指定的ROI权重，不是分母变动。所有 raw/相对结果见 `audit/RECOMPUTE.json`。

命名警告：`REPORT.md:25` 所说“原始CPU现任”不准确。`incumbent_distance_mm` 来自历史的 CPU→geo→official 合成现任（历史 `surface-evidence.../evaluate.py:94` 至 `evaluate.py:110`），CPU仅定义 primary mask。因此 `new_harm_vs_original` 是相对该历史合成现任，不是相对原始CPU坐标的损伤计数。该问题不改变已报告数字，但需改名/释义。

## C. Artifacts, claims, seals and tracker — WARN

所有主结果文件存在；报告中的MSE、降幅、场景均值、五个位置距离、替换数、损伤数、缺失有限数和观测统计均吻合。独立复算验证了7×512行、105+18坐标、63个决策、104组有理数端点收益界和12条分臂 transition。21条观测及全部119条原图扫描可逐字段精确重现：90参考—源段/164峰、77直接扫描/493段/612峰、28 star轨迹/13 cycle轨迹，非空分别6/21与3/21。数学副审查直接重算12个仿真实例的NCC、匹配集、收益和24个臂决策；数值及内容哈希一致。

需修正：

1. `refine-logs/EXPERIMENT_TRACKER.md:5` 至 `:8` 仍写 T01/O01 IN_PROGRESS、I01/A01 TODO；它与报告已完成描述不一致。保留历史初版，另补实际完成/审计状态及时间。
2. `REPORT.md:72` 写数学核11项测试；最终 `theory/test_kernel.py:62`、`:69` 已加入两项，实际13项通过。`theory/RUNLOG.md:3` 至 `:4` 正确解释首次11项和数值加固。主报告应报最终13项，同时保留首轮历史。
3. 初始 `PREPARATION_SEAL.json:9` 的 PROTOCOL 哈希为437a...，当前及正式锁为a92c...（`RUN_LOCK.json:8`）。这个唯一哈希不一致已在 `MANIFEST.md:6`、`REPORT.md:75` 披露，正式观测/推理锁均一致，未发现封存后改推理代码的证据；但没有找到初始协议文本快照，不能从最终文件精确验证“仅口径澄清”的全部差异。不要把初始封存描述为全部仍有效；补存实际可恢复的旧版本及准确变更记录，不能事后伪造初始文本。
4. 理论内置 `theory/verify_artifacts.py:19` 至 `:27` 主要核对存储矩阵/子集/标志，不独立从原图重算NCC、几何区间和收益，也不验证所有代码/图像哈希。其PASS应按这个范围理解。本审计的数学副审查已补足本次实例的这些核查，但不能把该内置程序单独描述为全面重现。

准备器及观测代码本身可重复覆盖一些输出，理论 `mechanism.py:149`、`:159`、`:186` 会重写产物；本审计未运行这些入口。现有哈希证明当前一致性，不能等同不可篡改历史存档。

## D. Dead-code / executed metrics — PASS

本轮唯一聚合指标函数 `metrics` 确实在七臂循环中调用（`experiment.py:176`），其所有字段出现在结果JSON且可独立复算。选择器调用真实世界坐标收益核（`experiment.py:89`），并用下界>0筛选（`:91`）；空集保留而非真空接受（`:84`）。理论机制调用 `gain_bounds` 和 `strictly_improves`（`theory/mechanism.py:174` 至 `:183`）。历史评价的距离、ROI统计、配对统计和聚合均在 `surface-evidence.../evaluate.py:317`、`:328`、`:426`、`:430`、`:459`、`:467` 被调用并写出。

没有发现被宣称但从未运行的主评价指标。该判断结合代码调用链、输出字段及独立复算，不是仅凭函数存在。只读重跑本轮5项集成测试通过，数学副审查13项核测试通过；观测SELFTEST保存5项通过及四个禁读负测试记录。

## E. Scope, causality, mathematics and novelty — WARN (qualified evidence only)

真实回放仅2个已暴露场景、4个ROI、21个残余请求和固定105个候选；新场景确认没有运行。当前报告没有冒称综合benchmark、SOTA或普遍改进（`REPORT.md:2`、`:36`、`:68`、`:86`）。官方double比较的预算不同，无法支持同预算优越性。

中点定理及任意3D冻结候选的仿射推广正确：在同一目标、正确射线、覆盖真值的非空有限闭区间并上，端点给出精确极值（`theory/THEORY.md:7` 至 `:33`）。有理数收益运算及向外舍入对输入浮点所代表数成立（`theory/kernel.py:24` 至 `:47`、`:65` 至 `:82`）。它不自动覆盖输入相机误差、对应误差、峰抽样误差或物理身份不确定性。

有限峰、1mm采样、假定padding、未形变patch和一向source1→source2约束不是完整物理可行集合（`observation/extract.py:154`、`:160`、`:223`、`:270`；`observation/README.md:15`、`:19`）。几何区间本身未严格向外舍入：数学副审查在仿真66个被接受对应端点中发现60个相对精确反解向内舍入，最大约9.43e-16长度单位；本批24个臂决策均不变。这已在 `theory/THEORY.md:33` 披露，故不是现有结果失败，但“整个物理集合已数值认证”不成立。

仿真正例通过预置p、p+0.6v、p−0.6v图案构造NCC非传递性（`theory/mechanism.py:27` 至 `:41`）。它支持“实测源—源约束能够提供额外信息”的存在性命题；3种子仅改变纹理/噪声，不是12种独立机制，更不是自然场景成功率。遮挡错层与共同相机尺度错误的反例也重现，说明一致轨迹不保证真值覆盖。该装置不同于真实提取器，报告对此标注正确。

真实cycle仅2次替换，一好一坏；相对star撤回两次大修复并避免一次坏修复，仍保留另一次损伤。预定主验收为cycle MSE同时优于star与photo且无新增photo好点损伤（`PROTOCOL.json:38`）；实际为false（`evaluation/RESULTS.json:180`）。所以真实增量优势尚不成立。star 29.63%总体下降是本批开发结果，scan122反而退化，且新增2处photo好点损伤。禁止将其升级为普适“保留有效几何”的保证。

数学新颖性没有得到本审计证明；当前协议正确禁止声称新定理（`PROTOCOL.json:41`）。

## F. Evaluation type — PASS

- 真实固定坐标回放：**real_gt**，子类 `reused_exact_coordinates / local nearest-laser-vertex`；不是完整官方点云benchmark或同物理层深度GT。
- 世界渲染的12实例：**simulation_only**；真值来自预先固定场景交点，不来自估计器输出。
- 原图NCC/共同轨迹及集合非空率：无GT的观测一致性诊断，可归为 **self_supervised_proxy**；不能单独转译成真实精度或身份标签。
- 未发现未标注的 **synthetic_proxy** 被当成真实GT；没有human_eval。

## Supplemental review: posthoc observation diagnostic

主审计完成后新增的 `observation/DIAGNOSTIC.md` 和 `verify_run.py` 已直接阅读，未将作者解释视为证明。本审计独立复算了scan118/query5三个非峰采样k240/241/247、scan122/query75三个非峰采样k328/334/335，全部满足既定NCC≥0.6、像素差≤1.5px、非空支撑交集及原四观测几何检查；它们不在原局部峰列表，却分别覆盖原star支撑区间。对应的所有原峰在像素关被拒。故两个空cycle确实包含有限峰代表造成的过度收缩，不能解释为原始源—源阈值证据不相容（`observation/DIAGNOSTIC.md:60`、`:99`；`observation/extract.py:287`）。

query95两个近邻集合的最高直接NCC复算为0.585218704和0.583360828，均低于0.6；query48既有直接峰NCC为0.732638487、像素差0.404517523，确实通过。诊断的这三个核心结论得到支持。它们只说明当前网格/patch/有限节点规则，不能辨别物理层，且query5和75同时出现同类缺失，补全集合本身不提供区分好坏修正的信息。

几何检查还有明确语义边界：最小二乘三角化深度只需位于完整公知域，并未要求位于最终射线支撑区间（`observation/extract.py:218`、`:294`、`:298`）。例如两个query5见证的三角化Z约650.4402和657.5531分别落在支撑区间[650.5,651.5]和[656.5,657.5]之外。诊断 `:111` 已披露，故“通过原工程检查”不是“存在一个统一物理点满足全部连续约束”的证书。

新增 `verify_run.py:49`、`:75` 是只读MSE/计数/有理数决策一致性检查；它不验证真实身份或完整观测覆盖，其输出说明与范围一致。它没有覆盖全部MAE/scene/ROI字段，不能取代本审计的全字段复算。审计结论仍为WARN，未改变任何选择器、阈值或封存结果。

## Action items and claim impact

优先修正跟踪表、最终测试数量及历史合成现任的命名；保持封存结果和初始报告快照不变。对准备协议变更保留可核验版本，若旧文本不可恢复，应明确保留溯源缺口。今后引用理论内置验证器时准确描述其范围。

| Claim | Impact |
|---|---|
| 本批star相对photo的ROI等权MSE下降29.6318%，伴随2处新增损伤 | Supported, development-only |
| 本批cycle下降3.8720%，仍有1处新增损伤，主验收失败 | Supported |
| cycle相对star/同预算官方方法优越 | Unsupported |
| 源—源实测约束可以排除某些星形歧义 | Supported as simulation existence construction |
| 该信息增量已使真实回放更好地保留有效几何 | Unsupported by this run |
| 仿射端点规则对正确覆盖集合成立 | Supported with stated same-target/model assumptions |
| 区间已经覆盖真实物理目标或保证最近激光距离不变差 | Unsupported |
| 原始CPU坐标的新增损伤已由new_harm_vs_original统计 | Unsupported wording; actual baseline is historical composed incumbent |
| 两场景之外泛化、SOTA、新理论或全局最优 | Unsupported / not claimed |
| 首次协议从未变动且完整可追溯 | Unsupported; final seals consistent, initial protocol text absent |

结论可以推进为“可复核但未通过主验收的开发实验”。没有证据支持部署本轮cycle或把star对照事后升级为已确认方法。

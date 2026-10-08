# 实验完整性审计

审计日期：2026-10-08。目标已核验为主机 `liekkas` 的 `/srv/slam-research/grf/map-denoise/runs/mixed-pixel-20261008T041249Z`。下列未写绝对目录的 `文件:行号` 均相对于该目录。

审阅身份：独立于方法实现的 Codex 审计代理 `/root/mixed_integrity_audit`，`review_independence=same-family`，`acceptance_status=provisional`。这是同家族临时审阅，不是外部认证。仅 `audit/` 由审计代理写入；未改方法/评价源码、未生成数据、未运行完整实验。曾尝试额外公式复审，但工具以 agent thread limit 拒绝，未据此声称第二份独立复审。

调度者提供的原始请求参数为 `requested_model=gpt-6-astra`、`requested_reasoning=ultra`、`fork_turns=none`；代理创建成功。这是调度记录，不是本代理对实际后端型号的独立证明。

**整体完整性结论：WARN。** A/B/C/D/F 通过；E 保留范围警告。当前主结果没有发现伪造真值、以最终预测极值美化分数、遗漏拒绝样本、哈希不符或错误汇总。科学结论另行判断：E1 方法门槛未通过，不能把完整性通过解释为方法有效或允许进入 E2。

## A. Ground Truth Provenance：PASS（附隔离限定）

E1 的 `true_depth=600` 和 `background_depth=900` 来自生成器固定场景参数，而非任何模型预测；图像由世界射线与两个平面交点、独立矩形归属、连续纹理和九射线平均生成。证据：`fixtures.py:58`、`fixtures.py:77`、`fixtures.py:128`、`fixtures.py:141`、`fixtures.py:153`、`fixtures.py:158`、`data/truth/metadata.json:44`。几何误差由封存选择与该真值的绝对差计算，真值在预测封存校验后才用于评价：`evaluate.py:45`、`evaluate.py:51`、`evaluate.py:52`、`evaluate.py:56`。

普通估计器 API 仅接收参考图、训练图、两相机和公共参数，没有评分图、世界标签、目标深度或候选参数：`predictor.py:317`、`run.py:134`、`run.py:136`。评分图只进入冻结预测的损失计算：`run.py:155`、`predictor.py:193`、`predictor.py:204`。`observed/` 只含图像、相机和匿名 ID；truth/oracle 另存：`fixtures.py:214`、`fixtures.py:216`、`fixtures.py:219`、`fixtures.py:226`。Oracle 辅助量显式特权且没有目标深度或源图真实归属：`fixtures.py:169`、`fixtures.py:174`、`fixtures.py:223`；审计逐 NPZ 核验该字段合同。

**隔离限定：** `verify_lock()` 在 `run.py:54`–`58` 对所有 input 文件读取字节做哈希；ordinary/oracle 在 `run.py:105` 调用后，到 `run.py:121` 才安装 Python audit hook。因此不能声称普通进程从未打开 truth/oracle 的字节，也不能声称 OS 沙箱隔离。已核查估计没有语义读取这些字节；guard 安装后的普通读取日志仅含 observed NPZ，truth/oracle 试读被拒绝：`ordinary_stage/OBSERVATIONS.json:6541`、`ordinary_stage/OBSERVATIONS.json:6545`。Oracle 的 truth 试读被拒绝：`oracle_stage/OBSERVATIONS.json:3727`。`run.py:78` 本身明确说明不是 OS sandbox。

E0 的一项解析零误差是自洽检查：`e0.py:102` 直接选取解析 prediction 的真深度行作为 observed；该项不能单独证明独立预测性能。它已明确限定为数值 sanity：`e0/RESULTS.json:852`。额外独立实现测试分别从生成器和直接针孔公式生成观测，再检查共同预测器：`test_integration.py:35`、`test_integration.py:135`、`test_integration.py:140`。因此没有把 E0 自洽分数冒充 E1 外部真值成绩。

## B. Score Normalization：PASS

原始 `L` 是共同整数传感器像素上的 MSE：`predictor.py:193`、`predictor.py:204`。训练尺度来自供体留块残差，含二像素隔离带，每表面至少四块、十二点；`sigma=max(1,1.4826*MAD)`：`predictor.py:241`、`predictor.py:254`、`predictor.py:261`、`predictor.py:395`、`predictor.py:412`。它不是最终预测曲线最大值或模型成绩自归一化。

两折按像素数合并 MSE；共同方差 `sum(n_f*sigma_f^2)/sum(n_f)` 用于四个混合模型臂，Oracle 复用普通估计的尺度；`J=L/sigma²`：`run.py:97`、`run.py:140`、`run.py:143`、`run.py:158`。原始 loss、normalized loss、fold loss、pixel counts、sigma values 全部同时保存：`run.py:172`。9/4 阈值在封存 `METHOD.json:8`、`METHOD.json:9` 中，不能解释为概率保证。

审计复核全部保存曲线的分母、归一化、接受集和区间，均匹配。72/108 个 ED 配置因尺度缺失 KEEP，没有从几何分母删掉；另 18/108 为 flat：`evaluation/SUMMARY.json:1282`。原始曲线仍保存并完成排序；因此“没有支持”不等于“没有信息”。

## C. Result File Existence：PASS

执行日志记录了单独的 prepare、ordinary、oracle、infer 和 evaluate 进程，全部退出码 0：`logs/COMMANDS.json:16`、`logs/COMMANDS.json:28`、`logs/COMMANDS.json:40`、`logs/COMMANDS.json:52`、`logs/COMMANDS.json:64`。两预测进程各完成 36 世界，infer 产生 648 决策：`logs/02.log:36`、`logs/03.log:36`、`logs/04.log:1`。28 项测试成功：`logs/00.log:31`。

已核对 19 个 source/protocol 文件、75 个输入文件和 663 个历史文件；ordinary seal 的 253 个工件、oracle seal 的 217 个工件、决策 seal 与评价 seal 均匹配当前字节。参考封存结构：`LOCK.json:4`、`LOCK.json:25`、`decisions/SEAL.json:3`、`decisions/SEAL.json:4`、`evaluation/SEAL.json:3`。648 唯一 `(world,arm,incumbent)` 组合全部存在，没有重复顶替缺行，CSV 与 JSON 数字一致。

审计程序不导入方法、生成器或评价模块，独立重算 P、全部逐行几何指标、candidate ranking、按初态/机制汇总、保留率、拒绝原因和 E1 gate。**4,128 项全部通过**：`audit/VERIFICATION_v2.json:5`。旧 30 例的 MAE=66，最大 NCC 差 5.551115123125783e-16；结果与输入哈希亦复核：`legacy_replay/RESULTS.json:2`。

主结果复算为：现任 540/660 时 ED MAE 均 56.6666667，N 均 10；现任 600 时 ED MAE 6.6666667、损伤 4/36，N 无损伤；ED 与 EF 的主终点相同。证据：`evaluation/SUMMARY.json:43`、`evaluation/SUMMARY.json:21`、`evaluation/SUMMARY.json:125`、`evaluation/SUMMARY.json:103`、`evaluation/SUMMARY.json:207`、`evaluation/SUMMARY.json:185`。E2 gate=false 已独立重算：`evaluation/SUMMARY.json:1298`、`audit/VERIFICATION_v2.json:418`。

补充诊断是评价后的只读重放，未替换原预测：`diagnostics/replay.py:1`、`diagnostics/replay.py:19`、`diagnostics/replay.py:31`。576 个 pixel 工件补齐逐像素残差、前景/背景/边界分解和 pairwise/regret。**7,179 项独立诊断检查全部通过**，含独立射线/矩形公式对真实归属的重算、区域 MSE、共同 ROI、固定 alpha、封存 fold loss 和 pairwise/regret；最大 loss 差为 0：`audit/DIAGNOSTIC_VERIFICATION.json:2`。此处没有调用完整预测/生成程序。

审计自身首轮 verifier 漏了旧区间规则中的半网格扩展/裁剪/合并，产生假差异；核对历史 `support.py:120` 后修正独立实现，首轮 `audit/VERIFICATION.json` 保留不覆盖，v2 注明修订：`audit/VERIFICATION_v2.json:427`。这是审计程序修复，实验实现与结果未改。

## D. Dead Code Detection / Test Sensitivity：PASS

评价函数 `stats`、`compare`、`rank_curve` 在流水线确实被调用并产出保存字段：`evaluate.py:62`、`evaluate.py:68`、`evaluate.py:69`、`evaluate.py:70`、`evaluate.py:100`、`evaluate.py:112`。原始损失曲线也确实使用共同 `predict_curve`：`run.py:155`。未发现定义但未运行、随后被当成绩报告的指标函数。

固定臂在现任深度冻结每条子射线的归属和前后顺序，但保留每个候选的前景纹理投影：`predictor.py:174`、`predictor.py:180`、`predictor.py:181`、`predictor.py:183`。固定/动态的解析纹理差和独立 expected ownership 测试会检测“冻结整个 warp”错误：`test_predictor.py:41`、`test_predictor.py:46`、`test_integration.py:155`、`test_integration.py:163`。F=B、非零基线、原始误差无 sigma 仍存在的测试分别见 `test_predictor.py:52`、`test_predictor.py:62`、`test_integration.py:75`。拒绝不删分母和两折加权也有测试：`test_runner.py:7`、`test_runner.py:10`。这些测试只支持已测数值合同，不证明迁移性能。

## E. Scope Assessment：WARN

本次是 E1 开发装置：四机制、两宽度、三个复用种子，24 设计组加 12 背景配对形成 36 世界；三个初态、六臂共 648 决策。定义见 `fixtures.py:17`、`fixtures.py:18`、`fixtures.py:19`、`fixtures.py:184`；结果诚实注明开发且背景配对/折不独立：`evaluation/SUMMARY.json:3`–`7`。始终真深度 600、相同 3×3 核、规则矩形、准确相机；这不支持跨深度、跨形状、真实图像或独立确认。

**36 世界实际只有 30 个不同观测张量。** flat_equal 与 textured_single 各三个跨宽度重复对；前者 F=B 不携带几何，后者根本不使用宽度。原因见 `fixtures.py:92`、`fixtures.py:107`、`FIXTURE_NOTES.md:9`；逐字节比对结果见 `audit/VERIFICATION_v2.json:428`。因此 24 应称“设计组”，不应称“24 个独立信息样本”。正确现任的四个损伤配置为 w002/w018 与 w014/w026，实际两个不同观测张量；不能报告四个独立场景失败。

四个混合模型臂共用每折 182 个原始评分像素，候选/初态不能改变 ROI；合同见 `predictor.py:91`、`run.py:155`，重放逐坐标核验通过。full9 使用 9×9=81 个参考采样点和 NCC，故只能称同三图/相机/扫描/候选的系统比较，不能称六臂完全相同评分像素或分母；见 `baseline.py:20` 及历史 `/srv/slam-research/grf/map-denoise/runs/surface-owned-support-20261008T022918Z/support.py:75`。EF/ED 及 OF/OD 配对才隔离覆盖变化贡献。

原始排序中 ED/OD 在边界世界给出真候选优势，而 F=B 曲线全域平坦；这支持本装置中的信息通道。不能由此声称修复系统成功：72/108 尺度拒绝、4/36 正确现任损伤、两方向均输给 full9，且没有 ED 对 EF 的系统增量。相应门槛明确 false，E2/E3/E4 和真实确认未由这些结果支持。

## F. Evaluation Type Classification：PASS — simulation_only

E1、E0 和旧回放均应按仿真/解析开发证据解释。E1 标签已正确登记：`METHOD.json:15`、`evaluation/SUMMARY.json:3`。Oracle 使用生成器的真辅助量，仅用于特权机制诊断，不能称可部署成绩或统一性能上界：`fixtures.py:169`、`fixtures.py:224`。没有真实数据 GT、人评或独立真实确认。

## 主张影响与交付限定

- “在这个合成开发装置，动态覆盖保留了 raw candidate discrimination”：支持，须连同固定臂、F=B 负控及配对单位解释。
- “当前 ED+原 P 可以保留正确点并胜过 full9”：不支持，且本次已观察到反例。
- “实验数值和工件可复算”：支持，确定性检查共 11,307 项通过；检查数不是统计样本数。
- “从未打开特权文件 / OS 隔离 / 外部认证 / 独立迁移成功”：不支持。

## 最终报告与图件核查

已读 `REPORT.md`、`CHECKPOINT.md`、`COMMANDS.md`、`EXPERIMENT_TRACKER.md`、`diagnostics/figures.py`、`figures/MANIFEST.json`，并目检三张 PNG。报告主表（`REPORT.md:25`）、30 个有信息世界/27 个不同观测的限定（`REPORT.md:11`）、重复配置（`REPORT.md:21`）、四损伤配置/两个不同图像（`REPORT.md:33`）、尺度拒绝（`REPORT.md:60`）、w002 候选损失/区间/P 选择（`REPORT.md:66`、`REPORT.md:72`）都与封存数据一致。明确没有进入 E2，且最小残差只作为开发诊断（`REPORT.md:91`、`REPORT.md:95`）。报告没有把负结果写成通过。Tracker 将已执行的 E1 标记 DONE/endpoint gate FAILED、审计 DONE/WARN、E2–E4 NOT RUN：`EXPERIMENT_TRACKER.md:8`、`EXPERIMENT_TRACKER.md:10`、`EXPERIMENT_TRACKER.md:11`。

图件显示原始 MSE 或几何 MAE，没有自归一化；描述性例子明确标注事后选择，没有置信区间或独立样本推断：`diagnostics/figures.py:38`、`diagnostics/figures.py:50`、`diagnostics/figures.py:62`、`diagnostics/figures.py:75`、`figures/MANIFEST.json:15`。图源与三 PNG 哈希匹配 manifest。E0 面积误差实际为 2.2798214319164067e-7，报告舍入正确；预检 RSS=1,467,656 KiB 约 1.40 GiB，报告正确注明不是完整管线峰值（`preflight/RESULTS.json:72`、`REPORT.md:80`）。

现有报告已披露本审计的范围警告；保留 WARN 是提醒证据只能支持有限装置内的结论，并不要求改变封存结果或继续未获门槛允许的实验。

最终 `REPORT.md` 已将临时审计状态更新为 11,307 项确定性检查通过及 WARN（`REPORT.md:85`）；最终核验 SHA-256 为 `a93fee6740483d3c34a3a71f47aacb669d11fff0ba33fa8355afa85dd3fd60c2`。完整原始审计响应另存 `audit/response.txt`。

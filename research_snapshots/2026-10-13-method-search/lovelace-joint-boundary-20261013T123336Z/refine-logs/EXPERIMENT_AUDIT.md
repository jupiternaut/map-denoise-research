# A14 实验诚实度审计

日期：2026-10-13T13:26:49Z。目标：liekkas 上 `/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z`；已核对其物理解析路径，未使用副本。审查者：fresh GPT-6-Astra / ultra，agent_id=`/root/a14_integrity_review`（调用未返回 UUID）。review_independence=same-family；acceptance_status=provisional。

## 总体结论：WARN；未发现阻断本轮限定实测主张的问题

主要剩余问题是**事前源码冻结不覆盖生成器独立核验链的完整动态依赖闭包**：56个登记项当前哈希全部匹配，但另有11个动态导入的祖先验证器未登记。不能据此声称完整运行闭包均事前冻结；当前哈希也不能倒填成历史证据。最新版 REPORT.md:81 已披露。此问题限制历史来源保证，不否定本次独立重建成功的18世界数据、630条预测误差或1091次排除证明。

C1、C2在18世界、3种子、同一模拟物理族内得到限定支持。没有真实照片/点云、跨机制泛化、外部SOTA、总体统计显著性或全面优于fine/legacy@24的证据。原稿的“三种负控”已经改为同一 full×lowcontrast 机制的三个世界（REPORT.md:31、66）；旧稿保留，未改正式数据。

## 确定性复核范围

独立审计脚本只用标准库及冻结的独立证明验证器，不运行原生成器、渲染器、求解器或正式矩阵。它复放全部随机潜变量/任务ID，独立用Fraction重建矩形投影、遮挡面积、7776个照片分量及相机/外观测量，再从全部90个job/result/receipt重算630行评分和各汇总。全部通过，比较最大差异0；结果见 independent_audit.json:42、52、834。脚本全文已审阅。

一次运行的完整进程rusage为1.058661 CPU秒（脚本测量段1.016570199秒，墙钟1.019260546秒），低于8秒。四线程变量均为1，-B与PYTHONDONTWRITEBYTECODE=1在脚本中断言；CPU硬限8秒。证据：independent_audit.py:24、30；independent_audit.json:936。未联网、安装、GPU或改动实现/祖先目录。

路径缩写仅用于下列源码证据：A12=`/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z`；A13=`/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z`；BI=`/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z`。

## A—F 必检项

| 项目 | 状态 | 直接证据与判读 |
|---|---|---|
| A 真值来源 | PASS | A12/data_builder.py:119先抽真实深度/相机/材质/噪声，:134前向生成照片，:141独立精确前向复核，:159才生成现任。无方法输出作真值。BI/appearance-variation-1/prepare_variation.py:40、BI/baseline_geometry.py:107是被调用的有理数前向实现。独立复放全部18事实及前向观测一致（independent_audit.py:118、204；independent_audit.json:834）。冻结来源限制单列G。 |
| B 原误差与分母 | PASS | experiment.py:113、118直接使用真实深度与原始现任/输出算AE、SE；A13/run_experiment.py:174按固定样本数求MSE/MAE。每配置全部126、扰动108、正确18、负控21输入，失败不筛除。相对identity改善的分母是原始identity MSE=22.00089308712955；“86.3584%补差”明确为同批base与legacy@24差额，未冒充总误差降幅（REPORT.md:46；independent_audit.json:827）。没有以输出最大值/最小值自归一化。 |
| C 文件/数值/完成状态 | PASS | 90组job/result/receipt及180个状态文件完整，630预测唯一，所有receipt结果SHA一致，METRICS/SUMMARY/COSTS/ANALYSIS全量重算一致（independent_audit.py:266、285、330、341；independent_audit.json:44）。REPORT.md:35、56数值与正式文件对应。refine-logs/EXPERIMENT_TRACKER.md:10—13正确将执行/核验列DONE、尚在进行的fresh审查列IN_PROGRESS，无把TODO当结果。 |
| D 执行路径/死代码 | PASS | experiment.py:103→score→core.aggregate及analyze.py:7→main均有对应完整产物。实际joint搜索走search_method.py:57—89内联四tile提案并调用JointContext.check_cell；joint_kernel.py:82的可选propose_cell只用于开发测试，不是正式计时路径，未把它的未执行输出当正式结果。证书构造与独立核验均有实际记录。 |
| E 样本/主张范围 | PASS（严格限域） | data/confirm/GENERATION.json:2—8与全部facts对应3个种子、每种子6条件，共18world。REPORT.md:3、29明确simulation_only同族新参数；统计单位为18world，六个随机扰动不当独立world。严格说生成器会拒绝幅度重复/偶然对称，六个扰动有联合筛选；不应进一步称为无条件独立样本（A12/data_builder.py:106）。3个负控是一个机制的3个实例。没有不确定区间或显著性检验，限定为这批确认数据的描述性比较。 |
| F 评价类型 | PASS：simulation_only | A12/task_bundle/TASK.md:7—15定义固定针孔、矩形、材质和有界误差模型；truth来自模拟潜变量前向，不是自然数据GT或预测构造的synthetic_proxy。idea-stage/docs/research_contract.md:9、REPORT.md:3给出正确分类。 |

## G. 冻结与方法信息流：WARN / PASS

登记56项源码/契约哈希均保持不变，SOURCE_FREEZE哈希与GENERATION、PREDICTIONS_READY引用一致（independent_audit.py:171；independent_audit.json:43、919）。冻结UTC为13:08:53.500593Z，生成UTC为13:08:54.782252Z。experiment.py:25继承旧清单并枚举顶层.py，A13/run_experiment.py:57只遍历sys.modules；祖先验证器用module_from_spec/exec_module但不登记sys.modules，因此遗漏下列11个动态依赖：

- `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1/verify_appearance.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1/verify_variation.py:34`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/recovery-1/verify_baseline.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1/verify_appearance.py:36`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/ablation-1/verify_ablation.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1/verify_appearance.py:39`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1/verify_point.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/independent-anchors-1/verify_anchors.py:37`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/verify_polytope.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/point-feasibility-1/verify_point.py:32`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z/verify_structured.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/recovery-1/verify_baseline.py:42`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z/verify_bridge.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-structured-view-bias-20261012T022500Z/verify_structured.py:37`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-view-bias-20261011T092300Z/verify_view_bias.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z/verify_bridge.py:39`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-shared-ablation-20261011T055200Z/verify_ablation.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-view-bias-20261011T092300Z/verify_view_bias.py:33`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-shared-appearance-20261011T052300Z/shared_verification.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-shared-ablation-20261011T055200Z/verify_ablation.py:35`。
- `/srv/slam-research/grf/map-denoise/runs/lovelace-nuisance-20261011T044500Z/verification/arithmetic.py`；调用证据 `/srv/slam-research/grf/map-denoise/runs/lovelace-shared-appearance-20261011T052300Z/shared_verification.py:35`。

这些遗漏影响“原生成时独立核验器整个闭包未变”的历史证明。此次没有回填SOURCE_FREEZE，也没有通过事后重跑正式矩阵制造事前证据。审计JSON保存的是审计时哈希，明确为post-hoc；独立Fraction前向重建给出了另一个当前数据一致性检查，但不能证明未记录的历史字节状态。

方法信息流为PASS：experiment.py:47、53只把公开三字段与现任任务分派；A13/worker.py:23调用prepare时仅给observation，:33每次decode只给一个incumbent；search_method.py、joint_kernel.py、A12/free方法均无truth文件读取。experiment.py:66先保存全部预测，:83全量无真值核验，:108才读truth评分。结果哈希、90个job时间序列及文件时间支持该顺序（independent_audit.py:463）。这是代码/产物证据，不是操作系统强制访问隔离或不可篡改日志；COMMANDS.md:28已说明。

## H. 四域证明、完整分区与保证边界：PASS

joint_verify.py:77—90独立构造精确中点的四个闭笛卡尔积域，要求全部四份有序证书，绑定输入哈希、分支、精确深度、camera_box、变量顺序及row_count；:92—118重建每个非零权重所需的物理行，用严格正有理数盒矛盾接受。只重建稀疏证书支持行是合法的：任意必要不等式子集若已矛盾，整个系统必矛盾；未使用的零权重行不是被遗漏的证书义务。

A13/verify_certificates.py:327—392检查所有排除、每分支[540,660]的闭区间完整分区、无内部空隙/重叠及保留集合凸包；experiment.py:89—97遍历全部54个base/fine/joint状态并检查解码。独立重验54/54状态，1091/1091排除通过：1025整域、66四域、264子证书（independent_audit.json:862）。五配置原始有理数真值也全部18/18落在输出包围内；此处不用浮点真值代替有理数比较（:840）。

保证是对声明物理模型及公开binary64读数舍入原像的必要条件包围，不是精确物理可行集。legacy经A12/library/methods.py:16把公开浮点字符串转换为有理数，且未经过本轮新验证器；新核A13/cert_kernel.py:49用舍入原像闭区间。REPORT.md:75、79已区分，不能声称五配置同一形式保证。

## I. 成本、失败与负控：PASS（正式成本）；开发成本为有限记录

正式冷调用由A13/run_experiment.py:85—107包住新进程启动至退出，包含import、prepare、7次decode、序列化与写盘；import不藏在warm结果外。失败有显式identity回退并保留所有tasks；A13/worker.py:21、30、39记录超时/解码失败，experiment.py:128的分母不排除失败。90调用实际零进程失败、630预测零failure，五配置overrun>budget+0.25均0；未把“本轮未出现拒证”说成所有情形不会拒证（REPORT.md:75）。

全部三个负控world、每配置21输入在总体分母；均保持identity，未暗删难例。54个快臂均boundary_resolution。子进程CPU244.303552秒、冷调用墙钟合计244.401903669秒、批次墙钟244.555334514秒；后三者含义不同（independent_audit.json:874）。outputs/time_predict.txt:1的父子进程总CPU=244.53，time_freeze_generate.txt:1=0.89，time_verify_score_analyze.txt:1=0.95，合计246.37秒（外部time按0.01秒舍入），没有把内外计时重复相加。fresh独立算术另记1.058661 CPU秒。570子进程与30附属预算不是统一自动硬截止，REPORT.md:79明确说明。

开发诊断三个JSON记录19.758354787+1.351518312+0.282511663=21.392384762 CPU秒；smoke三receipt记录1.093138 CPU秒。测试失败和重试在DEVELOPMENT_CHECKS.md:6—8显式披露，但初次测试约0.88CPU秒和部分测试墙时只有该开发记录，缺少每次测试的机器可读完整CPU receipt；不将其升级为精确完整开发总成本。正式冷推理比较不受此影响。

## 主张影响

| 主张 | 判读 | 必要限定 |
|---|---|---|
| C1 | supported_with_qualifier | joint扰动MSE6.794983958823187 < base6.880609897601817，平均冷调用0.579479837秒 < 同预算legacy4.460871882秒，证书/包围无失败；满足预定门槛。joint比base耗时增加41.6693%，比legacy@24误差高0.199454%，不能说同时全面胜出。 |
| C2 | supported_with_qualifier | joint均值比fine6.845114501592527低0.732355%；66个mode=tiles实际排除来自整盒浮点未排除后四子域证书接受的代码路径（search_method.py:62—88），不是只细化深度阈值。但不证明这66个整盒精确系统均可行，也不作各因素贡献的可加因果分解。 |
| 86.36%差额回收 | supported_with_qualifier | 同批base到legacy@24的MSE差额分母明确，复算86.3584076411%；fine35.7990338171%。它不是总MSE下降86%。 |
| 真实场景/跨机制/原创新方法 | unsupported if asserted | 当前报告未作这些断言；四相机分割已存在于BI/continuous-tightening-1/tightening_solver.py:44。 |

配对核对（更好/相同/更差）：joint相对base为41/67/0输入、12/6/0世界；fine为37/33/38输入、8/3/7世界；legacy@6为75/33/0输入、15/3/0世界；legacy@24为23/33/52输入、4/3/11世界。每世界六扰动数相同，world SSE与world MSE比较同号。18正确输入全部保持；joint的18包围皆为base子集，13个严格更窄但只有12world的这组扰动MSE改善。证据：ANALYSIS.json:3—53；independent_audit.json:827；REPORT.md:56—66。

## 行动与归档

保留本WARN与simulation_only/same-family/provisional标识；无需为本审计改实现或重跑矩阵。后续若要宣称完整事前来源保证，必须在未来实验开始前登记完整动态导入闭包，本轮不能追溯补造。开发成本不声称精确完整总额。

本报告核对的REPORT.md与REPORT_20261013_132200.md均为SHA256 `9e2bfcd14388cfd7afbdc16b0a1a7b1ae68d02464e6c577b50ff4ba553e53a6d`。完整私有请求/响应/meta保存在项目内审计trace目录，未公开。审查源码/产物哈希与缺失依赖审计时哈希见同名JSON。

补记（2026-10-13T13:34:05Z）：直接对比了最终REPORT_20261013_133200.md/latest与已审阅132200版本；只改审查完成状态、明确11项遗漏、增加已核验数字/成本和扰动联合筛选限定，未扩大科学主张。当前两文件SHA256均为 `7b900d677186ab248493c3dd3481c9533d7d84560aa609736fdefbc4a065c41a`。正文对tracker状态及REPORT哈希的引用是13:26:49审阅快照；最终tracker已完成。未重跑任何实验或核验。

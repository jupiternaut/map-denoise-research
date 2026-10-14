# A11–A14：数学计算通路的构造、认证与修订

发布日期：2026-10-13。源主机：`liekkas`。目标仓库：`jupiternaut/map-denoise-research`，`main`。

本次发布已有实验，不重跑研究、不改旧报告、不改部署默认。原始几何目标仍是**保留有效几何、修复错误**；新增研究对象是实现该目标的计算机制本身。

后续更新（2026-10-14）：[A15纯容差、边界检查与证明复用](LOVELACE_A15_20261014.md)已单独归档。它在本页A14同批观测上补齐joint .001对照，MSE降至6.763968；旧38个fine比较损失全部消除，legacy比较损失52→15。以下表格保留A14原始运行结果。

## 研究链及入口

| 阶段 | 本轮改变的对象 | 直接入口 |
|---|---|---|
| A11 | 累积排除证书，继承已经收紧的外集 | [报告](../research_snapshots/2026-10-13-method-search/lovelace-evidence-accumulation-20261013T051931Z/REPORT.md) · [条件关系](../research_snapshots/2026-10-13-method-search/lovelace-evidence-accumulation-20261013T051931Z/THEORY.md) |
| A12 | 固定方法／固定方法库／自由构造三臂；重新组织约束构建与端点求解 | [自由构造](../research_snapshots/2026-10-13-method-search/lovelace-representation-search-20261013T080031Z/arms/free/METHOD.md) · [审查与实测](../research_snapshots/2026-10-13-method-search/lovelace-representation-search-20261013T080031Z/EXPERIMENT_AUDIT.md) |
| A13 | 浮点寻找排除提案，有理数重建并独立核验证书 | [报告](../research_snapshots/2026-10-13-method-search/lovelace-certified-boundary-20261013T093552Z/REPORT.md) · [证书构造](../research_snapshots/2026-10-13-method-search/lovelace-certified-boundary-20261013T093552Z/CERTIFICATE_METHOD.md) |
| A14 | 用精确反例定位相关性损失，接入固定四相机域的析取必要约束 | [报告](../research_snapshots/2026-10-13-method-search/lovelace-joint-boundary-20261013T123336Z/REPORT.md) · [诊断](../research_snapshots/2026-10-13-method-search/lovelace-joint-boundary-20261013T123336Z/DIAGNOSIS.md) · [推导](../research_snapshots/2026-10-13-method-search/lovelace-joint-boundary-20261013T123336Z/JOINT_DERIVATION.md) |

A11复用36观测；A12、A13、A14各自使用不同的18个新参数世界。不要横跨批次比较MSE绝对值，或将每世界多个初态算作独立场景。

## 最新同批结果

A14五配置共90次冷调用、630预测；共同identity的108扰动输入MSE为22.0008930871 mm²。18个正确输入单列。

| 配置 | 扰动MSE（mm²） | 每世界平均冷调用秒 |
|---|---:|---:|
| base@6 | 6.8806098976 | 0.4090369704 |
| fine@6 | 6.8451145016 | 0.4901049659 |
| joint@6 | 6.7949839588 | 0.5794798374 |
| legacy@6 | 7.0167816263 | 4.4608718820 |
| legacy@24 | 6.7814580708 | 7.6383898815 |

joint相对identity：75改善、33不变、0恶化；18正确输入保持。相对base：41更好、67相同、0更差。相对fine：37更好、33相同、38更差。相对legacy@24：23更好、33相同、52更差。均值和逐任务优势不是同一个主张。

joint比legacy@24快13.18倍，MSE高0.1995%；相对base改善1.2445%且成本增加41.67%。恢复旧长配置与base间差额的86.36%只描述这一小差额，不是全误差下降率。legacy是内部旧求解器，不是外部成熟MVS。

[逐输入CSV](../research_snapshots/2026-10-13-method-search/lovelace-joint-boundary-20261013T123336Z/outputs/formal-v1/METRICS.csv) · [汇总](../research_snapshots/2026-10-13-method-search/lovelace-joint-boundary-20261013T123336Z/outputs/formal-v1/SUMMARY.json) · [成本](../research_snapshots/2026-10-13-method-search/lovelace-joint-boundary-20261013T123336Z/outputs/formal-v1/COSTS.json) · [原始独立核验](../research_snapshots/2026-10-13-method-search/lovelace-joint-boundary-20261013T123336Z/outputs/formal-v1/INDEPENDENT_VERIFICATION.json)

## 这属于哪种“寻找更好的数学方法”

不是单纯把同一求解器运行更久：A12重构约束与搜索组织，A13改变寻找证明与接受证明的分工，A14改变松弛所保留的相机联合关系。但A12仍回到深度外集＋HULL同族，A14四域也来自祖先方法。已经有自主构造计算通路、诊断和修订的实例；尚无自动发现新表示族或跨任务普遍方法发现能力的证据。

一个可操作的T–Π–O记录方式是：T固定观测、允许世界、目标量、损失和预算；Π登记观测编码、变量/依赖、消元与松弛、求解器、解码器和验证器；O交付预测、证书/UNKNOWN、实耗和失败。每个Π必须说明哪些步骤等价、哪些只给必要条件、哪些引入新假设。此处是研究组织建议，不是已经证明的新通用理论。

## 快照与公开范围

新档案位于 `research_snapshots/2026-10-13-method-search/`，不向旧十根快照目录追加文件，从而不破坏其精确文件集清单。

本次封存1,590个文件，来源合计93,748,262字节，发布合计63,018,571字节；8个大文本无损压缩。另有44个私有审查/缓存文件按原因排除。所有纳入来源在导出前后摘要一致。

- 四个完整科学运行根：代码、协议、开发/确认数据、模拟评价真值、保存预测、证明、原始报告、失败及更正。
- `private/truth.json` 是模拟评价标签，不是账号或个人隐私；公开用于复算，不意味着方法曾获准读取它。
- 私有 `.aris` 审稿轨迹、运行缓存排除；逐文件原因与摘要见[清单](LOVELACE_METHODS_20261013_MANIFEST.json)。公开审查结论保留，引用私有trace的历史链接仍可能不可用。
- 历史 `AGENTS.md` 仅在发布副本重命名为 `SOURCE_AGENTS.md`，不把旧指令变成读者当前权限。
- 大于1 MiB的JSON/CSV/log无损gzip，历史报告原名链接可能需要在同目录查找`.gz`；每项保存压缩前后SHA-256和字节数。
- 不是原始真实数据集、环境、完整聊天或GitBook网页备份。此次不重建GitBook。

## 复核与重跑边界

在仓库根目录运行（Python标准库即可）：

```bash
python3 -B publication/verify_lovelace_methods_20261013.py
python3 -B -m unittest discover -s publication -p 'test_verify_lovelace_methods_20261013.py' -v
```

新便携复核只从发布目录读取，检查文件集合、存储与解压后摘要，再从保存的A14任务、预测、评价真值与调用收据重算终点/成本。**这是对封存记录的算术复算，不是再次生成18世界，不是重跑求解器，也不是独立重证所有几何证书。** A13/A14原始证明审计代码与收据另行保留。

本次实际复核PASS：630预测、全部汇总与90调用成本最大差异0；新检查程序15项测试通过，含文件丢失、额外文件、字节篡改，以及重新封存后伪造数值的拒绝测试。旧快照2,280文件与原6项测试仍通过。[发布核验收据](LOVELACE_METHODS_20261013_VALIDATION.json)保存范围与程序摘要。

原始研究代码保留主机路径、祖先模块导入和私有review gate，不保证直接在另一台机器执行。完整求解重跑需明确适配来源路径与环境；本次没有悄悄修改科学源码让它“看起来可运行”。新导出脚本也仅适用于已验证的liekkas路径。

## 不抹去的记录缺口

1. A12正式记录和审查已完成，但原根目录缺最终REPORT/CHECKPOINT、tracker仍滞后。以保存预测/评分和EXPERIMENT_AUDIT为准，不为历史伪造完成时间。
2. A14事前SOURCE_FREEZE中56项一致，但漏记11项动态导入的生成核验依赖；原始WARN保留。本次较晚的归档摘要不能补成事前冻结证据。
3. 开发测试没有每次完整CPU收据，不能声称已精确衡量全部研发成本。冷调用费用不包含此前Agent构造与历史开发。
4. legacy将浮点字符串有理数化，新证书核包围binary64舍入原像，五臂读取相同公开数值但形式观测编码保证不完全对称。
5. 本轮只在已知物理族内确认，语义审查是same-family/provisional；不把它写成真实点云迁移或外部论文审稿接受。

下一项研究应由可检查的约束缺口驱动，而不是为了追到零差额不断扩大预算；本次发布未执行新的后续实验。

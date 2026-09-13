# V19–V22：研究进展、结果与可复核证据

2026-09-13，发布自 liekkas 原工程。此页是实验完成后的发布索引，不改写原协议或结果。

## 各轮解决了什么

| 阶段 | 主要动作 | 实验判断 | 报告 |
|---|---|---|---|
| V19 | 交叉拟合与后验导出两线 | A未升级；B有用途差异，未实现明显提精度 | [V19](exploration_v19/REPORT.md) |
| AgentRx适配审计 | 定位候选搜索退化与跨任务推断 | 不是运行官方AgentRx，也不是将所有负结果视为代理失败 | [审计](research_audits/agentrx_20260913/工程审计_AgentRx.md) |
| V20 | 保留旧候选；各训练折独立重建候选链 | 恢复V18水平，修复不等于新增精度收益 | [V20](exploration_v20/REPORT.md) |
| V21 | scan24公开网格＋独立DTU参考 | 整片平行面适配退化，不支持真实迁移升级 | [V21](real_closure_v21/REPORT.md) |
| V22 | 逐点移动局部曲面＋多尺度修正；新增scan37 | 减轻V18损伤，未达到预定综合升级标准 | [V22](reconstruction_v22/REPORT.md) |

V18的受控薄板族收益没有因迁移失败而自动作废，也没有因此获得真实地图适用性保证。
没有证明局部或全局最优。V22属于经典MLS/局部多项式回归构造，不宣称新的通用数学原理。

## V22 新场景关键结果

scan37：24个不共享源顶点的1024点片区，合计24,576点，约占输入网格3.13%。
下表为局部片区等权均值，不是全场景DTU官方总分。scan24的18个旧片区只作暴露诊断。

| 方法 | 点到参考MAE（mm）↓ | 1mm参考召回↑ |
|---|---:|---:|
| 不处理 | 0.437246 | 92.358% |
| 冻结V18 | 0.459480 | 87.688% |
| APSS2 | 0.436517 | 92.169% |
| RIMLS2 | 0.437229 | 92.219% |
| V22预定主候选 | 0.434631 | 92.025% |
| 同位移RMS的常数阻尼对照 | 0.434344 | 92.037% |

主候选对不处理的MAE改善约0.598%，但召回与F-score下降；相对常数阻尼未显示独有收益。
预设5%改善信号未达到。不能把2.6微米的距离均值差解释为已经证明相同量级的物理精度提升。
42片区×9方法共378输出，当前输出指标独立重算最大差为1.11e-16。

## 可在GitHub直接阅读的运行证据

- [V19 摘要](evidence/progress_v22/runs/parallel-v19-9ob96cyr/SUMMARY.json)、[确认表](evidence/progress_v22/runs/parallel-v19-9ob96cyr/confirmation_ALL_RESULTS.csv)。
- [V20 独立复核](evidence/progress_v22/runs/candidate-repair-v20-qndibmbc/INDEPENDENT_REVIEW.json)、[确认表](evidence/progress_v22/runs/candidate-repair-v20-qndibmbc/confirmation_RESULTS.csv)、[完整已保存候选池诊断](evidence/progress_v22/runs/candidate-repair-v20-qndibmbc/posthoc_raw_R/COMPLETE_ARCHIVED_POOL_ORACLE.json)。后者使用GT事后选取，不能作为可部署算法成绩。
- [V21 摘要](evidence/progress_v22/runs/real-closure-v21-dhb1ebbx/SUMMARY.json)、[确认表](evidence/progress_v22/runs/real-closure-v21-dhb1ebbx/confirmation_RESULTS.json)。其中无有效参考支持的片区不是程序执行失败。
- [V22 摘要](evidence/progress_v22/runs/reconstruction-v22-qayc8gft/SUMMARY.json)、[全部确认分数](evidence/progress_v22/runs/reconstruction-v22-qayc8gft/confirmation_RESULTS.json)、[最终审计](evidence/progress_v22/runs/reconstruction-v22-qayc8gft/FINAL_AUDIT.json)。
- [新增导出清单与SHA256](evidence/progress_v22/EXPORT_MANIFEST.json)、[scan37下载来源](evidence/progress_v22/data_provenance/scan37/)。数据与逐点预测仍留在原机，不包含在Git中。
- AgentRx原文：[论文](https://arxiv.org/pdf/2602.02475v2)、[作者代码](https://github.com/microsoft/AgentRx)。本工程只提供适配审计，不提供官方模型运行结果。

## 验证例外与快照说明

V22原 `verify.py` 的历史APSS逐点重放检查未通过：最大坐标差约0.001318mm，超过其1e-5mm容差。
V18/RIMLS历史重放一致。此失败未删除、未修改容差；新增 `final_audit.py` 独立检查本轮实际输出并记录历史差异。
**不能将最终审计通过写成原验证器全部通过。** 原运行已有FINAL_AUDIT.json，脚本采用防覆盖写入，重跑前需按其输出约定处理，而不是直接覆盖历史证据。

本次发布只更新首页和仓库说明，未改实验算子、协议、评分或原运行记录。历史源锁包含这两份顶层文档，
所以发布后直接用旧源锁检查整个当前工作树会报告文档变化；其原始字节保留在
[发布前文档](evidence/progress_v22/pre_publication_docs/)，不伪造“所有当前文件哈希未变”。
旧V18提交 `2d9f7e5` 可回退查阅；本次新增提交保存V19–V22完整可提交源码及紧凑证据。

完整重跑仍依赖报告所列原机路径、Python环境和未随Git上传的数据。此仓库不是无需配置的一键复现包。
可先在安装NumPy/SciPy的环境运行最新算子测试：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m unittest discover -s reconstruction_v22 -p 'test_*.py' -v
```

本次发布不重新拟合模型，也不另写导师简报。后续研究决策以真实效果、同信息对照和结构代价共同判断。

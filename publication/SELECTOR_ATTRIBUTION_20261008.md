# 2026-10-08发布：轨迹证据与选择器贡献消融

源实验均在2026-10-07 UTC于liekkas完成；北京时间为10月8日。此发布只归档，不改算法或部署。

## 本次结论

固定两旧场景、四ROI、512请求/484原CPU有效主分母、21待处理位置和105候选。旧photo MSE为23.955251mm²；新star证据配均值P为16.722238（下降30.1939%），最坏收益R为16.856879（下降29.6318%）。P→门槛GP→R的增量分别−0.134641、0mm²；最强轨迹Q→R则增加0.147264mm²。不是普适因果贡献分账，不能把总改善都归功于新公式。

P四点改善、两点恶化、478点不变；仍伤害两个原本≤1mm的位置。旧法仅取消歧义拒绝会恶化105.0459%。前轮cycle约束撤回两次大修复，并不是总增益的贡献者。保留完整负结果和更正，默认identity不变。

- [最终报告](../research_snapshots/2026-10-07/selector-attribution-20261007T180539Z/REPORT.md)
- [结果](../research_snapshots/2026-10-07/selector-attribution-20261007T180539Z/evaluation/RESULTS.json)
- [贡献分解](../research_snapshots/2026-10-07/selector-attribution-20261007T180539Z/evaluation/ATTRIBUTION.csv)
- [审计](../research_snapshots/2026-10-07/selector-attribution-20261007T180539Z/EXPERIMENT_AUDIT.md)
- [前轮观测实验](../research_snapshots/2026-10-07/track-discrimination-20261007T160716Z/REPORT.md)

## 收录与排除

本次收录150个源文件，10,934,269字节：两轮代码、协议、输入/照片证据JSON、预测、CSV/JSON评价、理论及合成机制图、公开审计、报告与更正。源字节逐文件保留，仅AGENTS.md改名SOURCE_AGENTS.md。源/目的路径、大小、SHA256、58项排除及其原因见SELECTOR_ATTRIBUTION_20261008_MANIFEST.json。

不上传`.aris/`私有审稿提示/回复、bytecode缓存、NPY/NPZ等数组，亦未复制原始照片/激光数据、环境和GPU工作区。合成机制源码、摘要、场景描述和PNG保留，但合成数组缓存不收录；不能称本次为完整数据集或每份旧seal所指对象的全量备份。旧清单会引用未上传绝对路径；本次发布manifest才定义公开子集。许可证和第三方数据要求不变。

快照中的本地Markdown路径和`CHECKPOINT`“未上传”记录描述原实验状态，按证据原样保留；此发布文件描述后续上传事件。审计为same-family/provisional WARN，确定性计算核验通过不等于盲测或普适性。

## 可移植只读核验

仓库根目录，用Python标准库执行：

```bash
python -B publication/verify_selector_attribution_20261008.py
python -B publication/verify_selector_attribution_20261008.py --replay
```

第一条核对公开文件哈希、17臂×512逐点表、指标及5组贡献；第二条从归档GT-free INPUTS重现336个选择及84个历史ID断言。重放器仅将策略的历史kernel导入映射到随仓库发布的副本，原源码和原哈希不修改。公开入口不需要原机路径、激光数据或GPU，且阻止原机`/srv`和仓库外`/home`数据读取；这只是进程内读取约束，不是安全沙箱。

它不重新提取照片匹配、不重新做原始激光查询、不生成新实验。原机鲜审计已重新查询614个原始激光坐标行、123 active对象逐顶点穷举，最大差均0；公开核验器不冒充执行了这一原数据验证。实际本次发布测试见SELECTOR_ATTRIBUTION_20261008_VALIDATION.json。

此次只更新GitHub、实验导航和README；既有GitBook网页与局域网服务未重建。不包含工作树中既有的未跟踪第三方论文全文。

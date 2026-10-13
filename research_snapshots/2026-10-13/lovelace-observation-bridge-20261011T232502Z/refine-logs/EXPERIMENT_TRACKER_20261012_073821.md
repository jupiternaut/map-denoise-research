# Experiment Tracker

日期2026-10-12（Asia/Shanghai）。原目标：保留有效几何、修复不足。本批A+B仅定位修复力度损失，不改旧算法。

| 阶段 | 状态 | 实际证据 |
|---|---|---|
| M0 两作者实现与单测/冒烟 | RUNNING | runner源码已写，作者测试继续；独立verifier基础反例通过但fullschema未完成；根分析器状态分母夹具PASS |
| M1 非作者源码审查 | RUNNING | 隔离新代理规格SPEC PASS WITH IMPLEMENTATION GATES；最终源码审查尚未放行 |
| M2 固定正式批次 | NOT STARTED | 未生成正式输出；总5400秒主+核验窗口尚未启动 |
| M3 完整独立核验 | NOT STARTED | 待主批完整7776动作 |
| M4 配对结算报告 | PREPARED / NOT RUN | 独立分析器及事前识别说明已写，无新正式结果 |

当前修复：SUMMARY中的KEEP复用标签需为false；已通知作者修正与补反例。算法/参数不变。
无GPU、无安装、无API、无发布。历史根仍只读。

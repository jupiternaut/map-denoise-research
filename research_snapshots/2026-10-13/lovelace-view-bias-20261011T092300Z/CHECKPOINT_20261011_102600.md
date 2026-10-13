# 完成检查点

日期2026-10-11；host liekkas。
根目录：/srv/slam-research/grf/map-denoise/runs/lovelace-view-bias-20261011T092300Z。

## 完成

- 根科学.py及PROTOCOL运行锁未改，历史输入只读。
- 正式view-bias-v1：162图像、1458初态、5832动作，COMPLETE，1744.387秒。
- 独立核验：PASS，446.835秒，C1全部三档SUPPORTED。
- 两作者测试29项PASS；汇总状态回归2项PASS。
- 同图同界channel改善0；shared改善44/32/25、MAE↓6.57%/4.65%/2.66%，无新增损伤。
- strict全部不相容；channel白背景预筛先排除，共享传播没有运行。
- analysis-v1原件保留，字段更正后的analysis-v2是报告源。

## 阅读入口

REPORT.md、refine-logs/EXPERIMENT_RESULTS.md、COMMANDS.md。
数学与证据：MATHEMATICAL_NOTE.md、analysis/STRICT_FAILURE_AUDIT.md、verification/VERIFY.json（位于outputs/view-bias-v1）。

## 下一工作而非已测收益

完整消融审阅与本地可行性结算已完成，见NEXT_EXPERIMENT.md。下一主批先裁剪/宽界桥接与传播敏感性，随后公共视图偏差与简单背景校准匹配比较；全部六项候选保留。新算法尚未实现，后续应另开运行根，保持本批只读。不把旧收益差异全部归因β，不增加真实部署主张。

本批没有后台科学运行；核验和主批进程均已退出。没有GitHub/GitBook发布、GPU/API/安装或默认输出修改。私有.aris审阅追踪不公开。

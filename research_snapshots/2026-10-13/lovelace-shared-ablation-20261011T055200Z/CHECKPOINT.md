# 完成检查点

## 已完成

主机 liekkas，唯一目标 /srv/slam-research/grf/map-denoise/runs/lovelace-shared-ablation-20261011T055200Z。
六臂完整预算、独立核验、配对分析、中文报告已完成；没有运行中的实验进程。

关键结果：global_gray511 在 432 偏移初态上 115 改善 / 317 同分 / 0 恶化，MAE 1.164813（KEEP 1.5）。view 修复 52，RGB 94，pixel/channel 0。所有方法正确初态 54 KEEP、真深度覆盖 54、负控 72 KEEP、预算未决叶 0。完整预算没有改变旧 255 两臂的任何偏移输出。

## 文件与冻结

- REPORT.md 是完整结论入口；outputs/analysis-v1 是逐条件/配对分解。
- outputs/ablation-v1/verification/VERIFY.json 为 PASS。
- outputs/ablation-v1/SOURCE_LOCK.json 锁定 53 份科学源码/协议；不可修改根目录科学 .py 或 PROTOCOL.md 后仍沿用这份核验。
- 八份上轮输入和全部历史依赖只读，未覆盖。
- 后写报告与 analysis/summarize.py 不属于运行源锁，分析只读已验证数据；没有改变算法或阈值。
- .aris 是私有完整审查记录，不用于公开上传。

## 尚未运行

有界跨视图曝光/偏置压力、真正彩色生成、精确半空间替代、8轮/行序敏感性与真实场景迁移。不存在已完成这些工作的暗示。后续 planner 文件仅为计划，不能计入本批结果。

## 恢复与复现

先核对 hostname 和绝对目录，再读 REPORT.md、PROTOCOL.md、COMMANDS.md。
本批输出不可覆盖；需要重算时，为 run_ablation.py 指定本目录下全新输出子目录，然后为 verify_ablation.py 指定相同目录。
analysis/summarize.py 当前硬编码 ablation-v1 与 analysis-v1，且拒绝覆盖；只适用于本次封存的已完成分析。重算需要在新的 analysis 文件中明确来源与新目标，不能直接覆盖既有分析，也不能把本轮哈希套到改过的算法上。

## 研究决定

保留共同材料解释这一有效结构。下一优先级是外观变化的带界建模，而非重跑删除对照或改旧路由器。
真实部署默认、GitHub、GitBook均未改变；本轮未获发布授权。


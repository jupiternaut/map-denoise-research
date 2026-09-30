# 完成检查点

2026-10-01 01:09:49 +08:00；liekkas；本目录M0–M3完成，无遗留实验进程。

- 映射628afc09…d94cce冻结；AGENTS和EXPERIMENT_PLAN是映射来源的一部分，不再改写。里面的PLANNED是封存时状态，当前状态以本检查点与TRACKER为准。
- 固定512 query；497有效，15质量失败；24case和48个U×W单元预测完成。
- PREDICTIONS_SEALED.json与EVALUATION_PROTOCOL_LOCK.json在评价前创建。
- RESULTS.json含all_valid与paired_common；报告主表使用paired_common、四ROI等权。
- U1 native KEEP平均MSE150.121，对U0共同集453.456下降66.894%；C见证通过本回放联合验收，H未通过。
- 独立384项MSE等数值复算差0；fresh semantic reviewer创建失败如实记录。
- 下一步是正确相机下的成熟MVS初始化比较与后续独立场景验证，尚未执行。不改部署默认，没有Git push或GitBook发布。

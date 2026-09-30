# 审核状态

Fresh语义审稿：REVIEW_UNAVAILABLE。请求gpt-6-astra ultra、空历史代理时，工具返回`agent thread limit reached`；未将该失败写为PASS。

已有独立代理执行的确定性核验完成：旧162文件、新302预测文件、6评价锁；96条record、384项MSE、10,768数值字段均复算一致，最大差0。见evaluation/INDEPENDENT_NUMERIC_CHECK.json。该代理不是fresh，标签existing/same-family/provisional。

结果限于两个已曝光场景、四ROI、497/512固定像素。U整体变化含深度支持和8个context成员替换；W的候选与模型固定。评价使用数据集参考、原始mm²和KEEP为基准，无预测自归一化。

未改部署默认。完整内容性审稿仍可由后续新会话独立承担，当前不会以其不可用掩盖实测或捏造背书。


# A12 表示构造三臂试验

language: zh
host: liekkas

用户批准把数学表示本身作为搜索对象。本轮为独立的新工作区，不修改任何祖先实验。逻辑根 `/srv/slam-research/grf/map-denoise/runs/lovelace-representation-search-20261013T080031Z`；沿现有 runs 符号链接解析到 `/mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/` 下同名目录，不是另建副本。

范围：小规模 simulation_only 表示构造对照；只用已有 Python、单线程 CPU。无 GPU、付费 API、安装、部署或发布。原始物理模型及最终深度评价保持一致；允许表示与配套求解器改变。

角色信息边界：
- 自由构造者是 fork none 的新 Agent，仅阅读 `task_bundle/` 和自己的 `arms/free/`。不能读本目录其他臂、历史项目、隐藏评价或报告。
- 方法库选择者原拟另一新 Agent，但新建触及当时并发上限，已在正式前的EXPERIMENT_PLAN注册改为复用旧A11实现作者。可阅读 `task_bundle/`、`library/` 和自己的 `arms/library/`；不得读 free 臂或隐藏评价。其历史上下文在METHOD中披露，不声称三臂上下文对称。
- 固定臂适配器作者可读取冻结旧代码和本目录任务协议，只写其分派文件。
- 生成器、执行器与独立审查者按分工读取评价材料；正式全部预测保存后才打开 holdout 真值。

这是合作式信息隔离而非 OS 安全沙箱。记录实际暴露；不把一次构造试验说成通用发现能力或新数学定理。设计改动、失败和成本保留。编辑使用 apply_patch。

# 本轮交接

主机 liekkas；项目 /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1。
本轮V7主实验、真实表示诊断、72输出决策消融、数学/成本复核已结束。未开启新确认。

先读同目录 REPORT.md，再按需读 math/SEARCH_DIAGNOSIS.md、real_geometry/REPORT.md、
decision/REPORT.md。planning_v7 是原计划只读记录，不改其历史状态来抹去先后顺序。

保留原固定独立参考：24例表面MAE0.205928mm、对应RMS0.458664mm。
重关联b6硬refit为0.561227/0.795314mm；取消硬refit为0.403107/0.629086mm。
后者仅减少新方法损害，尚未优于旧法；层距另有代价。不要写“修复成功”。
同一真实面可按噪声分组，WLS再把组内非零误差均值固化。高GT组纯度不等于去噪好。
真实DA有局部轴表示失配，CY junction有多面线索；亚毫米留出残差不是恢复精度。

全部运行根：/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1
- 主例：reassociation-v7-primary-0frtnsm9（3例45输出，包含在后面的24例）
- 开发：reassociation-v7-development-vbwoojgf（24例360输出＋独立审计）
- 决策消融：decision-v7-nk7sm4yf（同24例72输出，先封存后评价）
- 真实诊断：real-geometry-v7-x5ximvmn；划分首版 rlp6r6j5 保留
- 成本：v7-warm-cost-ifprdsy7（输出一致，不是新增几何样本）
- 历史保护：session-v7-t01y9kft（8541旧文件未变；28项测试记录）

下一轮优先做单位法向＋偏移局部面的实际输出/注入恢复对照；若继续关联，先给出
不依赖同一点噪声重复自确认的跨来源证据。独立训练也不自动解决选择偏差。
不继续堆当前EM迭代、不把历史公开种子叫新确认，不默认启动GPU/新下载/Skill改写。

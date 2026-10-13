# CHECKPOINT：冻结确认全部完成

时间：2026-10-12 12:35 Asia/Shanghai。host=liekkas。
唯一新根：/srv/slam-research/grf/map-denoise/runs/lovelace-structured-bias-confirmation-20261012T035916Z。

## 用户目标与本轮范围

保留有效几何、修复不足。本轮是上轮冻结共同视图偏差算法的未见noise/pose种子确认，不是新算法开发。
seed61004、原prefix、六模板×三深度、九初态、clip一次、无额外calibration；KEEP/calibrated/joint三方法。旧根只读。

## 阶段完成

M0测试16/16 PASS；M1 fresh同族非作者审查PASS。
M2唯一正式批COMPLETE：18观测162任务、486动作、36冷求解。
M3独立机械核验PASS；一次1800秒原截止窗口内174.455秒完成。
M4配对结算、深度/模板异质性、冷成本、消融评估和已有动作轨迹读回完成。

科学文件冻结未改。收尾独立只读检查SOURCE_LOCK 78/78、INPUTS 4/4匹配；成功记录PASS，提前失败记录保留。该完整性读回不是新增实验。

## 主结论

144偏移初态，calibrated改善16，joint改善23；MSE 2.883046→2.653448 mm²（再下降7.96%）。
joint相对KEEP MSE下降22.10%，改善23/144=15.97%，未改善121/144=84.03%，恶化0。
“改善”指误差下降，不是精确恢复真值。18正确初态与24偏移负控全部KEEP；真深度/生成分支覆盖18/18。
增量6/7新增改善、96.62%配对MSE收益在第一档深度；第三档0改善且全部原初态仍在联合凸包内。cold CPU增加29.69%。

## 更正与范围

17/18观测时提前启动核验，被RUNNING状态拒绝。原FAIL保留在 outputs/confirmation-v1/verification-premature-attempt；主批完成后同一deadline内核验PASS，无求解重跑或源码修改。
同族合成一个新seed，不是独立真实场景、跨成像迁移、总体显著性或普遍零损伤保证。历史物理核仍共享，封存依赖本机历史路径。
无后台科学进程；未发布GitHub/GitBook，未部署，不调用GPU/API。

## 交付

- REPORT.md：最终目标—方法—结果—机制—下一步。
- PROTOCOL.md、REGISTRATION.json：首次新标签前冻结。
- outputs/confirmation-v1：全量输出、指标、成本、预算与核验。
- analysis/MECHANISM_TRACE_20261012_123000.json：现有18观测/162配对读回，0新求解。
- refine-logs/ABLATION_ASSESSMENT.md：已有关键删除对照与未来条件性问题。
- CORRECTION.md：提前核验的流程更正。
- MANIFEST.md、.aris：版本与完整审查请求/最终回复。

## 下一步边界

本轮注册确认结束；不追加seed、raw、轮数、LP或学习器。
若继续扩大科学主张，另立协议检验共同依赖在新成像条件中能否产生足够窄且正确的深度支持。现有“不动”不证明物理不可辨识。


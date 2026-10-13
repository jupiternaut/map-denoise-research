# A7 检查点：数值诊断有收获，正式批次未验收

2026-10-13 07:12 Asia/Shanghai；host liekkas。
root: /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1
output: /srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/common-pose-feasibility-1/outputs/pose-v1

原目标保留有效几何、修复不足。正式STAGE_STATUS=INCOMPLETE，不得改称COMPLETE/PASS或原输出补跑。

## 已做

- 54夹具PASS；同族暂定源码review PASS；161源码/9输入锁定。
- 单批324任务/972分支全部计算并保存36checkpoint及完整结果。
- 固定seed连续偏移精确检查；零偏移重拟合；共同合法偏移搜索+严格有理前向参数。
- 正式结束检查因root事后统计脚本临时出现在根目录 '*.py' 自动锁定集合而失败。原求解器/输入未改。脚本已移至refine-logs，当前锁集合恢复。
- 使用独立verify_pose的原核验原语作独立保存记录诊断；所有公共记录通过再读private覆盖。仅新诊断收据，未改原失败状态；仍用原3600秒截止，不重置。
- saved-record numeric audit=SAVED_RECORD_NUMERICAL_PASS，formal_acceptance=false；36真实前向生成参数全覆盖。

## 科学事实

主40：17完整错误解释、22UNKNOWN、1旧排除；零偏移6；固定seed0。
全部288错误初态：103完整解释、150UNKNOWN、35旧排除；零偏移45。
正确36：24完整解释、12UNKNOWN、0旧排除；事后真实参数36/36均有效。
全部324/972：127任务完整解释、162UNKNOWN、35旧排除；275/198/499分支。
所有473旧可行分支seed不能完成，但重拟合后275完整解释。无新排除、无新几何修复。

## 下一步（未执行）

新的外观锚定×相机锚定2×2：固定已验证错误解释，用额外可观测误差界分开信息与求解价值。不得把private真实参数当校准。先设计/夹具/review，另建阶段，不在本输出续写。

本阶段结束；没有后台任务、没有发布、旧A6封存未改。
